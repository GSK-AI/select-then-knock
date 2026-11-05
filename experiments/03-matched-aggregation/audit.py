"""Recompute saved metrics; optionally compare with Gaussian support-recovery records.

Run from the project root:
    python experiments/03-matched-aggregation/audit.py \
      --input experiments/03-matched-aggregation/outputs/full

The optional reference records are seed directories written by
01-gaussian-support-recovery (arm keys stk_headline and stk_secondary).

Frozen comparisons are diagnostic: different numerical environments can change
frequencies even when the code and statistical settings match. This script does
not overwrite experimental records or silently substitute frozen results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from monitor import stage_summary
from run import aggregate_support, file_hash, summarize_cell, support_metrics, write_json


def audit(root, require_full=False, reference_results=None):
    if reference_results is not None and not reference_results.is_dir():
        raise ValueError("reference_results must be an existing directory of Gaussian support-recovery seed records")
    checks = []
    cells = set()
    paired_hashes = {}
    final_fit_flags = []
    for path in sorted(root.glob("k-*/seed-*/*.json")):
        record = json.loads(path.read_text())
        assert record["complete"], f"Incomplete record: {path}"
        signature = record["signature"]
        params, design = signature["parameters"], signature["design"]
        splits, support = record["splits"], record["S_star"]
        assert len(splits) == params["n_splits"]
        selected, counts = aggregate_support(splits, design["p"], params["eta"])
        assert selected.tolist() == record["selected"]
        assert {str(j): float(counts[j] / len(splits)) for j in np.flatnonzero(counts)} == record["pi_nonzero"]
        expected_seeds = np.random.SeedSequence(signature["split_base_seed"]).generate_state(len(splits))
        assert [split["split_seed"] for split in splits] == expected_seeds.tolist()
        for split in splits:
            assert split["metrics"] == support_metrics(split["selected"], support)
        assert record["summary"] == summarize_cell(splits, selected, support)
        if "lasso_fits" in record:
            assert record["convergence_by_stage"] == stage_summary(record["lasso_fits"])
            for row in record["lasso_fits"]:
                if row["stage"] == "held_out" and (not row["ok"] or row["final_n_iter"] >= row["max_iter"]):
                    final_fit_flags.append({"path": str(path), "fit": row})
        check = {"path": str(path), "sha256": file_hash(path), "metrics_recomputed": True}
        k, seed, preset = design["k"], signature["dataset_seed"], signature["preset"]
        assert (k, seed, preset) not in cells, "Duplicate cell"
        cells.add((k, seed, preset))
        if (k, seed) in paired_hashes:
            assert paired_hashes[k, seed] == signature["dataset_sha256"], "Configuration inputs differ"
        paired_hashes[k, seed] = signature["dataset_sha256"]
        if require_full:
            assert not signature["smoke_only"]
            assert design == {"n": 800, "p": 5000, "rho": 0.5, "snr": 0.5 if k else None, "k": k}
            expected = {
                "n_splits": 25,
                "n_bootstrap": 200,
                "n_alphas": 20,
                "n_l1_ratios": 3,
                "cv_folds": 3,
                "per_run_fdr": 0.5,
                "fdr": 0.1,
                "min_knockoff_candidates": 10 if preset == "headline" else 20,
                "eta": 0.5,
                "knockoff_offset": 1,
                "run_diagnostics": False,
                "solver_backend": "celer",
                "alpha_range": [0.005, 0.5],
                "split_frac": 0.5 if preset == "headline" else 0.7,
            }
            assert all(params[key] == value for key, value in expected.items()), "Unexpected full-run settings"
            minimum = expected["min_knockoff_candidates"]
            for split in splits:
                assert split["candidate_count"] >= minimum
                assert split["screening_rows"] == round(800 * expected["split_frac"])
                assert split["testing_rows"] + split["screening_rows"] == 800
        frozen_path = reference_results / f"seed-{seed}" / "results.json" if reference_results is not None else None
        if reference_results is not None and not frozen_path.is_file():
            check["reference_record_missing"] = str(frozen_path)
        if (
            design == {"n": 800, "p": 5000, "rho": 0.5, "snr": 0.5, "k": k}
            and frozen_path is not None
            and frozen_path.exists()
        ):
            frozen = json.loads(frozen_path.read_text())
            original = frozen["cells"][f"hard_rho05/k={k}"]["seeds"][str(seed)]
            arm = original["arms"][f"stk_{preset}"]
            check["frozen_comparison"] = {
                "k": k,
                "seed": seed,
                "preset": preset,
                "path": str(frozen_path),
                "sha256": file_hash(frozen_path),
                "same_dataset_hash": original["dataset_sha256"] == signature["dataset_sha256"],
                "same_true_support": original["S_star"] == support,
                "same_selected_support": arm["selected_indices"] == record["selected"],
                "same_frequencies": arm["pi_nonzero"] == record["pi_nonzero"],
                "same_per_split_sizes": arm["per_split_sizes"] == [len(split["selected"]) for split in splits],
                "added_features": sorted(set(record["selected"]) - set(arm["selected_indices"])),
                "removed_features": sorted(set(arm["selected_indices"]) - set(record["selected"])),
                "fdp_difference": record["summary"]["aggregate"]["fdp"] - arm["fdp"],
                "power_difference": record["summary"]["aggregate"]["power"] - arm["power"],
            }
        checks.append(check)
    if not checks:
        raise ValueError("No records to audit")
    if require_full:
        expected_cells = {
            (k, seed, preset)
            for k in (1, 2, 3, 5, 10, 15, 20)
            for seed in range(42, 52)
            for preset in ("headline", "secondary")
        } | {(0, seed, preset) for seed in range(42, 72) for preset in ("headline", "secondary")}
        assert cells == expected_cells, (
            f"Grid mismatch: missing={expected_cells - cells}, extra={cells - expected_cells}"
        )
        # A plain runner produces no sweep progress file. Check the complete grid
        # above, and require identical recorded source fingerprints instead.
        source_maps = {
            json.dumps(json.loads(Path(check["path"]).read_text())["provenance"]["source_sha256"], sort_keys=True)
            for check in checks
        }
        assert len(source_maps) == 1, "Full-grid records have different source fingerprints"
    log = root.parent / f"{root.name}.log"
    report = {"checks": checks, "full_grid_verified": require_full, "held_out_final_fit_flags": final_fit_flags}
    if log.exists():
        text = log.read_text()
        report["log"] = {
            "path": str(log),
            "sha256": file_hash(log),
            "convergence_warning_lines": text.count("ConvergenceWarning:"),
            "note": "Warning lines are not counts of failed fits; CV path warnings may repeat.",
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument(
        "--reference-results", type=Path, help="Optional Gaussian support-recovery (01) seed-record directory"
    )
    args = parser.parse_args()
    report = audit(args.input, require_full=args.require_full, reference_results=args.reference_results)
    write_json(args.input / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
