"""Matched aggregation: aggregate STK support versus its own individual splits.

Run from the project root (see experiments/README.md for the full commands):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python experiments/03-matched-aggregation/run.py --seeds 42,43

Each normal multi-split fit records its individual held-out selections. These
are the matched single-split outputs: the aggregate uses exactly the same fits.
This compares a random single split at q_split=0.5 with frequency aggregation,
not a separately calibrated single-split method at aggregate FDR 0.1.

Defaults are the reported signal grid: n=800, p=5000, rho=0.5, total SNR=0.5,
supports 1,2,3,5,10,15,20, both multi-split presets (headline, secondary), and
outputs under outputs/full. The null grid uses --supports 0 with 30 seeds.
k=0 uses the existing independent-outcome global-null generator.
No lineage component is present in these Gaussian generators.
--smoke deliberately reduces fitting budgets and dimensions for validation only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from monitor import monitor_lasso, stage_summary

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LIB_DIR = HERE.parent
sys.path.insert(0, str(LIB_DIR))
import _lib  # noqa: E402

from select_then_knock import SelectThenKnock  # noqa: E402


class RecordedSTK(SelectThenKnock):
    """Observe the existing split implementation; do not replace its logic."""

    def _one_split(self, X_arr, y_arr, cut, split_seed, per_run_fdr, knockoff_filter):
        observed = {}

        def observed_filter(X, y, **kwargs):
            result = knockoff_filter(X, y, **kwargs)
            observed.update(
                candidate_count=int(X.shape[1]),
                testing_rows=int(X.shape[0]),
                fallback_used=bool(result.fallback_used),
            )
            return result

        started = time.perf_counter()
        selected = super()._one_split(X_arr, y_arr, cut, split_seed, per_run_fdr, observed_filter)
        if not hasattr(self, "recorded_splits_"):
            self.recorded_splits_ = []
        self.recorded_splits_.append(
            {
                "split_seed": int(split_seed),
                "screening_rows": int(cut),
                "selected": np.asarray(selected, dtype=int).tolist(),
                "elapsed_s": time.perf_counter() - started,
                "candidate_count": observed.get("candidate_count", 0),
                "testing_rows": observed.get("testing_rows", int(X_arr.shape[0] - cut)),
                "fallback_used": observed.get("fallback_used"),
                "held_out_filter_called": bool(observed),
            }
        )
        return selected


def support_metrics(selected, support):
    selected, support = set(map(int, selected)), set(map(int, support))
    tp = len(selected & support)
    fp = len(selected - support)
    return {
        "n_selected": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": len(support - selected),
        "fdp": fp / max(1, len(selected)),
        "power": tp / len(support) if support else None,
        "exact_recovery": int(selected == support),
        "firing": int(bool(selected)),
    }


def aggregate_support(splits, p, eta):
    if not splits:
        raise ValueError("Cannot aggregate zero splits")
    counts = np.zeros(p, dtype=int)
    for split in splits:
        indices = np.asarray(split["selected"], dtype=int)
        if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= p):
            raise ValueError("Invalid Original-space split selections")
        counts[indices] += 1
    return np.flatnonzero(counts / len(splits) >= eta), counts


def summarize_cell(splits, selected, support):
    metrics = [support_metrics(split["selected"], support) for split in splits]
    means, sds = {}, {}
    for key in metrics[0]:
        values = [row[key] for row in metrics if row[key] is not None]
        means[key] = float(np.mean(values)) if values else None
        sds[key] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    aggregate = support_metrics(selected, support)
    return {
        "single_split_mean": means,
        "single_split_sd_within_dataset": sds,
        "aggregate": aggregate,
        "paired_difference_aggregate_minus_single_mean": {
            key: aggregate[key] - means[key] if means[key] is not None else None for key in aggregate
        },
        "held_out_fallback_count": sum(split["fallback_used"] is True for split in splits),
        "splits_without_held_out_filter": sum(not split["held_out_filter_called"] for split in splits),
        "mean_single_split_core_elapsed_s": float(np.mean([split["elapsed_s"] for split in splits])),
    }


def fingerprint_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def provenance():
    import select_then_knock

    source = Path(select_then_knock.__file__).resolve().parent
    return {
        "commit": _lib.git_sha(),
        "source_path": str(source),
        "source_sha256": {str(path.relative_to(source)): file_hash(path) for path in sorted(source.rglob("*.py"))},
        "runner_sha256": file_hash(Path(__file__)),
        "monitor_sha256": file_hash(HERE / "monitor.py"),
        "lib_sha256": file_hash(LIB_DIR / "_lib.py"),
        "git_status": _lib.git_status(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_build": np.show_config(mode="dicts"),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")},
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    }


def run_cell(args, seed, k, preset, source):
    if k == 0:
        data = _lib.make_null_dataset(args.p, args.rho, seed, n=args.n)
    else:
        data = _lib.generate_blocked_synthetic(
            n=args.n,
            n_blocks=args.p // 20,
            block_size=20,
            rho=args.rho,
            k_true=k,
            beta_true=_lib.beta_for_k(k),
            snr=args.snr,
            seed=seed,
        )
    X, y, support = data["X"], data["y"], data["S_star"]
    overrides = {
        "random_state": seed,
        "run_diagnostics": False,
        "solver_backend": args.solver,
        "n_jobs": args.n_jobs,
        "n_splits": args.n_splits,
    }
    if args.smoke:
        overrides.update(n_bootstrap=4, n_alphas=3, n_l1_ratios=2, cv_folds=2)
    model = RecordedSTK.from_preset(preset, task="regression", **overrides)
    signature = {
        "design": {"n": args.n, "p": args.p, "rho": args.rho, "snr": args.snr if k else None, "k": k},
        "dataset_seed": seed,
        "split_base_seed": seed,
        "preset": preset,
        "smoke_only": args.smoke,
        "parameters": model.get_params(),
        "dataset_sha256": fingerprint_arrays(X, y, support),
        "source_sha256": source["source_sha256"],
        "runner_sha256": source["runner_sha256"],
        "monitor_sha256": source["monitor_sha256"],
        "numpy_build": source["numpy_build"],
        "lib_sha256": source["lib_sha256"],
        "python": source["python"],
        "packages": source["packages"],
        "thread_environment": source["thread_environment"],
    }
    # Normalize tuples in sklearn parameters before comparing to loaded JSON.
    signature = json.loads(json.dumps(signature))
    path = args.out / f"k-{k}" / f"seed-{seed}" / f"{preset}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["signature"] != signature:
            raise ValueError(f"Refusing to reuse mismatched result: {path}")
        if existing.get("complete"):
            print(f"SKIP {path}", flush=True)
            return
    record = {"signature": signature, "provenance": source, "S_star": support.tolist(), "complete": False}
    write_json(path, record)
    started = time.perf_counter()
    lasso_fits = []
    with monitor_lasso(lasso_fits):
        model.fit(X, y)
    elapsed = time.perf_counter() - started
    splits = model.recorded_splits_
    selected, counts = aggregate_support(splits, args.p, model.eta)
    public_selected = model.get_support(indices=True, space="original")
    if len(splits) != model.n_splits or not np.array_equal(selected, public_selected):
        raise AssertionError("Recorded split aggregation does not reproduce the public estimator output")
    np.testing.assert_array_equal(counts / model.n_splits, model.selection_frequencies_)
    for split in splits:
        split["metrics"] = support_metrics(split["selected"], support)
    record.update(
        complete=True,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_s=elapsed,
        splits=splits,
        selected=selected.tolist(),
        pi_nonzero={str(j): float(counts[j] / model.n_splits) for j in np.flatnonzero(counts)},
        summary=summarize_cell(splits, selected, support),
        lasso_fits=lasso_fits,
        convergence_by_stage=stage_summary(lasso_fits),
    )
    write_json(path, record)
    summary = record["summary"]
    print(
        f"DONE k={k} seed={seed} {preset} elapsed={elapsed:.1f}s "
        f"single={summary['single_split_mean']} aggregate={summary['aggregate']} "
        f"fallbacks={summary['held_out_fallback_count']}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--supports", default="1,2,3,5,10,15,20")
    parser.add_argument("--presets", default="headline,secondary")
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--p", type=int, default=5000)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--snr", type=float, default=0.5)
    parser.add_argument("--n-splits", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--solver", choices=("sklearn", "celer"), default="celer")
    parser.add_argument("--out", type=Path, default=HERE / "outputs" / "full")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.seeds = [int(value) for value in args.seeds.split(",")]
    args.supports = [int(value) for value in args.supports.split(",")]
    args.presets = args.presets.split(",")
    if args.p <= 0 or args.p % 20 or any(k < 0 or k > args.p // 20 for k in args.supports):
        parser.error("p must be positive and divisible by 20; k must fit in the number of blocks")
    if args.n < 10 or args.n_splits < 1 or not 0 <= args.rho < 1 or args.snr <= 0:
        parser.error("Invalid sample count, partition count, correlation, or SNR")
    if not set(args.presets) <= {"headline", "secondary"}:
        parser.error("Use the headline and/or secondary presets")
    return args


def main():
    args = parse_args()
    source = provenance()
    if args.solver == "celer":
        source["packages"]["celer"] = importlib.metadata.version("celer")
    print(f"Matched aggregation: seeds={args.seeds} k={args.supports} presets={args.presets}", flush=True)
    for k in args.supports:
        for seed in args.seeds:
            for preset in args.presets:
                print(f"START k={k} seed={seed} preset={preset}", flush=True)
                run_cell(args, seed, k, preset, source)


if __name__ == "__main__":
    main()
