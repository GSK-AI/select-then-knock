"""Merge Gaussian support-recovery results into outputs/results.json and print the summary.

Reads seed shards under outputs/seed-*/results.json when present; otherwise it
validates the single outputs/results.json written by an unsharded run.py.

Usage from the project root:

    python experiments/01-gaussian-support-recovery/merge_and_summarize.py [outputs_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REGIME = "hard_rho05"

base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "outputs"
files = sorted(base.glob("seed-*/results.json"))
if not files and (base / "results.json").exists():
    files = [base / "results.json"]
if not files:
    raise RuntimeError(f"no results under {base}/seed-*/results.json or {base}/results.json")

merged = None
seeds: set[int] = set()
arms_run: list[str] = []
expected_arms = {"lasso_cv", "stk_pooled", "stk_headline", "stk_secondary", "stabl"}
multi_split_presets = {"stk_headline": "headline", "stk_secondary": "secondary"}
for path in files:
    data = json.loads(path.read_text())
    if merged is None:
        merged = {"meta": dict(data["meta"]), "cells": {}}
    for key in ("stk_commit", "snapshot_id", "run_sha256", "lib_sha256"):
        if data["meta"].get(key) != merged["meta"].get(key):
            raise RuntimeError(f"metadata mismatch for {key} in {path}")
    for cell_key, cell in data["cells"].items():
        cell_metadata = {key: value for key, value in cell.items() if key != "seeds"}
        merged_cell = merged["cells"].setdefault(cell_key, cell_metadata)
        merged_cell.setdefault("seeds", {})
        overlap = set(merged_cell["seeds"]) & set(cell["seeds"])
        if overlap:
            raise RuntimeError(f"duplicate seeds for {cell_key}: {sorted(overlap)}")
        merged_cell["seeds"].update(cell["seeds"])
    seeds.update(int(seed) for seed in data["meta"].get("seeds", []))
    for arm in data["meta"].get("arms_run", []):
        if arm not in arms_run:
            arms_run.append(arm)

assert merged is not None
merged["meta"]["seeds"] = sorted(seeds)
merged["meta"]["arms_run"] = arms_run
merged["meta"]["merged_from"] = [str(path.relative_to(base)) for path in files]

expected_cells = {f"{REGIME}/k={k}" for k in (1, 2, 3, 5, 10, 15, 20)}
if set(merged["cells"]) != expected_cells:
    raise RuntimeError(f"cell mismatch: got {sorted(merged['cells'])}")
for cell_key, cell in merged["cells"].items():
    if len(cell["seeds"]) != 10:
        raise RuntimeError(f"{cell_key} has {len(cell['seeds'])} seeds, expected 10")
    for seed, record in cell["seeds"].items():
        missing = expected_arms - set(record["arms"])
        if missing:
            raise RuntimeError(f"{cell_key}, seed={seed} missing arms {sorted(missing)}")
        for arm, preset in multi_split_presets.items():
            arm_record = record["arms"][arm]
            if arm_record.get("ok") and arm_record.get("preset") != preset:
                raise RuntimeError(f"{cell_key}, seed={seed}, arm={arm} has preset={arm_record.get('preset')}")
            if arm_record.get("ok") and arm_record.get("run_diagnostics") is not False:
                raise RuntimeError(f"{cell_key}, seed={seed}, arm={arm} ran diagnostics")

output = base / "results.json"
output.write_text(json.dumps(merged, indent=2))
print(f"merged {len(files)} files -> {output}\n")


def mean(values):
    return sum(values) / len(values) if values else float("nan")


header = (
    f"{'cell':18s} {'arm':13s} {'|S|':>5s} {'FDP_mean':>9s} {'FDP_max':>8s} "
    f"{'pow_mean':>8s} {'pow_min':>7s} {'exact':>7s} {'t_mean':>7s} {'ok':>3s}"
)
print(header)
print("-" * len(header))
for cell_key, cell in merged["cells"].items():
    for arm in arms_run:
        successful = [record["arms"][arm] for record in cell["seeds"].values() if record["arms"][arm].get("ok")]
        if not successful:
            missing_values = " ".join((f"{'--':>5s}", f"{'--':>9s}", *(f"{'--':>8s}" for _ in range(5))))
            print(f"{cell_key:18s} {arm:13s} {missing_values} {0:>3d}")
            continue
        print(
            f"{cell_key:18s} {arm:13s} "
            f"{mean([record['n_selected'] for record in successful]):5.1f} "
            f"{mean([record['fdp'] for record in successful]):9.3f} "
            f"{max(record['fdp'] for record in successful):8.3f} "
            f"{mean([record['power'] for record in successful]):8.2f} "
            f"{min(record['power'] for record in successful):7.2f} "
            f"{mean([float(record['exact_recovery']) for record in successful]):7.2f} "
            f"{mean([record['elapsed_s'] for record in successful]):7.0f} "
            f"{len(successful):3d}"
        )
