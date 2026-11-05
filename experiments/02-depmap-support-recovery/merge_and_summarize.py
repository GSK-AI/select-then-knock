"""Merge DepMap support-recovery results and summarize.

Reads seed shards under outputs/k-*/seed-*/results.json when present, otherwise
the single outputs/results.json written by an unsharded run.py.

Usage from the project root:

    python experiments/02-depmap-support-recovery/merge_and_summarize.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", nargs="?", default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--expected-seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--expected-k", default="3,10")
    parser.add_argument("--arms", default="lasso_cv,stk_pooled,stk_headline,stk_secondary,stabl")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "results.json"))
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values) if values else float("nan")


args = parse_args()
base = Path(args.outputs)
expected_seeds = {int(value) for value in args.expected_seeds.split(",") if value}
expected_k = {int(value) for value in args.expected_k.split(",") if value}
expected_arms = tuple(value for value in args.arms.split(",") if value)
files = sorted(base.glob("k-*/seed-*/results.json"))
if not files and (base / "results.json").exists():
    files = [base / "results.json"]
if not files:
    raise RuntimeError(f"no results under {base}/k-*/seed-*/results.json or {base}/results.json")

merged = None
arms_run: list[str] = []
metadata_keys = (
    "stk_commit",
    "snapshot_id",
    "run_sha256",
    "lib_sha256",
    "depmap_release",
    "data_sha256",
    "matrix_sha256",
    "row_ids_sha256",
    "feature_names_sha256",
    "n",
    "p",
    "row_seed",
    "snr",
    "min_support_std",
    "max_support_abs_corr",
)
for path in files:
    data = json.loads(path.read_text())
    if merged is None:
        merged = {"meta": dict(data["meta"]), "cells": {}}
    for key in metadata_keys:
        if data["meta"].get(key) != merged["meta"].get(key):
            raise RuntimeError(f"metadata mismatch for {key} in {path}")
    for cell_key, cell in data["cells"].items():
        merged_cell = merged["cells"].setdefault(cell_key, {"k": cell["k"], "seeds": {}})
        overlap = set(merged_cell["seeds"]) & set(cell["seeds"])
        if overlap:
            raise RuntimeError(f"duplicate seeds for {cell_key}: {sorted(overlap)}")
        merged_cell["seeds"].update(cell["seeds"])
    for arm in data["meta"].get("arms_run", []):
        if arm not in arms_run:
            arms_run.append(arm)

assert merged is not None
if {cell["k"] for cell in merged["cells"].values()} != expected_k:
    raise RuntimeError(f"k-grid mismatch: got {sorted(cell['k'] for cell in merged['cells'].values())}")
for cell_key, cell in merged["cells"].items():
    seeds = {int(seed) for seed in cell["seeds"]}
    if seeds != expected_seeds:
        raise RuntimeError(f"{cell_key} seeds={sorted(seeds)}, expected={sorted(expected_seeds)}")
    for seed, record in cell["seeds"].items():
        missing = set(expected_arms) - set(record["arms"])
        if missing:
            raise RuntimeError(f"{cell_key}, seed={seed} missing arms {sorted(missing)}")
        for arm in (item for item in ("stk_headline", "stk_secondary") if item in expected_arms):
            arm_record = record["arms"][arm]
            if arm_record.get("ok") and arm_record.get("preset") != arm.removeprefix("stk_"):
                raise RuntimeError(f"{cell_key}, seed={seed}, arm={arm} has preset={arm_record.get('preset')}")
            if arm_record.get("ok") and arm_record.get("run_diagnostics") is not False:
                raise RuntimeError(f"{cell_key}, seed={seed}, arm={arm} ran diagnostics")

merged["meta"]["k_values"] = sorted(expected_k)
merged["meta"]["seeds"] = sorted(expected_seeds)
merged["meta"]["arms_run"] = arms_run
merged["meta"]["merged_from"] = [str(path.relative_to(base)) for path in files]
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(merged, indent=2))
print(f"merged {len(files)} shards -> {output}\n")

header = (
    f"{'k':>3s} {'arm':14s} {'ok':>3s} {'|S|':>6s} {'FDP_mean':>9s} {'FDP_max':>8s} "
    f"{'power':>7s} {'exact':>7s} {'time_s':>9s}"
)
print(header)
print("-" * len(header))
for cell in sorted(merged["cells"].values(), key=lambda item: item["k"]):
    for arm in expected_arms:
        records = [record["arms"][arm] for record in cell["seeds"].values()]
        successful = [record for record in records if record.get("ok")]
        print(
            f"{cell['k']:3d} {arm:14s} {len(successful):3d} "
            f"{mean([record['n_selected'] for record in successful]):6.1f} "
            f"{mean([record['fdp'] for record in successful]):9.3f} "
            f"{max((record['fdp'] for record in successful), default=float('nan')):8.3f} "
            f"{mean([record['power'] for record in successful]):7.2f} "
            f"{mean([float(record['exact_recovery']) for record in successful]):7.2f} "
            f"{mean([record['elapsed_s'] for record in records if record.get('elapsed_s') is not None]):9.1f}"
        )
