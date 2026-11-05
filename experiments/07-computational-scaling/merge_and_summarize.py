"""Merge computational-scaling results, attach optional GNU-time memory, and summarize.

Reads shards under outputs/p-*/seed-*/results.json, as written by
run_resource_grid.sh, when present; otherwise the single outputs/results.json
written by an unsharded run.py. Only shards carry GNU-time memory reports.

Usage from the project root:

    python experiments/07-computational-scaling/merge_and_summarize.py

By default the merger requires all ten seeds for Secondary STK, while direct
Model-X may be incomplete at large p when it exceeds its time or memory limit.
run.py stops each arm at TIMEOUT_<ARM> seconds (defaults: stk_secondary 14400,
direct_model_x 7200) and records it as failed; DIRECT_MODEL_X_MEMORY_GB
optionally caps direct Model-X memory on Linux (default 0, no cap).

In the summary, "complete" counts successful runs, which give |S|, FDP, power,
and candidate sizes. "failed" counts failed and timed-out runs. time_s (mean)
and max_GiB (maximum) include the failed runs: their time to failure or timeout
and their memory up to that point. max_GiB is GNU time's peak resident set size
of the largest single process, not the sum over worker processes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", nargs="?", default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--expected-seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--expected-p", default="500,2000,5000,20000")
    parser.add_argument("--regular-arms", default="stk_secondary")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "results.json"))
    return parser.parse_args()


def parse_max_rss(path: Path) -> int | None:
    if not path.exists():
        return None
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text())
    return int(match.group(1)) if match else None


def mean(values):
    return sum(values) / len(values) if values else float("nan")


args = parse_args()
base = Path(args.outputs)
expected_seeds = {int(value) for value in args.expected_seeds.split(",") if value}
expected_p = {int(value) for value in args.expected_p.split(",") if value}
regular_arms = tuple(value for value in args.regular_arms.split(",") if value)
files = [
    path
    for path in sorted(base.glob("p-*/seed-*/results.json"))
    if int(path.parent.parent.name.removeprefix("p-")) in expected_p
]
if not files and (base / "results.json").exists():
    files = [base / "results.json"]
if not files:
    raise RuntimeError(f"no results under {base}/p-*/seed-*/results.json or {base}/results.json")

merged = None
arms_run: list[str] = []
for path in files:
    data = json.loads(path.read_text())
    if merged is None:
        merged = {"meta": dict(data["meta"]), "cells": {}}
    for key in ("stk_commit", "snapshot_id", "run_sha256", "lib_sha256", "n", "k", "snr", "rho"):
        if data["meta"].get(key) != merged["meta"].get(key):
            raise RuntimeError(f"metadata mismatch for {key} in {path}")
    for cell_key, cell in data["cells"].items():
        metadata = {key: value for key, value in cell.items() if key != "seeds"}
        merged_cell = merged["cells"].setdefault(cell_key, {**metadata, "seeds": {}})
        overlap = set(merged_cell["seeds"]) & set(cell["seeds"])
        if overlap:
            raise RuntimeError(f"duplicate seeds for {cell_key}: {sorted(overlap)}")
        for seed, record in cell["seeds"].items():
            for arm, arm_record in record["arms"].items():
                rss = parse_max_rss(path.parent / f"resources-{arm}.txt")
                if rss is not None:
                    arm_record["max_rss_kb"] = rss
            merged_cell["seeds"][seed] = record
    for arm in data["meta"].get("arms_run", []):
        if arm not in arms_run:
            arms_run.append(arm)

assert merged is not None
if {cell["p"] for cell in merged["cells"].values()} != expected_p:
    raise RuntimeError(f"p-grid mismatch: got {sorted(cell['p'] for cell in merged['cells'].values())}")
for cell_key, cell in merged["cells"].items():
    seeds = {int(seed) for seed in cell["seeds"]}
    if seeds != expected_seeds:
        raise RuntimeError(f"{cell_key} seeds={sorted(seeds)}, expected={sorted(expected_seeds)}")
    for seed, record in cell["seeds"].items():
        missing = set(regular_arms) - set(record["arms"])
        if missing:
            raise RuntimeError(f"{cell_key}, seed={seed} missing regular arms {sorted(missing)}")

merged["meta"]["p_values"] = sorted(expected_p)
merged["meta"]["seeds"] = sorted(expected_seeds)
merged["meta"]["arms_run"] = arms_run
merged["meta"]["merged_from"] = [str(path.relative_to(base)) for path in files]
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(merged, indent=2))
print(f"merged {len(files)} result files -> {output}\n")

header = (
    f"{'p':>6s} {'arm':14s} {'complete':>8s} {'failed':>6s} {'valid_mx':>8s} {'|S|':>6s} "
    f"{'FDP':>7s} {'power':>7s} {'time_s':>9s} {'cand':>8s} {'max_GiB':>8s}"
)
print(header)
print("-" * len(header))
for cell in sorted(merged["cells"].values(), key=lambda item: item["p"]):
    observed_arms = sorted({arm for record in cell["seeds"].values() for arm in record["arms"]})
    for arm in observed_arms:
        records = [record["arms"][arm] for record in cell["seeds"].values() if arm in record["arms"]]
        successful = [record for record in records if record.get("ok")]
        valid = [record for record in successful if record.get("valid_model_x", True)]
        candidates = [mean(record["candidate_sizes"]) for record in successful if record.get("candidate_sizes")]
        rss = [record["max_rss_kb"] / 1024**2 for record in records if record.get("max_rss_kb")]
        print(
            f"{cell['p']:6d} {arm:14s} {len(successful):8d} {len(records) - len(successful):6d} {len(valid):8d} "
            f"{mean([record['n_selected'] for record in successful]):6.1f} "
            f"{mean([record['fdp'] for record in successful]):7.3f} "
            f"{mean([record['power'] for record in successful]):7.2f} "
            f"{mean([record['elapsed_s'] for record in records if record.get('elapsed_s') is not None]):9.1f} "
            f"{mean(candidates):8.1f} {max(rss) if rss else float('nan'):8.2f}"
        )
