"""Merge Gaussian global-null results into outputs/results.json and print the summary.

Reads seed shards under outputs/seed-*/results.json when present; otherwise it
validates the single outputs/results.json written by an unsharded run.py.
Cells are keyed by their label (rho=0.0, p=500, ...). The summary reprints
run.py's P(fire) and mean/max |S_hat| table.

Usage from the project root:

    python experiments/06-gaussian-global-null/merge_and_summarize.py [outputs_dir]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import _summary  # noqa: E402

base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "outputs"
files = sorted(glob.glob(str(base / "seed-*/results.json")))
if not files and (base / "results.json").exists():
    files = [str(base / "results.json")]
if not files:
    raise RuntimeError(f"no results under {base}/seed-*/results.json or {base}/results.json")

first = json.loads(Path(files[0]).read_text())
order = [c["cell"] for c in first["cells"]]
by_key: dict = {}
seeds: set[int] = set()
expected_arms = {"lasso_cv", "stk_pooled", "stk_headline", "stk_secondary", "stabl"}

for f in files:
    d = json.loads(Path(f).read_text())
    for c in d["cells"]:
        key = c["cell"]
        if key not in by_key:
            by_key[key] = {k: v for k, v in c.items() if k != "seeds"}
            by_key[key]["seeds"] = {}
        overlap = set(by_key[key]["seeds"]) & set(c["seeds"])
        if overlap:
            raise RuntimeError(f"duplicate seeds for {key}: {sorted(overlap)}")
        by_key[key]["seeds"].update(c["seeds"])
    seeds.update(int(s) for s in d["meta"].get("seeds", []))

merged = {"meta": dict(first["meta"]), "cells": [by_key[k] for k in order]}
merged["meta"]["seeds"] = sorted(seeds)
merged["meta"]["merged_from"] = [str(Path(f).relative_to(base)) for f in files]

if seeds != set(range(42, 72)):
    raise RuntimeError(f"seed mismatch: got {sorted(seeds)}")
for cell in merged["cells"]:
    if set(int(seed) for seed in cell["seeds"]) != seeds:
        raise RuntimeError(f"{cell['cell']} has an incomplete seed grid")
    for seed, record in cell["seeds"].items():
        if not record.get("dataset_sha256"):
            raise RuntimeError(f"{cell['cell']}, seed={seed} missing dataset fingerprint")
        missing = expected_arms - set(record["arms"])
        if missing:
            raise RuntimeError(f"{cell['cell']}, seed={seed} missing arms {sorted(missing)}")
        for arm in ("stk_headline", "stk_secondary"):
            arm_record = record["arms"][arm]
            if arm_record.get("ok") and arm_record.get("preset") != arm.removeprefix("stk_"):
                raise RuntimeError(f"{cell['cell']}, seed={seed}, arm={arm} has preset={arm_record.get('preset')}")
            if arm_record.get("ok") and arm_record.get("run_diagnostics") is not False:
                raise RuntimeError(f"{cell['cell']}, seed={seed}, arm={arm} ran diagnostics")

out = base / "results.json"
out.write_text(json.dumps(merged, indent=2))
print(f"merged {len(files)} seed-files -> {out}  ({len(seeds)} seeds, {len(merged['cells'])} cells)")

_summary(merged)
