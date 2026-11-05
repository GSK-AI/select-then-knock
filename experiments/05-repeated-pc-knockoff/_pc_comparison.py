"""Evaluation inputs from the PC-Knockoff comparison (04), shared by this study's scripts.

The PC-Knockoff comparison keeps a failed or timed-out arm as a record with ok = False.
By default the scripts stop and list those tasks; --exclude-failed drops every dataset
with a failed arm instead and reports how many were dropped per support size.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PC_COMPARISON_DIR = Path(__file__).resolve().parent.parent / "04-pc-knockoff-comparison"
DEFAULT_OUT = PC_COMPARISON_DIR / "outputs" / "eval"
ARMS = ("pc_knockoff", "stk_headline", "stk_secondary")
KS = [0, 1, 3, 10, 20]
TASKS = [(k, s) for k in KS for s in range(10000, 10030 if k == 0 else 10010)]


def add_input_argument(parser):
    parser.add_argument(
        "--pc-comparison-out",
        type=Path,
        default=DEFAULT_OUT,
        help="PC-Knockoff comparison (04) eval outputs (its --out); default %(default)s",
    )


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    add_input_argument(parser)
    parser.add_argument(
        "--exclude-failed",
        action="store_true",
        help="drop datasets with a failed PC-Knockoff comparison arm instead of stopping",
    )
    return parser.parse_args()


def load(out, exclude_failed, tasks=TASKS):
    """Return ({(k, seed, arm): record} for usable datasets, {k: number excluded})."""
    records, failed = {}, []
    for k, seed in tasks:
        for arm in ARMS:
            rec = json.loads((out / f"k-{k}" / f"seed-{seed}" / f"{arm}.json").read_text())
            if not rec.get("result", {}).get("ok"):
                failed.append((k, seed, arm))
            records[k, seed, arm] = rec
    if failed and not exclude_failed:
        listed = "\n".join(f"  k={k} seed={seed} arm={arm}" for k, seed, arm in failed)
        sys.exit(
            f"{len(failed)} failed PC-Knockoff comparison arm(s) in {out}:\n{listed}\n"
            "Rerun them or pass --exclude-failed."
        )
    dropped = {(k, seed) for k, seed, _ in failed}
    excluded = dict(sorted(Counter(k for k, _ in dropped).items()))
    if excluded:
        print(f"Excluded datasets with a failed PC-Knockoff comparison arm, per k: {excluded}", flush=True)
    return {key: rec for key, rec in records.items() if key[:2] not in dropped}, excluded
