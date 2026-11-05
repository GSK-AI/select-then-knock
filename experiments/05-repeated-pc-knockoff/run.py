"""Repeated PC: PC-Knockoff repeated over splits with STK's frequency aggregation.

Question: does generic repetition explain STK's advantage over single-split
PC-Knockoff (04-pc-knockoff-comparison)? We repeat the complete PC screen/test
procedure over M=25 random row partitions on the same 70 saved evaluation
datasets and keep features selected in at least eta=0.5 of the splits, exactly
as STK does.

Prespecified design (fixed before any result was seen):
  - allocations (n_screen, d) in {(200, 50), (200, 100), (400, 50), (400, 100)};
    (200, 50) is the PC-Knockoff comparison's allocation.
  - alphas {0.1, 0.2, 0.5}; M = 25; eta = 0.5 (13 of 25 splits).
  - split 0 reuses the PC-Knockoff comparison's partition/knockoff streams, so the
    (200, 50) single split at m=0 must reproduce it exactly (consistency check).
  - splits m >= 1 take (partition, knockoff) seeds from
    SeedSequence([seed, REPEAT_STREAM, m]).generate_state(2).
Each dataset's per-split selections are stored, so single-split metrics
(averaged over the 25 splits within a dataset, as in 03-matched-aggregation)
and the aggregate are computed from the same outputs.

Run from the project root: python experiments/05-repeated-pc-knockoff/run.py
Inputs are created by the PC-Knockoff comparison's evaluation runs; no saved
matrices are bundled. Pass --pc-comparison-out if that study was run with a
non-default --out.
An existing output is reused only if its signature (design, input array and
pc_knockoff.py hashes) matches; a mismatch is an error, an unreadable file is recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _pc_comparison import PC_COMPARISON_DIR, TASKS, add_input_argument  # noqa: E402

sys.path.insert(0, str(PC_COMPARISON_DIR))
import pc_knockoff as pk  # noqa: E402

OUT = HERE / "outputs"

ALLOCATIONS = [(200, 50), (200, 100), (400, 50), (400, 100)]
ALPHAS = [0.1, 0.2, 0.5]
M = 25
ETA = 0.5
REPEAT_STREAM = 29  # Fixed seed-stream label; changing it changes every split m >= 1.


def stream_seed(seed, stream):
    # Identical to run.stream_seed in 04-pc-knockoff-comparison.
    return int(np.random.SeedSequence([seed, stream]).generate_state(1)[0])


def split_seeds(seed, m):
    if m == 0:
        return stream_seed(seed, 1), stream_seed(seed, 2)
    state = np.random.SeedSequence([seed, REPEAT_STREAM, m]).generate_state(2)
    return int(state[0]), int(state[1])


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_dataset(task):
    k, seed, data_dir = task
    path = OUT / f"k-{k}_seed-{seed}.json"
    source = data_dir / f"k-{k}_seed-{seed}.npz"
    signature = {
        "allocations": [list(a) for a in ALLOCATIONS],
        "alphas": ALPHAS,
        "n_splits": M,
        "data_sha256": file_hash(source),
        "pc_knockoff_sha256": file_hash(PC_COMPARISON_DIR / "pc_knockoff.py"),
    }
    try:
        existing = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        existing = None
    if existing is not None:
        if existing.get("signature") != signature:
            raise ValueError(f"Refusing to reuse mismatched result (delete it to recompute): {path}")
        return str(path)
    data = np.load(source)
    X, y, support = data["X"], data["y"], data["S_star"].astype(int)
    record = {"k": k, "seed": seed, "signature": signature, "S_star": support.tolist(), "allocations": {}}
    started = time.perf_counter()
    for n_screen, d in ALLOCATIONS:
        splits = []
        t0 = time.perf_counter()
        for m in range(M):
            ps, ks = split_seeds(seed, m)
            r = pk.pc_knockoff(X, y, n_screen=n_screen, d=d, alphas=ALPHAS, partition_seed=ps, knockoff_seed=ks)
            splits.append(
                {
                    "support_screened": int(np.isin(support, r.candidates).sum()),
                    "selected": {str(a): r.selected[a].tolist() for a in ALPHAS},
                }
            )
        record["allocations"][f"{n_screen}-{d}"] = {
            "n_screen": n_screen,
            "d": d,
            "elapsed_s": time.perf_counter() - t0,
            "splits": splits,
        }
    record["elapsed_s"] = time.perf_counter() - started
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record))
    os.replace(temporary, path)
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    add_input_argument(parser)
    args = parser.parse_args()
    data = args.pc_comparison_out / "data"
    tasks = [(k, s, data) for k, s in TASKS]
    missing = [
        data / f"k-{k}_seed-{seed}.npz" for k, seed, _ in tasks if not (data / f"k-{k}_seed-{seed}.npz").is_file()
    ]
    if missing:
        parser.error(
            f"Missing {len(missing)} evaluation datasets. Run the PC-Knockoff comparison's eval grids first; "
            f"expected {data}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} datasets, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, p in enumerate(pool.map(run_dataset, tasks), 1):
            print(f"[{i}/{len(tasks)}] {Path(p).name}", flush=True)


if __name__ == "__main__":
    main()
