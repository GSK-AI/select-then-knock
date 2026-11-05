"""Gaussian global null: false selections of STK and sparse baselines.

k_true=0: y is generated independent of X (see _lib.generate_global_null).
Every selection is therefore a false positive by construction, so mean FDP is
not informative (it is exactly 0 or 1 per seed). The
metrics that matter are the FIRING PROBABILITY P(|S_hat|>0) and the
distribution of |S_hat| given a fire.

One-factor-at-a-time grid (fixed n=800, block_size=20; rho=0.5/p=5000 is the
shared baseline cell):
  - correlation axis: rho in {0.0, 0.3, 0.5, 0.8} at p=5000
  - density/p axis:   p in {500, 2000, 5000} at rho=0.5

Arms: LassoCV (``lasso_cv``), knockoff-based Stabl (``stabl``), Pooled STK
(``stk_pooled``), Headline STK (``stk_headline``), and Secondary STK
(``stk_secondary``).

Run from the project root:
    SEEDS=42 STK_SOLVER_BACKEND=celer python experiments/06-gaussian-global-null/run.py

For seed-sharded runs use SEEDS=<one seed> and a distinct OUTDIR per shard.
The runner writes incrementally to OUTDIR/results.json.

Env knobs: SEEDS, ARMS, OUTDIR.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LIB_DIR))
import _lib  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ["OUTDIR"]).resolve() if os.environ.get("OUTDIR") else HERE / "outputs"
RESULTS_JSON = OUT / "results.json"

_DEFAULT_SEEDS = ",".join(str(s) for s in range(42, 72))  # 42..71 (30 seeds)
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", _DEFAULT_SEEDS).split(","))
ARMS = tuple(
    a.strip()
    for a in os.environ.get("ARMS", "lasso_cv,stk_pooled,stk_headline,stk_secondary,stabl").split(",")
    if a.strip()
)

STK_SHA = os.environ.get("STK_SHA") or _lib.git_sha()

CELLS = [
    {"cell": "rho=0.0", "rho": 0.0, "p": 5000},
    {"cell": "rho=0.3", "rho": 0.3, "p": 5000},
    {"cell": "rho=0.5", "rho": 0.5, "p": 5000},
    {"cell": "rho=0.8", "rho": 0.8, "p": 5000},
    {"cell": "p=500", "rho": 0.5, "p": 500},
    {"cell": "p=2000", "rho": 0.5, "p": 2000},
]

TIMEOUTS = {"lasso_cv": 1200, "stk_pooled": 600, "stk_headline": 3000, "stk_secondary": 3000, "stabl": 3600}
MULTI_SPLIT_ARMS = {"stk_headline", "stk_secondary"}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_sha256(X: np.ndarray, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (X, y):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _write_results(results: dict) -> None:
    temporary = RESULTS_JSON.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2))
    temporary.replace(RESULTS_JSON)


def run_arm(arm: str, X, y, seed) -> dict:
    if arm == "lasso_cv":
        return _lib.arm_lasso_cv(X, y, seed, timeout_s=TIMEOUTS[arm])
    if arm == "stk_pooled":
        return _lib.arm_stk_pooled(X, y, seed, timeout_s=TIMEOUTS[arm])
    if arm in MULTI_SPLIT_ARMS:
        return _lib.arm_stk_multi_split(X, y, seed, preset=arm.removeprefix("stk_"), timeout_s=TIMEOUTS[arm])
    if arm == "stabl":
        return _lib.arm_stabl(X, y, seed, timeout_s=TIMEOUTS[arm])
    raise ValueError(f"unknown arm {arm}")


def _load() -> dict:
    if RESULTS_JSON.exists():
        return json.loads(RESULTS_JSON.read_text())
    return {
        "meta": {
            "experiment": "06-gaussian-global-null",
            "stk_commit": STK_SHA,
            "run_sha256": _file_sha256(Path(__file__)),
            "lib_sha256": _file_sha256(_LIB_DIR / "_lib.py"),
            "run_diagnostics": False,
            "arm_presets": {"stk_pooled": "pooled", "stk_headline": "headline", "stk_secondary": "secondary"},
            "n": _lib.N,
            "block_size": _lib.BLOCK_SIZE,
            "cell_defs": CELLS,
            "seeds": list(SEEDS),
            "arms_run": [],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "cells": [dict(c, seeds={}) for c in CELLS],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    print(f"STK commit: {STK_SHA}", flush=True)
    print(f"seeds={SEEDS} arms={ARMS}", flush=True)

    results = _load()
    for a in ARMS:
        if a not in results["meta"]["arms_run"]:
            results["meta"]["arms_run"].append(a)
    by_key = {c["cell"]: c for c in results["cells"]}

    for cdef in CELLS:
        cell = by_key[cdef["cell"]]
        for seed in SEEDS:
            d = _lib.make_null_dataset(cdef["p"], cdef["rho"], seed)
            X, y = d["X"], d["y"]
            fingerprint = _dataset_sha256(X, y)
            srec = cell["seeds"].setdefault(str(seed), {"dataset_sha256": fingerprint, "arms": {}})
            recorded_fingerprint = srec.setdefault("dataset_sha256", fingerprint)
            if recorded_fingerprint != fingerprint:
                raise RuntimeError(f"dataset mismatch for {cdef['cell']}, seed={seed}")
            line = [f"[{cdef['cell']:8s} seed={seed}]"]
            for arm in ARMS:
                existing = srec["arms"].get(arm)
                if existing and existing.get("ok"):
                    line.append(f"{arm}=SKIP")
                    continue
                res = run_arm(arm, X, y, seed)
                if not res.get("ok"):
                    srec["arms"][arm] = {
                        "ok": False,
                        "error": res.get("error"),
                        "timed_out": res.get("timed_out", False),
                        "elapsed_s": res.get("elapsed_s"),
                    }
                    line.append(f"{arm}={'TIMEOUT' if res.get('timed_out') else 'ERR'}")
                else:
                    sel = res["selected"]
                    arec = {
                        "ok": True,
                        "elapsed_s": res["elapsed_s"],
                        "n_selected": len(sel),
                        "selected_indices": sel,
                    }
                    if arm == "lasso_cv":
                        arec["alpha_cv"] = res.get("alpha_cv")
                    if arm in MULTI_SPLIT_ARMS:
                        arec["pi_nonzero"] = res.get("pi_nonzero", {})
                        for key in (
                            "per_split_sizes",
                            "n_splits",
                            "eta",
                            "preset",
                            "split_frac",
                            "min_knockoff_candidates",
                            "run_diagnostics",
                        ):
                            arec[key] = res.get(key)
                    if arm == "stabl":
                        arec["min_fdr"] = res.get("min_fdr")
                    srec["arms"][arm] = arec
                    line.append(f"{arm}:|S|={len(sel)} t={res['elapsed_s']:.0f}s")
                _write_results(results)
            print("  ".join(line) + f"  ({time.perf_counter() - t_start:.0f}s)", flush=True)

    _summary(results)
    print(f"\nDONE in {time.perf_counter() - t_start:.0f}s -> {RESULTS_JSON}", flush=True)


def _summary(results: dict) -> None:
    print("\n=== SUMMARY: P(fire) and |S_hat| given a completed run, per cell x arm ===", flush=True)
    print(f"{'cell':8s} {'arm':13s} {'n':>3s} {'P(fire)':>8s} {'mean|S|':>8s} {'max|S|':>7s}", flush=True)
    for cell in results["cells"]:
        arms = sorted({a for s in cell["seeds"].values() for a in s["arms"]})
        for arm in arms:
            nsels = []
            for s in cell["seeds"].values():
                a = s["arms"].get(arm, {})
                if not a.get("ok"):
                    continue
                nsels.append(a["n_selected"])
            if not nsels:
                print(f"{cell['cell']:8s} {arm:13s} {0:>3d} {'--':>8s} {'--':>8s} {'--':>7s}", flush=True)
                continue
            p_fire = sum(1 for n in nsels if n > 0) / len(nsels)
            print(
                f"{cell['cell']:8s} {arm:13s} {len(nsels):>3d} {p_fire:8.2f} {np.mean(nsels):8.2f} {max(nsels):>7d}",
                flush=True,
            )


if __name__ == "__main__":
    main()
