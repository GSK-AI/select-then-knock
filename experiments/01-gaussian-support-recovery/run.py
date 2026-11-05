"""Gaussian support recovery across true support sizes.

Block-correlated Gaussian features (regime `hard_rho05`: SNR 0.5, within-block
correlation 0.5). Full design is n=800, p=5000, k in {1,2,3,5,10,15,20}, seeds
42-51, comparing Headline STK, Secondary STK, Pooled STK, LassoCV, and
knockoff-based Stabl.

Arms can run together in one environment or in separate passes appending to the
same per-seed JSON. A SHA-256 fingerprint checks that the generated data agree.

Run from the project root:
    STK_SOLVER_BACKEND=celer python experiments/01-gaussian-support-recovery/run.py

Environment knobs: SEEDS, K_SWEEP, ARMS, OUTDIR, STK_SHA, SNAPSHOT_ID,
STK_WORKTREE_DIRTY.
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

REGIME = "hard_rho05"

SEEDS = tuple(int(value) for value in os.environ.get("SEEDS", "42,43,44,45,46,47,48,49,50,51").split(","))
KS = tuple(int(value) for value in os.environ.get("K_SWEEP", "1,2,3,5,10,15,20").split(","))
ARMS = tuple(
    value.strip()
    for value in os.environ.get("ARMS", "lasso_cv,stk_pooled,stk_headline,stk_secondary,stabl").split(",")
    if value.strip()
)
STK_SHA = os.environ.get("STK_SHA") or _lib.git_sha()
SNAPSHOT_ID = os.environ.get("SNAPSHOT_ID", "local")
WORKTREE_DIRTY = os.environ.get("STK_WORKTREE_DIRTY", "unknown")

TIMEOUTS = {
    "lasso_cv": 1200,
    "stk_pooled": 3600,
    "stk_headline": 7200,
    "stk_secondary": 7200,
    "stabl": 3600,
}
MULTI_SPLIT_ARMS = {"stk_headline", "stk_secondary"}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_sha256(X: np.ndarray, y: np.ndarray, support: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (X, y, support):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _write_results(results: dict) -> None:
    temporary = RESULTS_JSON.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2))
    temporary.replace(RESULTS_JSON)


def run_arm(arm: str, X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    if arm == "lasso_cv":
        return _lib.arm_lasso_cv(X, y, seed, timeout_s=TIMEOUTS[arm])
    if arm == "stk_pooled":
        return _lib.arm_stk_pooled(X, y, seed, timeout_s=TIMEOUTS[arm])
    if arm in MULTI_SPLIT_ARMS:
        return _lib.arm_stk_multi_split(X, y, seed, preset=arm.removeprefix("stk_"), timeout_s=TIMEOUTS[arm])
    if arm == "stabl":
        return _lib.arm_stabl(X, y, seed, timeout_s=TIMEOUTS[arm])
    raise ValueError(f"unknown arm {arm}")


def _new_results() -> dict:
    return {
        "meta": {
            "experiment": "01-gaussian-support-recovery",
            "stk_commit": STK_SHA,
            "snapshot_id": SNAPSHOT_ID,
            "worktree_dirty": WORKTREE_DIRTY,
            "run_sha256": _file_sha256(Path(__file__)),
            "lib_sha256": _file_sha256(_LIB_DIR / "_lib.py"),
            "run_diagnostics": False,
            "arm_presets": {"stk_pooled": "pooled", "stk_headline": "headline", "stk_secondary": "secondary"},
            "n": _lib.N,
            "p": _lib.P,
            "regime": REGIME,
            "regime_params": _lib.REGIME_DEFS[REGIME],
            "k_sweep": list(KS),
            "seeds": list(SEEDS),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "arms_run": [],
        },
        "cells": {},
    }


def _load() -> dict:
    if not RESULTS_JSON.exists():
        return _new_results()
    results = json.loads(RESULTS_JSON.read_text())
    expected = _new_results()["meta"]
    for key in ("stk_commit", "snapshot_id", "run_sha256", "lib_sha256", "n", "p", "regime", "k_sweep"):
        if results["meta"].get(key) != expected[key]:
            raise RuntimeError(f"cannot append results: metadata mismatch for {key}")
    return results


def _exact_recovery(selected: list[int], support: np.ndarray) -> bool:
    return set(int(index) for index in selected) == set(int(index) for index in support)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"STK commit: {STK_SHA}", flush=True)
    print(f"snapshot={SNAPSHOT_ID} dirty={WORKTREE_DIRTY}", flush=True)
    print(f"regime={REGIME} {_lib.REGIME_DEFS[REGIME]}", flush=True)
    print(f"seeds={SEEDS} k={KS} arms={ARMS}", flush=True)

    results = _load()
    for arm in ARMS:
        if arm not in results["meta"]["arms_run"]:
            results["meta"]["arms_run"].append(arm)

    for k in KS:
        cell_key = f"{REGIME}/k={k}"
        cell = results["cells"].setdefault(
            cell_key,
            {
                "regime": REGIME,
                "k": k,
                "snr": _lib.REGIME_DEFS[REGIME]["snr"],
                "rho": _lib.REGIME_DEFS[REGIME]["rho"],
                "seeds": {},
            },
        )
        for seed in SEEDS:
            dataset = _lib.make_dataset(REGIME, k, seed)
            X = dataset["X"]
            y = dataset["y"]
            support = dataset["S_star"]
            blocks = dataset["block_assignments"]
            fingerprint = _dataset_sha256(X, y, support)

            seed_record = cell["seeds"].setdefault(
                str(seed),
                {
                    "S_star": support.tolist(),
                    "realized_snr": dataset["realized_snr"],
                    "dataset_sha256": fingerprint,
                    "arms": {},
                },
            )
            if seed_record["dataset_sha256"] != fingerprint:
                raise RuntimeError(f"dataset mismatch for {cell_key}, seed={seed}")

            line = [f"[{cell_key} seed={seed}] SNR={dataset['realized_snr']:.2f}"]
            for arm in ARMS:
                existing = seed_record["arms"].get(arm)
                if existing and existing.get("ok"):
                    line.append(f"{arm}=SKIP")
                    continue
                result = run_arm(arm, X, y, seed)
                if not result.get("ok"):
                    seed_record["arms"][arm] = {
                        "ok": False,
                        "error": result.get("error"),
                        "traceback": result.get("traceback"),
                        "timed_out": result.get("timed_out", False),
                        "elapsed_s": result.get("elapsed_s"),
                    }
                    line.append(f"{arm}={'TIMEOUT' if result.get('timed_out') else 'ERR'}")
                else:
                    metrics = _lib.compute_metrics(result["selected"], support, blocks)
                    arm_record = {
                        "ok": True,
                        "elapsed_s": result["elapsed_s"],
                        "exact_recovery": _exact_recovery(result["selected"], support),
                        **metrics,
                    }
                    if arm == "lasso_cv":
                        arm_record["alpha_cv"] = result.get("alpha_cv")
                    if arm in MULTI_SPLIT_ARMS:
                        arm_record["pi_nonzero"] = result.get("pi_nonzero", {})
                        arm_record["per_split_sizes"] = result.get("per_split_sizes", [])
                        arm_record["n_splits"] = result.get("n_splits")
                        arm_record["eta"] = result.get("eta")
                        arm_record["preset"] = result.get("preset")
                        arm_record["split_frac"] = result.get("split_frac")
                        arm_record["min_knockoff_candidates"] = result.get("min_knockoff_candidates")
                        arm_record["run_diagnostics"] = result.get("run_diagnostics")
                    if arm == "stabl":
                        arm_record["min_fdr"] = result.get("min_fdr")
                    seed_record["arms"][arm] = arm_record
                    line.append(
                        f"{arm}:|S|={metrics['n_selected']} FDP={metrics['fdp']:.2f} "
                        f"pow={metrics['power']:.2f} exact={arm_record['exact_recovery']} "
                        f"t={result['elapsed_s']:.0f}s"
                    )
                _write_results(results)
            print("  ".join(line) + f"  ({time.perf_counter() - started:.0f}s)", flush=True)

    print(f"DONE in {time.perf_counter() - started:.0f}s -> {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
