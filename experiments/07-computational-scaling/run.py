"""Computational scaling of Secondary STK versus direct full-p Model-X.

Measures statistical performance, wall-clock time, and the number of features
entering knockoff construction as p grows through {500, 2000, 5000, 20000}.
The fixed statistical setting is n=800, k=3, SNR=0.5, rho=0.8, with seeds
42-51. Arms are Secondary STK (``stk_secondary``) and direct full-p Model-X
knockoffs without screening (``direct_model_x``).

Run from the project root, for example:
    P_VALUES=500 SEEDS=42 ARMS=stk_secondary STK_SOLVER_BACKEND=celer \
      python experiments/07-computational-scaling/run.py

This implementation omits computation whose output was discarded, so fresh
runtime and memory numbers are lower than the historical ones and do not
reproduce them. Environment knobs: N_SAMPLES, P_VALUES, K, SNR, RHO, SEEDS,
ARMS, OUTDIR, STK_SHA, SNAPSHOT_ID, STK_WORKTREE_DIRTY, TIMEOUT_STK_SECONDARY,
TIMEOUT_DIRECT_MODEL_X, and DIRECT_MODEL_X_MEMORY_GB.
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

N = int(os.environ.get("N_SAMPLES", "800"))
P_VALUES = tuple(int(value) for value in os.environ.get("P_VALUES", "500,2000,5000,20000").split(","))
K = int(os.environ.get("K", "3"))
SNR = float(os.environ.get("SNR", "0.5"))
RHO = float(os.environ.get("RHO", "0.8"))
SEEDS = tuple(int(value) for value in os.environ.get("SEEDS", "42,43,44,45,46,47,48,49,50,51").split(","))
ARMS = tuple(
    value.strip() for value in os.environ.get("ARMS", "stk_secondary,direct_model_x").split(",") if value.strip()
)

STK_SHA = os.environ.get("STK_SHA") or _lib.git_sha()
SNAPSHOT_ID = os.environ.get("SNAPSHOT_ID", "local")
WORKTREE_DIRTY = os.environ.get("STK_WORKTREE_DIRTY", "unknown")

DEFAULT_TIMEOUTS = {
    "stk_secondary": 14400,
    "direct_model_x": 7200,
}
TIMEOUTS = {
    arm: int(os.environ.get(f"TIMEOUT_{arm.upper()}", str(default))) for arm, default in DEFAULT_TIMEOUTS.items()
}
DIRECT_MODEL_X_MEMORY_GB = float(os.environ.get("DIRECT_MODEL_X_MEMORY_GB", "0"))


def _validate_design() -> None:
    if K <= 0 or K > min(P_VALUES) // _lib.BLOCK_SIZE:
        raise ValueError("K must be positive and no larger than the number of blocks at the smallest p")
    for p in P_VALUES:
        if p <= 0 or p % _lib.BLOCK_SIZE:
            raise ValueError(f"p={p} must be a positive multiple of block_size={_lib.BLOCK_SIZE}")
    known = set(DEFAULT_TIMEOUTS)
    unknown = set(ARMS) - known
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")


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


def _make_dataset(p: int, seed: int) -> dict:
    return _lib.generate_blocked_synthetic(
        n=N,
        n_blocks=p // _lib.BLOCK_SIZE,
        block_size=_lib.BLOCK_SIZE,
        rho=RHO,
        k_true=K,
        beta_true=_lib.beta_for_k(K),
        snr=SNR,
        seed=seed,
    )


def _stk_instrumented_worker(X: np.ndarray, y: np.ndarray, seed: int, preset: str, queue) -> None:
    """Fit multi-split STK while observing the matrices passed to knockoff_filter."""
    started = time.perf_counter()
    try:
        from select_then_knock import SelectThenKnock

        class InstrumentedSelectThenKnock(SelectThenKnock):
            def _one_split(self, X_arr, y_arr, cut, split_seed, per_run_fdr, knockoff_filter):
                def observed_filter(X_knockoff, y_knockoff, *args, **kwargs):
                    self._observed_candidate_sizes.append(int(X_knockoff.shape[1]))
                    result = knockoff_filter(X_knockoff, y_knockoff, *args, **kwargs)
                    self._observed_fallbacks.append(bool(result.fallback_used))
                    return result

                return super()._one_split(X_arr, y_arr, cut, split_seed, per_run_fdr, observed_filter)

        stk = InstrumentedSelectThenKnock.from_preset(
            preset,
            task="regression",
            random_state=seed,
            solver_backend=os.environ.get("STK_SOLVER_BACKEND", "sklearn"),
            run_diagnostics=False,
        )
        stk._observed_candidate_sizes = []
        stk._observed_fallbacks = []
        stk.fit(X, y)
        selected = np.asarray(stk.get_support(indices=True, space="original"), dtype=int)

        extra = {
            "candidate_sizes": list(stk._observed_candidate_sizes),
            "fallback_flags": list(stk._observed_fallbacks),
            "per_split_selected_sizes": list(stk.multi_split_results_.per_split_sizes),
            "n_splits": int(stk.n_splits),
            "eta": float(stk.eta),
            "preset": preset,
            "split_frac": float(stk.split_frac),
            "min_knockoff_candidates": int(stk.multi_split_results_.min_knockoff_candidates),
            "run_diagnostics": bool(stk.run_diagnostics),
        }

        queue.put(
            {
                "ok": True,
                "elapsed_s": time.perf_counter() - started,
                "selected": selected.tolist(),
                **extra,
            }
        )
    except Exception as error:
        import traceback

        queue.put(
            {
                "ok": False,
                "elapsed_s": time.perf_counter() - started,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )


def _direct_model_x_worker(X: np.ndarray, y: np.ndarray, seed: int, fdr: float, memory_gb: float, queue) -> None:
    """Run direct Model-X under an optional child-only address-space cap."""
    started = time.perf_counter()
    try:
        memory_limit_enforced = False
        if memory_gb > 0 and sys.platform.startswith("linux"):
            import resource

            limit = int(memory_gb * 1024**3)
            _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if hard_limit != resource.RLIM_INFINITY:
                limit = min(limit, hard_limit)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            memory_limit_enforced = True

        from select_then_knock.phase4_knockoff import knockoff_filter

        result = knockoff_filter(
            X,
            y,
            task="regression",
            fdr=fdr,
            knockoff_type="gaussian",
            method="lasso",
            offset=1,
            random_seed=seed,
            knockoff_construction="equi",
        )
        queue.put(
            {
                "ok": True,
                "elapsed_s": time.perf_counter() - started,
                "selected": np.asarray(result.selected_features, dtype=int).tolist(),
                "fallback_used": bool(result.fallback_used),
                "memory_limit_gb": memory_gb or None,
                "memory_limit_enforced": memory_limit_enforced,
            }
        )
    except Exception as error:
        import traceback

        queue.put(
            {
                "ok": False,
                "elapsed_s": time.perf_counter() - started,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "memory_limited": isinstance(error, MemoryError),
                "memory_limit_gb": memory_gb or None,
                "memory_limit_enforced": memory_limit_enforced,
            }
        )


def run_arm(arm: str, X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    if arm == "stk_secondary":
        return _lib.run_in_child(_stk_instrumented_worker, (X, y, seed, "secondary"), timeout_s=TIMEOUTS[arm])
    if arm == "direct_model_x":
        result = _lib.run_in_child(
            _direct_model_x_worker,
            (X, y, seed, 0.1, DIRECT_MODEL_X_MEMORY_GB),
            timeout_s=TIMEOUTS[arm],
        )
        if result.get("ok"):
            result["candidate_sizes"] = [int(X.shape[1])]
        return result
    raise ValueError(f"unknown arm {arm}")


def _new_results() -> dict:
    return {
        "meta": {
            "experiment": "07-computational-scaling",
            "stk_commit": STK_SHA,
            "snapshot_id": SNAPSHOT_ID,
            "worktree_dirty": WORKTREE_DIRTY,
            "run_sha256": _file_sha256(Path(__file__)),
            "lib_sha256": _file_sha256(_LIB_DIR / "_lib.py"),
            "n": N,
            "p_values": list(P_VALUES),
            "k": K,
            "snr": SNR,
            "rho": RHO,
            "block_size": _lib.BLOCK_SIZE,
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
    keys = (
        "stk_commit",
        "snapshot_id",
        "run_sha256",
        "lib_sha256",
        "n",
        "p_values",
        "k",
        "snr",
        "rho",
        "block_size",
    )
    for key in keys:
        if results["meta"].get(key) != expected[key]:
            raise RuntimeError(f"cannot append results: metadata mismatch for {key}")
    return results


def _write(results: dict) -> None:
    temporary = RESULTS_JSON.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2))
    temporary.replace(RESULTS_JSON)


def _exact_recovery(selected: list[int], support: np.ndarray) -> bool:
    return set(int(index) for index in selected) == set(int(index) for index in support)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _validate_design()
    started = time.perf_counter()
    print(f"STK commit: {STK_SHA}", flush=True)
    print(f"snapshot={SNAPSHOT_ID} dirty={WORKTREE_DIRTY}", flush=True)
    print(f"n={N} p={P_VALUES} k={K} snr={SNR} rho={RHO} seeds={SEEDS} arms={ARMS}", flush=True)

    results = _load()
    for arm in ARMS:
        if arm not in results["meta"]["arms_run"]:
            results["meta"]["arms_run"].append(arm)

    for p in P_VALUES:
        cell_key = f"p={p}"
        cell = results["cells"].setdefault(
            cell_key,
            {"p": p, "n": N, "k": K, "snr": SNR, "rho": RHO, "seeds": {}},
        )
        for seed in SEEDS:
            dataset = _make_dataset(p, seed)
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
                        "memory_limited": result.get("memory_limited", False),
                        "memory_limit_gb": result.get("memory_limit_gb"),
                        "memory_limit_enforced": result.get("memory_limit_enforced", False),
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
                    for key in (
                        "candidate_sizes",
                        "fallback_flags",
                        "per_split_selected_sizes",
                        "n_splits",
                        "eta",
                        "fallback_used",
                        "memory_limit_gb",
                        "memory_limit_enforced",
                        "preset",
                        "split_frac",
                        "min_knockoff_candidates",
                        "run_diagnostics",
                    ):
                        if key in result:
                            arm_record[key] = result[key]
                    if arm == "direct_model_x" and result.get("fallback_used"):
                        arm_record["valid_model_x"] = False
                    elif arm == "stk_secondary":
                        arm_record["valid_model_x"] = not any(result.get("fallback_flags", []))
                    seed_record["arms"][arm] = arm_record
                    candidates = result.get("candidate_sizes", [])
                    candidate_text = f" cand_mean={np.mean(candidates):.1f}" if candidates else ""
                    line.append(
                        f"{arm}:|S|={metrics['n_selected']} FDP={metrics['fdp']:.2f} "
                        f"pow={metrics['power']:.2f} t={result['elapsed_s']:.0f}s{candidate_text}"
                    )
                _write(results)
            print("  ".join(line) + f"  ({time.perf_counter() - started:.0f}s)", flush=True)

    print(f"DONE in {time.perf_counter() - started:.0f}s -> {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
