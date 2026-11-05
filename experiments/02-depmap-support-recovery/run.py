"""Semi-synthetic support recovery on user-supplied public DepMap expression.

Uses real DepMap 26Q1 log(TPM+1) expression as X and plants a sparse linear
response with known support. A fixed 800-row subset is shared by every task.
For each support seed, the k=3 support is a prefix of the k=10 support, pairing
the sparse and moderately dense comparisons. Eligible support genes have
sample standard deviation >=0.1; planted genes have pairwise absolute
correlation <=0.5. All nonconstant genes remain candidate features.

Run from the project root:
    DEPMAP_EXPRESSION_PATH=/path/to/expression.csv STK_SOLVER_BACKEND=celer \
      python experiments/02-depmap-support-recovery/run.py

No data is bundled or downloaded automatically. Environment knobs: DEPMAP_EXPRESSION_PATH,
EXPECTED_DATA_SHA256, N_SAMPLES, ROW_SEED, K_VALUES, SEEDS, SNR,
MIN_SUPPORT_STD, MAX_SUPPORT_ABS_CORR, ARMS, OUTDIR, STK_SHA, SNAPSHOT_ID,
STK_WORKTREE_DIRTY, and TIMEOUT_<ARM>.
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
import pandas as pd

_LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LIB_DIR))
import _lib  # noqa: E402

HERE = Path(__file__).resolve().parent
_expression_path = os.environ.get("DEPMAP_EXPRESSION_PATH")
DATA_PATH = Path(_expression_path).expanduser().resolve() if _expression_path else None
OUT = Path(os.environ["OUTDIR"]).resolve() if os.environ.get("OUTDIR") else HERE / "outputs"
RESULTS_JSON = OUT / "results.json"

N_SAMPLES = int(os.environ.get("N_SAMPLES", "800"))
ROW_SEED = int(os.environ.get("ROW_SEED", "2026"))
KS = tuple(int(value) for value in os.environ.get("K_VALUES", "3,10").split(",") if value)
SEEDS = tuple(int(value) for value in os.environ.get("SEEDS", "42,43,44,45,46,47,48,49,50,51").split(","))
SNR = float(os.environ.get("SNR", "2.0"))
MIN_SUPPORT_STD = float(os.environ.get("MIN_SUPPORT_STD", "0.1"))
MAX_SUPPORT_ABS_CORR = float(os.environ.get("MAX_SUPPORT_ABS_CORR", "0.5"))
ARMS = tuple(
    value.strip()
    for value in os.environ.get("ARMS", "lasso_cv,stk_pooled,stk_headline,stk_secondary,stabl").split(",")
    if value.strip()
)
EXPECTED_DATA_SHA256 = os.environ.get("EXPECTED_DATA_SHA256", "").lower()
STK_SHA = os.environ.get("STK_SHA") or _lib.git_sha()
SNAPSHOT_ID = os.environ.get("SNAPSHOT_ID", "local")
WORKTREE_DIRTY = os.environ.get("STK_WORKTREE_DIRTY", "unknown")
DEPMAP_RELEASE = os.environ.get("DEPMAP_RELEASE", "26Q1")

DEFAULT_TIMEOUTS = {
    "lasso_cv": 3600,
    "stk_pooled": 7200,
    "stk_headline": 43200,
    "stk_secondary": 43200,
    "stabl": 43200,
}
TIMEOUTS = {
    arm: int(os.environ.get(f"TIMEOUT_{arm.upper()}", str(timeout))) for arm, timeout in DEFAULT_TIMEOUTS.items()
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _strings_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_design() -> None:
    if DATA_PATH is None or not DATA_PATH.is_file():
        raise FileNotFoundError("Set DEPMAP_EXPRESSION_PATH to a normalized expression CSV; no data is bundled")
    if N_SAMPLES <= 1:
        raise ValueError("N_SAMPLES must exceed one")
    if not KS or min(KS) <= 0:
        raise ValueError("K_VALUES must contain positive integers")
    if SNR <= 0 or MIN_SUPPORT_STD < 0:
        raise ValueError("SNR must be positive and MIN_SUPPORT_STD nonnegative")
    if not 0 <= MAX_SUPPORT_ABS_CORR <= 1:
        raise ValueError("MAX_SUPPORT_ABS_CORR must be in [0, 1]")
    unknown = set(ARMS) - set(DEFAULT_TIMEOUTS)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")


def load_expression() -> dict:
    """Load one fixed row subset and remove only zero-variance candidates."""
    data_sha256 = _file_sha256(DATA_PATH)
    if EXPECTED_DATA_SHA256 and data_sha256 != EXPECTED_DATA_SHA256:
        raise RuntimeError(f"input SHA-256 mismatch: got {data_sha256}, expected {EXPECTED_DATA_SHA256}")

    frame = pd.read_csv(DATA_PATH, index_col=0)
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("DepMap row identifiers and gene symbols must be unique")
    if N_SAMPLES > len(frame):
        raise ValueError(f"N_SAMPLES={N_SAMPLES} exceeds the {len(frame)} available rows")

    rng = np.random.default_rng(ROW_SEED)
    row_positions = np.sort(rng.choice(len(frame), size=N_SAMPLES, replace=False))
    frame = frame.iloc[row_positions]
    row_ids = frame.index.astype(str).to_numpy()
    feature_names = frame.columns.astype(str).to_numpy()
    X = frame.to_numpy(dtype=np.float64, copy=True)
    del frame

    if not np.isfinite(X).all():
        raise ValueError("DepMap expression matrix contains non-finite values")
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    keep = stds > 0
    X = X[:, keep]
    means = means[keep]
    stds = stds[keep]
    feature_names = feature_names[keep]

    return {
        "X": X,
        "means": means,
        "stds": stds,
        "row_ids": row_ids,
        "feature_names": feature_names,
        "data_sha256": data_sha256,
        "matrix_sha256": _array_sha256(X),
        "row_ids_sha256": _strings_sha256(row_ids),
        "feature_names_sha256": _strings_sha256(feature_names),
        "n_input_features": int(keep.size),
        "n_zero_variance_removed": int((~keep).sum()),
    }


def choose_support(X: np.ndarray, means: np.ndarray, stds: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Choose measurable, nonredundant genes without using the outcome."""
    eligible = np.flatnonzero(stds >= MIN_SUPPORT_STD)
    order = np.random.default_rng(seed).permutation(eligible)
    selected: list[int] = []
    standardized: list[np.ndarray] = []
    for index in order:
        z = (X[:, index] - means[index]) / stds[index]
        if all(abs(float(np.dot(z, prior) / X.shape[0])) <= MAX_SUPPORT_ABS_CORR for prior in standardized):
            selected.append(int(index))
            standardized.append(z)
            if len(selected) == k:
                return np.asarray(selected, dtype=int)
    raise RuntimeError(f"could select only {len(selected)} of {k} support genes under the correlation cap")


def make_dataset(expression: dict, k: int, seed: int) -> dict:
    X = expression["X"]
    support = choose_support(X, expression["means"], expression["stds"], k, seed)
    beta = _lib.beta_for_k(k)
    X_support = (X[:, support] - expression["means"][support]) / expression["stds"][support]
    signal = X_support @ beta
    sigma = float(np.sqrt(signal.var(ddof=0) / SNR))
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 24, 2026]))
    noise = noise_rng.standard_normal(X.shape[0]) * sigma
    y = signal + noise
    return {
        "X": X,
        "y": y,
        "support": support,
        "beta": beta,
        "sigma": sigma,
        "realized_snr": float(signal.var(ddof=0) / max(noise.var(ddof=0), 1e-12)),
        "dataset_sha256": _array_sha256(X, y, support),
    }


def run_arm(arm: str, X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    if arm == "lasso_cv":
        return _lib.arm_lasso_cv(X, y, seed, timeout_s=TIMEOUTS[arm])
    if arm == "stk_pooled":
        return _lib.arm_stk_pooled(X, y, seed, timeout_s=TIMEOUTS[arm])
    if arm in {"stk_headline", "stk_secondary"}:
        return _lib.arm_stk_multi_split(X, y, seed, preset=arm.removeprefix("stk_"), timeout_s=TIMEOUTS[arm])
    if arm == "stabl":
        return _lib.arm_stabl(X, y, seed, timeout_s=TIMEOUTS[arm])
    raise ValueError(f"unknown arm {arm}")


def _new_results(expression: dict) -> dict:
    return {
        "meta": {
            "experiment": "02-depmap-support-recovery",
            "stk_commit": STK_SHA,
            "snapshot_id": SNAPSHOT_ID,
            "worktree_dirty": WORKTREE_DIRTY,
            "run_sha256": _file_sha256(Path(__file__)),
            "lib_sha256": _file_sha256(_LIB_DIR / "_lib.py"),
            "run_diagnostics": False,
            "arm_presets": {"stk_pooled": "pooled", "stk_headline": "headline", "stk_secondary": "secondary"},
            "depmap_release": DEPMAP_RELEASE,
            "data_path_name": DATA_PATH.name,
            "data_sha256": expression["data_sha256"],
            "matrix_sha256": expression["matrix_sha256"],
            "row_ids_sha256": expression["row_ids_sha256"],
            "feature_names_sha256": expression["feature_names_sha256"],
            "n": int(expression["X"].shape[0]),
            "p": int(expression["X"].shape[1]),
            "n_input_features": expression["n_input_features"],
            "n_zero_variance_removed": expression["n_zero_variance_removed"],
            "row_seed": ROW_SEED,
            "snr": SNR,
            "min_support_std": MIN_SUPPORT_STD,
            "max_support_abs_corr": MAX_SUPPORT_ABS_CORR,
            "k_values": list(KS),
            "seeds": list(SEEDS),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "arms_run": [],
        },
        "cells": {},
    }


def _load(expression: dict) -> dict:
    expected = _new_results(expression)
    if not RESULTS_JSON.exists():
        return expected
    results = json.loads(RESULTS_JSON.read_text())
    keys = (
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
    for key in keys:
        if results["meta"].get(key) != expected["meta"].get(key):
            raise RuntimeError(f"cannot append results: metadata mismatch for {key}")
    return results


def _write(results: dict) -> None:
    temporary = RESULTS_JSON.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2))
    temporary.replace(RESULTS_JSON)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _validate_design()
    started = time.perf_counter()
    print(f"loading {DATA_PATH}", flush=True)
    expression = load_expression()
    print(
        f"DepMap {DEPMAP_RELEASE}: X={expression['X'].shape}, "
        f"removed_constant={expression['n_zero_variance_removed']}, data_sha256={expression['data_sha256']}",
        flush=True,
    )
    print(f"k={KS} seeds={SEEDS} arms={ARMS} SNR={SNR}", flush=True)

    results = _load(expression)
    for arm in ARMS:
        if arm not in results["meta"]["arms_run"]:
            results["meta"]["arms_run"].append(arm)

    for k in KS:
        cell_key = f"k={k}"
        cell = results["cells"].setdefault(cell_key, {"k": k, "seeds": {}})
        for seed in SEEDS:
            dataset = make_dataset(expression, k, seed)
            support = dataset["support"]
            support_names = expression["feature_names"][support]
            seed_record = cell["seeds"].setdefault(
                str(seed),
                {
                    "support": support.tolist(),
                    "support_names": support_names.tolist(),
                    "beta": dataset["beta"].tolist(),
                    "sigma": dataset["sigma"],
                    "realized_snr": dataset["realized_snr"],
                    "dataset_sha256": dataset["dataset_sha256"],
                    "arms": {},
                },
            )
            if seed_record["dataset_sha256"] != dataset["dataset_sha256"]:
                raise RuntimeError(f"dataset mismatch for {cell_key}, seed={seed}")

            line = [f"[{cell_key} seed={seed}] SNR={dataset['realized_snr']:.2f}"]
            for arm in ARMS:
                existing = seed_record["arms"].get(arm)
                if existing and existing.get("ok"):
                    line.append(f"{arm}=SKIP")
                    continue
                result = run_arm(arm, dataset["X"], dataset["y"], seed)
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
                    metrics = _lib.compute_metrics_flat(result["selected"], support)
                    selected = set(int(index) for index in result["selected"])
                    arm_record = {
                        "ok": True,
                        "elapsed_s": result["elapsed_s"],
                        "exact_recovery": selected == set(support.tolist()),
                        **metrics,
                    }
                    for key in (
                        "alpha_cv",
                        "pi_nonzero",
                        "per_split_sizes",
                        "n_splits",
                        "eta",
                        "min_fdr",
                        "preset",
                        "split_frac",
                        "min_knockoff_candidates",
                        "run_diagnostics",
                    ):
                        if key in result:
                            arm_record[key] = result[key]
                    seed_record["arms"][arm] = arm_record
                    line.append(
                        f"{arm}:|S|={metrics['n_selected']} FDP={metrics['fdp']:.2f} "
                        f"pow={metrics['power']:.2f} exact={arm_record['exact_recovery']} "
                        f"t={result['elapsed_s']:.0f}s"
                    )
                _write(results)
            print("  ".join(line) + f"  ({time.perf_counter() - started:.0f}s)", flush=True)

    print(f"DONE in {time.perf_counter() - started:.0f}s -> {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
