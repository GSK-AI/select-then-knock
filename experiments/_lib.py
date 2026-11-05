"""Shared generators, metrics, and comparator adapters for the seven experiments.

Only Gaussian/null generation, linear support benchmarks, and the evaluated arms
are retained. There is no private data ingestion or cloud access.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty as QueueEmpty

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N = 800
P = 5000
N_BLOCKS = 250
BLOCK_SIZE = 20
REGIME_DEFS = {
    "hard_rho05": {"snr": 0.5, "rho": 0.5},
}


def beta_for_k(k):
    if k == 3:
        return np.array([1.0, 0.8, 0.6])
    return np.linspace(1.0, 0.5, k)


def git_sha():
    try:
        args = ["git", "rev-parse", "HEAD"]
        return subprocess.check_output(args, cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_status():
    try:
        return subprocess.check_output(
            ["git", "status", "--short"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not a git checkout"


def source_fingerprints():
    source = ROOT / "select_then_knock"
    return {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*.py"))
    }


def generate_blocked_synthetic(n, n_blocks, block_size, rho, k_true, beta_true, snr, seed):
    """Preserve the original order of random draws and matrix operations."""
    rng = np.random.default_rng(seed)
    p = n_blocks * block_size
    Sigma_block = np.full((block_size, block_size), rho)
    np.fill_diagonal(Sigma_block, 1.0)
    L = np.linalg.cholesky(Sigma_block)
    Z = rng.standard_normal((n, p))
    X = np.empty_like(Z)
    for b in range(n_blocks):
        s = b * block_size
        e = s + block_size
        X[:, s:e] = Z[:, s:e] @ L.T
    block_assignments = np.repeat(np.arange(n_blocks), block_size)
    causal_blocks = rng.choice(n_blocks, size=k_true, replace=False)
    causal_offsets = rng.integers(0, block_size, size=k_true)
    S_star = np.sort(causal_blocks * block_size + causal_offsets)
    signal = X[:, S_star] @ beta_true
    sigma_eps = float(np.sqrt(signal.var(ddof=0) / snr))
    eps = rng.standard_normal(n) * sigma_eps
    y = signal + eps
    return {
        "X": X,
        "y": y,
        "S_star": S_star,
        "block_assignments": block_assignments,
        "sigma_eps": sigma_eps,
        "realized_snr": float(signal.var(ddof=0) / max(eps.var(ddof=0), 1e-12)),
    }


def make_dataset(regime, k, seed):
    cfg = REGIME_DEFS[regime]
    return generate_blocked_synthetic(
        n=N,
        n_blocks=N_BLOCKS,
        block_size=BLOCK_SIZE,
        rho=cfg["rho"],
        k_true=k,
        beta_true=beta_for_k(k),
        snr=cfg["snr"],
        seed=seed,
    )


def generate_global_null(n, n_blocks, block_size, rho, seed):
    rng = np.random.default_rng(seed)
    p = n_blocks * block_size
    Sigma_block = np.full((block_size, block_size), rho)
    np.fill_diagonal(Sigma_block, 1.0)
    L = np.linalg.cholesky(Sigma_block)
    Z = rng.standard_normal((n, p))
    X = np.empty_like(Z)
    for b in range(n_blocks):
        s = b * block_size
        e = s + block_size
        X[:, s:e] = Z[:, s:e] @ L.T
    y = rng.standard_normal(n)
    return {"X": X, "y": y, "S_star": np.array([], dtype=int)}


def make_null_dataset(p, rho, seed, *, n=N, block_size=BLOCK_SIZE):
    if p % block_size != 0:
        raise ValueError(f"p={p} must be a multiple of block_size={block_size}")
    return generate_global_null(n, p // block_size, block_size, rho, seed)


def compute_metrics_flat(selected, S_star):
    selected = np.asarray(selected, dtype=int)
    star, sel = set(int(i) for i in S_star), set(int(i) for i in selected)
    n_sel = int(selected.size)
    tp = len(sel & star)
    fp = n_sel - tp
    return {
        "n_selected": n_sel,
        "tp": tp,
        "fp": fp,
        "fdp": fp / max(n_sel, 1),
        "power": tp / max(len(star), 1),
        "k_true": len(star),
        "selected_indices": selected.tolist(),
    }


def compute_metrics(selected, S_star, block_assignments):
    selected = np.asarray(selected, dtype=int)
    star, sel = set(int(i) for i in S_star), set(int(i) for i in selected)
    n_sel = int(selected.size)
    tp = len(sel & star)
    fp = n_sel - tp
    causal_blocks = set(int(b) for b in block_assignments[S_star])
    sel_blocks = set(int(b) for b in block_assignments[selected]) if n_sel else set()
    return {
        "n_selected": n_sel,
        "tp": tp,
        "fp": fp,
        "fdp": float(fp / max(n_sel, 1)),
        "power": float(tp / max(len(star), 1)),
        "block_recall": len(sel_blocks & causal_blocks),
        "n_causal_blocks": len(causal_blocks),
        "selected_indices": selected.tolist(),
    }


def run_in_child(target, args, *, timeout_s):
    """Preserve spawn isolation, wall-clock limits, and failure records."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, queue))
    t0 = time.perf_counter()
    proc.start()
    proc.join(timeout=timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {"ok": False, "timed_out": True, "elapsed_s": time.perf_counter() - t0, "timeout_s": timeout_s}
    try:
        out = queue.get(timeout=5)
    except QueueEmpty:
        return {
            "ok": False,
            "timed_out": False,
            "elapsed_s": time.perf_counter() - t0,
            "error": f"child exited with code {proc.exitcode}, no result",
        }
    out["timed_out"] = False
    return out


def _stk_worker(X, y, seed, preset, overrides, queue):
    started = time.perf_counter()
    try:
        from select_then_knock import SelectThenKnock

        overrides = {
            "solver_backend": os.environ.get("STK_SOLVER_BACKEND", "sklearn"),
            **overrides,
            "run_diagnostics": False,
        }
        model = SelectThenKnock.from_preset(preset, task="regression", random_state=seed, **overrides)
        model.fit(X, y)
        result = {
            "ok": True,
            "elapsed_s": time.perf_counter() - started,
            "selected": model.get_support(indices=True, space="original").tolist(),
        }
        if model.selection_mode != "pooled":
            Pi = model.selection_frequencies_
            result.update(
                pi_nonzero={int(i): float(Pi[i]) for i in np.flatnonzero(Pi)},
                per_split_sizes=list(model.multi_split_results_.per_split_sizes),
                n_splits=int(model.n_splits),
                eta=float(model.eta),
                preset=preset,
                split_frac=float(model.split_frac),
                min_knockoff_candidates=int(model.multi_split_results_.min_knockoff_candidates),
                run_diagnostics=False,
            )
        queue.put(result)
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


def arm_stk_pooled(X, y, seed, *, overrides=None, timeout_s=600):
    return run_in_child(_stk_worker, (X, y, seed, "pooled", overrides or {}), timeout_s=timeout_s)


def arm_stk_multi_split(X, y, seed, *, preset, overrides=None, timeout_s=2400):
    return run_in_child(_stk_worker, (X, y, seed, preset, overrides or {}), timeout_s=timeout_s)


def _lasso_worker(X, y, seed, queue):
    started = time.perf_counter()
    try:
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler

        Xs = StandardScaler().fit_transform(X)
        model = LassoCV(cv=5, n_alphas=100, max_iter=10000, n_jobs=-1, random_state=seed)
        model.fit(Xs, y)
        queue.put(
            {
                "ok": True,
                "elapsed_s": time.perf_counter() - started,
                "selected": np.flatnonzero(model.coef_ != 0).tolist(),
                "alpha_cv": float(model.alpha_),
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


def arm_lasso_cv(X, y, seed, *, timeout_s=300):
    return run_in_child(_lasso_worker, (X, y, seed), timeout_s=timeout_s)


def _patch_scipy_for_knockpy():
    """The evaluated compatibility shim for knockpy 1.3.5 and newer SciPy."""
    import types

    import scipy.sparse.linalg as ssl

    try:
        if hasattr(ssl.eigen, "arpack") and hasattr(ssl.eigen.arpack, "ArpackNoConvergence"):
            return
    except Exception:
        pass
    from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence

    arpack = types.ModuleType("scipy.sparse.linalg.eigen.arpack")
    arpack.ArpackNoConvergence = ArpackNoConvergence
    eigen = types.ModuleType("scipy.sparse.linalg.eigen")
    eigen.arpack = arpack
    ssl.eigen = eigen
    sys.modules["scipy.sparse.linalg.eigen"] = eigen
    sys.modules["scipy.sparse.linalg.eigen.arpack"] = arpack


def _stabl_fit(X, y, seed, n_bootstraps, n_lambda):
    _patch_scipy_for_knockpy()
    import pandas as pd
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from stabl.stabl import Stabl

    X_raw = np.asarray(X, dtype=float)
    X_scaled = StandardScaler().fit(X_raw).transform(X_raw)
    frame = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(X_scaled.shape[1])])
    model = Stabl(
        base_estimator=Lasso(max_iter=int(1e4)),
        lambda_grid="auto",
        n_lambda=n_lambda,
        n_bootstraps=n_bootstraps,
        artificial_type="knockoff",
        artificial_proportion=1.0,
        sample_fraction=0.5,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(frame, y)
    return {
        "selected": np.asarray(model.get_support(indices=True), dtype=int).tolist(),
        "min_fdr": float(model.min_fdr_),
    }


def arm_stabl(X, y, seed, *, n_bootstraps=300, n_lambda=10, timeout_s=1800):
    """Knockoff-based Stabl with the evaluated in-process fit and SIGALRM timeout (Linux/macOS)."""
    import signal

    class FitTimeout(Exception):
        pass

    def handler(signum, frame):
        raise FitTimeout()

    t0 = time.perf_counter()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(timeout_s))
    try:
        out = _stabl_fit(X, y, seed, n_bootstraps, n_lambda)
        out.update(ok=True, elapsed_s=time.perf_counter() - t0)
        return out
    except FitTimeout:
        return {"ok": False, "timed_out": True, "elapsed_s": time.perf_counter() - t0, "timeout_s": timeout_s}
    except Exception as error:
        import traceback

        return {
            "ok": False,
            "elapsed_s": time.perf_counter() - t0,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
