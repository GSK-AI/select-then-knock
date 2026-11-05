"""Single-split PC-Knockoff (Liu, Ke, Liu and Li, 2022, JASA 117:428-443).

Python 3 port of the paper-linked author code, github.com/TwoLittle/PC_Screen
at revision 47c5f9d0ad981f4c8ebd5ddc8ef1f60c2ab132fe (PC_screen.py sha256
3aaab460978ea8d45922fa1cf4b692125c5f8677f95a005535bd158398ee1de8). The upstream
repository has no license file, so this module reimplements the procedure and
copies no upstream code. Attribution: Liu et al. (2022), DOI 10.1080/01621459.2020.1783274.

Algorithm 1 of the paper, as implemented here:

1. Split rows once into a screening subsample (n_screen) and a testing subsample.
2. Rank features by the screening-sample projection correlation PC(X_j, y) and keep
   the top d, with the published restriction 2*d < n_test.
3. On testing rows, build fixed-X equicorrelated knockoffs for the d features, as in
   the author's get_equi_features, and compute W_j = PC(X_j, y)^2 - PC(X~_j, y)^2.
4. Apply knockoff+ (offset 1) at each alpha. No selection if no threshold qualifies.

Exact computational shortcut (not an approximation). For scalar x and pivot row r,
the author's angle is a_klr = (pi/2) * (1 - s_k * s_l) with s = sign(x - x_r): angle
0 for the same side, pi for opposite sides, and pi/2 when either difference is zero
(the author code sets cos = 0 for zero-norm differences). Double-centering over (k, l)
removes the constant, so A_r = -(pi/2) c c' with c = s - mean(s), and

    sum_{k,l,r} A_klr B_klr = (pi/2)^2 * sum_r (c_r^x . c_r^y)^2.

This turns the O(n^3) memory and time per feature into O(n^2). The constants cancel
in PC^2. tests/ check this against the brute-force definition and the upstream code.

Diagonal convention. The paper's (2.4) defines a_klr = 0 when k = r or l = r; the
author code gives pi/2 there. Both are exact here (`convention`). The comparator uses
the author-code convention; the paper convention is an O(1/n) correction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

UPSTREAM_REVISION = "47c5f9d0ad981f4c8ebd5ddc8ef1f60c2ab132fe"
UPSTREAM_PC_SCREEN_SHA256 = "3aaab460978ea8d45922fa1cf4b692125c5f8677f95a005535bd158398ee1de8"
EQUI_SLACK = 1e-5  # author code: s = min(1, 2*lambda_min) - 0.00001
CONVENTIONS = ("author_code", "paper")


# ── Projection correlation ──────────────────────────────────────────────────
def _centered_signs(v):
    """Rows r: sign(v - v_r), centered over the row. Shape (..., n, n)."""
    s = np.sign(v[..., None, :] - v[..., :, None])
    return s - s.mean(axis=-1, keepdims=True)


def _pcov_terms(cx, cy, convention):
    """Unscaled sum_{k,l,r} A B for centered sign matrices (constants dropped)."""
    total = np.einsum("...rk,...rk->...r", cx, cy) ** 2
    if convention == "paper":
        # Paper: a_klr = 0 on the k = r or l = r cross. Relative to the author
        # convention this subtracts (pi/2)(e_r 1' + 1 e_r' - e_r e_r'); after double
        # centering only + (pi/2) e e' survives, e = e_r - 1/n. With c_r = 0 - mean(s):
        # A_r = (pi/2)(-c c' + e e'), and e.c = c_r, e.e = 1 - 1/n.
        n = cx.shape[-1]
        idx = np.arange(n)
        cxr, cyr = cx[..., idx, idx], cy[..., idx, idx]
        ee = 1.0 - 1.0 / n
        total = total - cxr**2 - cyr**2 + ee**2
    elif convention != "author_code":
        raise ValueError(f"convention must be one of {CONVENTIONS}")
    return total.sum(axis=-1)


def pc_squared(x, y, convention="author_code"):
    """Sample projection correlation squared, PC^2 in (2.5), for 1-D x and y.

    Returns 0.0 when either self-covariance is zero (paper and code: 0/0 = 0), which
    covers constant inputs.
    """
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("sample sizes do not match.")
    return float(pc_squared_columns(x[:, None], y, convention=convention)[0])


def pc_squared_columns(X, y, convention="author_code", batch=64):
    """PC(X_j, y)^2 for every column of X, sharing the y terms. Exact, O(p n^2)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("sample sizes do not match.")
    cy = _centered_signs(y)
    syy = _pcov_terms(cy, cy, convention)
    out = np.zeros(X.shape[1])
    for start in range(0, X.shape[1], batch):
        cx = _centered_signs(X[:, start : start + batch].T)
        sxx = _pcov_terms(cx, cx, convention)
        sxy = _pcov_terms(cx, cy[None], convention)
        denom = np.sqrt(sxx * syy)
        out[start : start + batch] = np.divide(sxy, denom, out=np.zeros_like(sxy), where=denom > 0)
    return out


# ── Fixed-X equicorrelated knockoffs (author get_equi_features) ─────────────
def equicorrelated_knockoffs(X, rng):
    """Port of the author's get_equi_features with an explicit random stream.

    Columns are scaled to unit norm without centering, Sigma = Xs'Xs, and
    s = min(1, 2 lambda_min(Sigma)) - 1e-5. The returned knockoff satisfies the
    fixed-X identities X~'X~ = Sigma and Xs'X~ = Sigma - s I (Barber and Candes, 2015).
    Upstream orthonormalizes [Xs, N] by classical Gram-Schmidt; this uses Householder
    QR with columns signed so diag(R) > 0, which is the same basis in exact arithmetic
    and more stable. N ~ iid N(0, 1) of shape (n, d), drawn as rng.standard_normal.

    Note: the paper prints (3.2) as h_j = 2 lambda_min v 1 (the maximum). The author
    code and Barber and Candes use the minimum; the maximum would not give a valid
    Gram matrix in general. We follow the code.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if 2 * d > n:
        raise ValueError(f"fixed-X knockoffs need n >= 2d; got n={n}, d={d}")
    scale = np.sqrt(np.sum(X**2, axis=0))
    if np.any(scale == 0):
        raise ValueError("zero-norm column in the knockoff family")
    Xs = X / scale
    sigma = Xs.T @ Xs
    lambda_min = float(np.linalg.eigvalsh(sigma)[0])
    s = min(1.0, 2.0 * lambda_min) - EQUI_SLACK
    if s <= 0:
        raise np.linalg.LinAlgError(f"singular Gram matrix: lambda_min={lambda_min:.3g}")
    sigma_inv = np.linalg.inv(sigma)
    C = np.linalg.cholesky(2.0 * s * np.eye(d) - s * s * sigma_inv).T
    noise = rng.standard_normal((n, d))
    Q, R = np.linalg.qr(np.hstack([Xs, noise]))
    Q = Q * np.sign(np.diag(R))
    U = Q[:, d:]
    knockoffs = Xs @ (np.eye(d) - s * sigma_inv) + U @ C
    diagnostics = {
        "lambda_min": lambda_min,
        "s": s,
        "gram_error": float(np.max(np.abs(knockoffs.T @ knockoffs - sigma))),
        "cross_error": float(np.max(np.abs(Xs.T @ knockoffs - (sigma - s * np.eye(d))))),
    }
    return knockoffs, diagnostics


# ── Knockoff+ threshold (3.6) ───────────────────────────────────────────────
def knockoff_plus_threshold(W, alpha, offset=1):
    """T = min{t in |W| minus 0 : (offset + #{W <= -t}) / #{W >= t} <= alpha}; inf if none."""
    W = np.asarray(W, dtype=float)
    for t in np.sort(np.unique(np.abs(W[W != 0]))):
        positives = np.sum(W >= t)
        if positives and (offset + np.sum(W <= -t)) / positives <= alpha:
            return float(t)
    return float("inf")


# ── Procedure ───────────────────────────────────────────────────────────────
@dataclass
class PCKnockoffResult:
    screen_rows: np.ndarray
    test_rows: np.ndarray
    screen_pc2: np.ndarray  # all p features, screening rows
    candidates: np.ndarray  # top d, original indices, rank order
    W: np.ndarray  # aligned with candidates
    pc2_original: np.ndarray
    pc2_knockoff: np.ndarray
    thresholds: dict
    selected: dict  # alpha -> sorted original indices
    knockoff_diagnostics: dict
    timings_s: dict = field(default_factory=dict)


def pc_knockoff(X, y, *, n_screen, d, alphas, partition_seed, knockoff_seed, convention="author_code"):
    """Run Algorithm 1 once. The PC evidence is shared by every alpha."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    n_test = n - n_screen
    if not 0 < n_screen < n or not 0 < d <= p:
        raise ValueError("invalid n_screen or d")
    if not 2 * d < n_test:
        raise ValueError(f"published restriction 2d < n_test fails: d={d}, n_test={n_test}")
    timings = {}

    started = time.perf_counter()
    perm = np.random.default_rng(partition_seed).permutation(n)
    screen_rows, test_rows = np.sort(perm[:n_screen]), np.sort(perm[n_screen:])
    timings["partition"] = time.perf_counter() - started

    # Screening reads only screening rows. Ties in PC are broken by feature index.
    started = time.perf_counter()
    screen_pc2 = pc_squared_columns(X[screen_rows], y[screen_rows], convention=convention)
    candidates = np.argsort(-screen_pc2, kind="stable")[:d]
    timings["screening"] = time.perf_counter() - started

    started = time.perf_counter()
    X_test, y_test = X[np.ix_(test_rows, candidates)], y[test_rows]
    knockoffs, diagnostics = equicorrelated_knockoffs(X_test, np.random.default_rng(knockoff_seed))
    timings["knockoffs"] = time.perf_counter() - started

    started = time.perf_counter()
    pc2_original = pc_squared_columns(X_test, y_test, convention=convention)
    pc2_knockoff = pc_squared_columns(knockoffs, y_test, convention=convention)
    W = pc2_original - pc2_knockoff
    timings["statistics"] = time.perf_counter() - started

    started = time.perf_counter()
    thresholds, selected = {}, {}
    for alpha in alphas:
        t = knockoff_plus_threshold(W, alpha, offset=1)
        thresholds[alpha] = t
        selected[alpha] = np.sort(candidates[W >= t]) if np.isfinite(t) else np.array([], dtype=int)
    timings["thresholds"] = time.perf_counter() - started

    return PCKnockoffResult(
        screen_rows=screen_rows,
        test_rows=test_rows,
        screen_pc2=screen_pc2,
        candidates=candidates,
        W=W,
        pc2_original=pc2_original,
        pc2_knockoff=pc2_knockoff,
        thresholds=thresholds,
        selected=selected,
        knockoff_diagnostics=diagnostics,
        timings_s=timings,
    )
