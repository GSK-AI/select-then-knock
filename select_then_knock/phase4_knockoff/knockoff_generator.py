"""Estimated Gaussian equicorrelated knockoffs and their numerical diagnostics."""

import warnings

import numpy as np
from scipy import linalg


def _floor_min_eigenvalue(M, floor):
    min_eig = linalg.eigvalsh(M)[0]
    if min_eig < floor:
        M = M + np.eye(M.shape[0]) * (floor - min_eig)
    return M


def compute_equi_s_matrix(Sigma):
    p = Sigma.shape[0]
    eigenvalues = linalg.eigvalsh(Sigma)
    lambda_min = eigenvalues[0]
    s_val = min(2 * lambda_min * 0.9999, 1.0)
    if s_val <= 0:
        warnings.warn(f"Sigma not positive definite (min eigenvalue={lambda_min:.4f}). Using small positive S.")
        s_val = 1e-3
    S = np.diag(np.ones(p) * s_val)
    min_eig = linalg.eigvalsh(2 * Sigma - S)[0]
    if min_eig < -1e-6:
        warnings.warn(f"2*Sigma - S not PSD (min eigenvalue={min_eig:.4f}). Knockoffs may not be valid.")
    return S


def generate_gaussian_knockoffs(X, random_state=None):
    """Keep the original draw sequence, covariance regularization, and fallback.

    These are approximate, estimated second-order knockoffs. A numerical failure
    returns permuted columns and a true fallback flag; it does not certify inference.
    """
    n, p = X.shape
    rng = np.random.default_rng(random_state)
    mu = X.mean(axis=0)
    X_centered = X - mu
    Sigma = (X_centered.T @ X_centered) / n
    Sigma = _floor_min_eigenvalue(Sigma, 1e-4)
    S = compute_equi_s_matrix(Sigma)
    try:
        Sigma_inv_S = linalg.solve(Sigma, S, assume_a="pos")
        X_centered = X - mu
        mu_tilde = X - X_centered @ Sigma_inv_S
        V_tilde = 2 * S - S @ Sigma_inv_S
        V_tilde = 0.5 * (V_tilde + V_tilde.T)
        Z = rng.standard_normal((n, p))
        L = linalg.cholesky(V_tilde, lower=True)
        X_knockoff = mu_tilde + Z @ L.T
    except linalg.LinAlgError as error:
        warnings.warn(f"Knockoff generation failed ({error}). Returning random permutations as fallback.")
        X_knockoff = np.zeros_like(X)
        for j in range(p):
            perm = rng.permutation(n)
            X_knockoff[:, j] = X[perm, j]
        S = np.diag(np.ones(p) * 0.5)
        return X_knockoff, S, True
    return X_knockoff, S, False


def validate_knockoffs(X, X_knockoff, tol=1e-6, S=None):
    """Check finite values and S feasibility; moment errors are descriptive only."""
    if X_knockoff.shape != X.shape:
        return {"valid": False, "error": "Shape mismatch"}
    if not np.isfinite(X).all() or not np.isfinite(X_knockoff).all():
        return {"valid": False, "error": "Non-finite values"}
    n, p = X.shape
    mu = X.mean(axis=0)
    X_centered = X - mu
    X_knockoff_centered = X_knockoff - mu
    Sigma = _floor_min_eigenvalue((X_centered.T @ X_centered) / n, 1e-4)
    Gram_knockoff = X_knockoff_centered.T @ X_knockoff_centered / n
    Cross_gram = X_centered.T @ X_knockoff_centered / n
    gram_diff = np.abs(Gram_knockoff - Sigma).max()
    if S is None:
        return {"valid": True, "gram_diff": float(gram_diff), "cross_diff": None, "tol": tol}
    S = np.asarray(S, dtype=float)
    if S.shape != (p, p) or not np.isfinite(S).all():
        return {
            "valid": False,
            "error": "Invalid S shape or values",
            "gram_diff": float(gram_diff),
            "cross_diff": None,
            "tol": tol,
        }
    cross_diff = np.abs(Cross_gram - (Sigma - S)).max()
    symmetry_error = np.abs(S - S.T).max()
    off_diagonal_error = np.abs(S - np.diag(np.diag(S))).max()
    min_s, max_s = float(np.diag(S).min()), float(np.diag(S).max())
    min_psd_eigenvalue = float(linalg.eigvalsh(2 * Sigma - S)[0])
    valid = (
        symmetry_error <= tol
        and off_diagonal_error <= tol
        and min_s >= -tol
        and max_s <= 1.0 + tol
        and min_psd_eigenvalue >= -tol
    )
    return {
        "valid": valid,
        "gram_diff": float(gram_diff),
        "cross_diff": float(cross_diff),
        "symmetry_error": float(symmetry_error),
        "off_diagonal_error": float(off_diagonal_error),
        "min_s": min_s,
        "max_s": max_s,
        "min_psd_eigenvalue": min_psd_eigenvalue,
        "tol": tol,
    }
