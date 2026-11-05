"""Gaussian knockoff filter with preserved scaling, threshold, and failure policy."""

from dataclasses import dataclass

import numpy as np

from select_then_knock.logging import get_logger

from .fdr_control import estimate_fdr, knockoff_threshold, select_features_knockoff
from .feature_statistics import compute_lasso_path_statistics
from .knockoff_generator import generate_gaussian_knockoffs, validate_knockoffs

logger = get_logger(__name__)


@dataclass
class KnockoffResults:
    """Indices are local to the candidate matrix; threshold ratios are descriptive."""

    selected_features: np.ndarray
    W_statistics: np.ndarray
    threshold: float
    fdr_target: float
    fdr_estimate: float
    X_knockoff: np.ndarray
    n_selected: int
    knockoff_type: str
    method: str
    fallback_used: bool = False


def knockoff_filter(
    X,
    y,
    task="regression",
    fdr=0.1,
    knockoff_type="gaussian",
    method="lasso",
    offset=1,
    random_seed=None,
    knockoff_construction="equi",
):
    """Apply the evaluated Gaussian/equicorrelated/regression procedure.

    Estimated Gaussian construction does not establish exact Model-X validity.
    A permutation fallback is retained and reported, not treated as valid inference.
    """
    if (task, knockoff_type, method, knockoff_construction) != ("regression", "gaussian", "lasso", "equi"):
        raise ValueError("Only regression with Gaussian equicorrelated knockoffs and lasso statistics is supported")
    col_std = X.std(axis=0, ddof=0)
    col_std[col_std == 0] = 1.0
    X_std = X / col_std
    X_knockoff, S, fallback_used = generate_gaussian_knockoffs(X_std, random_state=random_seed)
    if fallback_used:
        logger.warning("Knockoff construction used the permutation fallback; no error-control guarantee applies")
    validation = validate_knockoffs(X_std, X_knockoff, S=S)
    if not validation["valid"]:
        logger.warning("Knockoff construction failed validation: %s", validation)
    W = compute_lasso_path_statistics(X_std, X_knockoff, y, task=task)
    T = knockoff_threshold(W, fdr=fdr, offset=offset)
    selected = select_features_knockoff(W, fdr=fdr, offset=offset)
    return KnockoffResults(
        selected_features=selected,
        W_statistics=W,
        threshold=T,
        fdr_target=fdr,
        fdr_estimate=estimate_fdr(W, selected, offset=offset),
        X_knockoff=X_knockoff * col_std,
        n_selected=len(selected),
        knockoff_type=knockoff_type,
        method=method,
        fallback_used=fallback_used,
    )
