"""Knockoff threshold arithmetic; its ratio is not an STK aggregate FDR estimate."""

import numpy as np


def knockoff_threshold(W, fdr=0.1, offset=1):
    """Return the smallest qualifying observed nonzero absolute statistic."""
    if not 0 < fdr < 1:
        raise ValueError(f"FDR must be in (0, 1), got {fdr}")
    abs_W = np.abs(W)
    thresholds = np.sort(np.unique(abs_W[abs_W > 0]))
    for t in thresholds:
        num_selected = np.sum(W >= t)
        if num_selected == 0:
            continue
        num_knockoffs = np.sum(W <= -t)
        if (num_knockoffs + offset) / num_selected <= fdr:
            return float(t)
    return np.inf


def select_features_knockoff(W, fdr=0.1, offset=1):
    T = knockoff_threshold(W, fdr=fdr, offset=offset)
    if np.isinf(T):
        return np.array([], dtype=int)
    return np.where(W >= T)[0]


def estimate_fdr(W, selected, offset=1):
    """Return the historical threshold ratio, retained for record compatibility."""
    if len(selected) == 0:
        return 0.0
    T = W[selected].min()
    num_knockoffs = np.sum(W <= -T)
    return float(min((num_knockoffs + offset) / len(selected), 1.0))
