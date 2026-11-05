"""Candidate selection using subsample frequencies and the evaluated calibration."""

from dataclasses import dataclass

import numpy as np

from .bootstrap import compute_stability_scores, stability_bootstrap
from .threshold_selection import suggest_threshold_pfer


@dataclass
class StabilityResults:
    stability_scores: np.ndarray
    selection_matrix: np.ndarray
    stable_features: np.ndarray
    threshold: float
    n_bootstrap: int
    n_stable: int
    alpha: float
    l1_ratio: float


def select_stable_features(stability_scores, threshold=0.6):
    return np.where(stability_scores >= threshold)[0]


def stability_selection(
    X,
    y,
    task,
    alpha,
    l1_ratio,
    n_bootstrap=100,
    subsample_ratio=0.5,
    target_pfer=1.0,
    random_state=None,
    n_jobs=1,
    solver_backend="sklearn",
):
    selection_matrix = stability_bootstrap(
        X,
        y,
        task=task,
        alpha=alpha,
        l1_ratio=l1_ratio,
        n_bootstrap=n_bootstrap,
        subsample_ratio=subsample_ratio,
        threshold=1e-5,
        random_state=random_state,
        n_jobs=n_jobs,
        solver_backend=solver_backend,
    )
    scores = compute_stability_scores(selection_matrix)
    avg_selected = float(selection_matrix.sum(axis=1).mean())
    threshold = suggest_threshold_pfer(target_pfer=target_pfer, n_features=X.shape[1], avg_selected=avg_selected)
    stable_features = select_stable_features(scores, threshold)
    return StabilityResults(
        scores,
        selection_matrix,
        stable_features,
        threshold,
        n_bootstrap,
        len(stable_features),
        alpha,
        l1_ratio,
    )
