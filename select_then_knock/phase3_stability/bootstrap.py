"""The evaluated regression subsampling sequence and failure denominator."""

import warnings

import numpy as np
from joblib import Parallel, delayed

from select_then_knock.logging import get_logger

from ..phase2_tuning import ElasticNetWrapper

logger = get_logger(__name__)


def _fit_on_bootstrap_checked(X, X_boot, y_boot, task, alpha, l1_ratio, threshold, random_state, solver_backend):
    """Return the selection mask and success flag; failed fits are not empty evidence."""
    try:
        wrapper = ElasticNetWrapper(
            task=task,
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state,
            solver_backend=solver_backend,
        )
        model = wrapper._create_model(X_boot.shape[0])
        model.fit(X_boot, y_boot)
        coef = np.asarray(model.coef_)
        selected_indices = np.where(np.abs(coef) > threshold)[0]
        selected = np.zeros(X.shape[1], dtype=int)
        selected[selected_indices] = 1
        return selected, True
    except Exception as error:
        warnings.warn(f"Bootstrap fit failed: {error}. Dropping this iteration.")
        return np.zeros(X.shape[1], dtype=int), False


def stability_bootstrap(
    X,
    y,
    task,
    alpha,
    l1_ratio,
    n_bootstrap=100,
    subsample_ratio=0.5,
    threshold=1e-5,
    random_state=None,
    n_jobs=1,
    solver_backend="sklearn",
):
    """Draw all row indices first, preserving the original RandomState seed schedule.

    Sampling is without replacement. A failed fit is excluded from the denominator;
    all failed fits raise. Parallel failure falls back to the same sequential fits.
    """
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0. Use 1 for sequential or -1 for all cores.")
    n_subsample = int(len(y) * subsample_ratio)
    all_indices = []
    for i in range(n_bootstrap):
        seed = None if random_state is None else random_state + i
        rng = np.random.RandomState(seed)
        all_indices.append(rng.choice(len(y), size=n_subsample, replace=False))

    def single_bootstrap(i):
        seed = None if random_state is None else random_state + i
        indices = all_indices[i]
        return _fit_on_bootstrap_checked(
            X,
            X[indices],
            y[indices],
            task=task,
            alpha=alpha,
            l1_ratio=l1_ratio,
            threshold=threshold,
            random_state=seed,
            solver_backend=solver_backend,
        )

    if n_jobs != 1 and n_bootstrap >= 50:
        try:
            results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
                delayed(single_bootstrap)(i) for i in range(n_bootstrap)
            )
        except Exception as error:
            logger.warning("Parallel execution failed: %s. Falling back to sequential.", error)
            results = [single_bootstrap(i) for i in range(n_bootstrap)]
    else:
        results = [single_bootstrap(i) for i in range(n_bootstrap)]
    selection_matrix = np.array([mask for mask, _ in results])
    succeeded = np.array([ok for _, ok in results], dtype=bool)
    n_failed = int((~succeeded).sum())
    if n_failed:
        logger.warning("%d/%d bootstrap fits failed and were excluded from the denominator", n_failed, len(succeeded))
        if n_failed == len(succeeded):
            raise RuntimeError(f"All {n_failed} bootstrap fits failed; stability selection has no data to score.")
        selection_matrix = selection_matrix[succeeded]
    return selection_matrix


def compute_stability_scores(selection_matrix):
    return selection_matrix.mean(axis=0)
