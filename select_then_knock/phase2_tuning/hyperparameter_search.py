"""The evaluated ElasticNetCV search, without redundant prediction refits."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import ElasticNetCV


@dataclass
class TuningResults:
    optimal_alpha: float
    optimal_l1_ratio: float


def tune_hyperparameters(
    X,
    y,
    task="regression",
    alpha_range=(0.005, 0.5),
    l1_ratio_range=(0.1, 0.9),
    n_alphas=20,
    n_l1_ratios=9,
    cv_folds=5,
    max_iter=10000,
    tol=1e-4,
    random_state=None,
):
    """Use the original alpha grid, inner folds, and solver parameters.

    Only the chosen penalty parameters are retained; no prediction-score output
    or extra best-model fit is computed.
    """
    if task != "regression":
        raise ValueError("Only regression is supported")
    alphas = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas)
    l1_ratios_list = np.linspace(l1_ratio_range[0], l1_ratio_range[1], n_l1_ratios).tolist()
    model = ElasticNetCV(
        l1_ratio=l1_ratios_list,
        alphas=alphas,
        cv=cv_folds,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_jobs=-1,
    )
    try:
        model.fit(X, y)
        return TuningResults(float(model.alpha_), float(model.l1_ratio_))
    except Exception as error:
        raise RuntimeError(f"Elastic Net tuning failed: {error}. Check your data and parameter ranges.") from error
