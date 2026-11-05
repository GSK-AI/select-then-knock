"""The evaluated regression LassoCV coefficient-difference statistic."""

import warnings

import numpy as np
from sklearn.linear_model import LassoCV


def compute_lasso_path_statistics(X, X_knockoff, y, task="regression"):
    if task != "regression":
        raise ValueError("Only regression is supported")
    p = X.shape[1]
    X_augmented = np.column_stack([X, X_knockoff])
    model = LassoCV(cv=5, max_iter=10000, n_jobs=1, random_state=42)
    try:
        model.fit(X_augmented, y)
        coef = model.coef_
        return np.abs(coef[:p]) - np.abs(coef[p:])
    except Exception as error:
        warnings.warn(f"Lasso path failed: {error}. Returning zero statistics.")
        return np.zeros(p)
