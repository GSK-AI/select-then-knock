"""The evaluated regression solver settings, without prediction APIs."""

import importlib
import warnings

from sklearn.linear_model import ElasticNet


class ElasticNetWrapper:
    """Create the solver used by each stability-selection replicate."""

    def __init__(
        self,
        task="regression",
        alpha=0.5,
        l1_ratio=0.5,
        max_iter=10000,
        tol=1e-4,
        random_state=None,
        solver_backend="sklearn",
    ):
        self.task = task
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.solver_backend = solver_backend

    def _create_model(self, n_samples):
        if self.task != "regression":
            raise ValueError("Only regression is supported")
        if self.solver_backend == "celer":
            try:
                celer = importlib.import_module("celer")
                return celer.ElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
            except ImportError:
                warnings.warn("celer not installed, falling back to sklearn")
        elif self.solver_backend != "sklearn":
            raise ValueError("solver_backend must be 'sklearn' or 'celer'")
        return ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            selection="random",
        )
