"""Selection-only STK: pooled selection or repeated held-out knockoff filtering.

The cross-split aggregate has no proved FDR or PFER guarantee. Candidate counts
are floors, not caps.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from .logging import get_logger
from .phase2_tuning import tune_hyperparameters
from .phase3_stability import stability_selection
from .preprocessing import apply_preprocessing_to_new_data, preprocess_for_modeling

logger = get_logger(__name__)


@dataclass
class MultiSplitResults:
    """Multi-split records in Original feature space."""

    selection_frequencies: np.ndarray
    selected_features: np.ndarray
    n_splits: int
    eta: float
    per_run_fdr: float
    split_frac: float
    per_split_sizes: list[int] = field(default_factory=list)
    min_knockoff_candidates: int = 0


class SelectThenKnock(BaseEstimator):
    """Return candidate and selected-feature indices, without prediction refits.

    ``headline`` is Headline STK (50/50 rows, candidate floor 10).
    ``secondary`` is Secondary STK (70/30 rows, floor 20). Both use 25 splits,
    frequency threshold 0.5, and per-split knockoff level 0.5. These are empirical
    nomination configurations, not implementations of a proved aggregate guarantee.
    ``pooled`` is the same-sample Pooled STK comparison configuration.
    """

    def __init__(
        self,
        task="regression",
        run_diagnostics=False,
        variance_threshold=0.0,
        alpha_range=(0.005, 0.5),
        l1_ratio_range=(0.25, 0.99),
        n_alphas=20,
        n_l1_ratios=10,
        cv_folds=5,
        max_iter=10000,
        tol=1e-4,
        solver_backend="sklearn",
        n_bootstrap=1000,
        subsample_ratio=0.5,
        target_pfer=7.0,
        fdr=0.1,
        knockoff_offset=1,
        min_knockoff_candidates=None,
        selection_mode="pooled",
        n_splits=25,
        eta=0.5,
        per_run_fdr=0.5,
        split_frac=0.5,
        random_state=None,
        n_jobs=-1,
        verbose=False,
    ):
        self.task = task
        self.run_diagnostics = run_diagnostics
        self.variance_threshold = variance_threshold
        self.alpha_range = alpha_range
        self.l1_ratio_range = l1_ratio_range
        self.n_alphas = n_alphas
        self.n_l1_ratios = n_l1_ratios
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.tol = tol
        self.solver_backend = solver_backend
        self.n_bootstrap = n_bootstrap
        self.subsample_ratio = subsample_ratio
        self.target_pfer = target_pfer
        self.fdr = fdr
        self.knockoff_offset = knockoff_offset
        self.min_knockoff_candidates = min_knockoff_candidates
        self.selection_mode = selection_mode
        self.n_splits = n_splits
        self.eta = eta
        self.per_run_fdr = per_run_fdr
        self.split_frac = split_frac
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    _PRESETS = {
        "pooled": {
            "n_bootstrap": 500,
            "fdr": 0.1,
            "knockoff_offset": 1,
            "target_pfer": 7.0,
        },
        "headline": {
            "selection_mode": "multi_split",
            "knockoff_offset": 1,
            "fdr": 0.1,
            "min_knockoff_candidates": 10,
            "per_run_fdr": 0.5,
            "eta": 0.5,
            "n_splits": 25,
            "split_frac": 0.5,
            "n_bootstrap": 200,
            "alpha_range": (0.005, 0.5),
            "n_alphas": 20,
            "n_l1_ratios": 3,
            "cv_folds": 3,
        },
        "secondary": {
            "selection_mode": "multi_split",
            "knockoff_offset": 1,
            "fdr": 0.1,
            "min_knockoff_candidates": 20,
            "per_run_fdr": 0.5,
            "eta": 0.5,
            "n_splits": 25,
            "split_frac": 0.7,
            "n_bootstrap": 200,
            "alpha_range": (0.005, 0.5),
            "n_alphas": 20,
            "n_l1_ratios": 3,
            "cv_folds": 3,
        },
    }

    @classmethod
    def from_preset(cls, preset, task="regression", **overrides):
        if preset not in cls._PRESETS:
            raise ValueError(f"Unknown preset {preset!r}; use one of {list(cls._PRESETS)}")
        return cls(**{**cls._PRESETS[preset], "task": task, **overrides})

    def fit(self, X, y, feature_names=None):
        """Select features. Every fit starts with fresh fitted state."""
        if self.task != "regression" or self.run_diagnostics:
            raise ValueError("Only regression with disabled outlier diagnostics is supported")
        if self.solver_backend not in {"sklearn", "celer"}:
            raise ValueError("solver_backend must be 'sklearn' or 'celer'")
        if self.selection_mode not in {"pooled", "multi_split"}:
            raise ValueError("selection_mode must be 'pooled' or 'multi_split'")
        original_level = logger.level
        try:
            logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
            self._preprocess_impl(X, y, feature_names)
            self.phase2_results_ = None
            self.phase3_results_ = None
            self.phase3_expanded_features_ = None
            self.phase4_results_ = None
            self.selection_frequencies_ = None
            self.multi_split_results_ = None
            self.selected_features_ = None
            self.optimal_alpha_ = None
            self.optimal_l1_ratio_ = None
            if self.selection_mode == "pooled":
                return self._fit_impl()
            return self._fit_multi_split(X, y)
        finally:
            logger.setLevel(original_level)

    def _preprocess_impl(self, X, y, feature_names=None):
        X_arr = check_array(X, accept_sparse=False, ensure_2d=True)
        y_arr = np.asarray(y).ravel()
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError(f"X and y have inconsistent lengths: {X_arr.shape[0]} vs {len(y_arr)}")
        self.n_samples_in_, self.n_features_in_ = X_arr.shape
        self.feature_names_in_ = None if feature_names is None else np.asarray(feature_names)
        self.preprocessing_results_ = preprocess_for_modeling(
            X_arr,
            y_arr,
            task=self.task,
            feature_names=feature_names,
            variance_threshold=self.variance_threshold,
        )
        self.y_scale_ = self.preprocessing_results_.y_scale

    def _fit_impl(self):
        pre = self.preprocessing_results_
        X_clean, y_clean = pre.X_processed, pre.y_processed
        self.phase2_results_ = tune_hyperparameters(
            X_clean,
            y_clean,
            task=self.task,
            alpha_range=self.alpha_range,
            l1_ratio_range=self.l1_ratio_range,
            n_alphas=self.n_alphas,
            n_l1_ratios=self.n_l1_ratios,
            cv_folds=self.cv_folds,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self.optimal_alpha_ = self.phase2_results_.optimal_alpha
        self.optimal_l1_ratio_ = self.phase2_results_.optimal_l1_ratio
        self.phase3_results_ = stability_selection(
            X_clean,
            y_clean,
            task=self.task,
            alpha=self.optimal_alpha_,
            l1_ratio=self.optimal_l1_ratio_,
            n_bootstrap=self.n_bootstrap,
            subsample_ratio=self.subsample_ratio,
            target_pfer=self.target_pfer,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            solver_backend=self.solver_backend,
        )
        candidates = self._maybe_expand_phase3_for_knockoff(self.phase3_results_.stable_features)
        if getattr(self, "_candidates_only", False):
            return self
        from .phase4_knockoff import knockoff_filter

        if len(candidates):
            self.phase4_results_ = knockoff_filter(
                X_clean[:, candidates],
                y_clean,
                task=self.task,
                fdr=self.fdr,
                offset=self.knockoff_offset,
                random_seed=self.random_state,
            )
            self.selected_features_ = candidates[self.phase4_results_.selected_features]
        else:
            self.selected_features_ = np.array([], dtype=int)
        return self

    def _effective_min_knockoff_candidates(self):
        requested = self.min_knockoff_candidates
        if requested is None:
            return int(np.ceil(self.knockoff_offset / self.fdr))
        if isinstance(requested, bool) or not isinstance(requested, (int, np.integer)) or requested < 1:
            raise ValueError(f"min_knockoff_candidates must be a positive int or None, got {requested!r}")
        return int(requested)

    def _maybe_expand_phase3_for_knockoff(self, phase3_indices):
        """Preserve floor expansion even when the child's unused filter is skipped."""
        minimum = self._effective_min_knockoff_candidates()
        self.phase3_expanded_features_ = None
        if self.knockoff_offset == 0 and self.min_knockoff_candidates is None:
            return phase3_indices
        if len(phase3_indices) >= minimum:
            return phase3_indices
        scores = self.phase3_results_.stability_scores
        top_k = min(minimum, len(scores))
        expanded = np.sort(np.argsort(scores)[-top_k:])
        self.phase3_expanded_features_ = expanded.copy()
        return expanded

    def _fit_multi_split(self, X, y):
        from .phase4_knockoff import knockoff_filter

        n_splits, eta, level, split_frac = self.n_splits, self.eta, self.per_run_fdr, self.split_frac
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if not 0.0 < eta <= 1.0:
            raise ValueError(f"eta must be in (0, 1], got {eta}")
        if not 0.0 < level < 1.0:
            raise ValueError(f"per_run_fdr must be in (0, 1), got {level}")
        if not 0.0 < split_frac < 1.0:
            raise ValueError(f"split_frac must be in (0, 1), got {split_frac}")
        minimum = self._effective_min_knockoff_candidates()
        required = int(np.ceil(self.knockoff_offset / level))
        if minimum < required:
            raise ValueError(f"Candidate floor {minimum} is below the {required} needed at per_run_fdr={level}")
        X_arr = check_array(X, accept_sparse=False, ensure_2d=True)
        y_arr = np.asarray(y).ravel()
        n_samples, n_features = X_arr.shape
        base = self.random_state if self.random_state is not None else int(np.random.default_rng().integers(0, 2**31))
        split_seeds = np.random.SeedSequence(base).generate_state(n_splits)
        cut = int(round(n_samples * split_frac))
        if not 0 < cut < n_samples:
            raise ValueError(f"split_frac={split_frac} yields an empty screening or testing partition")
        counts = np.zeros(n_features, dtype=int)
        per_split_sizes = []
        for split_seed in split_seeds:
            selected = self._one_split(X_arr, y_arr, cut, int(split_seed), level, knockoff_filter)
            per_split_sizes.append(int(selected.size))
            if selected.size:
                counts[selected] += 1
        frequencies = counts / n_splits
        selected_original = np.flatnonzero(frequencies >= eta)
        self.selection_frequencies_ = frequencies
        self.multi_split_results_ = MultiSplitResults(
            frequencies,
            selected_original,
            n_splits,
            eta,
            level,
            split_frac,
            per_split_sizes,
            minimum,
        )
        original_from_clean = np.flatnonzero(self.preprocessing_results_.feature_mask)
        clean_from_original = {int(original): clean for clean, original in enumerate(original_from_clean)}
        selected_clean = []
        for original in selected_original:
            clean = clean_from_original.get(int(original))
            if clean is None:
                logger.warning("Selected original feature %d was removed by the global variance filter", original)
            else:
                selected_clean.append(clean)
        self.selected_features_ = np.array(sorted(selected_clean), dtype=int)
        return self

    def _one_split(self, X_arr, y_arr, cut, split_seed, per_run_fdr, knockoff_filter):
        """Screen on A; filter on B; return Original-space indices.

        The child still tunes, standardizes, subsamples, and expands its candidate
        floor. Only its discarded pooled filter and prediction refits are omitted.
        """
        rng = np.random.default_rng(split_seed)
        permutation = rng.permutation(X_arr.shape[0])
        a, b = permutation[:cut], permutation[cut:]
        child = self.__class__(**{**self.get_params(), "selection_mode": "pooled", "random_state": split_seed})
        child._candidates_only = True
        child.fit(X_arr[a], y_arr[a])
        candidates = child.phase3_expanded_features_
        if candidates is None:
            candidates = child.phase3_results_.stable_features
        candidates = np.asarray(candidates, dtype=int)
        original_from_clean = np.flatnonzero(child.preprocessing_results_.feature_mask)
        original_candidates = original_from_clean[candidates] if candidates.size else np.array([], dtype=int)
        if not original_candidates.size:
            return np.array([], dtype=int)
        X_test = apply_preprocessing_to_new_data(X_arr[b], child.preprocessing_results_)[:, candidates]
        result = knockoff_filter(
            X_test,
            y_arr[b],
            task=self.task,
            fdr=per_run_fdr,
            knockoff_type="gaussian",
            method="lasso",
            offset=self.knockoff_offset,
            random_seed=split_seed,
            knockoff_construction="equi",
        )
        return original_candidates[result.selected_features]

    def get_support(self, indices=False, space="preprocessed"):
        """Return a mask or indices in Original or Preprocessed space."""
        check_is_fitted(self, ["selected_features_", "preprocessing_results_"])
        if self.selected_features_ is None:
            raise RuntimeError("SelectThenKnock is not fitted")
        if space == "original":
            original_indices = np.where(self.preprocessing_results_.feature_mask)[0]
            selected = original_indices[self.selected_features_]
            size = self.n_features_in_
        elif space == "preprocessed":
            selected = self.selected_features_
            size = self.preprocessing_results_.X_processed.shape[1]
        else:
            raise ValueError("space must be 'original' or 'preprocessed'")
        if indices:
            return selected
        mask = np.zeros(size, dtype=bool)
        mask[selected] = True
        return mask
