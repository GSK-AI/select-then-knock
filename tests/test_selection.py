"""Tests for the retained selection, scaling, mapping, and failure contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import linalg

from select_then_knock import SelectThenKnock
from select_then_knock.phase3_stability import bootstrap
from select_then_knock.phase3_stability.threshold_selection import suggest_threshold_pfer
from select_then_knock.phase4_knockoff.fdr_control import knockoff_threshold
from select_then_knock.phase4_knockoff.knockoff_generator import generate_gaussian_knockoffs, validate_knockoffs
from select_then_knock.preprocessing import apply_preprocessing_to_new_data, preprocess_for_modeling


@pytest.mark.parametrize("preset,fraction,floor", [("headline", 0.5, 10), ("secondary", 0.7, 20)])
def test_evaluated_presets(preset, fraction, floor):
    model = SelectThenKnock.from_preset(preset)
    assert model.selection_mode == "multi_split"
    assert model.min_knockoff_candidates == floor
    assert (model.n_splits, model.eta, model.per_run_fdr, model.knockoff_offset) == (25, 0.5, 0.5, 1)
    assert model.split_frac == fraction
    assert model._effective_min_knockoff_candidates() == floor
    assert (model.n_bootstrap, model.n_alphas, model.n_l1_ratios, model.cv_folds) == (200, 20, 3, 3)
    assert model.alpha_range == (0.005, 0.5)
    assert model.l1_ratio_range == (0.25, 0.99)
    assert model.target_pfer == 7.0
    assert not model.run_diagnostics


def test_unused_capabilities_are_absent():
    model = SelectThenKnock()
    for method in ("predict", "predict_proba", "score", "fit_pyramid", "load", "summary"):
        assert not hasattr(model, method)


def test_scale_only_preprocessing_and_held_out_mapping():
    X = np.array([[0.0, 7.0, 1.0], [2.0, 7.0, 3.0], [4.0, 7.0, 5.0]])
    y = np.array([10.0, 20.0, 30.0])
    result = preprocess_for_modeling(X, y, feature_names=["a", "constant", "c"])
    assert result.feature_mask.tolist() == [True, False, True]
    assert result.feature_names.tolist() == ["a", "c"]
    assert result.X_processed[0, 0] == 0
    np.testing.assert_allclose(result.X_processed.std(axis=0), 1.0)
    np.testing.assert_allclose(result.y_processed, y / y.std())
    assert result.y_scale == y.std()
    np.testing.assert_allclose(apply_preprocessing_to_new_data(X, result), result.X_processed)


def test_constant_target_is_not_divided_by_zero():
    X = np.arange(20, dtype=float).reshape(10, 2)
    result = preprocess_for_modeling(X, np.ones(10))
    assert result.y_scale == 1.0
    np.testing.assert_array_equal(result.y_processed, np.ones(10))


def test_candidate_floor_is_not_a_cap():
    model = SelectThenKnock.from_preset("headline")
    scores = np.linspace(0.6, 1, 30)
    model.phase3_results_ = SimpleNamespace(stability_scores=scores)
    natural = np.arange(30)
    np.testing.assert_array_equal(model._maybe_expand_phase3_for_knockoff(natural), natural)
    expanded = model._maybe_expand_phase3_for_knockoff(np.array([29]))
    np.testing.assert_array_equal(expanded, np.arange(20, 30))
    np.testing.assert_array_equal(model.phase3_expanded_features_, expanded)
    model.phase3_results_ = SimpleNamespace(stability_scores=np.arange(4))
    assert len(model._maybe_expand_phase3_for_knockoff(np.array([], dtype=int))) == 4


@pytest.mark.parametrize("requested", [True, 0, -1, 3.5])
def test_invalid_candidate_floors_are_rejected(requested):
    with pytest.raises(ValueError):
        SelectThenKnock(min_knockoff_candidates=requested)._effective_min_knockoff_candidates()


def test_calibration_and_knockoff_threshold_boundaries():
    assert suggest_threshold_pfer(7, 100, 5) == pytest.approx(0.5 + 25 / 1400)
    assert suggest_threshold_pfer(7, 100, 0) == 0.51
    assert suggest_threshold_pfer(1, 10, 100) == 0.99
    W = np.array([5.0, 4, 3, 2, 1, -0.5, 0])
    assert knockoff_threshold(W, 0.4) == 0.5
    assert knockoff_threshold(W, 0.2) == 1.0
    assert knockoff_threshold(W, 0.1) == np.inf
    assert knockoff_threshold(np.zeros(5), 0.5) == np.inf


def test_failed_bootstraps_are_excluded(monkeypatch):
    calls = 0

    def fit_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        return (np.array([1, 0]) if calls % 2 else np.zeros(2, dtype=int)), bool(calls % 2)

    monkeypatch.setattr(bootstrap, "_fit_on_bootstrap_checked", fit_once)
    result = bootstrap.stability_bootstrap(np.ones((10, 2)), np.arange(10), "regression", 0.1, 0.5, n_bootstrap=4)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.mean(axis=0), [1, 0])
    monkeypatch.setattr(bootstrap, "_fit_on_bootstrap_checked", lambda *a, **k: (np.zeros(2), False))
    with pytest.raises(RuntimeError, match="All 4"):
        bootstrap.stability_bootstrap(np.ones((10, 2)), np.arange(10), "regression", 0.1, 0.5, n_bootstrap=4)


def test_gaussian_draws_and_numerical_fallback(monkeypatch):
    X = np.random.default_rng(4).normal(size=(60, 5))
    first, S, fallback = generate_gaussian_knockoffs(X, random_state=7)
    second, _, _ = generate_gaussian_knockoffs(X, random_state=7)
    np.testing.assert_array_equal(first, second)
    assert not fallback and validate_knockoffs(X, first, S=S)["valid"]

    def failed_cholesky(*args, **kwargs):
        raise linalg.LinAlgError("forced numerical failure")

    monkeypatch.setattr(linalg, "cholesky", failed_cholesky)
    with pytest.warns(UserWarning, match="fallback"):
        permuted, _, fallback = generate_gaussian_knockoffs(X, random_state=7)
    assert fallback
    rng = np.random.default_rng(7)
    rng.standard_normal(X.shape)  # The attempted Gaussian draw occurs before fallback.
    expected = np.column_stack([X[rng.permutation(len(X)), j] for j in range(X.shape[1])])
    np.testing.assert_array_equal(permuted, expected)


@pytest.mark.parametrize("preset", ["pooled", "headline", "secondary"])
def test_small_real_fit_and_fresh_refit_state(preset):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 30))
    X[:, 0] = 0
    y = 4 * X[:, 8] + 3 * X[:, 12] + 2 * X[:, 20] + 0.2 * rng.normal(size=100)
    model = SelectThenKnock.from_preset(
        preset,
        random_state=42,
        n_jobs=1,
        n_bootstrap=4,
        n_alphas=3,
        n_l1_ratios=2,
        cv_folds=2,
        n_splits=3,
    ).fit(X, y)
    selected = model.get_support(indices=True, space="original")
    assert 0 not in selected
    assert model.get_support(space="original").shape == (30,)
    assert model.get_support().shape == (29,)
    X2 = X[:, :25]
    model.fit(X2, y)
    assert model.n_features_in_ == 25
    assert model.get_support(space="original").shape == (25,)


def test_skip_discarded_filters_but_keep_held_out_calls(monkeypatch):
    import select_then_knock.phase4_knockoff as phase4

    calls = []

    def observed_filter(X, y, **kwargs):
        calls.append(X.shape)
        return SimpleNamespace(selected_features=np.array([0, 1], dtype=int))

    monkeypatch.setattr(phase4, "knockoff_filter", observed_filter)
    rng = np.random.default_rng(12)
    X, y = rng.normal(size=(80, 30)), rng.normal(size=80)
    model = SelectThenKnock.from_preset(
        "headline",
        random_state=1,
        n_jobs=1,
        n_splits=3,
        n_bootstrap=3,
        n_alphas=3,
        n_l1_ratios=2,
        cv_folds=2,
    ).fit(X, y)
    assert len(calls) == 3  # One held-out call, not two filters per repetition.
    assert all(n == 40 and p >= 10 for n, p in calls)
    assert model.phase2_results_ is None  # No unused full-data tuning/refit.
