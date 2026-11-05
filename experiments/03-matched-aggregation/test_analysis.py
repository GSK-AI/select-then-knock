"""Checks for coordinate mapping, aggregation, and the replication unit."""

import numpy as np
import pytest
from monitor import monitor_lasso
from report import paired_interval, wilson
from run import RecordedSTK, SelectThenKnock, aggregate_support, summarize_cell, support_metrics
from summarize import estimate


def test_threshold_and_false_discovery_counts():
    splits = [{"selected": [1, 3]}] * 12 + [{"selected": [1, 4]}] * 13
    selected, counts = aggregate_support(splits, 8, 0.5)
    assert selected.tolist() == [1, 4]
    assert counts[3] == 12
    assert counts[4] == 13
    assert support_metrics(selected, [1, 2]) == {
        "n_selected": 2,
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "fdp": 0.5,
        "power": 0.5,
        "exact_recovery": 0,
        "firing": 1,
    }
    assert support_metrics([], [])["power"] is None


def test_average_fdp_not_fdp_of_average_counts():
    splits = [
        {"selected": [0], "fallback_used": False, "held_out_filter_called": True, "elapsed_s": 1},
        {"selected": [0, 1, 2], "fallback_used": False, "held_out_filter_called": True, "elapsed_s": 1},
    ]
    summary = summarize_cell(splits, [0], [0])
    assert summary["single_split_mean"]["fdp"] == pytest.approx(1 / 3)
    assert estimate([summary["single_split_mean"]["fdp"]])["se_across_datasets"] is None


def test_uncertainty_uses_dataset_count():
    interval = paired_interval({"mean": 0.2, "se_across_datasets": 0.1, "n_datasets": 10})
    assert interval == pytest.approx([-0.0262157163, 0.4262157163])
    lower, upper = wilson(0, 30)
    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(0.1135133932)
    assert paired_interval({"mean": 0.2, "se_across_datasets": None, "n_datasets": 1}) is None


@pytest.mark.parametrize("preset", ["headline", "secondary"])
def test_recording_preserves_public_fit_and_original_space(preset):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 20))
    X[:, 0] = 0  # forces a non-identity Original -> Preprocessed map
    y = 3 * X[:, 8] + 2 * X[:, 12] + rng.normal(size=100)
    settings = dict(
        random_state=42,
        run_diagnostics=False,
        n_jobs=1,
        n_splits=3,
        n_bootstrap=4,
        n_alphas=3,
        n_l1_ratios=2,
        cv_folds=2,
        solver_backend="sklearn",
    )
    numerical_records = []
    with monitor_lasso(numerical_records):
        observed = RecordedSTK.from_preset(preset, **settings).fit(X, y)
    plain = SelectThenKnock.from_preset(preset, **settings).fit(X, y)
    np.testing.assert_array_equal(observed.selection_frequencies_, plain.selection_frequencies_)
    selected, counts = aggregate_support(observed.recorded_splits_, X.shape[1], observed.eta)
    np.testing.assert_array_equal(selected, plain.get_support(indices=True, space="original"))
    np.testing.assert_array_equal(counts / 3, plain.selection_frequencies_)
    assert counts[0] == 0
    assert sum(row["stage"] == "held_out" for row in numerical_records) == 3
    assert all(row["ok"] for row in numerical_records)
    assert all(row["max_iter"] == 10000 and row["tol"] == 1e-4 for row in numerical_records)
    # M=1 uses precisely the first generated split seed; this checks that an
    # observed split is an actual public single-split fit, not a new algorithm.
    single = SelectThenKnock.from_preset(preset, **{**settings, "n_splits": 1}).fit(X, y)
    np.testing.assert_array_equal(
        observed.recorded_splits_[0]["selected"],
        single.get_support(indices=True, space="original"),
    )
