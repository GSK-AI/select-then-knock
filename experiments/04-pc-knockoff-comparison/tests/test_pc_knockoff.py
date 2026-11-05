"""Mathematical checks for pc_knockoff.py.

Run from the project root:

    python -m pytest experiments/04-pc-knockoff-comparison/tests -q

Upstream comparisons need the pinned author code at
experiments/04-pc-knockoff-comparison/outputs/upstream/pc_screen/PC_screen.py (not redistributed;
no upstream code is bundled). They are skipped when it is absent.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pc_knockoff as pk  # noqa: E402

UPSTREAM = HERE.parent / "outputs" / "upstream" / "pc_screen" / "PC_screen.py"


def brute_pc2(x, y, convention="author_code"):
    """Direct (2.4)-(2.5): arccos of normalized differences, double centering, n^-3 sums."""
    x, y = np.asarray(x, float).reshape(len(x), -1), np.asarray(y, float).reshape(len(y), -1)
    n = len(x)

    def angles(Z):
        a = np.zeros((n, n, n))
        for r in range(n):
            for k in range(n):
                for m in range(n):
                    if convention == "paper" and (k == r or m == r):
                        continue
                    u, v = Z[k] - Z[r], Z[m] - Z[r]
                    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                    cos = 0.0 if nu * nv == 0 else np.clip(u @ v / (nu * nv), -1, 1)
                    a[k, m, r] = np.arccos(cos)
        return a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean((0, 1), keepdims=True)

    A, B = angles(x), angles(y)
    sxy, sxx, syy = (A * B).sum() / n**3, (A * A).sum() / n**3, (B * B).sum() / n**3
    return 0.0 if sxx * syy == 0 else sxy / np.sqrt(sxx * syy)


def load_upstream():
    if not UPSTREAM.exists():
        pytest.skip("pinned upstream PC_screen.py not present")
    assert hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() == pk.UPSTREAM_PC_SCREEN_SHA256
    spec = importlib.util.spec_from_file_location("upstream_pc_screen", UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(9)
    return {
        "continuous": (x, rng.standard_normal(9)),
        "linear": (x, 2 * x + 1),
        "monotone": (x, np.exp(x)),
        "non_monotone": (np.linspace(-1, 1, 9), np.linspace(-1, 1, 9) ** 2),
        "ties": (np.round(x), np.round(rng.standard_normal(9), 0)),
        "constant_x": (np.ones(9), x),
        "constant_y": (x, np.full(9, 3.0)),
    }


@pytest.mark.parametrize("convention", pk.CONVENTIONS)
@pytest.mark.parametrize("name", list(cases()))
def test_fast_pc_matches_bruteforce_definition(name, convention):
    x, y = cases()[name]
    assert pk.pc_squared(x, y, convention) == pytest.approx(brute_pc2(x, y, convention), abs=1e-12)


def test_known_values_and_not_pearson():
    c = cases()
    assert pk.pc_squared(*c["linear"]) == pytest.approx(1.0)
    assert pk.pc_squared(*c["monotone"]) == pytest.approx(1.0)  # rank-invariant
    assert pk.pc_squared(*c["constant_x"]) == 0.0
    assert pk.pc_squared(*c["constant_y"]) == 0.0
    x, y = c["non_monotone"]
    assert abs(np.corrcoef(x, y)[0, 1]) < 1e-12
    assert pk.pc_squared(x, y) > 0.05  # detects dependence Pearson misses


def test_columns_batch_matches_single():
    rng = np.random.default_rng(1)
    X, y = rng.standard_normal((15, 7)), rng.standard_normal(15)
    X[:, 3] = 0.5  # constant feature
    expected = [pk.pc_squared(X[:, j], y) for j in range(7)]
    np.testing.assert_allclose(pk.pc_squared_columns(X, y, batch=3), expected, atol=1e-14)
    assert expected[3] == 0.0


@pytest.mark.parametrize("name", list(cases()))
def test_matches_upstream_pc_squared_once(name):
    up = load_upstream()
    x, y = cases()[name]
    ours = pk.pc_squared(x, y)
    # Upstream returns PC (a square root); square it exactly once.
    assert ours == pytest.approx(up.projection_corr_1d(x[:, None], y[:, None]) ** 2, abs=1e-12)
    with np.errstate(invalid="ignore"):
        general = up.projection_corr(x[:, None], y[:, None])
    if name.startswith("constant"):
        # Upstream defect: the general path leaves rounding-level S_xx and takes the
        # sqrt of a negative ratio. The 1-D path and the paper's 0/0 = 0 give 0.
        assert np.isnan(general) and ours == 0.0
    else:
        assert ours == pytest.approx(general**2, abs=1e-12)


class _NumpyLegacyStream:
    def __init__(self, seed):
        self._state = np.random.RandomState(seed)

    def standard_normal(self, size):
        return self._state.randn(*size)


def test_knockoffs_match_upstream_and_identities():
    up = load_upstream()
    rng = np.random.default_rng(2)
    X = rng.standard_normal((40, 6)) + 0.3 * rng.standard_normal((40, 1))
    ours, diag = pk.equicorrelated_knockoffs(X, _NumpyLegacyStream(7))
    np.random.seed(7)
    theirs = up.get_equi_features(X)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)
    assert diag["gram_error"] < 1e-8 and diag["cross_error"] < 1e-8
    assert 0 < diag["s"] <= 1


def test_knockoff_guards():
    with pytest.raises(ValueError):
        pk.equicorrelated_knockoffs(np.ones((10, 6)), np.random.default_rng(0))  # n < 2d
    X = np.random.default_rng(0).standard_normal((20, 3))
    X[:, 1] = 0.0
    with pytest.raises(ValueError):
        pk.equicorrelated_knockoffs(X, np.random.default_rng(0))
    X[:, 1] = X[:, 0]
    with pytest.raises(np.linalg.LinAlgError):
        pk.equicorrelated_knockoffs(X, np.random.default_rng(0))


def test_statistic_swap_flips_sign():
    rng = np.random.default_rng(3)
    X, Xk, y = rng.standard_normal((30, 4)), rng.standard_normal((30, 4)), rng.standard_normal(30)
    W = pk.pc_squared_columns(X, y) - pk.pc_squared_columns(Xk, y)
    Xs, Xks = X.copy(), Xk.copy()
    Xs[:, [1, 2]], Xks[:, [1, 2]] = Xk[:, [1, 2]], X[:, [1, 2]]
    Ws = pk.pc_squared_columns(Xs, y) - pk.pc_squared_columns(Xks, y)
    np.testing.assert_allclose(Ws, W * np.array([1, -1, -1, 1]), atol=1e-14)


def test_knockoff_plus_threshold():
    W = np.array([5.0, 4.0, 3.0, 2.0, 1.0, -0.5, 0.0])
    # t=0.5: (1+1)/5 = 0.4; t=1: (1+0)/5 = 0.2 -> first qualifying t at alpha 0.2 is 1.
    assert pk.knockoff_plus_threshold(W, 0.2) == 1.0
    assert pk.knockoff_plus_threshold(W, 0.4) == 0.5
    assert pk.knockoff_plus_threshold(W, 0.25) == 1.0
    # Offset one needs >= ceil(1/alpha) positives: 5 positives cannot reach 0.1.
    assert pk.knockoff_plus_threshold(W, 0.1) == np.inf
    assert pk.knockoff_plus_threshold(np.zeros(5), 0.5) == np.inf
    assert pk.knockoff_plus_threshold(np.array([-1.0, -2.0]), 0.5) == np.inf


def _signal_data(seed=4, n=160, p=60):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X[:, 5] = 1.0  # constant feature
    y = 2 * X[:, 37] + np.sin(3 * X[:, 11]) + X[:, 50] + 0.3 * rng.standard_normal(n)
    return X, y


def test_procedure_rows_screening_mapping_and_determinism():
    X, y = _signal_data()
    kw = dict(n_screen=50, d=12, alphas=(0.1, 0.2, 0.5), partition_seed=1, knockoff_seed=2)
    res = pk.pc_knockoff(X, y, **kw)
    assert len(np.intersect1d(res.screen_rows, res.test_rows)) == 0
    assert len(res.screen_rows) + len(res.test_rows) == len(y)
    assert 37 in res.candidates and 5 not in res.candidates
    assert 37 in res.selected[0.5]
    for alpha, sel in res.selected.items():
        assert set(sel) <= set(res.candidates)
        # Original-space mapping: recompute W on the selected original columns.
        idx = [list(res.candidates).index(j) for j in sel]
        assert np.all(res.W[idx] >= res.thresholds[alpha])
    # Screening cannot inspect held-out outcomes.
    y2 = y.copy()
    y2[res.test_rows] = np.random.default_rng(9).standard_normal(len(res.test_rows))
    res2 = pk.pc_knockoff(X, y2, **kw)
    np.testing.assert_array_equal(res.screen_pc2, res2.screen_pc2)
    np.testing.assert_array_equal(res.candidates, res2.candidates)
    # Deterministic rerun.
    res3 = pk.pc_knockoff(X, y, **kw)
    np.testing.assert_array_equal(res.W, res3.W)
    assert {a: s.tolist() for a, s in res.selected.items()} == {a: s.tolist() for a, s in res3.selected.items()}


def test_published_restriction_enforced():
    X, y = _signal_data(n=100)
    with pytest.raises(ValueError, match="2d < n_test"):
        pk.pc_knockoff(X, y, n_screen=50, d=25, alphas=(0.1,), partition_seed=0, knockoff_seed=0)
