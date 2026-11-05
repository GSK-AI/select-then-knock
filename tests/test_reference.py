"""Optional, bounded numerical comparisons with a user-supplied reference source.

Set STK_REFERENCE_SRC to the directory containing the reference select_then_knock
package. Both processes use the same interpreter and dependency environment.
No historical results or datasets are copied, and no full benchmark is fitted.

The reference source uses the historical preset names and per-split method name;
HISTORICAL_PRESETS and HISTORICAL_SPLIT_METHOD translate the current names.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = os.environ.get("STK_REFERENCE_SRC")
REFERENCE_LIB = os.environ.get("STK_REFERENCE_LIB")
# Current name -> name in the historical reference source.
HISTORICAL_PRESETS = {"pooled": "default", "headline": "derandomized", "secondary": "multisplit"}
HISTORICAL_SPLIT_METHOD = "_derand_one_split"
SPLIT_METHOD = "_one_split"
WORKER = r"""
import json, sys, warnings
from pathlib import Path
import numpy as np
from select_then_knock import SelectThenKnock
import select_then_knock
warnings.filterwarnings("ignore")
cfg = json.loads(sys.argv[1])
origin = Path(select_then_knock.__file__).resolve()
assert origin.is_relative_to(Path(sys.argv[2]).resolve()), origin
rng = np.random.default_rng(123)
p = 8 if cfg["case"] == "few" else 60
X = rng.normal(size=(120, p))
X[:, 0] = 0
if p > 20:
    X[:, 19] = 7
    X[:, 4] *= 20
y = rng.normal(size=120)
if cfg["case"] == "signal":
    y = 4 * X[:, 7] + 3 * X[:, 12] + 2 * X[:, 30] + 0.15 * y
class Observed(SelectThenKnock):
    candidates = []
    splits = []
    def _maybe_expand_phase3_for_knockoff(self, indices, *args, **kwargs):
        result = super()._maybe_expand_phase3_for_knockoff(indices, *args, **kwargs)
        original = np.flatnonzero(self.preprocessing_results_.feature_mask)[result]
        Observed.candidates.append({"seed": self.random_state, "indices": original.tolist(),
            "scores": self.phase3_results_.stability_scores.tolist(),
            "bootstrap": self.phase3_results_.selection_matrix.tolist(),
            "threshold": self.phase3_results_.threshold})
        return result
def observed_split(self, X, y, cut, seed, level, filter):
    detail = {}
    def observed_filter(Xk, yk, **kw):
        result = filter(Xk, yk, **kw)
        detail.update(W=result.W_statistics.tolist(), X=Xk.tolist(),
            fallback=bool(result.fallback_used), local=result.selected_features.tolist())
        return result
    selected = getattr(super(Observed, self), cfg["split_method"])(X, y, cut, seed, level, observed_filter)
    Observed.splits.append({"seed": seed, "selected": selected.tolist(), **detail})
    return selected
setattr(Observed, cfg["split_method"], observed_split)
model = Observed.from_preset(cfg["preset"], random_state=42, n_jobs=1,
    n_splits=cfg.get("n_splits", 3), n_bootstrap=6, n_alphas=4, n_l1_ratios=3,
    cv_folds=3, solver_backend=cfg["solver"]).fit(X, y)
result = {"selected": model.get_support(indices=True, space="original").tolist(),
    "candidates": Observed.candidates, "splits": Observed.splits,
    "frequencies": None if model.selection_frequencies_ is None else model.selection_frequencies_.tolist()}
print(json.dumps(result))
"""


def run_source(source, config, cwd, historical=False):
    """Run WORKER on one source; ``historical`` selects the reference source's names."""
    config = {
        **config,
        "preset": HISTORICAL_PRESETS[config["preset"]] if historical else config["preset"],
        "split_method": HISTORICAL_SPLIT_METHOD if historical else SPLIT_METHOD,
    }
    env = {**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"}
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    output = subprocess.run(
        [sys.executable, "-B", "-c", WORKER, json.dumps(config), str(source)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert output.returncode == 0, output.stderr
    return json.loads(output.stdout)


@pytest.mark.reference
@pytest.mark.skipif(not REFERENCE, reason="Set STK_REFERENCE_SRC to compare with the preserved reference source")
@pytest.mark.parametrize("solver", ["sklearn", "celer"])
@pytest.mark.parametrize("preset", ["pooled", "headline", "secondary"])
@pytest.mark.parametrize("case", ["signal", "null", "few"])
def test_reference_selections_candidates_and_frequencies(solver, preset, case, tmp_path):
    if solver == "celer":
        pytest.importorskip("celer")
    config = {"solver": solver, "preset": preset, "case": case}
    old, new = run_source(REFERENCE, config, tmp_path, historical=True), run_source(ROOT, config, tmp_path)
    assert old["selected"] == new["selected"]
    assert old["frequencies"] == new["frequencies"]
    assert len(old["candidates"]) == len(new["candidates"])
    for a, b in zip(old["candidates"], new["candidates"], strict=True):
        assert a["seed"] == b["seed"] and a["indices"] == b["indices"]
        assert a["bootstrap"] == b["bootstrap"]
        np.testing.assert_allclose(a["scores"], b["scores"], rtol=0, atol=0)
        assert a["threshold"] == b["threshold"]
    for a, b in zip(old["splits"], new["splits"], strict=True):
        assert a["selected"] == b["selected"] and a["local"] == b["local"]
        assert a["fallback"] == b["fallback"]
        np.testing.assert_allclose(a["W"], b["W"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a["X"], b["X"], rtol=0, atol=0)


@pytest.mark.reference
@pytest.mark.skipif(not REFERENCE, reason="Set STK_REFERENCE_SRC to compare the 25-split threshold")
def test_reference_twenty_five_split_aggregation(tmp_path):
    config = {"solver": "sklearn", "preset": "headline", "case": "signal", "n_splits": 25}
    old, new = run_source(REFERENCE, config, tmp_path, historical=True), run_source(ROOT, config, tmp_path)
    assert old["selected"] == new["selected"]
    assert old["frequencies"] == new["frequencies"]
    assert [r["selected"] for r in old["splits"]] == [r["selected"] for r in new["splits"]]


def reference_helper():
    spec = importlib.util.spec_from_file_location("reference_benchmark_helpers", REFERENCE_LIB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.reference
@pytest.mark.skipif(not REFERENCE_LIB, reason="Set STK_REFERENCE_LIB to the preserved benchmark helper file")
@pytest.mark.parametrize("null", [False, True])
def test_reference_generators_and_metrics_are_identical(null):
    import _lib

    original = reference_helper()
    if null:
        old = original.generate_global_null(80, 6, 10, 0.5, 42)
        new = _lib.generate_global_null(80, 6, 10, 0.5, 42)
    else:
        kwargs = dict(
            n=80, n_blocks=6, block_size=10, rho=0.5, k_true=3, beta_true=original.beta_for_k(3), snr=0.5, seed=42
        )
        old, new = original.generate_blocked_synthetic(**kwargs), _lib.generate_blocked_synthetic(**kwargs)
    assert set(old) == set(new)
    for key in old:
        np.testing.assert_array_equal(old[key], new[key])
    assert original.compute_metrics_flat([1, 3], old["S_star"]) == _lib.compute_metrics_flat([1, 3], new["S_star"])
    blocks = np.repeat(np.arange(6), 10)
    assert original.compute_metrics([1, 3], old["S_star"], blocks) == _lib.compute_metrics(
        [1, 3], new["S_star"], blocks
    )


@pytest.mark.reference
@pytest.mark.skipif(not REFERENCE_LIB, reason="Set STK_REFERENCE_LIB to compare the Lasso adapter")
def test_reference_lasso_adapter_selections():
    from types import SimpleNamespace

    import _lib

    original = reference_helper()
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 20))
    y = 4 * X[:, 7] + 2 * X[:, 12] + rng.normal(size=80)
    old, new = [], []
    original._lasso_worker(X, y, 42, SimpleNamespace(put=old.append))
    _lib._lasso_worker(X, y, 42, SimpleNamespace(put=new.append))
    assert old[0]["ok"] and new[0]["ok"]
    assert old[0]["selected"] == new[0]["selected"]
    assert old[0]["alpha_cv"] == new[0]["alpha_cv"]


@pytest.mark.reference
@pytest.mark.skipif(not REFERENCE_LIB, reason="Set STK_REFERENCE_LIB to compare the Stabl adapter")
def test_reference_stabl_adapter_preprocessing_and_settings(monkeypatch):
    import _lib

    stabl = pytest.importorskip("stabl.stabl")
    original = reference_helper()
    records = []

    class ObservedStabl:
        def __init__(self, **kwargs):
            kwargs["base_estimator"] = kwargs["base_estimator"].get_params()
            self.record = {"parameters": kwargs}
            self.min_fdr_ = 0.2
            records.append(self.record)

        def fit(self, frame, y):
            self.record.update(X=frame.to_numpy().tolist(), columns=frame.columns.tolist(), y=y.tolist())
            return self

        def get_support(self, indices):
            assert indices
            return np.array([1, 3])

    monkeypatch.setattr(stabl, "Stabl", ObservedStabl)
    rng = np.random.default_rng(9)
    X, y = rng.normal(size=(20, 5)), rng.normal(size=20)
    old = original._stabl_fit(X, y, "knockoff", 42, 300, 10)
    new = _lib._stabl_fit(X, y, 42, 300, 10)
    assert old == new
    assert records[0] == records[1]
