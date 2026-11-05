"""Import portability and small helper/interface checks, without running study grids."""

import importlib.util
import inspect
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import _lib
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES = [
    "01-gaussian-support-recovery",
    "02-depmap-support-recovery",
    "03-matched-aggregation",
    "04-pc-knockoff-comparison",
    "05-repeated-pc-knockoff",
    "06-gaussian-global-null",
    "07-computational-scaling",
]


@contextmanager
def study_module(study, module="run"):
    """Isolate sibling script imports whose historic module names overlap."""
    names = ("run", "summarize", "monitor", "report", "pc_knockoff", "_pc_comparison")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    old_path = sys.path[:]
    directory = ROOT / "experiments" / study
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(module, directory / f"{module}.py")
        obj = importlib.util.module_from_spec(spec)
        sys.modules[module] = obj
        spec.loader.exec_module(obj)
        yield obj
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = old_path


@pytest.mark.parametrize("study", STUDIES)
def test_every_runner_imports_from_this_project(study):
    with study_module(study) as module:
        assert Path(module.__file__).resolve().is_relative_to(ROOT)
        if hasattr(module, "_lib"):
            assert Path(module._lib.__file__).resolve() == ROOT / "experiments" / "_lib.py"
        if hasattr(module, "SelectThenKnock"):
            assert Path(inspect.getfile(module.SelectThenKnock)).resolve().is_relative_to(ROOT)


@pytest.mark.parametrize("study", ["03-matched-aggregation", "04-pc-knockoff-comparison"])
def test_provenance_does_not_require_git_or_an_installed_project(study):
    with study_module(study) as module:
        data = module.provenance() if study.startswith("03") else module.provenance({"solver": "sklearn"})
        key = "source_sha256" if study.startswith("03") else "stk_source_sha256"
        assert data[key] == _lib.source_fingerprints()


def test_missing_depmap_input_is_actionable(monkeypatch):
    with study_module("02-depmap-support-recovery") as module:
        monkeypatch.setattr(module, "DATA_PATH", None)
        with pytest.raises(FileNotFoundError, match="DEPMAP_EXPRESSION_PATH"):
            module._validate_design()


def test_pc_comparison_task_grids_are_complete():
    directory = ROOT / "experiments/04-pc-knockoff-comparison/configs"
    cells = [tuple(map(int, line.split())) for line in (directory / "tasks_eval.txt").read_text().splitlines()]
    assert len(cells) == len(set(cells)) == 70
    assert sum(k == 0 for k, _ in cells) == 30
    assert sum(k != 0 for k, _ in cells) == 40


def test_missing_supplied_reference_directory_is_reported(tmp_path):
    with study_module("03-matched-aggregation", "audit") as module:
        with pytest.raises(ValueError, match="reference_results"):
            module.audit(tmp_path, reference_results=tmp_path / "missing-reference")


def test_latex_tables_stay_with_their_experiment_and_keep_both_labels(tmp_path, monkeypatch):
    with study_module("05-repeated-pc-knockoff", "summarize") as module:
        monkeypatch.setattr(module, "RES", tmp_path)
        methods = [
            f"PC {allocation} a={a}"
            for allocation in ("200-50", "200-100", "400-50", "400-100")
            for a in (0.1, 0.2, 0.5)
        ] + ["Headline STK", "Secondary STK"]
        conditions = [
            {
                "method": m,
                "k": k,
                "agg_power": 0.5,
                "agg_fdp": 0.1,
                "single_power": 0.6,
                "single_fdp": 0.2,
                "single_firing": 0.2,
                "screened_fraction": 0.8,
            }
            for m in methods
            for k in (0, 1, 3, 10, 20)
        ]
        module.write_latex({"conditions": conditions, "null": [{"method": m, "fired": 0, "n": 30} for m in methods]})
        text = (tmp_path / "tables/repeated_pc_main.tex").read_text()
        assert "Headline STK" in text and "Secondary STK" in text and "Repeated PC" in text
    with study_module("05-repeated-pc-knockoff", "recurrence_bounds") as module:
        monkeypatch.setattr(module, "RES", tmp_path)
        methods = ["Headline STK", "Secondary STK", "PC 200-50", "PC 200-100", "PC 400-50", "PC 400-100"]
        rows = [
            {
                "method": m,
                "k": k,
                "rho_all": 0.5,
                "rho_sig": 0.5,
                "single_power": 0.6,
                "agg_power": 0.5,
                "power_lo": 0.0,
                "power_hi": 1.0,
            }
            for m in methods
            for k in (3, 10, 20)
        ]
        (tmp_path / "recurrence_bounds.json").write_text(json.dumps({"rows": rows}))
        module.write_table()
        compact = (tmp_path / "tables/recurrence_bounds.tex").read_text()
        full = (tmp_path / "tables/recurrence_bounds_all.tex").read_text()
        assert "Headline STK" in compact and "Secondary STK" not in compact
        assert "Headline STK" in full and "Secondary STK" in full


def test_stabl_constructor_contract_if_available():
    stabl = pytest.importorskip("stabl.stabl")
    parameters = inspect.signature(stabl.Stabl).parameters
    assert {"base_estimator", "lambda_grid", "n_lambda", "n_bootstraps", "artificial_type", "sample_fraction"} <= set(
        parameters
    )


def test_resource_helper_refuses_existing_cells_without_running_models(tmp_path):
    fake_time = tmp_path / "fake-time"
    fake_time.write_text('#!/bin/bash\nif [[ "$1" == "--version" ]]; then echo "GNU time"; exit 0; fi\nexit 99\n')
    fake_time.chmod(0o700)
    out = tmp_path / "outputs"
    cell = out / "p-500/seed-42"
    cell.mkdir(parents=True)
    env = {
        **os.environ,
        "GNU_TIME": str(fake_time),
        "OUT_ROOT": str(out),
        "P_GRID": "500",
        "SEED_GRID": "42",
        "ARM_GRID": "stk_secondary",
        "PYTHON": "must-not-be-called",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "experiments/07-computational-scaling/run_resource_grid.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "Refusing existing cell directory" in result.stderr
    assert list(cell.iterdir()) == []


def test_scaling_plot_reports_missing_memory_sidecars(tmp_path, monkeypatch):
    results = tmp_path / "results.json"
    arms = {arm: {"elapsed_s": 1.0} for arm in ("stk_secondary", "direct_model_x")}
    results.write_text(json.dumps({"cells": {"p-500": {"p": 500, "seeds": {"42": {"arms": arms}}}}}))
    with study_module("07-computational-scaling", "plot") as module:
        monkeypatch.setattr(sys, "argv", ["plot.py", "--results", str(results)])
        try:
            with pytest.raises(ValueError, match="run_resource_grid.sh"):
                module.main()
        finally:
            module.plt.close("all")
