"""Observe Phase-4 LassoCV numerical diagnostics without changing settings."""

import inspect
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from unittest.mock import patch

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV


@contextmanager
def monitor_lasso(records):
    original = LassoCV.fit

    def observed_fit(model, X, y, *args, **kwargs):
        functions = {frame.function for frame in inspect.stack()}
        stage = "held_out" if "observed_filter" in functions else "other"
        if "_one_split" not in functions:
            stage = "other"
        started = time.perf_counter()
        row = {
            "stage": stage,
            "shape": list(X.shape),
            "max_iter": model.max_iter,
            "tol": model.tol,
            "ok": False,
        }
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                result = original(model, X, y, *args, **kwargs)
            messages = Counter(str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning))
            row.update(
                ok=True,
                alpha=float(model.alpha_),
                final_dual_gap=float(model.dual_gap_),
                final_n_iter=int(model.n_iter_),
                convergence_warning_count=sum(messages.values()),
                warning_messages=dict(messages),
            )
            for item in caught:
                if not issubclass(item.category, ConvergenceWarning):
                    warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
            return result
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            row["elapsed_s"] = time.perf_counter() - started
            records.append(row)

    with patch.object(LassoCV, "fit", observed_fit):
        yield


def stage_summary(records):
    result = {}
    for stage in sorted({row["stage"] for row in records}):
        subset = [row for row in records if row["stage"] == stage]
        successful = [row for row in subset if row["ok"]]
        result[stage] = {
            "fits": len(subset),
            "errors": len(subset) - len(successful),
            "warning_count": sum(row["convergence_warning_count"] for row in successful),
            "fits_with_warnings": sum(row["convergence_warning_count"] > 0 for row in successful),
            "final_refits_at_iteration_limit": sum(row["final_n_iter"] >= row["max_iter"] for row in successful),
            "max_final_dual_gap": max((row["final_dual_gap"] for row in successful), default=None),
            "max_final_n_iter": max((row["final_n_iter"] for row in successful), default=None),
        }
    return result
