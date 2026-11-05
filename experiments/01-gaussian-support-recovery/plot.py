"""Plot Gaussian support recovery with and without the k=1 boundary at rho=0.5.

Run from the project root:

    python experiments/01-gaussian-support-recovery/plot.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

ARMS = ("stk_headline", "stk_secondary", "stabl", "stk_pooled", "lasso_cv")
LABELS = {
    "stk_headline": "Headline STK",
    "stk_secondary": "Secondary STK",
    "stabl": "Stabl",
    "stk_pooled": "Pooled STK",
    "lasso_cv": "LassoCV",
}
COLORS = {
    "stk_headline": "#0072B2",
    "stk_secondary": "#56B4E9",
    "stabl": "#D55E00",
    "stk_pooled": "#009E73",
    "lasso_cv": "#CC79A7",
}
STYLES = {
    "stk_headline": {"linestyle": "-", "marker": "^"},
    "stk_secondary": {"linestyle": "-", "marker": "^"},
    "stabl": {"linestyle": "--", "marker": "s"},
    "stk_pooled": {"linestyle": "-.", "marker": "D"},
    "lasso_cv": {"linestyle": ":", "marker": "o"},
}


def metric(cell: dict, arm: str, name: str) -> np.ndarray:
    return np.asarray(
        [record["arms"][arm][name] for record in cell["seeds"].values() if record["arms"][arm].get("ok")],
        dtype=float,
    )


def sorted_cells(path: Path) -> list[dict]:
    return sorted(json.loads(path.read_text())["cells"].values(), key=lambda cell: cell["k"])


def plot(cells: list[dict], stem: str, figure_dir: Path) -> None:
    ks = np.asarray([cell["k"] for cell in cells])
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for axis, name, ylabel in zip(axes, ("fdp", "power"), ("False discovery proportion", "Power"), strict=True):
        for arm in ARMS:
            values = [metric(cell, arm, name) for cell in cells]
            means = np.asarray([item.mean() for item in values])
            standard_errors = np.asarray([item.std(ddof=1) / np.sqrt(item.size) for item in values])
            axis.errorbar(
                ks,
                means,
                yerr=standard_errors,
                color=COLORS[arm],
                label=LABELS[arm],
                linewidth=1.4,
                markersize=5,
                capsize=2,
                markeredgecolor="white",
                markeredgewidth=0.5,
                **STYLES[arm],
            )
        axis.set_xlabel(r"True support size $k$")
        axis.set_ylabel(ylabel)
        axis.set_xticks(ks)
        axis.set_ylim(-0.03, 1.03)

    axes[0].set_title("(a) Empirical false discoveries")
    axes[1].set_title("(b) Support-recovery power")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.4,
    )
    figure.tight_layout(w_pad=1.6, rect=(0, 0.08, 1, 1))

    figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        output = figure_dir / f"{stem}.{extension}"
        figure.savefig(output, dpi=220, bbox_inches="tight")
        print(f"wrote {output}")
    plt.close(figure)


def main() -> None:
    global ARMS
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "outputs" / "results.json")
    parser.add_argument("--figure-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()

    cells = sorted_cells(args.results)
    available = set(next(iter(cells[0]["seeds"].values()))["arms"])
    ARMS = tuple(arm for arm in ARMS if arm in available)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plot(cells, "fig_gaussian_support_recovery_with_k1", args.figure_dir)
    plot([cell for cell in cells if cell["k"] != 1], "fig_gaussian_support_recovery_without_k1", args.figure_dir)


if __name__ == "__main__":
    main()
