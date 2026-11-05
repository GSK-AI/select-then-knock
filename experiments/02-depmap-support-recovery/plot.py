"""Plot FDP and power for the merged DepMap support-recovery results.

Style, colors, and arm labels intentionally mirror
``01-gaussian-support-recovery/plot.py`` so the two figures can be read side by
side; keep them in sync when either changes.

Run from the project root:

    python experiments/02-depmap-support-recovery/plot.py \
        --results experiments/02-depmap-support-recovery/results/results.json \
        --figure-dir experiments/02-depmap-support-recovery/figures
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


def metric(cell: dict, arm: str, name: str) -> np.ndarray:
    return np.asarray(
        [record["arms"][arm][name] for record in cell["seeds"].values() if record["arms"][arm].get("ok")],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results" / "results.json")
    parser.add_argument("--figure-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    cells = sorted(data["cells"].values(), key=lambda cell: cell["k"])
    available = set(next(iter(cells[0]["seeds"].values()))["arms"])
    arms = tuple(arm for arm in ARMS if arm in available)

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

    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    x = np.arange(len(cells), dtype=float)
    width = 0.16
    for axis, name, ylabel in zip(axes, ("fdp", "power"), ("False discovery proportion", "Power"), strict=True):
        for arm_index, arm in enumerate(arms):
            values = [metric(cell, arm, name) for cell in cells]
            means = np.asarray([item.mean() for item in values])
            standard_errors = np.asarray([item.std(ddof=1) / np.sqrt(item.size) for item in values])
            offset = (arm_index - (len(arms) - 1) / 2) * width
            axis.bar(
                x + offset,
                means,
                width,
                yerr=standard_errors,
                color=COLORS[arm],
                label=LABELS[arm],
                edgecolor="white",
                linewidth=0.4,
                error_kw={"linewidth": 0.8, "capsize": 1.8},
            )
            # A zero-height bar draws nothing, so an exactly-zero mean would read
            # as missing data rather than as the result it is. Mark it with a
            # flat stub in the arm colour and a neutral label above it.
            for position, mean in zip(x + offset, means, strict=True):
                if mean == 0.0:
                    axis.plot(
                        [position - width / 2, position + width / 2],
                        [0.0, 0.0],
                        color=COLORS[arm],
                        linewidth=1.8,
                        solid_capstyle="butt",
                        clip_on=False,
                    )
                    axis.text(
                        position,
                        0.025,
                        "0",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#333333",
                    )
        axis.set_xlabel(r"True support size $k$")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [str(cell["k"]) for cell in cells])
        axis.set_ylim(0, 1.03)

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
        handlelength=1.6,
    )
    figure.tight_layout(w_pad=1.6, rect=(0, 0.08, 1, 1))

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        output = args.figure_dir / f"fig_depmap_support_recovery.{extension}"
        figure.savefig(output, dpi=220, bbox_inches="tight")
        print(f"wrote {output}")
    plt.close(figure)


if __name__ == "__main__":
    main()
