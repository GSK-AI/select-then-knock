#!/usr/bin/env python
"""Figure for the Gaussian global-null experiment.

Reads outputs/results.json (merged across seeds) and writes one figure into
this experiment's figures/ subdir:

  figures/fig_gaussian_global_null.{pdf,png} -- P(fire) vs correlation
  (panel a, at p=5000) and vs p (panel b, at rho=0.5), with Wilson 95% CI
  error bars for all methods. Okabe-Ito colour-blind-safe palette, Times serif,
  embedded TrueType fonts.

Run:
  python experiments/06-gaussian-global-null/plot.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
C = {
    "stk_headline": "#0072B2",
    "stk_secondary": "#56B4E9",
    "stabl": "#D55E00",
    "stk_pooled": "#009E73",
    "lasso_cv": "#CC79A7",
}
LABEL = {
    "stk_pooled": "Pooled STK",
    "stk_headline": "Headline STK",
    "stk_secondary": "Secondary STK",
    "stabl": "Stabl",
    "lasso_cv": "LassoCV",
}
STYLE = {
    "stk_headline": dict(ls="-", marker="^"),
    "stk_secondary": dict(ls="-", marker="^"),
    "stabl": dict(ls="--", marker="s"),
    "stk_pooled": dict(ls="-.", marker="D"),
    "lasso_cv": dict(ls=":", marker="o"),
}
ARMS = ["stk_headline", "stk_secondary", "stabl", "stk_pooled", "lasso_cv"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load(results: Path) -> dict:
    d = json.loads(results.read_text())
    return {c["cell"]: c for c in d["cells"]}


def nsel_series(cell: dict, arm: str) -> list[int]:
    """Per-seed |S_hat| for one arm in one cell (ok seeds only)."""
    out = []
    for sd in cell["seeds"].values():
        a = sd["arms"].get(arm)
        if not a or not a.get("ok"):
            continue
        out.append(a["n_selected"])
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (p_hat, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def fire_stats(cell: dict, arm: str) -> tuple[float, float, float, int]:
    vals = nsel_series(cell, arm)
    n = len(vals)
    k = sum(1 for v in vals if v > 0)
    p, lo, hi = wilson_ci(k, n)
    return p, lo, hi, n


def _panel(ax, cells_ordered: list[dict], xvals: list[float], xlabel: str, log_x: bool = False):
    for arm in ARMS:
        ys, lo_err, hi_err = [], [], []
        for cell in cells_ordered:
            p, lo, hi, n = fire_stats(cell, arm)
            ys.append(p)
            lo_err.append(p - lo)
            hi_err.append(hi - p)
        ax.errorbar(
            xvals,
            ys,
            yerr=[lo_err, hi_err],
            color=C[arm],
            label=LABEL[arm],
            capsize=2,
            elinewidth=0.8,
            markeredgewidth=0.6,
            markeredgecolor="white",
            **STYLE[arm],
        )
    ax.set_xlabel(xlabel)
    if log_x:
        ax.set_xscale("log")
    ax.set_xticks(xvals)
    if log_x:
        ax.set_xticklabels([str(int(x)) for x in xvals])
        ax.minorticks_off()
    ax.set_ylim(-0.03, 1.05)


RHO_CELL_KEYS = ["rho=0.0", "rho=0.3", "rho=0.5", "rho=0.8"]  # at p=5000
P_CELL_KEYS = ["p=500", "p=2000", "rho=0.5"]  # at rho=0.5; rho=0.5 doubles as the p=5000 point


def fig_gaussian_global_null(cells: dict, figure_dir: Path) -> None:
    rho_cells = [cells[k] for k in RHO_CELL_KEYS if k in cells]
    rho_vals = [c["rho"] for c in rho_cells]

    p_cells_unsorted = [cells[k] for k in P_CELL_KEYS if k in cells]
    p_cells, p_vals = zip(*sorted(((c, c["p"]) for c in p_cells_unsorted), key=lambda t: t[1]))
    p_cells, p_vals = list(p_cells), list(p_vals)

    fig, (axr, axp) = plt.subplots(1, 2, figsize=(6.6, 2.8))
    _panel(axr, rho_cells, rho_vals, r"correlation $\rho$ (p=5000)")
    axr.set_ylabel("P(fire)")
    axr.set_title("(a) vs. within-block correlation")

    _panel(axp, p_cells, p_vals, r"$p$ ($\rho$=0.5)", log_x=True)
    axp.set_title("(b) vs. dimension $p$")

    handles, labels = axr.get_legend_handles_labels()
    order = [labels.index(LABEL[a]) for a in ARMS]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        ncol=5,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.6,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(w_pad=1.6, rect=(0, 0.06, 1, 1))
    figure_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = figure_dir / f"fig_gaussian_global_null.{ext}"
        fig.savefig(out)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "outputs" / "results.json")
    parser.add_argument("--figure-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    cells = load(args.results)
    available = set(next(iter(next(iter(cells.values()))["seeds"].values()))["arms"])
    ARMS = [arm for arm in ARMS if arm in available]
    fig_gaussian_global_null(cells, args.figure_dir)
