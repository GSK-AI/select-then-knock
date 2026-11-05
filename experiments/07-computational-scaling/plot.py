"""Plot runtime, memory, and candidate reduction for the computational-scaling study.

Run after merging resource-grid shards, from the project root:

    python experiments/07-computational-scaling/plot.py \
        --results experiments/07-computational-scaling/results/results.json \
        --figure-dir experiments/07-computational-scaling/figures

Runtime and memory include failed and timed-out runs by default: a failed run
contributes its time to failure or timeout and its memory up to that point.
Pass --successful-only to plot completed runs alone. An arm with no runs at
some p, such as direct Model-X stopped by its time or memory limit, is drawn
only at the p values where it has data. Memory is GNU time's "Maximum resident
set size": the peak of the largest single process, not the sum over worker
processes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
labels = {
    "stk_secondary": "Secondary STK",
    "direct_model_x": "Direct Model-X",
}
colors = {
    "stk_secondary": "#009E73",
    "direct_model_x": "#000000",
}


def arm_records(cell: dict, arm: str, successful_only: bool = False) -> list[dict]:
    return [
        record["arms"][arm]
        for record in cell["seeds"].values()
        if arm in record["arms"] and (record["arms"][arm].get("ok") or not successful_only)
    ]


def arm_values(cell: dict, arm: str, field: str, successful_only: bool = False) -> list[float]:
    return [record[field] for record in arm_records(cell, arm, successful_only) if record.get(field) is not None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results" / "results.json")
    parser.add_argument("--figure-dir", type=Path, default=HERE / "figures")
    parser.add_argument(
        "--successful-only",
        action="store_true",
        help="exclude failed and timed-out runs from runtime and memory (default: include them)",
    )
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    cells = sorted(data["cells"].values(), key=lambda cell: cell["p"])
    p_values = np.array([cell["p"] for cell in cells])
    tick_labels = [f"{p // 1000}k" if p >= 1000 else str(p) for p in p_values]

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
    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))

    for arm in labels:
        timed = [(cell["p"], arm_values(cell, arm, "elapsed_s", args.successful_only)) for cell in cells]
        timed = [(p, values) for p, values in timed if values]
        if timed:
            arm_p = [p for p, _ in timed]
            medians = [np.median(values) for _, values in timed]
            lower = [np.percentile(values, 25) for _, values in timed]
            upper = [np.percentile(values, 75) for _, values in timed]
            axes[0].plot(arm_p, medians, marker="o", color=colors[arm], label=labels[arm])
            axes[0].fill_between(arm_p, lower, upper, color=colors[arm], alpha=0.12)

        memory = []
        for cell in cells:
            if arm_records(cell, arm) and not arm_values(cell, arm, "max_rss_kb"):
                raise ValueError(
                    f"Missing memory records for {arm}; use run_resource_grid.sh and merge its GNU-time sidecars"
                )
            values = arm_values(cell, arm, "max_rss_kb", args.successful_only)
            if values:
                memory.append((cell["p"], max(values) / 1024**2))
        if memory:
            axes[1].plot(*zip(*memory), marker="o", color=colors[arm], label=labels[arm])

    means = []
    for cell in cells:
        values = [size for record in arm_records(cell, "stk_secondary") for size in record.get("candidate_sizes", [])]
        if values:
            means.append((cell["p"], np.mean(values)))
    if means:
        axes[2].plot(*zip(*means), marker="o", color=colors["stk_secondary"], label=labels["stk_secondary"])
    axes[2].plot(
        p_values,
        p_values,
        linestyle="--",
        color=colors["direct_model_x"],
        label=labels["direct_model_x"],
    )

    ylabels = ("Wall-clock seconds", "Peak resident memory (GiB)", "Features entering knockoff construction")
    titles = ("(a) End-to-end runtime", "(b) Peak memory", "(c) Candidate-space reduction")
    for axis, ylabel, title in zip(axes, ylabels, titles, strict=True):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(p_values, tick_labels)
        axis.set_xlabel(r"Input features $p$")
        axis.set_ylabel(ylabel)
        axis.set_title(title)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    handles2, legend_labels2 = axes[2].get_legend_handles_labels()
    by_label = dict(zip(legend_labels + legend_labels2, handles + handles2))
    figure.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.14, 1, 1))

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        output = args.figure_dir / f"fig_computational_scaling.{suffix}"
        figure.savefig(output, dpi=220, bbox_inches="tight")
        print(f"wrote {output}")
    plt.close(figure)


if __name__ == "__main__":
    main()
