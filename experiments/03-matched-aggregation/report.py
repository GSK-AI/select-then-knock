"""Produce paired uncertainty, null calibration, and full-grid figures.

Run only after audit.py --require-full succeeds. All uncertainty uses dataset
seeds as replicates. Per-condition paired t intervals are descriptive and are
not adjusted for multiple support sizes. Null aggregate Wilson intervals use
30 datasets, never the 750 overlapping individual splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from summarize import summarize


def paired_interval(estimate):
    mean, se, n = estimate["mean"], estimate["se_across_datasets"], estimate["n_datasets"]
    if se is None:
        return None
    width = float(t.ppf(0.975, n - 1)) * se
    return [mean - width, mean + width]


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return [max(0.0, float(center - half)), min(1.0, float(center + half))]


def report(root, destination):
    audit = json.loads((root / "audit.json").read_text())
    if not audit["full_grid_verified"]:
        raise ValueError("Run the full-grid audit first")
    summary, table = summarize(root)
    notes = [
        table,
        "## Paired differences",
        "",
        "Aggregate minus mean individual-split performance, paired within each dataset. "
        "95% t intervals use ten dataset seeds per signal condition and are not multiplicity-adjusted.",
        "",
        "| Preset | k | FDP difference [95% interval] | Power difference [95% interval] |",
        "|---|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        for metric, estimate in cell["paired_difference_aggregate_minus_single_mean"].items():
            estimate["paired_95_t_interval"] = paired_interval(estimate)
        if cell["design"]["k"]:
            entries = []
            for metric in ("fdp", "power"):
                estimate = cell["paired_difference_aggregate_minus_single_mean"][metric]
                lower, upper = estimate["paired_95_t_interval"]
                entries.append(f"{estimate['mean']:+.3f} [{lower:+.3f}, {upper:+.3f}]")
            notes.append(f"| {cell['preset']} | {cell['design']['k']} | " + " | ".join(entries) + " |")
        else:
            n = len(cell["seeds"])
            successes = round(cell["aggregate"]["firing"]["mean"] * n)
            cell["aggregate_firing_wilson95"] = wilson(successes, n)
    notes.extend(["", "## Null calibration", ""])
    for cell in summary["cells"]:
        if cell["design"]["k"] == 0:
            lower, upper = cell["aggregate_firing_wilson95"]
            notes.append(
                f"- {cell['preset']}: mean single-split firing "
                f"{cell['single_split_mean']['firing']['mean']:.3f}; aggregate firing "
                f"{cell['aggregate']['firing']['mean']:.3f}, Wilson 95% interval [{lower:.3f}, {upper:.3f}], "
                f"across {len(cell['seeds'])} datasets."
            )
    notes.extend(["", "## Numerical diagnostics", ""])
    for preset in ("headline", "secondary"):
        records = [cell for cell in summary["cells"] if cell["preset"] == preset]
        fallback_count = sum(cell["held_out_fallback_count"] for cell in records)
        for stage in ("held_out",):
            stages = [cell["convergence_by_stage"].get(stage, {}) for cell in records]
            warnings = sum(row.get("warning_count", 0) for row in stages)
            limits = sum(row.get("final_refits_at_iteration_limit", 0) for row in stages)
            errors = sum(row.get("errors", 0) for row in stages)
            notes.append(
                f"- {preset}, {stage}: {warnings} CV warning lines, {limits} final iteration-limit hits, "
                f"{errors} fit errors."
            )
        notes.append(f"- {preset}: {fallback_count} held-out permutation fallbacks.")
    comparisons = [row["frozen_comparison"] for row in audit["checks"] if "frozen_comparison" in row]
    notes.extend(
        [
            "",
            "## Historical comparison",
            "",
            (
                f"{sum(row['same_selected_support'] for row in comparisons)}/{len(comparisons)} aggregate supports "
                "match the supplied Gaussian support-recovery supports. These results are not pooled with the historical results."
                if comparisons
                else "No matching historical reference records were supplied; no historical equivalence is claimed."
            ),
        ]
    )
    for comparison in comparisons:
        if not comparison["same_selected_support"]:
            notes.append(
                f"- {comparison['preset']}, k={comparison['k']}, seed={comparison['seed']}: "
                f"added {comparison['added_features']}, removed {comparison['removed_features']}; "
                f"FDP difference {comparison['fdp_difference']:+.3f}, "
                f"power difference {comparison['power_difference']:+.3f}."
            )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (destination / "summary.md").write_text("\n".join(notes) + "\n")
    (destination / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for preset, color, label in (("headline", "#0072B2", "Headline STK"), ("secondary", "#D55E00", "Secondary STK")):
        cells = sorted(
            [cell for cell in summary["cells"] if cell["preset"] == preset and cell["design"]["k"]],
            key=lambda cell: cell["design"]["k"],
        )
        for arm, style, suffix in (("single_split_mean", "--", "single split"), ("aggregate", "-", "aggregate")):
            for axis, metric in zip(axes, ("fdp", "power"), strict=True):
                axis.errorbar(
                    [cell["design"]["k"] for cell in cells],
                    [cell[arm][metric]["mean"] for cell in cells],
                    yerr=[cell[arm][metric]["se_across_datasets"] for cell in cells],
                    color=color,
                    linestyle=style,
                    marker="o",
                    capsize=2,
                    label=f"{label}: {suffix}",
                )
    for axis, ylabel in zip(axes, ("Mean FDP", "Mean power"), strict=True):
        axis.set(xlabel="True support size k", ylabel=ylabel, ylim=(-0.02, 1.05), xticks=[1, 2, 3, 5, 10, 15, 20])
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    for extension in ("pdf", "png"):
        fig.savefig(destination / f"matched_aggregation.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(notes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.input, args.output)


if __name__ == "__main__":
    main()
