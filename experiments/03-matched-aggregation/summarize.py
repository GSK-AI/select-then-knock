"""Summarize matched ablations using DATASETS, not splits, as replicates.

Run from the project root:
    python experiments/03-matched-aggregation/summarize.py \
      --input experiments/03-matched-aggregation/outputs/full

Writes summary.json and summary.md inside the input run directory (or --output). Single-split
metrics are first averaged within each dataset; paired differences and standard
errors are then computed across dataset seeds. With one seed, SE is undefined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def estimate(values):
    values = [value for value in values if value is not None]
    return {
        "mean": float(np.mean(values)) if values else None,
        "se_across_datasets": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
        "n_datasets": len(values),
    }


def summarize(root):
    groups = defaultdict(list)
    common = None
    inputs = []
    for path in sorted(root.glob("k-*/seed-*/*.json")):
        record = json.loads(path.read_text())
        if not record.get("complete"):
            raise ValueError(f"Incomplete run: {path}")
        signature = record["signature"]
        comparison = {
            key: signature[key]
            for key in (
                "source_sha256",
                "runner_sha256",
                "lib_sha256",
                "smoke_only",
                "python",
                "packages",
                "thread_environment",
            )
        }
        comparison["monitor_sha256"] = signature.get("monitor_sha256")
        comparison["numpy_build"] = signature.get("numpy_build")
        if common is not None and common != comparison:
            raise ValueError("Cannot combine different source/environment versions or smoke and full runs")
        common = comparison
        key = (signature["preset"], signature["design"]["k"])
        groups[key].append(record)
        inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not groups:
        raise ValueError(f"No completed cell records in {root}")
    output = {
        "replication_unit": "dataset seed",
        "smoke_only": common["smoke_only"],
        "provenance": common,
        "input_records": inputs,
        "cells": [],
    }
    lines = [
        "# Matched aggregation",
        "",
        "Single-split results average individual split metrics within each dataset. "
        "Splits are not treated as independent datasets. Differences are aggregate minus single-split mean.",
        "",
        "| Preset | k | Datasets | Single FDP | Aggregate FDP | Single power | Aggregate power | "
        "Single firing | Aggregate firing | Held-out fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (preset, k), records in sorted(groups.items()):
        first = records[0]["signature"]
        reference_params = {key: value for key, value in first["parameters"].items() if key != "random_state"}
        seeds = [record["signature"]["dataset_seed"] for record in records]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate dataset seed")
        for record in records:
            params = {key: value for key, value in record["signature"]["parameters"].items() if key != "random_state"}
            if record["signature"]["design"] != first["design"] or params != reference_params:
                raise ValueError("Cannot combine mismatched designs or fitting settings")
        cell = {"preset": preset, "design": first["design"], "seeds": seeds, "parameters": reference_params}
        for arm in ("single_split_mean", "aggregate", "paired_difference_aggregate_minus_single_mean"):
            cell[arm] = {
                metric: estimate([record["summary"][arm][metric] for record in records])
                for metric in records[0]["summary"][arm]
            }
        cell["held_out_fallback_count"] = sum(record["summary"]["held_out_fallback_count"] for record in records)
        cell["total_elapsed_s"] = sum(record["elapsed_s"] for record in records)
        cell["convergence_by_stage"] = {}
        for stage in ("held_out", "other"):
            rows = [record.get("convergence_by_stage", {}).get(stage) for record in records]
            rows = [row for row in rows if row is not None]
            if rows:
                cell["convergence_by_stage"][stage] = {
                    key: sum(row[key] for row in rows)
                    for key in (
                        "fits",
                        "errors",
                        "warning_count",
                        "fits_with_warnings",
                        "final_refits_at_iteration_limit",
                    )
                }
        output["cells"].append(cell)

        def formatted(arm, metric):
            value = cell[arm][metric]["mean"]
            return "NA" if value is None else f"{value:.3f}"

        vals = [
            formatted(arm, metric)
            for metric in ("fdp", "power", "firing")
            for arm in ("single_split_mean", "aggregate")
        ]
        lines.append(
            f"| {preset} | {k} | {len(records)} | " + " | ".join(vals) + f" | {cell['held_out_fallback_count']} |"
        )
    lines.extend(["", "Power is undefined under the global null."])
    if any(len(cell["seeds"]) == 1 for cell in output["cells"]):
        lines.append("A one-dataset run does not estimate across-dataset uncertainty.")
    if common["smoke_only"]:
        lines.extend(["", "SMOKE TEST ONLY: fitting budgets were reduced; these results are not scientific evidence."])
    return output, "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional destination for the small canonical summary")
    args = parser.parse_args()
    output, markdown = summarize(args.input)
    destination = args.output or args.input
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    (destination / "summary.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
