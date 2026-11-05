"""Audit and summarize PC-Knockoff comparison records; write results/ tables and figures.

Run from the project root:

    python experiments/04-pc-knockoff-comparison/summarize.py --stage eval

Reads outputs/<stage>/ and the planned cells in configs/tasks_<stage>.txt. Every check
below must pass, or the script stops before writing summaries:

- every planned (k, seed, arm) record exists; failures are listed, not dropped;
- all arms of a dataset share one dataset hash, matching the saved .npz;
- all records share one runner, comparator, STK-source, _lib and package fingerprint;
- metrics recompute from the stored selections and S*;
- PC selections recompute from the stored W, candidates and knockoff+ (offset 1);
- STK selections recompute from the recorded splits at eta; split count is n_splits;
- no record mentions a celer fallback.

Datasets are the replicates. PC alphas share one evidence computation per dataset.
Outputs: a per-condition table (FDP, power, size, firing, runtime, memory), a
global-null firing figure, and a runtime/memory figure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pc_knockoff as pk  # noqa: E402
from run import ARMS, sha256_arrays, support_metrics  # noqa: E402

LABELS = {"stk_headline": "Headline STK", "stk_secondary": "Secondary STK", "pc_knockoff": "PC-Knockoff"}
COLORS = {"stk_headline": "#2a78d6", "stk_secondary": "#1baf7a", "pc_knockoff": "#eb6834"}
MARKERS = {"stk_headline": "o", "stk_secondary": "s", "pc_knockoff": "D"}
FINGERPRINT_KEYS = ("runner_sha256", "pc_knockoff_sha256", "stk_source_sha256", "lib_sha256", "packages", "python")


def wilson(successes, n, z=1.959964):
    if n == 0:
        return None, None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mean_ci(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return {"n": 0, "mean": None, "se": None, "ci95": None}
    if len(values) == 1:
        return {"n": 1, "mean": float(values[0]), "se": None, "ci95": None}
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    half = float(stats.t.ppf(0.975, len(values) - 1) * se)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "se": se,
        "ci95": [values.mean() - half, values.mean() + half],
    }


def load(stage):
    out = HERE / "outputs" / stage
    planned = [
        tuple(map(int, line.split())) for line in (HERE / "configs" / f"tasks_{stage}.txt").read_text().splitlines()
    ]
    records, problems = {}, []
    for k, seed in planned:
        for arm in ARMS:
            path = out / f"k-{k}" / f"seed-{seed}" / f"{arm}.json"
            if not path.exists():
                problems.append(f"missing {path.relative_to(out)}")
                continue
            records[(k, seed, arm)] = json.loads(path.read_text())
    return out, planned, records, problems


def audit(out, planned, records):
    errors, failures = [], []
    fingerprints = {
        json.dumps({key: r["signature"][key] for key in FINGERPRINT_KEYS}, sort_keys=True) for r in records.values()
    }
    if len(fingerprints) != 1:
        errors.append(f"{len(fingerprints)} distinct code/package fingerprints across records")
    configs = {
        json.dumps({key: v for key, v in r["signature"]["config"].items() if key != "timeout_s"}, sort_keys=True)
        for r in records.values()
    }
    if len(configs) != 1:
        errors.append(f"{len(configs)} distinct run configurations across records")
    for k, seed in planned:
        arms = {arm: records.get((k, seed, arm)) for arm in ARMS}
        hashes = {r["signature"]["dataset_sha256"] for r in arms.values() if r}
        if len(hashes) != 1:
            errors.append(f"k={k} seed={seed}: arms disagree on the dataset hash")
            continue
        with np.load(out / "data" / f"k-{k}_seed-{seed}.npz") as data:
            if sha256_arrays(data["X"], data["y"], data["S_star"]) != hashes.pop():
                errors.append(f"k={k} seed={seed}: saved arrays do not match the recorded hash")
        for arm, record in arms.items():
            if record is None:
                continue
            result, support = record["result"], record["S_star"]
            tag = f"k={k} seed={seed} {arm}"
            if not result.get("ok"):
                failures.append({"k": k, "seed": seed, "arm": arm, "error": result.get("error")})
                continue
            if any("celer" in w["warning"].lower() for w in result["warnings"]):
                errors.append(f"{tag}: celer warning recorded")
            cfg = record["signature"]["config"]
            if arm == "pc_knockoff":
                W, cand = np.asarray(result["W"]), np.asarray(result["candidates"])
                if len(cand) != cfg["d"] or not 2 * cfg["d"] < result["n_test_rows"] or not result["rows_disjoint"]:
                    errors.append(f"{tag}: candidate count, 2d < n_test or row disjointness fails")
                for alpha, point in result["operating_points"].items():
                    t = pk.knockoff_plus_threshold(W, float(alpha))
                    expected = np.sort(cand[W >= t]).tolist() if np.isfinite(t) else []
                    if expected != point["selected"]:
                        errors.append(f"{tag} alpha={alpha}: selection does not recompute from W")
                    if support_metrics(point["selected"], support) != point["metrics"]:
                        errors.append(f"{tag} alpha={alpha}: metrics do not recompute")
            else:
                splits, eta = result["splits"], result["parameters"]["eta"]
                counts = np.zeros(cfg["p"], dtype=int)
                for split in splits:
                    counts[split["selected"]] += 1
                if len(splits) != cfg["stk_splits"]:
                    errors.append(f"{tag}: {len(splits)} splits recorded")
                if np.flatnonzero(counts / len(splits) >= eta).tolist() != result["selected"]:
                    errors.append(f"{tag}: aggregate does not recompute from splits")
                if support_metrics(result["selected"], support) != result["metrics"]:
                    errors.append(f"{tag}: metrics do not recompute")
                if result["parameters"]["solver_backend"] != cfg["solver"]:
                    errors.append(f"{tag}: solver backend differs from configuration")
    return errors, failures


def operating_rows(records):
    """One row per (k, seed, method label) with metrics and resources."""
    rows = []
    for (k, seed, arm), record in records.items():
        result = record["result"]
        base = {"k": k, "seed": seed, "arm": arm, "ok": bool(result.get("ok"))}
        if not result.get("ok"):
            rows.append({**base, "method": arm})
            continue
        resources = {
            "wall_s": result["wall_s"],
            "core_s": sum(result["stage_elapsed_s"].values()),
            # Exact getrusage peak of the arm process vs sampled process-tree peak: STK's
            # n_jobs run as threads (one process), and sampling misses PC's sub-second spike.
            "peak_rss_mib": max(result["peak_rss_bytes"], result["peak_tree_rss_bytes"]) / 2**20,
            "peak_tree_rss_mib": result["peak_tree_rss_bytes"] / 2**20,
            "peak_self_rss_mib": result["peak_rss_bytes"] / 2**20,
            "baseline_rss_mib": result["peak_rss_after_load_bytes"] / 2**20,
            "max_tree_processes": result["max_tree_processes"],
            "warnings": sum(w["count"] for w in result["warnings"]),
        }
        if arm == "pc_knockoff":
            extra = {
                "support_screened": result["support_screened"],
                "n_positive_W": result["n_positive_W"],
                "knockoff_s": result["knockoff_diagnostics"]["s"],
                "stage_s": result["stage_elapsed_s"],
            }
            for alpha, point in result["operating_points"].items():
                rows.append({**base, "method": f"pc_knockoff@{alpha}", **point["metrics"], **resources, **extra})
        else:
            extra = {
                "fallback_count": result["fallback_count"],
                "mean_candidates": float(np.mean(result["candidate_counts"])),
                "max_candidates": int(np.max(result["candidate_counts"])),
            }
            rows.append({**base, "method": arm, **result["metrics"], **resources, **extra})
    return rows


def summarize(rows, planned):
    by = defaultdict(list)
    for row in rows:
        by[(row["k"], row["method"])].append(row)
    planned_per_k = defaultdict(int)
    for k, _ in planned:
        planned_per_k[k] += 1
    cells = []
    for (k, method), group in sorted(by.items(), key=lambda item: (item[0][0], item[0][1])):
        done = [row for row in group if row["ok"]]
        cell = {"k": k, "method": method, "planned": planned_per_k[k], "completed": len(done)}
        if done:
            fired = sum(row["firing"] for row in done)
            cell.update(
                fdp=mean_ci([row["fdp"] for row in done]),
                power=mean_ci([row["power"] for row in done]) if k else None,
                n_selected=mean_ci([row["n_selected"] for row in done]),
                tp=mean_ci([row["tp"] for row in done]),
                fp=mean_ci([row["fp"] for row in done]),
                exact_recovery=sum(row["exact_recovery"] for row in done) / len(done),
                firing={"count": fired, "rate": fired / len(done), "wilson95": wilson(fired, len(done))},
                abstention_rate=1 - fired / len(done),
                wall_s={"median": float(np.median([r["wall_s"] for r in done])), "max": max(r["wall_s"] for r in done)},
                core_s={"median": float(np.median([r["core_s"] for r in done])), "max": max(r["core_s"] for r in done)},
                peak_rss_mib={
                    "median": float(np.median([r["peak_rss_mib"] for r in done])),
                    "max": max(r["peak_rss_mib"] for r in done),
                },
                max_tree_processes=max(r["max_tree_processes"] for r in done),
                warnings_total=sum(r["warnings"] for r in done),
            )
            if method.startswith("pc_knockoff"):
                cell["support_screened_mean"] = float(np.mean([r["support_screened"] for r in done]))
                cell["sure_screening_rate"] = float(np.mean([r["support_screened"] == k for r in done])) if k else None
            else:
                cell["fallbacks_total"] = sum(r["fallback_count"] for r in done)
                cell["mean_candidates"] = float(np.mean([r["mean_candidates"] for r in done]))
        cells.append(cell)
    return cells


def fmt(stat, digits=2):
    if stat is None or stat["mean"] is None:
        return "–"
    text = f"{stat['mean']:.{digits}f}"
    return text + (f" ± {stat['se']:.{digits}f}" if stat["se"] is not None else "")


def markdown(stage, cells, failures):
    lines = [f"# PC-Knockoff comparison summary: {stage}", "", "Mean ± SE over datasets. Firing CI is Wilson 95%.", ""]
    lines += [
        "| k | method | done/planned | FDP | power | size | TP | FP | exact | firing (95% CI) | sure screen | "
        "median wall s | max peak RSS MiB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        if not c["completed"]:
            lines.append(f"| {c['k']} | {c['method']} | 0/{c['planned']} | | | | | | | | | | |")
            continue
        lo, hi = c["firing"]["wilson95"]
        screen = c.get("sure_screening_rate")
        lines.append(
            f"| {c['k']} | {c['method']} | {c['completed']}/{c['planned']} | {fmt(c['fdp'])} | {fmt(c['power'])} | "
            f"{fmt(c['n_selected'], 1)} | {fmt(c['tp'], 1)} | {fmt(c['fp'], 1)} | {c['exact_recovery']:.2f} | "
            f"{c['firing']['count']}/{c['completed']} ({lo:.2f}–{hi:.2f}) | "
            f"{'–' if screen is None else f'{screen:.2f}'} | {c['wall_s']['median']:.1f} | "
            f"{c['peak_rss_mib']['max']:.0f} |"
        )
    lines += ["", f"## Failures ({len(failures)})", ""]
    lines += [f"- k={f['k']} seed={f['seed']} {f['arm']}: {f['error']}" for f in failures] or ["None."]
    return "\n".join(lines) + "\n"


def figures(stage, cells, rows, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
    arm_of = lambda method: method.split("@")[0]  # noqa: E731
    null = [c for c in cells if c["k"] == 0 and c["completed"]]
    if null:
        fig, ax = plt.subplots(figsize=(3.4, 2.0))
        order = sorted(null, key=lambda c: (arm_of(c["method"]) != "stk_headline", c["method"]))
        for i, c in enumerate(order):
            arm = arm_of(c["method"])
            lo, hi = c["firing"]["wilson95"]
            rate = c["firing"]["rate"]
            ax.errorbar(
                i,
                rate,
                yerr=[[rate - lo], [hi - rate]],
                fmt=MARKERS[arm],
                ms=5,
                color=COLORS[arm],
                elinewidth=0.8,
                capsize=0,
            )
            ax.annotate(
                f"{c['firing']['count']}/{c['completed']}",
                (i, hi),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                color="#52514e",
            )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            [
                c["method"]
                .replace("pc_knockoff@", "PC-Knockoff\nα=")
                .replace("stk_headline", "Headline\nSTK")
                .replace("stk_secondary", "Secondary\nSTK")
                for c in order
            ],
            fontsize=6.5,
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("firing rate under global null")
        ax.grid(axis="y", color="#e6e5e0", lw=0.5)
        fig.tight_layout()
        fig.savefig(results / f"fig_null_firing_{stage}.pdf", bbox_inches="tight")
        fig.savefig(results / f"fig_null_firing_{stage}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Resources: one row per arm and dataset (PC alphas share one run).
    per_arm = defaultdict(list)
    for row in rows:
        if row["ok"] and (row["arm"] != "pc_knockoff" or row["method"] == "pc_knockoff@0.1"):
            per_arm[row["arm"]].append(row)
    if per_arm:
        fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.1))
        arms = [a for a in ("pc_knockoff", "stk_headline", "stk_secondary") if a in per_arm]
        rng = np.random.default_rng(0)
        for ax, key, label, log in (
            (axes[0], "wall_s", "end-to-end wall time per dataset (s)", True),
            (axes[1], "peak_rss_mib", "peak RSS per arm process (MiB)", False),
        ):
            for i, arm in enumerate(arms):
                values = np.array([r[key] for r in per_arm[arm]])
                ax.scatter(
                    i + rng.uniform(-0.12, 0.12, len(values)),
                    values,
                    s=9,
                    color=COLORS[arm],
                    marker=MARKERS[arm],
                    lw=0,
                    alpha=0.8,
                )
                ax.hlines(np.median(values), i - 0.25, i + 0.25, color="#0b0b0b", lw=1)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels([LABELS[a] for a in arms], fontsize=6.5)
            ax.set_ylabel(label, fontsize=7)
            if log:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            else:
                ax.set_ylim(bottom=0)
            ax.grid(axis="y", color="#e6e5e0", lw=0.5)
        fig.tight_layout()
        fig.savefig(results / f"fig_resources_{stage}.pdf", bbox_inches="tight")
        fig.savefig(results / f"fig_resources_{stage}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("smoke", "eval"), required=True)
    args = parser.parse_args()
    out, planned, records, missing = load(args.stage)
    errors, failures = audit(out, planned, records)
    if errors or missing:
        print("AUDIT FAILED:\n  " + "\n  ".join(errors + missing))
        sys.exit(1)
    rows = operating_rows(records)
    cells = summarize(rows, planned)
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    seeds = sorted({seed for _, seed in planned})
    payload = {
        "stage": args.stage,
        "seeds": [seeds[0], seeds[-1]],
        "planned_datasets": len(planned),
        "records": len(records),
        "missing_records": missing,
        "failures": failures,
        "audit": "passed: hashes, fingerprints, recomputed selections and metrics",
        "fingerprint": {
            key: next(iter(records.values()))["signature"][key]
            for key in FINGERPRINT_KEYS
            if key != "stk_source_sha256"
        },
        "config": next(iter(records.values()))["signature"]["config"],
        "hardware": sorted({r["provenance"]["hostname"] for r in records.values()}),
        "cells": cells,
    }
    name = f"summary_{args.stage}_seeds{seeds[0]}-{seeds[-1]}"
    (results / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    (results / f"{name}.md").write_text(markdown(args.stage, cells, failures))
    figures(args.stage, cells, rows, results)
    print(f"audit passed; {len(records)} records, {len(failures)} failures, {len(missing)} missing")
    print((results / f"{name}.md").read_text())


if __name__ == "__main__":
    main()
