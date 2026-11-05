"""Summarize repeated PC and pair it with the STK records of the PC-Knockoff comparison (04).

Writes results/summary.json, results/summary.md, and LaTeX tables under
results/tables/. Single-split metrics are averaged over the 25 splits within
each dataset (the 03-matched-aggregation convention); aggregates keep features
selected in >= ceil(M*eta) = 13 splits.

Run from the project root: python experiments/05-repeated-pc-knockoff/summarize.py
Stops before writing anything if a PC-Knockoff comparison arm failed (unless
--exclude-failed) or if split 0 does not reproduce its PC-Knockoff selections.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import _pc_comparison
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
RES = HERE / "results"
M, ETA = 25, 0.5
R = math.ceil(M * ETA)
ALPHAS = ["0.1", "0.2", "0.5"]
KS = _pc_comparison.KS
STK = {"Headline STK": "stk_headline", "Secondary STK": "stk_secondary"}


def metrics(selected, support):
    sel, sup = set(selected), set(support)
    tp = len(sel & sup)
    return {
        "fdp": (len(sel) - tp) / max(1, len(sel)),
        "power": tp / len(sup) if sup else float("nan"),
        "firing": float(len(sel) > 0),
        "size": float(len(sel)),
        "exact": float(sel == sup) if sup else float("nan"),
    }


def aggregate(split_sets):
    counts = {}
    for s in split_sets:
        for j in s:
            counts[j] = counts.get(j, 0) + 1
    return sorted(j for j, c in counts.items() if c >= R)


def mean_se(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def t_interval(diffs):
    d = np.asarray(diffs, dtype=float)
    m = d.mean()
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float(m), float(m), float(m)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(m), float(m - h), float(m + h)


def wilson(x, n, z=1.959964):
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def main():
    args = _pc_comparison.parse_args(__doc__)
    comparison, excluded = _pc_comparison.load(args.pc_comparison_out, args.exclude_failed)
    tasks = sorted({(k, seed) for k, seed, _ in comparison})
    records = [json.loads((OUT / f"k-{k}_seed-{seed}.json").read_text()) for k, seed in tasks]

    # Consistency: split 0 at (200, 50) must reproduce the single-split PC-Knockoff selections.
    mismatches = 0
    for rec in records:
        ref_ops = comparison[rec["k"], rec["seed"], "pc_knockoff"]["result"]["operating_points"]
        split0 = rec["allocations"]["200-50"]["splits"][0]["selected"]
        mismatches += sum(sorted(split0[a]) != sorted(ref_ops[a]["selected"]) for a in ALPHAS)
    if mismatches:
        sys.exit(f"{mismatches} split-0 selections differ from the single-split PC-Knockoff; no tables written.")
    RES.mkdir(exist_ok=True)

    per_dataset = {}
    for rec in records:
        k, seed, sup = rec["k"], rec["seed"], rec["S_star"]
        for alloc, block in rec["allocations"].items():
            for a in ALPHAS:
                sets = [s["selected"][a] for s in block["splits"]]
                single = [metrics(s, sup) for s in sets]
                single_mean = {key: float(np.mean([m[key] for m in single])) for key in single[0]}
                agg = metrics(aggregate(sets), sup)
                screened = float(np.mean([s["support_screened"] for s in block["splits"]])) / max(1, len(sup))
                per_dataset[(f"PC {alloc} a={a}", k, seed)] = {"single": single_mean, "agg": agg, "screened": screened}
        for name, arm in STK.items():
            ref = comparison[k, seed, arm]["result"]
            per_dataset[(name, k, seed)] = {"agg": metrics(ref["selected"], sup)}

    methods = sorted({key[0] for key in per_dataset})
    summary = {"split0_consistency_mismatches": mismatches, "conditions": [], "paired": [], "null": []}
    if excluded:
        summary["excluded_failed_datasets_per_k"] = {str(k): n for k, n in excluded.items()}
    for method in methods:
        for k in KS:
            seeds = sorted(s for (mm, kk, s) in per_dataset if mm == method and kk == k)
            ds = [per_dataset[(method, k, s)] for s in seeds]
            row = {"method": method, "k": k, "n": len(ds)}
            for level in ("single", "agg"):
                if level not in ds[0]:
                    continue
                for key in ("fdp", "power", "firing", "size", "exact"):
                    row[f"{level}_{key}"], row[f"{level}_{key}_se"] = mean_se([d[level][key] for d in ds])
            if "screened" in ds[0]:
                row["screened_fraction"] = float(np.mean([d["screened"] for d in ds]))
            summary["conditions"].append(row)
            if k == 0:
                fired = int(sum(d["agg"]["firing"] for d in ds))
                summary["null"].append(
                    {"method": method, "fired": fired, "n": len(ds), "wilson": wilson(fired, len(ds))}
                )

    # Paired: each STK preset minus each aggregated PC operating point, same datasets.
    for stk in STK:
        for method in methods:
            if not method.startswith("PC"):
                continue
            for k in [1, 3, 10, 20]:
                seeds = sorted(s for (mm, kk, s) in per_dataset if mm == method and kk == k)
                for key in ("fdp", "power"):
                    diffs = [
                        per_dataset[(stk, k, s)]["agg"][key] - per_dataset[(method, k, s)]["agg"][key] for s in seeds
                    ]
                    m, lo, hi = t_interval(diffs)
                    summary["paired"].append(
                        {"stk": stk, "pc": method, "k": k, "metric": key, "diff": m, "lo": lo, "hi": hi}
                    )

    (RES / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    lines = [
        "# Repeated PC summary",
        "",
        f"Split-0 consistency mismatches against single-split PC-Knockoff (should be 0): {mismatches}",
        "",
        *([f"Datasets excluded for a failed PC-Knockoff comparison arm, per k: {excluded}", ""] if excluded else []),
        "| Method | k | single FDP | agg FDP | single power | agg power | agg size | agg exact | screened |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["conditions"]:
        f = lambda key: f"{r[key]:.3f}" if key in r and not math.isnan(r[key]) else "NA"  # noqa: E731
        lines.append(
            f"| {r['method']} | {r['k']} | {f('single_fdp')} | {f('agg_fdp')} | {f('single_power')} | "
            f"{f('agg_power')} | {f('agg_size')} | {f('agg_exact')} | {f('screened_fraction')} |"
        )
    lines += ["", "## Global null (aggregate firing)", ""]
    for r in summary["null"]:
        lines.append(f"- {r['method']}: {r['fired']}/{r['n']}, Wilson [{r['wilson'][0]:.3f}, {r['wilson'][1]:.3f}]")
    lines += ["", "## Paired STK minus repeated PC (unadjusted 95% t intervals)", ""]
    lines += ["| STK | PC | k | metric | diff [95% CI] |", "|---|---|---:|---|---:|"]
    for r in summary["paired"]:
        lines.append(
            f"| {r['stk']} | {r['pc']} | {r['k']} | {r['metric']} | {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] |"
        )
    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    write_latex(summary)


def write_latex(summary):
    cond = {(r["method"], r["k"]): r for r in summary["conditions"]}
    null = {r["method"]: r for r in summary["null"]}

    def cell(method, k, level="agg"):
        r = cond[(method, k)]
        return f"{r[f'{level}_power']:.3f} / {r[f'{level}_fdp']:.3f}"

    # Main text: the 200/50 allocation, single-split PC-Knockoff vs repeated PC, and STK.
    rows = []
    for a in ALPHAS:
        m = f"PC 200-50 a={a}"
        rows.append(
            f"PC-Knockoff $\\alpha={a}$ & "
            + " & ".join(cell(m, k, "single") for k in (3, 10, 20))
            + f" & {cond[(m, 0)]['single_firing']:.3f} \\\\"
        )
    for a in ALPHAS:
        m = f"PC 200-50 a={a}"
        rows.append(
            f"Repeated PC $\\alpha={a}$ & "
            + " & ".join(cell(m, k) for k in (3, 10, 20))
            + f" & {null[m]['fired'] / null[m]['n']:.3f} \\\\"
        )
    # Headline STK is the primary configuration; retain the secondary row too.
    for m in STK:
        rows.append(
            f"{m} & " + " & ".join(cell(m, k) for k in (3, 10, 20)) + f" & {null[m]['fired'] / null[m]['n']:.3f} \\\\"
        )
    main = [
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "Method & $k=3$ & $k=10$ & $k=20$ & Null firing \\\\",
        "\\midrule",
        *rows[:3],
        "\\midrule",
        *rows[3:6],
        "\\midrule",
        *rows[6:],
        "\\bottomrule",
        "\\end{tabular}",
    ]
    tables = RES / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "repeated_pc_main.tex").write_text("\n".join(main) + "\n")

    # Appendix: every allocation and alpha, aggregate power / FDP and screening ceiling.
    full = [
        "\\begin{tabular}{@{}llccccc@{}}",
        "\\toprule",
        "$(n_1,d)$ & $\\alpha$ & $k=1$ & $k=3$ & $k=10$ & $k=20$ & Null \\\\",
        "\\midrule",
    ]
    for alloc in ("200-50", "200-100", "400-50", "400-100"):
        for a in ALPHAS:
            m = f"PC {alloc} a={a}"
            n1, d = alloc.split("-")
            full.append(
                f"({n1}, {d}) & {a} & "
                + " & ".join(cell(m, k) for k in (1, 3, 10, 20))
                + f" & {null[m]['fired']}/{null[m]['n']} \\\\"
            )
        if alloc != "400-100":
            full.append("\\addlinespace")
    full += ["\\bottomrule", "\\end{tabular}"]
    (tables / "repeated_pc_full.tex").write_text("\n".join(full) + "\n")

    screen = [
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "$(n_1,d)$ & $k=1$ & $k=3$ & $k=10$ & $k=20$ \\\\",
        "\\midrule",
    ]
    for alloc in ("200-50", "200-100", "400-50", "400-100"):
        m = f"PC {alloc} a=0.5"
        n1, d = alloc.split("-")
        screen.append(
            f"({n1}, {d}) & " + " & ".join(f"{cond[(m, k)]['screened_fraction']:.3f}" for k in (1, 3, 10, 20)) + " \\\\"
        )
    screen += ["\\bottomrule", "\\end{tabular}"]
    (tables / "repeated_pc_screen.tex").write_text("\n".join(screen) + "\n")


if __name__ == "__main__":
    main()
