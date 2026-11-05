"""Recurrence-sandwich diagnostics for STK (from 04-pc-knockoff-comparison) and repeated PC.

For split selections S_1..S_M with frequencies Pi_j, write A = sum_j Pi_j (mean split
size), I = sum_j Pi_j^2, rho = I / A (overlap ratio), and R = {j : Pi_j >= eta}.
The pathwise recurrence sandwich states, for any feature subset G with A_G > 0,
    (rho_G - eta) / (1 - eta) <= F_G = sum_{R cap G} Pi_j / A_G <= rho_G / eta,
and (I_G - eta A_G) / (1 - eta) <= |R cap G| <= I_G / eta^2.
This script checks the inequalities on every record and reports the quantities for
G = all features and G = planted support, together with single-split and aggregate
power, so that theory and the empirical comparison use identical datasets.
The pairwise ratio estimate rho_U = sum_{m != m'} |S_m cap S_m'| / ((M-1) sum_m |S_m|) of the
population overlap ratio is also reported (a ratio of unbiased sums, not itself unbiased).

Run from the project root: python experiments/05-repeated-pc-knockoff/recurrence_bounds.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _pc_comparison
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
RES = HERE / "results"
ETA = 0.5
TOL = 1e-12


def quantities(split_sets, group=None):
    counts: dict[int, int] = {}
    for s in split_sets:
        for j in s:
            if group is None or j in group:
                counts[j] = counts.get(j, 0) + 1
    m = len(split_sets)
    pi = np.array([c / m for c in counts.values()], dtype=float)
    a, i = float(pi.sum()), float((pi**2).sum())
    kept = pi >= ETA - TOL
    retained_mass, retained_count = float(pi[kept].sum()), int(kept.sum())
    sizes = [len(s if group is None else set(s) & group) for s in split_sets]
    pair = float(np.sum(np.array(list(counts.values()), dtype=float) ** 2) - sum(sizes))  # sum_{m != m'} overlaps
    rho_u = pair / ((m - 1) * sum(sizes)) if sum(sizes) else float("nan")
    out = {"A": a, "I": i, "rho": i / a if a else float("nan"), "rho_u": rho_u}
    out["F"] = retained_mass / a if a else float("nan")
    out["count"] = retained_count
    out["count_lo"] = (i - ETA * a) / (1 - ETA)
    out["count_hi"] = i / ETA**2
    if a:
        lo, hi = (out["rho"] - ETA) / (1 - ETA), out["rho"] / ETA
        assert lo - TOL <= out["F"] <= hi + TOL, (out, lo, hi)
        assert out["count_lo"] - 1e-9 <= retained_count <= out["count_hi"] + 1e-9, out
    return out


def methods_for(comparison, k, seed):
    support = set(comparison[k, seed, "stk_secondary"]["S_star"])
    sets = {}
    for arm, name in (("stk_headline", "Headline STK"), ("stk_secondary", "Secondary STK")):
        res = comparison[k, seed, arm]["result"]
        sets[name] = [s["selected"] for s in res["splits"]]
    pc = json.loads((OUT / f"k-{k}_seed-{seed}.json").read_text())
    for alloc in ("200-50", "200-100", "400-50", "400-100"):
        sets[f"PC {alloc}"] = [s["selected"]["0.5"] for s in pc["allocations"][alloc]["splits"]]
    return support, sets


def main():
    args = _pc_comparison.parse_args(__doc__)
    comparison, excluded = _pc_comparison.load(args.pc_comparison_out, args.exclude_failed)
    RES.mkdir(exist_ok=True)
    rows, checked = [], 0
    for k in (0, 1, 3, 10, 20):
        seeds = range(10000, 10030 if k == 0 else 10010)
        acc: dict[str, list] = {}
        for seed in seeds:
            if (k, seed, "stk_secondary") not in comparison:
                continue
            support, sets = methods_for(comparison, k, seed)
            for name, ss in sets.items():
                q_all = quantities(ss)
                checked += 1
                entry = {"all": q_all}
                if support:
                    q_sig = quantities(ss, support)
                    checked += 1
                    entry["sig"] = q_sig
                    entry["single_power"] = float(np.mean([len(set(s) & support) / len(support) for s in ss]))
                    entry["agg_power"] = q_sig["count"] / len(support)
                    entry["power_hi"] = min(1.0, q_sig["count_hi"] / len(support))
                    entry["power_lo"] = max(0.0, q_sig["count_lo"] / len(support))
                acc.setdefault(name, []).append(entry)
        for name, entries in acc.items():

            def mean(path, entries=entries):
                vals = []
                for e in entries:
                    v = e
                    for key in path:
                        v = v.get(key) if isinstance(v, dict) else None
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        vals.append(v)
                return float(np.mean(vals)) if vals else float("nan")

            row = {
                "k": k,
                "method": name,
                "mean_split_size": mean(("all", "A")),
                "rho_all": mean(("all", "rho")),
                "rho_u_all": mean(("all", "rho_u")),
                "F_all": mean(("all", "F")),
            }
            if k:
                row.update(
                    rho_sig=mean(("sig", "rho")),
                    single_power=mean(("single_power",)),
                    agg_power=mean(("agg_power",)),
                    power_lo=mean(("power_lo",)),
                    power_hi=mean(("power_hi",)),
                )
            if excluded.get(k):
                row["excluded_failed_datasets"] = excluded[k]
            rows.append(row)
            print(json.dumps({key: (round(v, 3) if isinstance(v, float) else v) for key, v in row.items()}))
    print(f"sandwich inequalities verified on {checked} (record, subset) pairs")
    (RES / "recurrence_bounds.json").write_text(json.dumps({"checked": checked, "rows": rows}, indent=1) + "\n")
    write_table()


def write_table():
    """LaTeX tables of recurrence diagnostics at k = 3, 10, 20.

    The compact table uses headline STK; the full table adds secondary STK.
    """
    rows = json.loads((RES / "recurrence_bounds.json").read_text())["rows"]
    pcs = ["PC 200-50", "PC 200-100", "PC 400-50", "PC 400-100"]
    label = {"Headline STK": "Headline STK", "Secondary STK": "Secondary STK"}
    label |= {n: "Repeated " + n.replace("-", "/") for n in pcs}
    tables = RES / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for filename, names in (
        ("recurrence_bounds.tex", ["Headline STK", *pcs]),
        ("recurrence_bounds_all.tex", ["Headline STK", "Secondary STK", *pcs]),
    ):
        lines = [
            "\\begin{tabular}{@{}rlcccccc@{}}",
            "\\toprule",
            "$k$ & Method & $\\rho_{\\mathrm{all}}$ & $\\rho_{S^\\star}$ & Split power & Aggregate power"
            " & Lower bound & Upper bound \\\\",
            "\\midrule",
        ]
        for k in (3, 10, 20):
            for n in names:
                r = next(x for x in rows if x["k"] == k and x["method"] == n)
                lines.append(
                    f"{k} & {label[n]} & {r['rho_all']:.2f} & {r['rho_sig']:.2f} & {r['single_power']:.3f} & "
                    f"{r['agg_power']:.3f} & {r['power_lo']:.3f} & {r['power_hi']:.3f} \\\\"
                )
            if k != 20:
                lines.append("\\addlinespace")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (tables / filename).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
