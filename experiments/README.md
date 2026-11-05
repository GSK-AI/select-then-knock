# Experiment guide

Run from the project root with the documented dependency environment active.
No saved results, arrays, expression files, or scheduler scripts are included.
Each experiment creates and owns its own outputs when you explicitly run it.

## Designs

Folders follow the order of the paper's benchmark table.

| Folder | Paper experiment | Design and compared methods |
|---|---|---|
| `01-gaussian-support-recovery` | Synthetic Gaussian | n=800, p=5000, block size 20, rho=0.5, SNR=0.5; supports 1,2,3,5,10,15,20; 10 seeds; Headline STK, Secondary STK, Pooled STK, Stabl, LassoCV |
| `02-depmap-support-recovery` | DepMap | Fixed 800-row public-expression subset; planted linear supports 3 and 10, SNR=2; 10 seeds; the same five methods |
| `03-matched-aggregation` | Matched aggregation | Aggregate support versus its own 25 individual splits; Headline and Secondary STK; synthetic Gaussian design; 10 signal seeds per support and 30 null seeds |
| `04-pc-knockoff-comparison` | Screened knockoffs | n=800, p=5000, rho=0.5, SNR=0.5; supports 0,1,3,10,20; 30 null and 10 signal seeds; Headline STK, Secondary STK, and single-split PC-Knockoff at levels 0.1,0.2,0.5 |
| `05-repeated-pc-knockoff` | Screened knockoffs (Repeated PC) | The same 70 saved datasets as 04; 25 PC splits; allocations 200/50, 200/100, 400/50, 400/100; levels 0.1,0.2,0.5; frequency threshold 0.5 |
| `06-gaussian-global-null` | Gaussian global null | n=800, targets independent of X; rho 0, 0.3, 0.5, 0.8 at p=5000 and p=500, 2000 at rho=0.5; 30 seeds; the same five methods as 01 |
| `07-computational-scaling` | Scaling | n=800, support 3, rho=0.8, SNR=0.5; p=500, 2000, 5000, 20000; 10 seeds; Secondary STK and Direct Model-X |

Method keys in records and commands:

| Key | Paper name |
|---|---|
| `stk_headline` | Headline STK (preset `headline`) |
| `stk_secondary` | Secondary STK (preset `secondary`) |
| `stk_pooled` | Pooled STK (preset `pooled`) |
| `stabl` | Stabl (knockoff-based) |
| `lasso_cv` | LassoCV |
| `direct_model_x` | Direct Model-X (all p features, no screening) |
| `pc_knockoff` | PC-Knockoff (single split) |

Repeated PC is computed by `05-repeated-pc-knockoff` and has no arm key.

## Running the full experiments

Full grids can be expensive. Set thread limits to suit your machine. Set
`STK_SOLVER_BACKEND=celer` for experiments 01, 02, 06, and 07 to use the reported
fast solver.

```sh
export STK_SOLVER_BACKEND=celer
python experiments/01-gaussian-support-recovery/run.py
DEPMAP_EXPRESSION_PATH=/path/to/expression.csv python experiments/02-depmap-support-recovery/run.py
python experiments/06-gaussian-global-null/run.py
python experiments/07-computational-scaling/run.py
```

These runners take the environment variable `ARMS` (comma-separated method keys)
to run a subset. Experiments 02 and 07 also take `TIMEOUT_<KEY>` (for example
`TIMEOUT_STK_SECONDARY`) to change a per-fit time limit.

Experiment 03 runs its signal and null grids into the same output root. The
defaults are the reported signal grid (supports 1,2,3,5,10,15,20, presets
`headline,secondary`, output `outputs/full`):

```sh
SIGNAL_SEEDS=42,43,44,45,46,47,48,49,50,51
NULL_SEEDS=$SIGNAL_SEEDS,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71
python experiments/03-matched-aggregation/run.py --seeds "$SIGNAL_SEEDS"
python experiments/03-matched-aggregation/run.py --supports 0 --seeds "$NULL_SEEDS"
```

Experiment 04 must save its Gaussian arrays before experiment 05 can run. Its
fixed task list is `04-pc-knockoff-comparison/configs/tasks_eval.txt`.

```sh
python experiments/04-pc-knockoff-comparison/run.py --stage eval --supports 0 --seeds 10000-10029
python experiments/04-pc-knockoff-comparison/run.py --stage eval --supports 1,3,10,20 --seeds 10000-10009
python experiments/04-pc-knockoff-comparison/summarize.py --stage eval
python experiments/05-repeated-pc-knockoff/run.py --workers 1
python experiments/05-repeated-pc-knockoff/summarize.py
python experiments/05-repeated-pc-knockoff/recurrence_bounds.py
```

If experiment 04 was run with a non-default `--out`, pass the same directory to
the experiment 05 scripts with `--pc-comparison-out`.

Optional reference-comparison tests need a separately supplied reference source.
They skip when it, or the optional celer solver, is missing:

```sh
STK_REFERENCE_SRC=/path/to/reference/source-parent python -B -m pytest tests/test_reference.py -q -p no:cacheprovider
```

## Output ownership and analysis

`outputs/` holds per-seed and generated artifacts. `results/` holds summaries and
tables. Plotters write to a local `figures/` folder by default. There is no shared
result folder. Bulky outputs are ignored by git.

For experiments 01, 02, 06, and 07, run `merge_and_summarize.py` and then `plot.py`
after the runner finishes. Each merger reads either the single `outputs/results.json`
from an unsharded run or seed shards in its own `outputs/` folder:

| Experiment | Shard layout (one runner call per shard) | Merged file read by `plot.py` |
|---|---|---|
| 01, 06 | `outputs/seed-<s>/` with `SEEDS=<s> OUTDIR=...` | `outputs/results.json` |
| 07 | `outputs/p-<p>/seed-<s>/` with `P_VALUES=<p> SEEDS=<s> OUTDIR=...` | `results/results.json` |
| 02 | `outputs/k-<k>/seed-<s>/` with `K_VALUES=<k> SEEDS=<s> OUTDIR=...` | `results/results.json` |

The mergers check the full seed and cell grid. Run `plot.py --help` for the
output path options.

Experiment 07's memory plot needs a GNU-time report for each method and cell.
`run_resource_grid.sh` uses one existing environment, refuses existing cell
directories, and creates the shard layout expected by the merger. It requires an
already available GNU time; set `GNU_TIME=gtime` on macOS only if that executable
is already installed. Direct Model-X has no memory cap by default; set
`DIRECT_MODEL_X_MEMORY_GB` to cap its address space on Linux.

```sh
bash experiments/07-computational-scaling/run_resource_grid.sh --help
# The following runs the full grid only when explicitly invoked:
bash experiments/07-computational-scaling/run_resource_grid.sh
python experiments/07-computational-scaling/merge_and_summarize.py
python experiments/07-computational-scaling/plot.py
```

The plain experiment 07 runner produces selection and runtime records without
memory reports. The plot then reports the missing inputs instead of drawing an
incomplete comparison.

For experiment 03, `summarize.py`, `audit.py`, and `report.py` compute the matched
statistics and diagnostics. `audit.py --reference-results /path/to/seed-records`
optionally compares with seed records written by experiment 01; these must not
replace newly generated results. Missing supplied records are reported.

For experiment 04, the summarizer requires every task in its stage's configuration
file and checks input-array hashes, common code and dependency fingerprints, split
aggregation, and PC threshold reconstruction. Failed fits remain reported.

For experiment 05, run `summarize.py` and then `recurrence_bounds.py`. Generated
tables stay in its own `results/tables/`.

## Runtime and memory

The runtimes and memory reported in the paper were measured with the historical
implementation. Its screening step also ran a pooled knockoff fit whose
selections were discarded. This release omits that fit. Selections are unchanged,
but fresh runtimes and memory will be lower. They are not a reproduction of the
reported resource numbers.

The paper's scaling experiment (07) ran with 16 CPUs and 64 GB for both methods.
Runtime and memory depend on the machine, thread settings, and solver.

## DepMap input interface

Set `DEPMAP_EXPRESSION_PATH` to your own prepared **CSV**:

- First column: unique sample identifiers, read as the row index.
- Remaining columns: unique feature names and finite numeric expression values.
- The experiment expects normalized DepMap 26Q1 log(TPM+1) expression.
- At least 800 rows are required for the default design; original row and column
  order matters for the seeded subset and support selection.

The runner validates identifiers, finite values, dimensions, and an optional
`EXPECTED_DATA_SHA256`. It records matrix, row-identifier, and feature-name hashes.
It does not normalize raw counts, download data, use credentials, or access private
storage. A differently prepared CSV is not an exact reproduction; keep its source,
preparation steps, and hash. DepMap access and redistribution remain subject to
its terms.

## Small software checks

Experiment 03 supports `--smoke`; experiment 04 supports `--stage smoke`. These
reduce dimensions and fitting budgets and must not be labelled full results.
Experiment 05 needs the complete saved evaluation grid from experiment 04, not its
smoke grid.

## Names used in earlier records

Frozen historical records and earlier versions of this code use older names.
Use this map when comparing them with records written by this release.

| Earlier name | Current name |
|---|---|
| Folder `22-hard-rho05-density-four-methods` | `01-gaussian-support-recovery` |
| Folder `24-depmap-linear-supports` | `02-depmap-support-recovery` |
| Folder `27-matched-single-split` | `03-matched-aggregation` |
| Folder `28-pc-knockoff-vs-stk` | `04-pc-knockoff-comparison` |
| Folder `29-repeated-pc-knockoff` | `05-repeated-pc-knockoff` |
| Folder `19-global-null-calibration` | `06-gaussian-global-null` |
| Folder `23-tractability-scaling` | `07-computational-scaling` |
| Preset `derandomized` | `headline` |
| Preset `multisplit` | `secondary` |
| Preset `default` | `pooled` |
| `selection_mode` `derandomized` or `multisplit` | `multi_split` |
| Arm `derandomized` or `stk_derandomized` | `stk_headline` |
| Arm `multisplit` or `stk_multisplit` | `stk_secondary` |
| Arm `m1` | `stk_pooled` |
| Arm `stabl_knock` | `stabl` |
| Arm `lasso` | `lasso_cv` |
| Arm `vanilla_mx` | `direct_model_x` |
| `VANILLA_MX_MEMORY_GB` | `DIRECT_MODEL_X_MEMORY_GB` |
| `derand_results_` | `multi_split_results_` |
| 05 option `--exp28-out` | `--pc-comparison-out` |

The historical `derandomized` preset recorded `min_knockoff_candidates` as unset
(an implicit floor of 10); `headline` sets 10 explicitly. Selections are the same.
