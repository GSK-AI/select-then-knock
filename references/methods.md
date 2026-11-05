# Method scope

This release preserves the reported regression selection paths while removing
work that does not contribute to candidate sets or final selected features.
It is a modified implementation, not a byte-identical historical source snapshot.

## What remains

- Raw-scale variance filtering and scale-only standardization of X and y.
- The original ElasticNetCV grids, inner folds, and solver settings.
- The original subsample seed sequence, successful-fit denominator, and threshold calibration.
- Candidate-floor expansion by stability score, without capping a larger natural set.
- Held-out Gaussian equicorrelated knockoff construction and lasso coefficient differences.
- The original minimum-threshold arithmetic, empty-selection policy, and numerical fallback.
- Cross-split frequencies and selection at `frequency >= eta`.
- Headline, Secondary, and Pooled STK (presets `headline`, `secondary`, `pooled`), and the
  LassoCV, Stabl, Direct Model-X, and PC-Knockoff comparator paths used by the experiments.

## Coordinate spaces

**Original:** columns exactly as supplied to `fit`.

**Preprocessed:** columns remaining after raw-scale variance filtering, followed
by division by feature standard deviation. The target is divided by its own
standard deviation; a constant target is left unchanged.

**Stable:** the candidate subset sent to knockoff filtering. Indices must map
Stable → Preprocessed → Original. Each split's screening mask and scaler are
applied to its held-out rows. Frequency tallies remain in Original space.

## Statistical interpretation

FDP is the realized false-selection proportion in one dataset. FDR is its
expectation across datasets and algorithmic randomness. A single run's FDP is
not FDR, and its false-selection count is not PFER.

The screening calibration parameter `target_pfer` is historical terminology.
Clipping its threshold and augmenting a candidate set do not establish PFER
control for that candidate set. The repeated STK aggregate has no proved FDR
or PFER guarantee; `per_run_fdr=0.5` is not a target aggregate FDR.

Gaussian covariance is estimated, not known. The retained permutation fallback
does not generally provide Model-X exchangeability. A selected feature is a
nomination for independent validation, not evidence of causality.

## Removed computation

- The multi-split parent's full-data tuning and final prediction-model fit.
- The screening child's discarded pooled knockoff result and prediction-model fit.
- Redundant tuned-model refits used only for baseline/prediction outputs.
- Coefficient interpretation, prediction APIs, classification, outlier diagnostics,
  pyramid fitting, optional outer CV, alternate STK knockoff/statistic methods,
  and unrelated data generators or cloud ingestion.

Candidate-floor expansion still runs even though the child's pooled filter is
skipped. Numerical failure paths for the retained computation remain. Runtime,
memory use, and discarded diagnostic outputs are intentionally allowed to change:
the paper's runtime and memory numbers were measured with the historical
implementation, which ran the discarded pooled fit, so fresh measurements are
lower and do not reproduce them.
Selection equivalence with the historical implementation was checked on small
inputs only, by `tests/test_reference.py`, which needs the unreleased historical source.

## Sources and attribution

- Meinshausen and Bühlmann (2010), *Stability selection*, JRSS B 72(4), 417–473.
- Barber and Candès (2015), *Controlling the false discovery rate via knockoffs*,
  Annals of Statistics 43(5), 2055–2085.
- Candès, Fan, Janson, and Lv (2018), *Panning for gold*, JRSS B 80(3), 551–577.
- Liu, Ke, Liu, and Li (2022), JASA 117, 428–443,
  DOI: 10.1080/01621459.2020.1783274. The local PC comparator is a reimplementation;
  no unlicensed author code is bundled.
- Ren, Wei, and Candès (2023), *Derandomizing knockoffs*. Repeated STK changes its
  screened family across partitions and is not the same fixed-family procedure.
