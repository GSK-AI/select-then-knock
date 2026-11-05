# Dependencies and third-party scope

## Common local environment

`requirements.txt` contains a 54-distribution dependency closure for the retained
implementation, all comparators, plotting, and tests. Versions were read from
existing Python 3.12.11 environments. Applicable declared constraints were checked
without installing, downloading, or resolving packages from a registry.

The existing core/comparison and Stabl environments share NumPy 2.5.1,
SciPy 1.18.0, scikit-learn 1.6.1, pandas 3.0.4, and celer 0.7.4. These common
versions motivate a single default environment. A clean, unified installation and
platform-specific wheel/build availability have **not** been validated.

Some broad dependencies remain because external comparators declare them.
For example, knockpy requires cvxpy even though STK's own SDP option was removed.
No private sibling repository, cloud library, or external source tree is vendored.

## Historical environments are not identical

These examples come from the frozen run records, not the new requirements file:

| Experiment | Python | NumPy | SciPy | scikit-learn | celer |
|---|---|---|---|---|---|
| 06 Gaussian global null, recorded comparator extension | 3.12.4 | 2.5.2 | 1.18.0 | 1.6.1 | Not recorded in that entry |
| 07 Computational scaling, scaling freeze | 3.12.4 | 2.5.2 | 1.18.1 | 1.6.1 | 0.7.4 |
| 04 PC-Knockoff comparison, reference comparison record | 3.12.1 | 2.5.1 | 1.18.0 | 1.6.1 | 0.7.4 |

This is a documented sample of environment metadata, not an audit of every
historical run. The same release number never establishes source or environment
equality. Selection comparisons while preparing this release used the same local interpreter
for both sources; they do not certify historical cross-platform reproducibility.

## Third-party attribution and licenses

- **Stabl:** external dependency pinned to commit
  `1d07f85a13cfbecb4f08ce21075bf4fbb8e34678` of
  `https://github.com/gregbellan/stabl`. Installed metadata identifies version
  1.0.0 and the `stabl_lw` branch. Its installed license text is **Clear BSD**,
  copyright 2023 SurgeCare. No Stabl source is copied into this project.
  Historical version strings alone do not prove that every old run used this
  exact revision; that equality has not been established for every run.
- **knockpy:** external version 1.3.5; installed license text is **MIT**,
  copyright Asher Spector. The original SciPy compatibility shim is retained.
- **PC author source:** linked upstream revision
  `47c5f9d0ad981f4c8ebd5ddc8ef1f60c2ab132fe` has no license in the inspected
  fixture. The scholarly attribution and reimplementation note remain in
  `pc_knockoff.py`; the author source is not redistributed. Tests requiring it skip.

These dependency licenses do not grant a license for this project's own code.
This project's own code is licensed under Apache License 2.0 (see `LICENSE`).
