# Select-then-Knock

Research code for multi-split feature selection: the selection method, experiment
runners, and tests. It runs from source and is not an installable library.
No datasets or saved results are included.

**Caveat:** the aggregate has no proved FDR or PFER guarantee. Multi-split STK is
not the published fixed-family derandomized-knockoff procedure. See
`references/methods.md`.

## Layout

```text
select_then_knock/   Selection code
experiments/         The paper's seven experiments, 01-07 (see experiments/README.md)
tests/               Tests
references/          Method scope and dependencies
```

## Setup

Python 3.12 on macOS or Linux. Run everything from the project root.

```sh
uv venv --python 3.12
uv pip sync --python .venv/bin/python requirements.txt
source .venv/bin/activate
```

## Usage

```python
from select_then_knock import SelectThenKnock

model = SelectThenKnock.from_preset("headline", random_state=42, solver_backend="celer")
model.fit(X, y)
selected = model.get_support(indices=True, space="original")
```

| Preset | Paper name | Screening fraction | Candidate floor |
|---|---|---:|---:|
| `headline` | Headline STK | 0.50 | 10 |
| `secondary` | Secondary STK | 0.70 | 20 |
| `pooled` | Pooled STK (same-sample reference) | none | 10 |

`headline` and `secondary` use 25 splits, a per-split knockoff level of 0.5, and a
frequency threshold of 0.5. The candidate floor is a minimum, not a cap. `pooled`
selects and tests on the same rows without splitting.

## Experiments

Each experiment has a `run.py` under `experiments/NN-*/`, numbered in the order of
the paper's benchmark table. Full grids can be expensive. See `experiments/README.md`
for designs, commands, data requirements, and the names used in earlier records.

Runtime and memory figures from fresh runs are not a reproduction of the reported
resource numbers; see "Runtime and memory" in `experiments/README.md`.

## Tests

```sh
python -m pytest -q
ruff check .
```

## License

Apache License 2.0. See `LICENSE`.
