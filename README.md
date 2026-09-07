# NeuroPace

Post-surgical patient monitoring from EEG. A synthetic signal generator, a
multi-head neural network that reads one five-second window of 19-channel EEG
and predicts **ailment**, **stress level**, and **mood state** at once, a
FastAPI service that serves it, and a mobile client that reads from it.

The EEG is simulated. There is no electrode hardware and no patient data
anywhere in this project — the generator produces signals whose ground truth is
known exactly, which is what makes the pipeline measurable.

```
neuropace.train  ──▶  artifacts/model.keras  ──▶  neuropace.api  ◀──  app/
```

## Layout

| Path | Responsibility | TensorFlow |
|---|---|---|
| `neuropace/config.py` | Label vocabulary, bands, electrodes, windowing geometry | no |
| `neuropace/synthetic.py` | Signal generation per condition, stress level, mood | no |
| `neuropace/dataset.py` | Recordings, windowing, stratified splits | no |
| `neuropace/features.py` | Band powers and spectral features | no |
| `neuropace/baselines.py` | XGBoost reference model | no |
| `neuropace/data.py` | `tf.data` input pipeline | yes |
| `neuropace/models.py` | CNN, LSTM, and hybrid architectures | yes |
| `neuropace/train.py` | Training CLI, seeding, run manifest | yes |
| `neuropace/evaluate.py` | Per-head metrics and confusion matrices | yes |
| `neuropace/api.py` | Inference service | lazily |
| `app/` | Expo client | — |
| `deploy/` | Dockerfile for the service | — |

The TensorFlow-free column is a deliberate boundary, not an accident. Everything
above the line is array work, so most of the suite runs without importing TF:
**13 tests in 0.75s**, against 6s for all 36, so a regression in the signal
generator surfaces without waiting on a TensorFlow import.

## The model

Three softmax heads on one shared trunk, because the labels are not independent
— stress and mood both modulate alpha and beta amplitude, so a shared
representation is the honest framing.

- **Input** `(1250, 19)` — 5s at 250Hz across the 10-20 electrode placement
- **`ailment`** (6) — normal, seizure, delayed_recovery, ischemia, hemorrhage, infection
- **`stress`** (5) — minimal, mild, moderate, high, severe
- **`mood`** (10) — Pain & Discomfort, Anxiety & Fear, … Focused Attention

## Quickstart

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,train,api]"
```

Train, evaluate, serve:

```sh
.venv/bin/python -m neuropace.train --epochs 50 --recordings-per-condition 50
.venv/bin/python -m neuropace.evaluate
.venv/bin/uvicorn neuropace.api:app --port 8080
```

Compare against the classical baseline:

```sh
.venv/bin/python -m neuropace.baselines --head ailment
```

Run the client:

```sh
cd app && npm install && npm start
```

Every run writes `artifacts/`: the model, a `run.json` manifest recording the
seed, resolved config, split sizes and library versions, the training history,
per-head `metrics.json`, and a confusion matrix per head.

## Results

**No trained weights ship with this repository, and this table is empty on
purpose.**

| Head | Classes | Chance | Accuracy | Macro-F1 |
|---|---|---|---|---|
| ailment | 6 | 0.167 | run `neuropace.evaluate` | run `neuropace.evaluate` |
| stress | 5 | 0.200 | run `neuropace.evaluate` | run `neuropace.evaluate` |
| mood | 10 | 0.100 | run `neuropace.evaluate` | run `neuropace.evaluate` |

The weights this project used to carry were produced by a training loop that
was silently broken (below), so any number quoted from them would have been
fiction. Rather than reprint them or invent replacements, the table names the
command that fills it. Macro-F1 sits beside accuracy because mood has ten
classes drawn uniformly at random, where accuracy alone flatters a model that
has learned nothing but the marginal distribution.

## Engineering notes

**The training loop never trained.** The data generator subclassed
`keras.utils.Sequence` and implemented:

```python
def __getitem__(self, idx):
    return next(iter(self.dataset_tf))
```

`iter()` builds a fresh iterator on every call, so this returns batch 0 forever
and ignores `idx` entirely. Measured on a 24-window fixture, the class reported
6 batches per epoch and yielded **1 distinct batch, covering 4 of 24 windows**;
at the shipped configuration that is roughly 0.5% of the training set, seen
repeatedly for every epoch of every run. Two copies of the class existed, in
separate modules, and they had diverged.

It was deleted rather than repaired: it subclassed `Sequence` but constructed a
`tf.data.Dataset` internally, so it was a broken wrapper around the mechanism
that already worked. `neuropace/data.py` returns that dataset directly.
`tests/test_data.py::test_successive_batches_differ` fails against the old
implementation and passes against the replacement.

Three more that mattered:

- **The leakage check checked nothing.** `set(train) & set(val) & set(test)` is
  a three-way intersection — empty almost regardless of whether any pair
  actually overlaps. Replaced with pairwise disjointness.
- **The API skewed against training.** Inference applied no normalization while
  the training pipeline standardized every window, so requests arrived on a
  scale the model had never seen. The service now standardizes identically.
- **Labels were duplicated** between the service and the config module, free to
  drift apart and return confidently mislabelled predictions. They are imported
  now, and `GET /labels` serves them to the client.

The service used to live in a loose `api/` directory outside the package, which
forced the tests to splice it onto `sys.path` before importing it. It is
`neuropace.api` now: one installable unit, no path manipulation, and the
Dockerfile installs the package rather than copying loose files.

## Tests

```sh
.venv/bin/python -m pytest -v          # 36 tests
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

The fast subset, which needs no TensorFlow:

```sh
.venv/bin/python -m pytest tests/test_config.py tests/test_dataset.py \
                          tests/test_features.py
```

The suite pins behaviour to properties rather than golden values: a 10Hz tone
must land in the alpha band and a 2Hz tone in delta; one epoch must cover every
window exactly once; the confusion-matrix diagonal must reproduce the reported
accuracy; the API must answer with and without a model on disk.
