# NeuroPace Hardening — Design

Date: 2026-09-07
Status: Approved

## Problem

NeuroPace was assembled from three hackathon directories. The merge fixed the
layout; this work fixes the contents. Three problems motivate it:

1. **The training loop never trained.** `EEGDataGenerator.__getitem__` in both
   `ml/dsc.py` and `ml/pp.py` returns `next(iter(self.dataset_tf))`, which
   rebuilds the iterator on every call and therefore yields batch 0 forever.
   The `idx` argument is ignored. Every reported accuracy came from a model that
   saw at most 32 windows, repeatedly.
2. **The repo is not legible.** Filenames (`dsc`, `sdg`, `ma`, `pp`, `smt`)
   encode nothing. Modules run training as an import side effect. There is a
   star import, a duplicated and diverging data-generator class, dead
   parameters, and no tests, lint, CI, or packaging.
3. **The committed weights are misleading.** `api/best_yet.h5` is the product of
   the broken loop. It is being deleted rather than shipped.

## Goal

A repository whose ML pipeline is correct, reproducible, and honestly evaluated,
supported by a clean inference service and a real (if small) client. Optimized
to be read by a technical reviewer.

Non-goal: training a model. No weights are produced by this work.

## Emphasis

Depth is concentrated in `neuropace/`. The API is clean but minimal. The app is
one well-built screen, not a product.

## Architecture

```
neuropace/            installable package
├── config.py         EEGConfig: labels, bands, electrodes       [no TF]
├── synthetic.py      signal generation                          [no TF]
├── dataset.py        Recording, Dataset, windowing, splits      [no TF]
├── data.py           tf.data pipeline: batching, augmentation   [TF]
├── models.py         build_model(cnn|lstm|hybrid)               [TF]
├── train.py          CLI trainer -> artifacts/                  [TF]
├── evaluate.py       metrics, confusion matrices                [TF]
└── baselines.py      XGBoost over band-power features           [no TF]

api/                  FastAPI service, lazy model load
app/                  Expo client, one screen
tests/                pytest suite
artifacts/            gitignored: models, plots, metrics
```

The TensorFlow-free boundary is load-bearing. Config, signal generation,
windowing and splitting are pure NumPy, so the bulk of the suite runs in about a
second and CI does not depend on a TensorFlow install to catch most regressions.

## Design decisions

### The data pipeline

`EEGDataGenerator` is deleted, not repaired. Both copies subclass
`keras.utils.Sequence`, ignore `idx`, and construct a `tf.data.Dataset`
internally — the class is a broken wrapper around the mechanism that already
does the work. It is replaced by:

```python
def make_dataset(recordings, config, *, augment=False, shuffle=False,
                 seed=None) -> tf.data.Dataset
```

Augmentation becomes `.map()` tensor ops — additive noise, amplitude scaling,
channel dropout, time shift — recovering the logic from the commented-out block
in `dsc.py`, which was the stronger implementation. Augmentation applies to the
training split only. Shuffling uses an explicit buffer and seed.

This removes in one change: the batch-0 bug, the duplicated class, and the dead
`augment` / `preprocess` parameters.

### Model artifacts

Models are build outputs, never repository contents. `artifacts/` is gitignored.
`train.py` writes `artifacts/model.keras` plus a run manifest. The API loads
lazily via an LRU-cached `get_model()`; absent a model file the service still
imports, starts, and passes its tests.

Contract:

- `GET /health` -> `200 {"status": "ok", "model_loaded": false, ...}`
- `POST /predict/` with no model -> `503`, detail naming the training command
- `POST /predict/` with wrong shape -> `400`, detail naming expected shape

### Reproducibility

A single `--seed` flag seeds Python, NumPy, and TensorFlow. The seed, the
resolved config, dataset size, and library versions are written to
`artifacts/run.json` alongside the model.

### Evaluation

`evaluate.py` loads a model and reports, per head: accuracy, macro-F1, a full
`classification_report`, and a confusion-matrix PNG, writing `metrics.json`.
Macro-F1 matters because the mood head has ten classes over a dataset whose
mood labels are drawn uniformly at random.

### Correctness fixes carried in the relevant commits

- Real leakage assertion: pairwise disjointness, checked on windows, not the
  current three-way intersection (`set(a) & set(b) & set(c)`), which is empty
  almost regardless of leakage.
- `build_model` accepts `learning_rate` and compiles once; callers stop
  recompiling immediately after construction.
- Per-head metrics passed as a dict, so results stop being read positionally.
- `copy.deepcopy` for the nested band-parameter dicts; the present `.copy()` is
  shallow and aliases the nested entries.
- `if __name__ == "__main__":` and argparse; importing a module no longer starts
  a training run.
- `from dsc import *` removed.
- Models saved in `.keras` format rather than legacy HDF5.
- API label lists imported from `neuropace.config` instead of duplicated in
  `api/main.py`, where they are already at risk of silent desync.

## Testing strategy

| Layer | Tests |
|---|---|
| `config` | label/band invariants, serialization round-trip |
| `synthetic` | shape, determinism under seed, band-power ordering reflects the requested condition |
| `dataset` | window count and stride, split ratios, pairwise disjointness |
| `data` | **batches differ across steps** (the regression test for the core bug), epoch covers every window, augmentation only on train |
| `models` | output shapes per head, input shape, single compile |
| `api` | health with and without a model, 503 path, 400 on bad shape |

The `data` test is the point of the exercise: it fails against the original
implementation and passes against the replacement.

## Delivery

Ten commits, each pushed separately:

1. Delete weights, gitignore `artifacts/`, API lazy-load
2. `ml/` -> `neuropace/` package, `pyproject.toml`
3. Fix the data pipeline, with tests
4. Trainer CLI, seeding, run manifest
5. `evaluate.py`, metrics and plots
6. XGBoost baselines sharing real feature extraction
7. API hardening, API tests
8. Expo screen
9. ruff, pytest config, GitHub Actions CI
10. README rewrite

## Honest-results constraint

No training happens in this work and the only weights are being deleted. The
README's results section therefore ships with the metrics table structure in
place and the figures unfilled, next to the exact command that reproduces them.
No invented numbers.
