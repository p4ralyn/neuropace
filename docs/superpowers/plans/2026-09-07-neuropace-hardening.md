# NeuroPace Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn NeuroPace from hackathon code with a silently broken training loop into a correct, reproducible, tested project that reads well to a technical reviewer.

**Architecture:** The `ml/` scripts become an installable `neuropace/` package split along a TensorFlow-free boundary — config, synthetic signal generation, and windowing/splitting are pure NumPy; only the tf.data pipeline, models, trainer, and evaluator import TensorFlow. The broken `EEGDataGenerator` class is deleted and replaced by a `make_dataset()` function returning a real `tf.data.Dataset`. Model weights become gitignored build outputs, and the API loads them lazily so it runs and tests without them.

**Tech Stack:** Python 3.11, TensorFlow 2.21, NumPy 2.x, scikit-learn, XGBoost, FastAPI, pytest, ruff, uv, Expo/React Native.

**Spec:** `docs/superpowers/specs/2026-09-07-neuropace-hardening-design.md`

## Global Constraints

- Python 3.11 exactly. TensorFlow does not support the system default 3.14. Create the environment with `uv venv --python 3.11 .venv`.
- Install with `uv pip install --python .venv/bin/python`. Plain `pip` is not on PATH in this environment.
- Every commit is pushed to `origin main` immediately after it is made.
- **No Claude co-authorship.** Commit messages must not contain `Co-Authored-By: Claude` or `Claude-Session:` trailers. Author and committer are `Pragun Kathuria <pragunkathuria30@gmail.com>`.
- No model training is performed by this work, and no weights are committed. `artifacts/` is gitignored.
- No invented metric values anywhere in docs. Where a number would go, put the command that produces it.
- Models are saved in `.keras` format, never legacy `.h5`.
- Run tests with `.venv/bin/python -m pytest`.

---

### Task 1: Delete the weights and make the API survive without them

**Files:**
- Delete: `api/best_yet.h5`
- Modify: `api/main.py` (full rewrite of model loading)
- Modify: `.gitignore`
- Test: none yet (API tests land in Task 7, once the package exists to import labels from)

**Interfaces:**
- Consumes: nothing
- Produces: `api/main.py` exposing `get_model()` (LRU-cached, returns `keras.Model | None`), `MODEL_PATH` (a `pathlib.Path` from the `NEUROPACE_MODEL` env var, default `artifacts/model.keras`)

- [ ] **Step 1: Append artifact ignores to `.gitignore`**

```
# Build outputs -- models, metrics, plots. Never commit these.
artifacts/
*.keras
*.h5
```

- [ ] **Step 2: Delete the weights**

```bash
git rm --cached api/best_yet.h5
rm -f api/best_yet.h5
```

- [ ] **Step 3: Rewrite `api/main.py` model loading**

```python
"""NeuroPace inference service."""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.environ.get("NEUROPACE_MODEL", "artifacts/model.keras"))
TRAIN_HINT = "python -m neuropace.train --help"

app = FastAPI(title="NeuroPace EEG API", version="0.1.0")


@lru_cache(maxsize=1)
def get_model():
    """Load the trained model, or return None if no artifact exists.

    Lazy so the service imports, starts, and answers /health with no model
    present -- which is what lets it be tested in CI without a training run.
    """
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf  # imported here to keep startup cheap

    return tf.keras.models.load_model(MODEL_PATH)


def require_model():
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"No model at {MODEL_PATH}. Train one: {TRAIN_HINT}",
        )
    return model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": get_model() is not None,
        "model_path": str(MODEL_PATH),
    }
```

- [ ] **Step 4: Verify the service imports with no model present**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0,'api'); import main; print(main.get_model())"`
Expected: prints `None`, no exception.

- [ ] **Step 5: Commit and push**

```bash
git add -A
git commit -m "Remove committed weights; load the model lazily

best_yet.h5 was produced by a training loop that replayed a single batch
(see the data generator's __getitem__), so its weights carry no signal.
Models become gitignored build outputs under artifacts/.

The API now loads lazily: it imports, starts, and serves /health with no
model present, which is what makes it testable without a training run."
git push origin main
```

---

### Task 2: Restructure `ml/` into an installable `neuropace/` package

This task is a move plus mechanical import rewrites. Behaviour changes land in Task 3. Splitting them keeps the diff reviewable: a reviewer can confirm Task 2 changed no logic.

**Files:**
- Create: `pyproject.toml`
- Create: `neuropace/__init__.py`
- Create: `neuropace/config.py` (the `EEGDataConfig` half of `ml/dsc.py`)
- Create: `neuropace/dataset.py` (the `EEGRecording` / `EEGDataset` half of `ml/dsc.py`)
- Create: `neuropace/synthetic.py` (from `ml/sdg.py`)
- Create: `neuropace/models.py` (from `ml/ma.py`)
- Create: `neuropace/baselines.py` (from `ml/experiments/xgb2.py`)
- Delete: `ml/` entirely, including `ml/pipeline.py`, `ml/pp.py`, `ml/experiments/xgb.py`, `ml/experiments/smt.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `neuropace.config.EEGConfig` — frozen dataclass; fields `sampling_rate: int = 250`, `channels: int = 19`, `recording_duration: float = 60.0`, `window_size: float = 5.0`, `window_overlap: float = 0.5`, `batch_size: int = 32`; properties `samples_per_recording: int`, `samples_per_window: int`, `window_step: int`, `num_ailments: int`, `num_stress_levels: int`, `num_moods: int`; methods `to_dict() -> dict`, `from_dict(d) -> EEGConfig`
  - `neuropace.config.AILMENT_CLASSES`, `STRESS_LEVELS`, `MOOD_STATES`, `FREQUENCY_BANDS`, `ELECTRODES_10_20` — module-level tuples/dicts
  - `neuropace.dataset.Recording` — dataclass with `data: np.ndarray`, `condition: str`, `condition_id: int`, `stress_level: str`, `stress_level_id: int`, `mood_state: str`, `mood_state_id: int`, `patient_id: str | None`, `metadata: dict`
  - `neuropace.synthetic.generate_recording(config, condition, *, stress_level=None, mood_state=None, patient_id=None, rng=None) -> Recording`
  - `neuropace.synthetic.generate_dataset(config, n_per_condition=20, *, rng=None) -> list[Recording]`
  - `neuropace.models.build_model(config, *, model_type="hybrid", output_type="all", learning_rate=1e-3) -> keras.Model`

- [ ] **Step 1: Write the failing config test**

```python
# tests/test_config.py
from neuropace.config import (
    AILMENT_CLASSES, MOOD_STATES, STRESS_LEVELS, EEGConfig,
)


def test_window_geometry():
    config = EEGConfig(sampling_rate=250, window_size=5.0, window_overlap=0.5)
    assert config.samples_per_window == 1250
    assert config.window_step == 625


def test_class_counts_match_label_lists():
    config = EEGConfig()
    assert config.num_ailments == len(AILMENT_CLASSES) == 6
    assert config.num_stress_levels == len(STRESS_LEVELS) == 5
    assert config.num_moods == len(MOOD_STATES) == 10


def test_roundtrips_through_dict():
    config = EEGConfig(channels=8, batch_size=4)
    assert EEGConfig.from_dict(config.to_dict()) == config
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuropace'`

- [ ] **Step 3: Write `neuropace/config.py`**

```python
"""Configuration and label vocabulary for EEG data.

Deliberately free of TensorFlow: this module and its label lists are imported
by the API, the tests, and the NumPy-only parts of the pipeline, none of which
should pay for a TF import.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

AILMENT_CLASSES: tuple[str, ...] = (
    "normal",
    "seizure",
    "delayed_recovery",
    "ischemia",
    "hemorrhage",
    "infection",
)

STRESS_LEVELS: tuple[str, ...] = (
    "minimal",
    "mild",
    "moderate",
    "high",
    "severe",
)

MOOD_STATES: tuple[str, ...] = (
    "Pain & Discomfort",
    "Anxiety & Fear",
    "Depression & Mental Fatigue",
    "Insomnia & Sleep Disturbances",
    "Cognitive Dysfunction",
    "Emotional Exhaustion",
    "Post-Surgery PTSD",
    "Relaxation & Recovery",
    "Deep Sleep",
    "Focused Attention",
)

# Canonical EEG frequency bands, in Hz.
FREQUENCY_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}

# International 10-20 electrode placement, in channel order.
ELECTRODES_10_20: tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
    "Pz", "P4", "T6", "O1", "O2",
)


@dataclass(frozen=True)
class EEGConfig:
    """Acquisition and windowing parameters for a run."""

    sampling_rate: int = 250
    channels: int = 19
    recording_duration: float = 60.0
    window_size: float = 5.0
    window_overlap: float = 0.5
    batch_size: int = 32

    def __post_init__(self) -> None:
        if not 0.0 <= self.window_overlap < 1.0:
            raise ValueError(
                f"window_overlap must be in [0, 1), got {self.window_overlap}"
            )
        if self.window_size > self.recording_duration:
            raise ValueError(
                f"window_size {self.window_size}s exceeds recording_duration "
                f"{self.recording_duration}s"
            )

    @property
    def samples_per_recording(self) -> int:
        return int(self.recording_duration * self.sampling_rate)

    @property
    def samples_per_window(self) -> int:
        return int(self.window_size * self.sampling_rate)

    @property
    def window_step(self) -> int:
        return int(self.samples_per_window * (1.0 - self.window_overlap))

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.samples_per_window, self.channels)

    @property
    def num_ailments(self) -> int:
        return len(AILMENT_CLASSES)

    @property
    def num_stress_levels(self) -> int:
        return len(STRESS_LEVELS)

    @property
    def num_moods(self) -> int:
        return len(MOOD_STATES)

    @property
    def electrode_names(self) -> tuple[str, ...]:
        if self.channels == len(ELECTRODES_10_20):
            return ELECTRODES_10_20
        return tuple(f"CH{i}" for i in range(self.channels))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EEGConfig":
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EEGConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
```

- [ ] **Step 4: Write `pyproject.toml` and `neuropace/__init__.py`**

```toml
[project]
name = "neuropace"
version = "0.1.0"
description = "EEG-based post-surgical patient monitoring"
requires-python = ">=3.11,<3.13"
dependencies = [
    "numpy>=2.0",
    "scipy>=1.11",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
train = ["tensorflow>=2.19", "matplotlib>=3.8", "xgboost>=2.0"]
api = ["fastapi>=0.115", "uvicorn[standard]>=0.30"]
dev = ["pytest>=8.0", "ruff>=0.6", "httpx>=0.27"]

[project.scripts]
neuropace-train = "neuropace.train:main"
neuropace-eval = "neuropace.evaluate:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["neuropace*"]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = ["ignore::DeprecationWarning"]
```

```python
# neuropace/__init__.py
"""EEG-based post-surgical patient monitoring."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Install the package and run the test**

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest tests/test_config.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Port `dataset.py`, `synthetic.py`, `models.py`, `baselines.py`**

Port rules, applied while moving:
- `ml/dsc.py` splits at the class boundary: `EEGDataConfig` became `config.py` in Step 3; `EEGRecording` and `EEGDataset` become `dataset.py`. The `EEGDataGenerator` class at the bottom of `ml/dsc.py` is **not** ported — it is the broken one, and Task 3 replaces it.
- `ml/dsc.py`'s `convert_to_tabular` is dropped: it writes one CSV row per sample per channel (285,000 rows per recording), nothing calls it, and Task 5's evaluation covers the inspection need it was reaching for.
- `ml/sdg.py` becomes `synthetic.py`. Replace `from dsc import *` with explicit imports. Replace `base_params.copy()` with `copy.deepcopy(base_params)` — the shallow copy aliases the nested per-band dicts. Thread an explicit `rng: np.random.Generator` through instead of calling module-level `np.random.*`, so runs are reproducible.
- `ml/ma.py` becomes `models.py`, with `build_model(config, *, model_type, output_type, learning_rate)` compiling exactly once and taking per-head metrics as a dict.
- `ml/experiments/xgb2.py` becomes `baselines.py`; `xgb.py` and `smt.py` are deleted. `smt.py` trains on `np.random.rand` placeholder data and asserts nothing; `xgb.py` is a strictly weaker version of `xgb2.py` (mean/std features only, no spectral features). Task 6 rewrites what remains.
- Every module gets an `if __name__ == "__main__":` guard or no top-level side effects at all. `ml/pipeline.py` and `ml/pp.py` are deleted; Task 4 rewrites them as one `train.py`.

- [ ] **Step 7: Confirm no logic drifted in the port**

Run: `.venv/bin/python -c "from neuropace.synthetic import generate_recording; from neuropace.config import EEGConfig; import numpy as np; r = generate_recording(EEGConfig(channels=4, recording_duration=2.0), 'seizure', rng=np.random.default_rng(0)); print(r.data.shape, r.condition, r.condition_id)"`
Expected: `(500, 4) seizure 1`

- [ ] **Step 8: Commit and push**

```bash
git add -A
git commit -m "Restructure ml/ scripts into an installable neuropace package

dsc/sdg/ma/pp/smt named nothing a reader could follow. The package splits
on a TensorFlow boundary: config, dataset and synthetic are pure NumPy so
the API and most tests never import TF.

Dropped: pipeline.py and pp.py (top-level scripts that ran a 75-epoch
training run on import; replaced by a CLI trainer next), xgb.py and smt.py
(weaker duplicates of xgb2.py; smt.py trained on np.random.rand), and
convert_to_tabular (285k CSV rows per recording, called by nothing).

Fixed in passing: from dsc import *, and a shallow .copy() of the nested
band-parameter dicts that aliased them."
git push origin main
```

---

### Task 3: Replace the broken data generator

This is the core of the work. The regression test must fail against the old implementation and pass against the new one.

**Files:**
- Create: `neuropace/data.py`
- Modify: `neuropace/dataset.py` (add `make_windows`, `split_indices`)
- Test: `tests/test_dataset.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `neuropace.config.EEGConfig`, `neuropace.dataset.Recording`
- Produces:
  - `neuropace.dataset.make_windows(recording, config) -> np.ndarray` of shape `(n_windows, samples_per_window, channels)`
  - `neuropace.dataset.split_indices(recordings, *, train=0.7, val=0.15, test=0.15, seed=42) -> tuple[list[int], list[int], list[int]]`, stratified on `condition_id`
  - `neuropace.data.windows_and_labels(recordings, config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` returning `(X, y_ailment, y_stress, y_mood)` with integer labels
  - `neuropace.data.make_dataset(recordings, config, *, augment=False, shuffle=False, seed=None) -> tf.data.Dataset` yielding `(features, {"ailment": ..., "stress": ..., "mood": ...})` one-hot

- [ ] **Step 1: Write the failing regression test**

```python
# tests/test_data.py
import numpy as np
import pytest

from neuropace.config import EEGConfig
from neuropace.synthetic import generate_dataset

tf = pytest.importorskip("tensorflow")

from neuropace.data import make_dataset, windows_and_labels  # noqa: E402


@pytest.fixture(scope="module")
def small():
    # Tiny on every axis: 2 windows per recording, 4 channels, 12 recordings.
    config = EEGConfig(
        sampling_rate=50,
        channels=4,
        recording_duration=3.0,
        window_size=2.0,
        window_overlap=0.5,
        batch_size=4,
    )
    recordings = generate_dataset(
        config, n_per_condition=2, rng=np.random.default_rng(0)
    )
    return config, recordings


def test_successive_batches_differ(small):
    """The bug this project was built to fix.

    The old EEGDataGenerator.__getitem__ returned next(iter(self.dataset_tf)),
    rebuilding the iterator every call, so every training step saw batch 0.
    """
    config, recordings = small
    dataset = make_dataset(recordings, config)
    batches = [x.numpy() for x, _ in dataset.take(3)]

    assert len(batches) == 3
    assert not np.array_equal(batches[0], batches[1]), "batch 1 repeats batch 0"
    assert not np.array_equal(batches[1], batches[2]), "batch 2 repeats batch 1"


def test_one_epoch_covers_every_window(small):
    config, recordings = small
    features, *_ = windows_and_labels(recordings, config)
    seen = sum(int(x.shape[0]) for x, _ in make_dataset(recordings, config))
    assert seen == features.shape[0]


def test_shapes_and_heads(small):
    config, recordings = small
    dataset = make_dataset(recordings, config)
    features, labels = next(iter(dataset))

    assert features.shape[1:] == tuple(config.input_shape)
    assert set(labels) == {"ailment", "stress", "mood"}
    assert labels["ailment"].shape[1] == config.num_ailments
    assert labels["stress"].shape[1] == config.num_stress_levels
    assert labels["mood"].shape[1] == config.num_moods


def test_augmentation_changes_only_when_requested(small):
    config, recordings = small
    plain = next(iter(make_dataset(recordings, config)))[0].numpy()
    again = next(iter(make_dataset(recordings, config)))[0].numpy()
    augmented = next(
        iter(make_dataset(recordings, config, augment=True, seed=0))
    )[0].numpy()

    np.testing.assert_array_equal(plain, again)
    assert not np.array_equal(plain, augmented)


def test_shuffle_is_seed_reproducible(small):
    config, recordings = small
    first = next(iter(make_dataset(recordings, config, shuffle=True, seed=7)))[0]
    second = next(iter(make_dataset(recordings, config, shuffle=True, seed=7)))[0]
    np.testing.assert_array_equal(first.numpy(), second.numpy())
```

```python
# tests/test_dataset.py
import numpy as np

from neuropace.config import EEGConfig
from neuropace.dataset import make_windows, split_indices
from neuropace.synthetic import generate_dataset


def test_window_count_and_stride():
    config = EEGConfig(
        sampling_rate=10, channels=2, recording_duration=5.0,
        window_size=2.0, window_overlap=0.5,
    )
    recording = generate_dataset(
        config, n_per_condition=1, rng=np.random.default_rng(0)
    )[0]
    windows = make_windows(recording, config)

    # 50 samples, 20-sample windows, stride 10 -> starts at 0,10,20,30 = 4
    assert windows.shape == (4, 20, 2)
    np.testing.assert_array_equal(windows[1], recording.data[10:30])


def test_splits_are_pairwise_disjoint_and_cover_everything():
    config = EEGConfig(channels=2, recording_duration=2.0, window_size=1.0)
    recordings = generate_dataset(
        config, n_per_condition=5, rng=np.random.default_rng(0)
    )
    train, val, test = split_indices(recordings, seed=42)

    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert sorted(train + val + test) == list(range(len(recordings)))


def test_split_is_stratified_by_condition():
    config = EEGConfig(channels=2, recording_duration=2.0, window_size=1.0)
    recordings = generate_dataset(
        config, n_per_condition=10, rng=np.random.default_rng(0)
    )
    train, _, _ = split_indices(recordings, seed=42)
    conditions = {recordings[i].condition_id for i in train}
    assert len(conditions) == 6
```

- [ ] **Step 2: Run and watch them fail**

Run: `.venv/bin/python -m pytest tests/test_data.py tests/test_dataset.py -v`
Expected: FAIL — `ImportError: cannot import name 'make_dataset'` / `'make_windows'`

- [ ] **Step 3: Add windowing and splitting to `neuropace/dataset.py`**

```python
def make_windows(recording: Recording, config: EEGConfig) -> np.ndarray:
    """Slice a recording into overlapping windows.

    Returns shape (n_windows, samples_per_window, channels). Any trailing
    samples that cannot fill a whole window are dropped.
    """
    width = config.samples_per_window
    step = config.window_step
    n_samples = recording.data.shape[0]

    if n_samples < width:
        return np.empty((0, width, recording.data.shape[1]), dtype=recording.data.dtype)

    starts = range(0, n_samples - width + 1, step)
    return np.stack([recording.data[s : s + width] for s in starts])


def split_indices(
    recordings: Sequence[Recording],
    *,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split recordings into train/val/test, stratified on condition.

    Splits by *recording*, never by window: windows from one recording share
    a signal and would leak across the boundary if split individually.
    """
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {train + val + test}")

    indices = list(range(len(recordings)))
    conditions = [r.condition_id for r in recordings]

    train_val, test_idx = train_test_split(
        indices, test_size=test, stratify=conditions, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_val,
        test_size=val / (train + val),
        stratify=[conditions[i] for i in train_val],
        random_state=seed,
    )
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)
```

- [ ] **Step 4: Write `neuropace/data.py`**

```python
"""tf.data input pipeline.

Replaces the EEGDataGenerator class that previously lived in dsc.py and pp.py.
That class subclassed keras.utils.Sequence but implemented

    def __getitem__(self, idx):
        return next(iter(self.dataset_tf))

which rebuilds the iterator on every call and so returns batch 0 forever,
ignoring idx. Training therefore saw a single batch, repeatedly, for every
epoch of every run. The class also duplicated -- and diverged -- across two
modules, and its augment/preprocess flags were dead.

It is replaced by a function, because it was only ever a broken wrapper
around a tf.data.Dataset that it built internally anyway.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tensorflow as tf

from .config import EEGConfig
from .dataset import Recording, make_windows

HEADS = ("ailment", "stress", "mood")


def windows_and_labels(
    recordings: Sequence[Recording], config: EEGConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten recordings into windows with their integer labels."""
    features, ailment, stress, mood = [], [], [], []

    for recording in recordings:
        windows = make_windows(recording, config)
        if windows.shape[0] == 0:
            continue
        features.append(windows)
        ailment.append(np.full(windows.shape[0], recording.condition_id))
        stress.append(np.full(windows.shape[0], recording.stress_level_id))
        mood.append(np.full(windows.shape[0], recording.mood_state_id))

    if not features:
        raise ValueError("no recording was long enough to yield a window")

    return (
        np.concatenate(features).astype(np.float32),
        np.concatenate(ailment),
        np.concatenate(stress),
        np.concatenate(mood),
    )


def _standardize(features: tf.Tensor) -> tf.Tensor:
    """Zero-mean, unit-variance per channel, per window."""
    mean = tf.reduce_mean(features, axis=0, keepdims=True)
    std = tf.math.reduce_std(features, axis=0, keepdims=True)
    return (features - mean) / (std + 1e-6)


def _augment(features: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """Noise, amplitude scaling, and channel dropout.

    Recovered from the commented-out block in dsc.py, which was the stronger
    implementation of the two that shipped. Applied to the training split only.
    """
    noise_seed, scale_seed, drop_seed = tf.unstack(
        tf.random.experimental.stateless_split(seed, num=3)
    )

    noise = tf.random.stateless_normal(tf.shape(features), noise_seed, stddev=0.05)
    features = features + noise

    scale = tf.random.stateless_uniform([], scale_seed, minval=0.9, maxval=1.1)
    features = features * scale

    keep = tf.cast(
        tf.random.stateless_uniform([tf.shape(features)[-1]], drop_seed) > 0.1,
        features.dtype,
    )
    return features * keep


def make_dataset(
    recordings: Sequence[Recording],
    config: EEGConfig,
    *,
    augment: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Build a batched tf.data pipeline over the given recordings.

    Yields (features, {"ailment": ..., "stress": ..., "mood": ...}) with
    one-hot labels, batched at config.batch_size.
    """
    features, ailment, stress, mood = windows_and_labels(recordings, config)

    labels = {
        "ailment": tf.keras.utils.to_categorical(ailment, config.num_ailments),
        "stress": tf.keras.utils.to_categorical(stress, config.num_stress_levels),
        "mood": tf.keras.utils.to_categorical(mood, config.num_moods),
    }

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=features.shape[0],
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda x, y: (_standardize(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment:
        rng = tf.random.Generator.from_seed(seed if seed is not None else 0)
        dataset = dataset.map(
            lambda x, y: (_augment(x, rng.make_seeds(1)[:, 0]), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_data.py tests/test_dataset.py -v`
Expected: all pass, `test_successive_batches_differ` included.

- [ ] **Step 6: Commit and push**

```bash
git add -A
git commit -m "Replace the data generator that trained on a single batch

EEGDataGenerator.__getitem__ was

    return next(iter(self.dataset_tf))

which rebuilds the iterator on every call, so every step of every epoch got
batch 0 and the idx argument was ignored. Both copies of the class -- one in
dsc.py, one diverged in pp.py -- had it, so every training run this repo has
ever done saw 32 windows total.

Deleted rather than repaired: it subclassed keras.utils.Sequence but built a
tf.data.Dataset internally, so it was a broken wrapper around the thing that
already worked. make_dataset() returns that dataset directly.

Augmentation is now real tf.map ops on the training split only, recovered
from the commented-out block in dsc.py. test_successive_batches_differ fails
against the old implementation and passes against this one.

Also replaces the leakage assertion, which was set(a) & set(b) & set(c) -- a
three-way intersection, empty almost regardless of leakage -- with pairwise
disjointness."
git push origin main
```

---

### Task 4: Trainer CLI with seeding and a run manifest

**Files:**
- Create: `neuropace/train.py`
- Create: `neuropace/seeding.py`
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `make_dataset`, `build_model`, `split_indices`, `generate_dataset`
- Produces:
  - `neuropace.seeding.seed_everything(seed: int) -> None`
  - `neuropace.train.train(config, *, n_per_condition, epochs, model_type, learning_rate, seed, out_dir) -> dict` — returns the manifest, writes `model.keras`, `run.json`, `history.json` into `out_dir`
  - `neuropace.train.main(argv=None) -> int` — argparse entry point

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_train_smoke.py
import json

import pytest

from neuropace.config import EEGConfig

pytest.importorskip("tensorflow")

from neuropace.train import train  # noqa: E402


def test_one_epoch_writes_model_and_manifest(tmp_path):
    """A real end-to-end run, sized to finish in seconds."""
    config = EEGConfig(
        sampling_rate=25, channels=2, recording_duration=2.0,
        window_size=1.0, window_overlap=0.5, batch_size=4,
    )
    manifest = train(
        config,
        n_per_condition=2,
        epochs=1,
        model_type="cnn",
        seed=0,
        out_dir=tmp_path,
    )

    assert (tmp_path / "model.keras").exists()
    assert (tmp_path / "run.json").exists()
    assert manifest["seed"] == 0
    assert manifest["n_windows"]["train"] > 0
    assert json.loads((tmp_path / "run.json").read_text())["seed"] == 0
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_train_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuropace.train'`

- [ ] **Step 3: Write `neuropace/seeding.py`**

```python
"""One place to make a run reproducible."""

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow, and pin TF to deterministic ops."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
```

- [ ] **Step 4: Write `neuropace/train.py`**

Requirements for the implementation:
- `train()` seeds first, generates recordings with `np.random.default_rng(seed)`, splits with `split_indices(..., seed=seed)`, builds train/val/test datasets with `augment=True` on train only and `shuffle=True` on train only.
- Callbacks: `EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)`, `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)`, `ModelCheckpoint(out_dir/"model.keras", monitor="val_loss", save_best_only=True)`.
- Writes `run.json` containing: `seed`, `config` (via `to_dict()`), `model_type`, `epochs`, `learning_rate`, `n_recordings`, `n_windows` per split, `python`, `tensorflow`, `numpy` versions, and a UTC `finished_at`.
- Writes `history.json` with `history.history` so Task 5 can plot curves without retraining.
- `main()` argparse flags, with these exact names: `--seed` (int, default 42), `--epochs` (int, default 50), `--recordings-per-condition` (int, default 50), `--model-type` (choice of `cnn`/`lstm`/`hybrid`, default `hybrid`), `--learning-rate` (float, default 1e-3), `--out` (Path, default `artifacts`), `--channels`, `--duration`, `--window`.
- Guarded by `if __name__ == "__main__": raise SystemExit(main())`. Nothing runs on import — the modules this replaces started a 75-epoch run when imported.

- [ ] **Step 5: Run the smoke test**

Run: `.venv/bin/python -m pytest tests/test_train_smoke.py -v`
Expected: PASS in under a minute.

- [ ] **Step 6: Verify the CLI help works and nothing trains on import**

Run: `.venv/bin/python -c "import neuropace.train; print('imported, nothing trained')" && .venv/bin/python -m neuropace.train --help`
Expected: the message, then the argparse help.

- [ ] **Step 7: Commit and push**

```bash
git add -A
git commit -m "Add a seeded trainer CLI with a run manifest

pipeline.py and pp.py were top-level scripts: importing either started a
75-epoch run. They also disagreed with each other on learning rate, epochs
and augmentation, and both recompiled the model that build_eeg_model had
already compiled.

One train.py with argparse, a --seed that reaches Python, NumPy and TF, and
a run.json recording seed, config, split sizes and library versions next to
the weights, so a result can be traced back to what produced it."
git push origin main
```

---

### Task 5: Honest evaluation

**Files:**
- Create: `neuropace/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `make_dataset`, `windows_and_labels`, a `model.keras` written by Task 4
- Produces: `neuropace.evaluate.evaluate(model, recordings, config) -> dict` mapping each head to `{"accuracy": float, "macro_f1": float, "report": dict, "confusion_matrix": list[list[int]]}`; `neuropace.evaluate.main(argv=None) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import numpy as np
import pytest

from neuropace.config import EEGConfig
from neuropace.synthetic import generate_dataset

pytest.importorskip("tensorflow")

from neuropace.evaluate import evaluate  # noqa: E402
from neuropace.models import build_model  # noqa: E402


def test_reports_every_head_with_sane_bounds():
    config = EEGConfig(
        sampling_rate=25, channels=2, recording_duration=2.0,
        window_size=1.0, batch_size=4,
    )
    recordings = generate_dataset(
        config, n_per_condition=2, rng=np.random.default_rng(0)
    )
    model = build_model(config, model_type="cnn")

    results = evaluate(model, recordings, config)

    assert set(results) == {"ailment", "stress", "mood"}
    for head, scores in results.items():
        assert 0.0 <= scores["accuracy"] <= 1.0, head
        assert 0.0 <= scores["macro_f1"] <= 1.0, head

    # Confusion matrix is square in the number of classes and totals the windows.
    matrix = np.array(results["ailment"]["confusion_matrix"])
    assert matrix.shape == (config.num_ailments, config.num_ailments)
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuropace.evaluate'`

- [ ] **Step 3: Write `neuropace/evaluate.py`**

Requirements:
- `evaluate()` runs `model.predict` over `make_dataset(..., augment=False, shuffle=False)`, takes `argmax` per head, and computes `accuracy_score`, `f1_score(average="macro")`, `classification_report(output_dict=True, zero_division=0)`, and `confusion_matrix` against the labels from `windows_and_labels`.
- Macro-F1 is reported alongside accuracy because the mood head has ten classes drawn uniformly at random; accuracy alone would flatter a model that has learned only the marginal.
- `save_confusion_matrices(results, config, out_dir)` writes one PNG per head via matplotlib, with class names on both axes.
- `main()` takes `--model` (default `artifacts/model.keras`), `--out` (default `artifacts`), `--seed`, `--recordings-per-condition`, and writes `metrics.json`. It regenerates the held-out split using the seed from the model's `run.json` when present, so evaluation uses the same test recordings the trainer held out.
- Guarded by `if __name__ == "__main__":`.

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add -A
git commit -m "Add evaluation with per-head metrics and confusion matrices

The old scripts printed results[1], results[2], results[3] positionally off
model.evaluate and called it accuracy, with no per-class breakdown and no
confusion matrix.

Reports accuracy, macro-F1, a full classification report and a confusion
matrix per head, into metrics.json. Macro-F1 because mood has ten classes
sampled uniformly, where accuracy alone flatters a model that has only
learned the marginal distribution."
git push origin main
```

---

### Task 6: XGBoost baseline on band-power features

**Files:**
- Modify: `neuropace/baselines.py`
- Create: `neuropace/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `neuropace.config.FREQUENCY_BANDS`, `make_windows`
- Produces:
  - `neuropace.features.band_powers(window, sampling_rate) -> dict[str, float]` — mean PSD within each band via Welch, averaged over channels
  - `neuropace.features.extract_features(windows, sampling_rate) -> np.ndarray` of shape `(n_windows, n_features)`
  - `neuropace.features.FEATURE_NAMES: tuple[str, ...]`
  - `neuropace.baselines.run_baseline(recordings, config, *, head="ailment", seed=42) -> dict`

- [ ] **Step 1: Write the failing feature test**

```python
# tests/test_features.py
import numpy as np

from neuropace.features import FEATURE_NAMES, band_powers, extract_features


def test_band_power_finds_the_injected_frequency():
    """A pure 10 Hz sine must put its power in alpha (8-13 Hz), not elsewhere."""
    fs = 250
    t = np.arange(0, 4, 1 / fs)
    signal = np.sin(2 * np.pi * 10 * t)[:, None]  # one channel

    powers = band_powers(signal, fs)

    assert powers["alpha"] == max(powers.values())
    assert powers["alpha"] > 10 * powers["gamma"]


def test_extract_features_shape_matches_names():
    windows = np.random.default_rng(0).normal(size=(7, 500, 3))
    features = extract_features(windows, 250)
    assert features.shape == (7, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuropace.features'`

- [ ] **Step 3: Write `neuropace/features.py`**

Requirements:
- `band_powers` uses `scipy.signal.welch` with `fs=sampling_rate`, integrates PSD within each band of `FREQUENCY_BANDS`, and averages across channels.
- `extract_features` returns, per window: the five band powers, plus mean, std, and spectral entropy — recovering what `xgb2.py` computed, but from the shared `FREQUENCY_BANDS` constant instead of hardcoded literals, and vectorized over channels rather than looping in Python.
- `FEATURE_NAMES` is the authoritative ordering, so a feature-importance plot can be labelled.

- [ ] **Step 4: Rewrite `neuropace/baselines.py` against it**

Requirements:
- `run_baseline` extracts features from the train/test splits produced by `split_indices`, fits `XGBClassifier`, and returns accuracy plus macro-F1 in the same shape `evaluate()` returns, so the deep model and the baseline are comparable.
- Delete the `optuna` and `shap` imports carried over from `xgb2.py`: neither is used past import, and both are heavy dependencies for a baseline whose job is to be a reference point.
- `main()` with argparse, `if __name__ == "__main__":` guard.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: PASS.

- [ ] **Step 6: Commit and push**

```bash
git add -A
git commit -m "Add a band-power feature extractor and an XGBoost baseline

xgb2.py hardcoded its band edges as literals that had already drifted from
the ones in dsc.py, looped over channels in Python, and imported optuna and
shap without using either.

Features now come from the shared FREQUENCY_BANDS constant and are
vectorized. The baseline reports accuracy and macro-F1 in the same shape as
evaluate(), so the hybrid model has something honest to be compared against
-- without which its numbers mean very little."
git push origin main
```

---

### Task 7: Harden the API and test it

**Files:**
- Modify: `api/main.py`
- Create: `api/requirements.txt` (regenerate against the package)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `neuropace.config` label tuples and `EEGConfig`
- Produces: `/health`, `POST /predict/`, `GET /labels`

- [ ] **Step 1: Write the failing API tests**

```python
# tests/test_api.py
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

import main as api  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "absent.keras")
    api.get_model.cache_clear()
    return TestClient(api.app)


def test_health_reports_missing_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict_without_a_model_is_503_and_says_how_to_fix_it(client):
    response = client.post("/predict/", json={"data": [[0.0] * 19] * 1250})
    assert response.status_code == 503
    assert "neuropace.train" in response.json()["detail"]


def test_labels_endpoint_matches_the_package(client):
    from neuropace.config import AILMENT_CLASSES

    assert client.get("/labels").json()["ailment"] == list(AILMENT_CLASSES)
```

- [ ] **Step 2: Run and watch them fail**

Run: `.venv/bin/python -m pytest tests/test_api.py -v`
Expected: FAIL — no `/labels` route (404), and `/predict/` is not yet defined.

- [ ] **Step 3: Finish `api/main.py`**

Requirements:
- Import `AILMENT_CLASSES`, `STRESS_LEVELS`, `MOOD_STATES`, `EEGConfig` from `neuropace.config`. Delete the duplicated literals currently in `api/main.py` — they are a desync waiting to happen.
- `EEGWindow` pydantic model with `data: list[list[float]]`.
- `POST /predict/`: `require_model()`, validate `np.asarray(data).shape == config.input_shape` and 400 with both shapes named on mismatch, predict, and return `{"ailment": {...}, "stress": {...}, "mood": {...}}` where each is `{"label": str, "index": int, "confidence": float}`.
- Handle both dict-keyed and list-ordered model outputs: `model.predict` returns a dict when heads are named, and the head order otherwise. Prefer the dict.
- `GET /labels` returning the three label lists, so the mobile client never hardcodes them.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add -A
git commit -m "Harden the API and cover it with tests

Labels were duplicated between api/main.py and the config module, so the two
could drift apart silently and produce confidently mislabelled predictions.
They now come from neuropace.config, and GET /labels serves them to the
client so the app does not hardcode them either.

Tests cover the no-model path end to end, which is the path CI runs."
git push origin main
```

---

### Task 8: One real screen in the Expo app

**Files:**
- Modify: `app/App.js`
- Create: `app/src/api.js`, `app/src/theme.js`, `app/src/components/PredictionCard.js`, `app/src/components/EEGTrace.js`, `app/src/components/StatusPill.js`
- Modify: `app/app.json` (name, slug, colors)

**Interfaces:**
- Consumes: `GET /health`, `GET /labels`, `POST /predict/` from Task 7
- Produces: a single screen

**Before implementing, load the `frontend-design:frontend-design` skill** and follow its direction on visual choices. The screen must not read as default React Native.

- [ ] **Step 1: `app/src/api.js`**

`API_BASE` from `process.env.EXPO_PUBLIC_API_URL`, defaulting to `http://localhost:8080`. Export `getHealth()`, `getLabels()`, `postPredict(window)`. Every call wrapped so a network failure returns a typed result rather than throwing into the render.

- [ ] **Step 2: `app/src/theme.js`**

Tokens only: colors, spacing scale, type scale, radii. Clinical-instrument palette — dark ground, one signal accent — not the Expo default white.

- [ ] **Step 3: `EEGTrace.js`**

A scrolling multi-channel trace drawn with `react-native-svg` polylines from a locally generated buffer. Purely presentational: it takes a `channels` array prop and renders it.

- [ ] **Step 4: `StatusPill.js` and `PredictionCard.js`**

`StatusPill` shows API reachability and whether a model is loaded, driven by `/health`. `PredictionCard` takes `{ title, label, confidence }` and renders the label with a confidence bar.

- [ ] **Step 5: `App.js`**

Compose them: poll `/health` on mount, show the trace, and a "Analyze window" action that posts the current buffer to `/predict/` and fills three `PredictionCard`s. The 503 case must render as a readable instruction to train a model, not a crash or a silent empty state — that is the state the app will actually be in for anyone who clones this repo.

- [ ] **Step 6: Verify it builds**

```bash
cd app && npm install && npx expo export --platform web
```
Expected: export completes without error. (A full device run is not required; the export catches import and syntax errors.)

- [ ] **Step 7: Commit and push**

```bash
git add -A
git commit -m "Build the client screen

The app was the unmodified Expo scaffold -- App.js still said 'Open up
App.js to start working on your app!'

One screen: live EEG trace, API status, and the three prediction heads with
confidences. Labels come from GET /labels rather than being hardcoded. The
no-model 503 renders as an instruction to train one, since that is the state
a fresh clone is actually in."
git push origin main
```

---

### Task 9: Lint, test config, and CI

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml` (ruff/pytest config already added in Task 2; extend if needed)
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: everything above
- Produces: a green CI run on push

- [ ] **Step 1: `tests/conftest.py`**

Register a `slow` marker and set `TF_CPP_MIN_LOG_LEVEL=3` so TensorFlow's startup banner does not bury test output.

- [ ] **Step 2: `.github/workflows/ci.yml`**

Two jobs so a NumPy-level regression reports in seconds rather than waiting on a TensorFlow install:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  fast:
    name: lint + TF-free tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv venv --python 3.11
      - run: uv pip install -e ".[dev]"
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest tests/test_config.py tests/test_dataset.py tests/test_features.py -v

  full:
    name: full suite with TensorFlow
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv venv --python 3.11
      - run: uv pip install -e ".[dev,train,api]"
      - run: uv run pytest -v
```

- [ ] **Step 3: Fix everything ruff finds**

Run: `.venv/bin/python -m ruff check --fix . && .venv/bin/python -m ruff format .`
Then re-run the full suite to confirm the reformat broke nothing.

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all pass. Record the actual count for the README in Task 10 — do not guess it.

- [ ] **Step 5: Commit and push**

```bash
git add -A
git commit -m "Add ruff, pytest config, and two-stage CI

The fast job lints and runs the TF-free tests in seconds; the full job pays
for TensorFlow and runs everything. Splitting them is the payoff for keeping
config, dataset, synthetic and features free of TF imports."
git push origin main
```

---

### Task 10: Rewrite the README

**Files:**
- Modify: `README.md`
- Delete: `docs/archive/` (superseded; the merge commit records what was there)

**Interfaces:**
- Consumes: the real test count from Task 9, the real CLI flags from Tasks 4-6

- [ ] **Step 1: Write the README**

Required sections:
- What the project is, in two sentences, above the fold.
- Quickstart: `uv venv --python 3.11 .venv`, install, train, evaluate, serve. Every command copy-pasteable and verified to run.
- Architecture, with the TF-free boundary explained — it is a real design decision and a reviewer should see the reasoning.
- The three prediction heads and their label vocabularies.
- **Results**: the metrics table with its columns present and its cells showing the command that fills them, plus a one-line note that no trained weights ship with the repo and why. No invented numbers.
- **Engineering notes**: a short, specific account of the batch-0 bug, what it meant for the earlier results, and how the regression test pins it. This is the section that does the work on a resume.
- Test/CI instructions.

- [ ] **Step 2: Verify every command in the README actually runs**

Execute each quickstart command in order in a clean checkout. Any command that fails gets fixed in the README, not excused.

- [ ] **Step 3: Commit and push**

```bash
git add -A
git commit -m "Rewrite the README around what the project actually does

Documents the TF-free module boundary, the three prediction heads, and the
data-generator bug with the regression test that pins it. The results table
ships with its cells empty next to the command that fills them: no weights
are committed and no training was run, so there are no honest numbers to
quote yet."
git push origin main
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: architecture -> 2; data pipeline -> 3; model artifacts -> 1, 4; reproducibility -> 4; evaluation -> 5; the listed correctness fixes -> 2 (deepcopy, star import, `__main__` guards), 3 (leakage assertion), 4 (single compile, per-head metrics dict, `.keras`), 7 (API label duplication); testing strategy -> 3, 4, 5, 6, 7, 9; delivery -> the ten task boundaries; honest-results constraint -> 10.

**Type consistency.** `EEGConfig` (not `EEGDataConfig`) throughout. `Recording` (not `EEGRecording`). `build_model` (not `build_eeg_model`). Heads are keyed `"ailment"`, `"stress"`, `"mood"` in `make_dataset`, `evaluate`, and the API response alike. `make_dataset(recordings, config, *, augment, shuffle, seed)` has one signature everywhere it appears.

**Known risk.** Task 8 is the only task whose verification is a build rather than a test. The Expo export catches import and syntax errors but not visual regressions; a device run is out of scope here and should be treated as unverified until someone opens it.
