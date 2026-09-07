"""EEG recordings, windowing, and dataset splits.

No TensorFlow here: windowing and splitting are array bookkeeping, and keeping
them TF-free lets the bulk of the test suite run without a TF import.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import train_test_split

from .config import EEGConfig


@dataclass
class Recording:
    """One EEG recording with its three labels."""

    data: np.ndarray  # (samples, channels)
    condition: str
    condition_id: int
    stress_level: str
    stress_level_id: int
    mood_state: str
    mood_state_id: int
    patient_id: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])


def make_windows(recording: Recording, config: EEGConfig) -> np.ndarray:
    """Slice a recording into overlapping windows.

    Returns shape (n_windows, samples_per_window, channels). Trailing samples
    that cannot fill a whole window are dropped.
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

    Splits by *recording*, never by window: windows from one recording share a
    signal, so splitting them individually would leak across the boundary.
    """
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {train + val + test}")

    indices = list(range(len(recordings)))
    conditions = [r.condition_id for r in recordings]

    # Stratification needs at least one recording per class in every split.
    n_classes = len(set(conditions))
    smallest = min(val, test) * len(recordings)
    if smallest < n_classes:
        raise ValueError(
            f"{len(recordings)} recordings is too few to stratify {n_classes} "
            f"classes across these ratios: the smallest split would hold "
            f"{smallest:.1f}. Generate at least "
            f"{int(np.ceil(n_classes / min(val, test)))} recordings."
        )

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


def subset(recordings: Sequence[Recording], indices: Sequence[int]) -> list[Recording]:
    """Select recordings by index."""
    return [recordings[i] for i in indices]
