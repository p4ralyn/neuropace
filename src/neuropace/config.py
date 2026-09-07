"""Configuration and label vocabulary for EEG data.

Deliberately free of TensorFlow: this module and its label lists are imported
by the API, the tests, and the NumPy-only parts of the pipeline, none of which
should pay for a TF import.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
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
    def from_dict(cls, data: dict) -> EEGConfig:
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> EEGConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))
