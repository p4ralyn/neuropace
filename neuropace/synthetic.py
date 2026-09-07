"""Synthetic EEG generation.

Signals are built as sums of band-limited sinusoids whose amplitudes are
modulated by the requested condition, stress level, and mood state, plus
condition-specific spatial and temporal structure (seizure bursts, hemisphere
suppression in hemorrhage) and per-channel noise.

The band adjustments follow standard qualitative EEG findings: slowing (raised
delta/theta) in ischemia and delayed recovery, alpha suppression with rising
beta under stress, delta dominance in deep sleep. They are a caricature of real
EEG, not a substitute for it -- the point is a dataset with learnable structure
whose ground truth is known exactly.

All randomness flows through an explicit numpy Generator so a run is
reproducible from its seed.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import numpy as np

from .config import AILMENT_CLASSES, MOOD_STATES, STRESS_LEVELS, EEGConfig
from .dataset import Recording

# Baseline amplitude and frequency ranges per band, before any modulation.
BASE_BANDS: dict[str, dict[str, float]] = {
    "delta": {"min_freq": 0.5, "max_freq": 4.0, "min_amp": 0.5, "max_amp": 1.5},
    "theta": {"min_freq": 4.0, "max_freq": 8.0, "min_amp": 0.3, "max_amp": 1.0},
    "alpha": {"min_freq": 8.0, "max_freq": 13.0, "min_amp": 0.5, "max_amp": 1.5},
    "beta": {"min_freq": 13.0, "max_freq": 30.0, "min_amp": 0.2, "max_amp": 0.8},
    "gamma": {"min_freq": 30.0, "max_freq": 80.0, "min_amp": 0.1, "max_amp": 0.4},
}


def _condition_bands(
    condition: str, rng: np.random.Generator, metadata: dict
) -> dict[str, dict[str, float]]:
    """Amplitude ranges for a condition, plus its metadata side effects.

    deepcopy, not copy: the nested per-band dicts must not be aliased back into
    BASE_BANDS. The original code used a shallow .copy() here.
    """
    bands = copy.deepcopy(BASE_BANDS)

    if condition == "normal":
        bands["alpha"].update(min_amp=0.8, max_amp=1.8)
        bands["beta"].update(min_amp=0.4, max_amp=1.0)

    elif condition == "seizure":
        bands["beta"].update(min_amp=1.0, max_amp=2.5)
        bands["gamma"].update(min_amp=0.8, max_amp=2.0)
        metadata["seizure_type"] = str(rng.choice(["focal", "generalized"]))
        metadata["seizure_severity"] = str(rng.choice(["mild", "moderate", "severe"]))

    elif condition == "delayed_recovery":
        bands["delta"].update(min_amp=1.2, max_amp=2.5)
        bands["beta"].update(min_amp=0.1, max_amp=0.4)
        bands["gamma"].update(min_amp=0.05, max_amp=0.2)

    elif condition == "ischemia":
        bands["delta"].update(min_amp=1.5, max_amp=3.0)
        bands["theta"].update(min_amp=0.8, max_amp=1.8)
        bands["alpha"].update(min_amp=0.2, max_amp=0.6)
        metadata["ischemia_location"] = str(
            rng.choice(["frontal", "temporal", "parietal", "occipital"])
        )
        metadata["ischemia_severity"] = str(rng.choice(["mild", "moderate", "severe"]))

    elif condition == "hemorrhage":
        bands["delta"].update(min_amp=1.3, max_amp=2.8)
        bands["theta"].update(min_amp=0.7, max_amp=1.6)
        bands["alpha"].update(min_amp=0.2, max_amp=0.5)
        metadata["hemorrhage_type"] = str(
            rng.choice(["subdural", "epidural", "subarachnoid", "intracerebral"])
        )
        metadata["hemorrhage_location"] = str(rng.choice(["left", "right"]))
        metadata["hemorrhage_severity"] = str(
            rng.choice(["mild", "moderate", "severe"])
        )

    elif condition == "infection":
        bands["delta"].update(min_amp=1.0, max_amp=2.0)
        bands["theta"].update(min_amp=1.0, max_amp=2.0)
        bands["beta"].update(min_amp=0.3, max_amp=0.9)
        metadata["infection_type"] = str(rng.choice(["bacterial", "viral", "fungal"]))
        metadata["infection_severity"] = str(rng.choice(["mild", "moderate", "severe"]))

    return bands


def _stress_gains(stress_level: str, rng: np.random.Generator) -> dict[str, float]:
    """Per-band amplitude multipliers for a stress level.

    Alpha falls and beta/gamma rise as stress increases -- the standard
    desynchronization pattern.
    """
    if stress_level == "minimal":
        return {
            "alpha": float(rng.uniform(1.2, 1.5)),
            "beta": float(rng.uniform(0.6, 0.8)),
            "gamma": float(rng.uniform(0.5, 0.7)),
        }
    if stress_level == "mild":
        return {
            "alpha": float(rng.uniform(0.8, 0.9)),
            "beta": float(rng.uniform(1.1, 1.3)),
        }
    if stress_level == "moderate":
        return {
            "alpha": float(rng.uniform(0.6, 0.8)),
            "beta": float(rng.uniform(1.3, 1.6)),
        }
    if stress_level == "high":
        return {
            "alpha": float(rng.uniform(0.4, 0.6)),
            "beta": float(rng.uniform(1.5, 1.8)),
            "gamma": float(rng.uniform(1.3, 1.6)),
        }
    if stress_level == "severe":
        return {
            "alpha": float(rng.uniform(0.2, 0.4)),
            "beta": float(rng.uniform(1.8, 2.2)),
            "gamma": float(rng.uniform(1.6, 2.0)),
        }
    return {}


def _mood_gains(mood_state: str, rng: np.random.Generator) -> dict[str, float]:
    """Per-band amplitude multipliers for a mood state."""
    if mood_state in ("Pain & Discomfort", "Anxiety & Fear"):
        return {
            "beta": float(rng.uniform(1.2, 1.5)),
            "gamma": float(rng.uniform(1.1, 1.4)),
        }
    if mood_state in ("Depression & Mental Fatigue", "Emotional Exhaustion"):
        return {
            "theta": float(rng.uniform(1.3, 1.6)),
            "alpha": float(rng.uniform(0.6, 0.8)),
            "beta": float(rng.uniform(0.7, 0.9)),
        }
    if mood_state == "Insomnia & Sleep Disturbances":
        return {
            "delta": float(rng.uniform(0.5, 0.7)),
            "beta": float(rng.uniform(1.2, 1.5)),
        }
    if mood_state == "Cognitive Dysfunction":
        return {
            "theta": float(rng.uniform(1.2, 1.5)),
            "alpha": float(rng.uniform(0.5, 0.7)),
        }
    if mood_state == "Post-Surgery PTSD":
        return {
            "beta": float(rng.uniform(1.3, 1.6)),
            "gamma": float(rng.uniform(1.2, 1.5)),
            "alpha": float(rng.uniform(0.5, 0.7)),
        }
    if mood_state == "Relaxation & Recovery":
        return {
            "alpha": float(rng.uniform(1.4, 1.7)),
            "beta": float(rng.uniform(0.6, 0.8)),
            "gamma": float(rng.uniform(0.5, 0.7)),
        }
    if mood_state == "Deep Sleep":
        return {
            "delta": float(rng.uniform(1.8, 2.2)),
            "theta": float(rng.uniform(0.5, 0.7)),
            "alpha": float(rng.uniform(0.3, 0.5)),
            "beta": float(rng.uniform(0.2, 0.4)),
            "gamma": float(rng.uniform(0.1, 0.3)),
        }
    if mood_state == "Focused Attention":
        return {
            "beta": float(rng.uniform(1.3, 1.6)),
            "alpha": float(rng.uniform(1.1, 1.3)),
        }
    return {}


def _seizure_burst(
    samples: int, t: np.ndarray, amplitude: float, rng: np.random.Generator
) -> np.ndarray:
    """A Hanning-enveloped high-frequency burst, as seen in seizure activity."""
    start = int(rng.integers(0, max(1, samples - samples // 4)))
    duration = int(rng.integers(max(1, samples // 10), max(2, samples // 4)))
    duration = min(duration, samples - start)

    envelope = np.zeros(samples)
    envelope[start : start + duration] = np.hanning(duration)

    frequency = float(rng.uniform(15.0, 25.0))
    return amplitude * 3.0 * envelope * np.sin(2 * np.pi * frequency * t)


def generate_recording(
    config: EEGConfig,
    condition: str,
    *,
    stress_level: str | None = None,
    mood_state: str | None = None,
    patient_id: str | None = None,
    rng: np.random.Generator | None = None,
) -> Recording:
    """Generate one synthetic recording for the given labels."""
    rng = rng if rng is not None else np.random.default_rng()

    if condition not in AILMENT_CLASSES:
        raise ValueError(f"unknown condition {condition!r}; expected {AILMENT_CLASSES}")
    stress_level = stress_level or str(rng.choice(STRESS_LEVELS))
    if stress_level not in STRESS_LEVELS:
        raise ValueError(f"unknown stress level {stress_level!r}")
    mood_state = mood_state or str(rng.choice(MOOD_STATES))
    if mood_state not in MOOD_STATES:
        raise ValueError(f"unknown mood state {mood_state!r}")

    samples = config.samples_per_recording
    channels = config.channels
    t = np.arange(samples) / config.sampling_rate

    metadata: dict = {
        "is_synthetic": True,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    bands = _condition_bands(condition, rng, metadata)
    gains = _stress_gains(stress_level, rng)
    mood = _mood_gains(mood_state, rng)
    for band, gain in mood.items():
        gains[band] = gains.get(band, 1.0) * gain
    metadata["band_gains"] = gains

    data = np.zeros((samples, channels))

    for channel in range(channels):
        signal = np.zeros(samples)

        for band, params in bands.items():
            gain = gains.get(band, 1.0)

            for _ in range(int(rng.integers(3, 6))):
                frequency = float(rng.uniform(params["min_freq"], params["max_freq"]))
                amplitude = (
                    float(rng.uniform(params["min_amp"], params["max_amp"])) * gain
                )

                bursting = condition == "seizure" and band in ("beta", "gamma")
                if bursting and rng.random() < 0.7:
                    signal += _seizure_burst(samples, t, amplitude, rng)

                phase = float(rng.uniform(0, 2 * np.pi))
                signal += amplitude * np.sin(2 * np.pi * frequency * t + phase)

        # Hemorrhage suppresses the affected hemisphere.
        if condition == "hemorrhage":
            left = metadata["hemorrhage_location"] == "left"
            affected = channel < channels // 2 if left else channel >= channels // 2
            if affected:
                signal *= float(rng.uniform(0.3, 0.6))

        signal += rng.normal(0, float(rng.uniform(0.05, 0.15)), samples)
        data[:, channel] = signal

    return Recording(
        data=data,
        condition=condition,
        condition_id=AILMENT_CLASSES.index(condition),
        stress_level=stress_level,
        stress_level_id=STRESS_LEVELS.index(stress_level),
        mood_state=mood_state,
        mood_state_id=MOOD_STATES.index(mood_state),
        patient_id=patient_id or f"SYN{int(rng.integers(1000, 9999))}",
        metadata=metadata,
    )


def generate_dataset(
    config: EEGConfig,
    n_per_condition: int = 20,
    *,
    rng: np.random.Generator | None = None,
) -> list[Recording]:
    """Generate a balanced dataset: n_per_condition recordings per ailment."""
    rng = rng if rng is not None else np.random.default_rng()
    return [
        generate_recording(config, condition, rng=rng)
        for condition in AILMENT_CLASSES
        for _ in range(n_per_condition)
    ]
