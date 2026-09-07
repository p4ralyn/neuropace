"""tf.data input pipeline.

Replaces the EEGDataGenerator class that lived in both dsc.py and pp.py. That
class subclassed keras.utils.Sequence but implemented

    def __getitem__(self, idx):
        return next(iter(self.dataset_tf))

which rebuilds the iterator on every call and so returns batch 0 forever,
ignoring idx. Training therefore saw a single batch, repeatedly, for every
epoch of every run. Both copies had it, and they had diverged in other ways.

It is replaced by a function rather than repaired, because it was only ever a
broken wrapper around a tf.data.Dataset that it constructed internally anyway.
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
    """Flatten recordings into windows with their integer labels.

    Returns (features, ailment, stress, mood). Every window inherits the labels
    of the recording it came from.
    """
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


def standardize(window: tf.Tensor) -> tf.Tensor:
    """Zero-mean, unit-variance per channel, within each window.

    Per window rather than per dataset: amplitude scale varies between
    recordings for reasons (electrode impedance, gain) that carry no
    diagnostic signal, and normalizing them away is the point.
    """
    mean = tf.reduce_mean(window, axis=0, keepdims=True)
    std = tf.math.reduce_std(window, axis=0, keepdims=True)
    return (window - mean) / (std + 1e-6)


def augment(window: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """Additive noise, amplitude scaling, and channel dropout.

    Recovered from the commented-out block in dsc.py, which was the stronger
    of the two implementations that shipped. Stateless ops keyed on an
    explicit seed, so an augmented epoch is reproducible.
    """
    noise_seed, scale_seed, drop_seed = tf.unstack(
        tf.random.experimental.stateless_split(seed, num=3)
    )

    window += tf.random.stateless_normal(tf.shape(window), noise_seed, stddev=0.05)
    window *= tf.random.stateless_uniform([], scale_seed, minval=0.9, maxval=1.1)

    # Drop ~10% of channels outright, mimicking a lead coming loose.
    keep = tf.cast(
        tf.random.stateless_uniform([tf.shape(window)[-1]], drop_seed) > 0.1,
        window.dtype,
    )
    return window * keep


def make_dataset(
    recordings: Sequence[Recording],
    config: EEGConfig,
    *,
    augment_data: bool = False,
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
        lambda x, y: (standardize(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment_data:
        generator = tf.random.Generator.from_seed(0 if seed is None else seed)

        def apply(x, y):
            return augment(x, generator.make_seeds(1)[:, 0]), y

        dataset = dataset.map(apply, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
