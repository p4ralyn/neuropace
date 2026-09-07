"""One place to make a run reproducible."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow, and ask TF for deterministic ops."""
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
