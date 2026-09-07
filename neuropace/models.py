"""Model architectures for multi-head EEG classification.

Three heads share one trunk: ailment, stress, and mood are predicted from the
same window. They are correlated in the data -- stress and mood both modulate
alpha and beta -- so a shared representation is the natural framing.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
)

from .config import EEGConfig

HEAD_NAMES = ("ailment", "stress", "mood")


def _cnn_trunk(inputs: tf.Tensor) -> tf.Tensor:
    x = inputs
    for filters in (64, 128, 256):
        x = Conv1D(filters, kernel_size=3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    return Dropout(0.3)(x)


def _lstm_trunk(inputs: tf.Tensor) -> tf.Tensor:
    x = Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    return Dropout(0.3)(x)


def _hybrid_trunk(inputs: tf.Tensor) -> tf.Tensor:
    """Convolutional and recurrent branches over the same input, concatenated.

    The convolutional branch picks up local waveform shape (spikes, bursts);
    the recurrent branch carries longer-range temporal structure.
    """
    cnn = inputs
    for filters in (64, 128):
        cnn = Conv1D(filters, kernel_size=3, activation="relu", padding="same")(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = GlobalAveragePooling1D()(cnn)

    lstm = Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
    lstm = Dropout(0.3)(lstm)
    lstm = Bidirectional(tf.keras.layers.LSTM(64))(lstm)

    x = Concatenate()([cnn, lstm])
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    return Dropout(0.3)(x)


TRUNKS = {"cnn": _cnn_trunk, "lstm": _lstm_trunk, "hybrid": _hybrid_trunk}


def build_model(
    config: EEGConfig,
    *,
    model_type: str = "hybrid",
    learning_rate: float = 1e-3,
) -> Model:
    """Build and compile a three-headed classifier.

    Compiled exactly once, here. The previous code compiled inside the builder
    and then again in every training script, with a different optimizer.
    """
    if model_type not in TRUNKS:
        raise ValueError(f"unknown model_type {model_type!r}; expected {list(TRUNKS)}")

    inputs = Input(shape=config.input_shape, name="eeg_window")
    trunk = TRUNKS[model_type](inputs)

    outputs = {
        "ailment": Dense(config.num_ailments, activation="softmax", name="ailment")(
            trunk
        ),
        "stress": Dense(config.num_stress_levels, activation="softmax", name="stress")(
            trunk
        ),
        "mood": Dense(config.num_moods, activation="softmax", name="mood")(trunk),
    }

    model = Model(inputs=inputs, outputs=outputs, name=f"neuropace_{model_type}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={name: "categorical_crossentropy" for name in HEAD_NAMES},
        metrics={name: ["accuracy"] for name in HEAD_NAMES},
    )
    return model
