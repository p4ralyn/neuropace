"""NeuroPace inference service.

Serves the multi-head model trained by `python -m neuropace.train`: one EEG
window in, three labelled predictions out.

The model is a build output, not a repository artifact, so this service is
written to be useful without one -- it imports, starts, and answers /health
with no weights present. That is what lets it be tested in CI without a
training run, and what makes a fresh clone explain itself rather than crash.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from neuropace.config import (
    AILMENT_CLASSES,
    MOOD_STATES,
    STRESS_LEVELS,
    EEGConfig,
)

MODEL_PATH = Path(os.environ.get("NEUROPACE_MODEL", "artifacts/model.keras"))
TRAIN_HINT = "python -m neuropace.train"

# Labels are imported, never retyped. The previous version kept its own copies
# alongside the ones in config.py, which is how a service starts returning
# confidently mislabelled predictions.
HEAD_LABELS: dict[str, tuple[str, ...]] = {
    "ailment": AILMENT_CLASSES,
    "stress": STRESS_LEVELS,
    "mood": MOOD_STATES,
}

app = FastAPI(
    title="NeuroPace EEG API",
    version="0.1.0",
    description="Multi-head EEG classification: ailment, stress, and mood.",
)


@lru_cache(maxsize=1)
def get_config() -> EEGConfig:
    """The config the model was trained with, if its manifest is beside it."""
    manifest = MODEL_PATH.parent / "config.json"
    if manifest.exists():
        return EEGConfig.load(manifest)
    return EEGConfig()


@lru_cache(maxsize=1)
def get_model():
    """Load the trained model, or return None if no artifact exists."""
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf  # imported lazily to keep startup cheap

    return tf.keras.models.load_model(MODEL_PATH)


def require_model():
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"No model at {MODEL_PATH}. Train one: {TRAIN_HINT}",
        )
    return model


class EEGWindow(BaseModel):
    """One window of EEG, shaped (samples_per_window, channels)."""

    data: list[list[float]] = Field(
        ...,
        description="Rows are time samples, columns are channels.",
    )


class Prediction(BaseModel):
    label: str
    index: int
    confidence: float


@app.get("/health")
def health() -> dict:
    """Liveness plus whether a model is actually loaded."""
    config = get_config()
    return {
        "status": "ok",
        "model_loaded": get_model() is not None,
        "model_path": str(MODEL_PATH),
        "expected_shape": list(config.input_shape),
    }


@app.get("/labels")
def labels() -> dict[str, list[str]]:
    """The label vocabulary, so clients never hardcode it."""
    return {head: list(values) for head, values in HEAD_LABELS.items()}


def _head(probabilities: np.ndarray, head: str) -> dict:
    index = int(np.argmax(probabilities))
    return {
        "label": HEAD_LABELS[head][index],
        "index": index,
        "confidence": float(probabilities[index]),
    }


@app.post("/predict/")
def predict(window: EEGWindow) -> dict[str, Prediction]:
    """Classify one EEG window across all three heads."""
    model = require_model()
    config = get_config()

    signal = np.asarray(window.data, dtype=np.float32)
    if signal.shape != tuple(config.input_shape):
        raise HTTPException(
            status_code=400,
            detail=(
                f"expected shape {tuple(config.input_shape)} "
                f"(samples, channels), got {signal.shape}"
            ),
        )

    # Same per-window standardization the training pipeline applies.
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    normalized = (signal - mean) / (std + 1e-6)

    outputs = model.predict(normalized[np.newaxis, ...], verbose=0)
    if not isinstance(outputs, dict):
        outputs = dict(zip(HEAD_LABELS, outputs, strict=True))

    return {head: _head(outputs[head][0], head) for head in HEAD_LABELS}
