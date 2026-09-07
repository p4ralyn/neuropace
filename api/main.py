"""NeuroPace inference service."""

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

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
