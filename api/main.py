"""NeuroPace inference service.

Serves best_yet.h5, the multi-output hybrid CNN/LSTM trained by ../ml/pipeline.py.
The model takes one EEG window of shape (1250, 19) -- 5 s at 250 Hz across 19
channels -- and emits three softmax heads: ailment, stress level, mood state.
"""

import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Label sets mirror EEGDataConfig defaults in ../ml/dsc.py. Keep them in sync.
AILMENT_CLASSES = [
    "normal", "seizure", "delayed_recovery", "ischemia", "hemorrhage", "infection",
]
STRESS_LEVELS = ["minimal", "mild", "moderate", "high", "severe"]
MOOD_STATES = [
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
]

MODEL_PATH = os.environ.get("MODEL_PATH", "best_yet.h5")
model = tf.keras.models.load_model(MODEL_PATH)
_, WINDOW_SAMPLES, CHANNELS = model.input_shape

app = FastAPI(title="NeuroPace EEG API")


class EEGWindow(BaseModel):
    # (WINDOW_SAMPLES x CHANNELS) EEG readings, i.e. 1250 rows of 19 floats.
    data: list


def _head(probabilities, labels):
    index = int(np.argmax(probabilities))
    return {
        "label": labels[index],
        "index": index,
        "confidence": float(probabilities[index]),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "window_samples": WINDOW_SAMPLES,
        "channels": CHANNELS,
    }


@app.post("/predict/")
def predict(window: EEGWindow):
    signal = np.asarray(window.data, dtype=np.float32)

    if signal.shape != (WINDOW_SAMPLES, CHANNELS):
        raise HTTPException(
            status_code=400,
            detail=f"expected shape ({WINDOW_SAMPLES}, {CHANNELS}), got {signal.shape}",
        )

    ailment, stress, mood = model.predict(signal[np.newaxis, ...], verbose=0)

    return {
        "ailment": _head(ailment[0], AILMENT_CLASSES),
        "stress": _head(stress[0], STRESS_LEVELS),
        "mood": _head(mood[0], MOOD_STATES),
    }
