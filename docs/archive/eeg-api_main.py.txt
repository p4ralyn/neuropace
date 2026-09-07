import os
from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from pydantic import BaseModel

# Load model
model = tf.keras.models.load_model("eeg_mood_model.h5")

app = FastAPI()

class EEGData(BaseModel):
    data: list  # List of EEG readings

@app.post("/predict/")
def predict_mood(eeg_data: EEGData):
    try:
        input_data = np.array(eeg_data.data).reshape(1, -1)
        prediction = model.predict(input_data)
        mood = np.argmax(prediction)
        return {"mood": int(mood)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # Use the PORT env variable
    uvicorn.run(app, host="0.0.0.0", port=port)
