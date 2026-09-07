# NeuroPace

Post-surgical patient monitoring from EEG. A synthetic EEG generator feeds a
multi-output neural network that predicts three things at once from a single
5-second window of 19-channel EEG: **ailment**, **stress level**, and **mood
state**. The trained model is served over HTTP and consumed by a mobile app.

```
ml/  ── trains ──>  api/best_yet.h5  ── serves ──>  api/ (FastAPI)  <── calls ──  app/ (Expo)
```

## Layout

| Path | What it is |
|---|---|
| `ml/dsc.py` | `EEGDataConfig`, `EEGDataset`, `EEGRecording` — data model and the label sets everything else derives from |
| `ml/sdg.py` | Synthetic EEG generation, per condition / stress level / mood state |
| `ml/ma.py` | `build_eeg_model(config, model_type, output_type)` — CNN, LSTM, or hybrid |
| `ml/pipeline.py` | End-to-end training run: generate → split → build → fit |
| `ml/pp.py` | Training variant with the augmenting data generator and GPU setup |
| `ml/experiments/` | XGBoost side-explorations on hand-extracted band-power features |
| `api/` | FastAPI inference service + the trained weights + Dockerfile |
| `app/` | Expo / React Native client |
| `docs/archive/` | Superseded code kept for reference only; not wired into anything |

## The model

`api/best_yet.h5` is the hybrid architecture with `output_type='all'`.

- **Input** `(1250, 19)` — 5 s window at 250 Hz, 19 electrodes
- **Outputs** three softmax heads:
  - `ailment_output` (6) — normal, seizure, delayed_recovery, ischemia, hemorrhage, infection
  - `stress_output` (5) — minimal, mild, moderate, high, severe
  - `mood_output` (10) — Pain & Discomfort, Anxiety & Fear, … Focused Attention

Label lists live in `ml/dsc.py` (`EEGDataConfig` defaults) and are mirrored in
`api/main.py`. Change one, change the other.

## Running it

### Training

```sh
cd ml
pip install -r requirements.txt
python pipeline.py
```

The XGBoost experiments import their siblings, so run them as modules from `ml/`:

```sh
python -m experiments.xgb
```

### Inference API

```sh
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

- `GET /health` → confirms the model loaded and reports the expected input shape
- `POST /predict/` → `{"data": [[...19 floats...] x 1250]}` returns a label,
  index, and confidence for each of the three heads

Container:

```sh
docker build -t neuropace-api api/
docker run -p 8080:8080 neuropace-api
```

### Mobile app

```sh
cd app
npm install
npm start
```

Currently the stock Expo scaffold — nothing is wired to the API yet. This is the
main outstanding piece of work.

## History

This was a hackathon project split across three directories: `NeuroPace/` (all
the ML work, untracked), `NeuroPace_1/` (a Flask stub plus the git remote and a
Windows venv committed into it), and `MyHackathonApp/` (an untouched Expo
scaffold, duplicated inside `NeuroPace_1/`). They are merged here. Dropped along
the way: the Flask stub, which rendered a template that did not exist and
duplicated the API's role; `model/eeg-api/`, a second FastAPI copy that loaded a
weights file (`eeg_mood_model.h5`) that exists nowhere in the project; and
`model/demo.py`, an unrelated prime-number exercise. The first two are in
`docs/archive/`.
