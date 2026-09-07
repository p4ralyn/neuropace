import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

import main as api  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client whose model path points at nothing -- the fresh-clone state."""
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "absent.keras")
    api.get_model.cache_clear()
    api.get_config.cache_clear()
    yield TestClient(api.app)
    api.get_model.cache_clear()
    api.get_config.cache_clear()


def test_health_reports_missing_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["expected_shape"] == [1250, 19]


def test_predict_without_a_model_is_503_and_says_how_to_fix_it(client):
    response = client.post("/predict/", json={"data": [[0.0] * 19] * 1250})
    assert response.status_code == 503
    assert "neuropace.train" in response.json()["detail"]


def test_labels_endpoint_matches_the_package(client):
    from neuropace.config import AILMENT_CLASSES, MOOD_STATES, STRESS_LEVELS

    body = client.get("/labels").json()
    assert body["ailment"] == list(AILMENT_CLASSES)
    assert body["stress"] == list(STRESS_LEVELS)
    assert body["mood"] == list(MOOD_STATES)


def test_malformed_payload_is_rejected(client):
    response = client.post("/predict/", json={"data": "not an array"})
    assert response.status_code == 422


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """A tiny real model on disk, so the prediction path is actually exercised."""
    tf = pytest.importorskip("tensorflow")

    from neuropace.config import EEGConfig
    from neuropace.models import build_model

    directory = tmp_path_factory.mktemp("served")
    config = EEGConfig(
        sampling_rate=25, channels=2, recording_duration=2.0, window_size=1.0
    )
    build_model(config, model_type="cnn").save(directory / "model.keras")
    config.save(directory / "config.json")
    assert tf is not None
    return directory, config


@pytest.fixture
def loaded_client(served, monkeypatch):
    directory, _ = served
    monkeypatch.setattr(api, "MODEL_PATH", directory / "model.keras")
    api.get_model.cache_clear()
    api.get_config.cache_clear()
    yield TestClient(api.app)
    api.get_model.cache_clear()
    api.get_config.cache_clear()


def test_health_reports_the_loaded_model(loaded_client, served):
    _, config = served
    body = loaded_client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["expected_shape"] == list(config.input_shape)


def test_predict_returns_a_label_per_head(loaded_client, served):
    _, config = served
    samples, channels = config.input_shape
    payload = {"data": [[0.1 * c for c in range(channels)] for _ in range(samples)]}

    response = loaded_client.post("/predict/", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"ailment", "stress", "mood"}

    from neuropace.config import AILMENT_CLASSES

    assert body["ailment"]["label"] in AILMENT_CLASSES
    assert 0.0 <= body["ailment"]["confidence"] <= 1.0
    assert body["ailment"]["index"] == AILMENT_CLASSES.index(body["ailment"]["label"])


def test_wrong_shape_is_400_and_names_both_shapes(loaded_client, served):
    _, config = served
    response = loaded_client.post("/predict/", json={"data": [[0.0] * 99] * 5})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert str(tuple(config.input_shape)) in detail
    assert "(5, 99)" in detail
