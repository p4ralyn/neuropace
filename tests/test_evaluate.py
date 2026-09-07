import numpy as np
import pytest

from neuropace.config import EEGConfig
from neuropace.synthetic import generate_dataset

pytest.importorskip("tensorflow")

from neuropace.evaluate import evaluate, save_confusion_matrices  # noqa: E402
from neuropace.models import build_model  # noqa: E402


@pytest.fixture(scope="module")
def fixture():
    config = EEGConfig(
        sampling_rate=25,
        channels=2,
        recording_duration=2.0,
        window_size=1.0,
        batch_size=4,
    )
    recordings = generate_dataset(
        config, n_per_condition=3, rng=np.random.default_rng(0)
    )
    model = build_model(config, model_type="cnn")
    return config, recordings, model


def test_reports_every_head_with_sane_bounds(fixture):
    config, recordings, model = fixture
    results = evaluate(model, recordings, config)

    assert set(results) == {"ailment", "stress", "mood"}
    for head, scores in results.items():
        assert 0.0 <= scores["accuracy"] <= 1.0, head
        assert 0.0 <= scores["macro_f1"] <= 1.0, head
        assert "report" in scores


def test_confusion_matrix_is_square_and_totals_the_windows(fixture):
    config, recordings, model = fixture
    results = evaluate(model, recordings, config)

    matrix = np.array(results["ailment"]["confusion_matrix"])
    assert matrix.shape == (config.num_ailments, config.num_ailments)

    from neuropace.data import windows_and_labels

    assert matrix.sum() == windows_and_labels(recordings, config)[0].shape[0]


def test_accuracy_agrees_with_the_confusion_matrix(fixture):
    """The diagonal over the total must reproduce the reported accuracy."""
    config, recordings, model = fixture
    results = evaluate(model, recordings, config)

    for head, scores in results.items():
        matrix = np.array(scores["confusion_matrix"])
        assert np.isclose(np.trace(matrix) / matrix.sum(), scores["accuracy"]), head


def test_saves_a_png_per_head(fixture, tmp_path):
    config, recordings, model = fixture
    results = evaluate(model, recordings, config)
    paths = save_confusion_matrices(results, config, tmp_path)

    assert len(paths) == 3
    for path in paths:
        assert path.exists() and path.stat().st_size > 0
