import numpy as np
import pytest

from neuropace.config import EEGConfig
from neuropace.synthetic import generate_dataset

pytest.importorskip("xgboost")

from neuropace.baselines import featurize, run_baseline  # noqa: E402


@pytest.fixture(scope="module")
def recordings():
    config = EEGConfig(
        sampling_rate=50, channels=2, recording_duration=4.0, window_size=2.0
    )
    return config, generate_dataset(
        config, n_per_condition=12, rng=np.random.default_rng(0)
    )


def test_featurize_labels_match_their_recordings(recordings):
    config, recs = recordings
    features, labels = featurize(recs[:3], config, "ailment")

    assert features.shape[0] == labels.shape[0]
    assert set(labels) == {r.condition_id for r in recs[:3]}


def test_baseline_beats_chance_on_ailment(recordings):
    """The generator encodes condition in band amplitudes, so band-power
    features must do better than the 1/6 a coin-flipping model would get."""
    config, recs = recordings
    results = run_baseline(recs, config, head="ailment", seed=42)

    assert results["accuracy"] > 1.0 / 6.0
    assert results["n_features"] == 8


def test_unknown_head_is_rejected(recordings):
    config, recs = recordings
    with pytest.raises(ValueError, match="unknown head"):
        run_baseline(recs, config, head="nonsense")
