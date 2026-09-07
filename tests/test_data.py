import numpy as np
import pytest

from neuropace.config import EEGConfig
from neuropace.synthetic import generate_dataset

pytest.importorskip("tensorflow")

from neuropace.data import make_dataset, windows_and_labels  # noqa: E402


@pytest.fixture(scope="module")
def small():
    """Tiny on every axis so the TF tests stay fast."""
    config = EEGConfig(
        sampling_rate=50,
        channels=4,
        recording_duration=3.0,
        window_size=2.0,
        window_overlap=0.5,
        batch_size=4,
    )
    recordings = generate_dataset(
        config, n_per_condition=2, rng=np.random.default_rng(0)
    )
    return config, recordings


def test_successive_batches_differ(small):
    """The bug this whole pass exists to fix.

    The old EEGDataGenerator.__getitem__ was

        return next(iter(self.dataset_tf))

    which rebuilds the iterator on every call, so every training step saw
    batch 0 and the idx argument was ignored entirely.
    """
    config, recordings = small
    batches = [x.numpy() for x, _ in make_dataset(recordings, config).take(3)]

    assert len(batches) == 3
    assert not np.array_equal(batches[0], batches[1]), "batch 1 repeats batch 0"
    assert not np.array_equal(batches[1], batches[2]), "batch 2 repeats batch 1"


def test_one_epoch_covers_every_window(small):
    config, recordings = small
    features, *_ = windows_and_labels(recordings, config)
    seen = sum(int(x.shape[0]) for x, _ in make_dataset(recordings, config))
    assert seen == features.shape[0]


def test_shapes_and_heads(small):
    config, recordings = small
    features, labels = next(iter(make_dataset(recordings, config)))

    assert tuple(features.shape[1:]) == tuple(config.input_shape)
    assert set(labels) == {"ailment", "stress", "mood"}
    assert labels["ailment"].shape[1] == config.num_ailments
    assert labels["stress"].shape[1] == config.num_stress_levels
    assert labels["mood"].shape[1] == config.num_moods


def test_labels_line_up_with_their_windows(small):
    """Window i must carry recording i's label, not a shuffled one."""
    config, recordings = small
    _, ailment, stress, mood = windows_and_labels(recordings, config)

    per_recording = len(ailment) // len(recordings)
    for i, recording in enumerate(recordings):
        block = slice(i * per_recording, (i + 1) * per_recording)
        assert set(ailment[block]) == {recording.condition_id}
        assert set(stress[block]) == {recording.stress_level_id}
        assert set(mood[block]) == {recording.mood_state_id}


def test_augmentation_changes_only_when_requested(small):
    config, recordings = small
    plain = next(iter(make_dataset(recordings, config)))[0].numpy()
    again = next(iter(make_dataset(recordings, config)))[0].numpy()
    augmented = next(iter(make_dataset(recordings, config, augment_data=True, seed=0)))[
        0
    ].numpy()

    np.testing.assert_array_equal(plain, again)
    assert not np.array_equal(plain, augmented)


def test_shuffle_is_seed_reproducible(small):
    config, recordings = small
    first = next(iter(make_dataset(recordings, config, shuffle=True, seed=7)))[0]
    second = next(iter(make_dataset(recordings, config, shuffle=True, seed=7)))[0]
    np.testing.assert_array_equal(first.numpy(), second.numpy())


def test_standardization_zero_means_each_channel(small):
    config, recordings = small
    features = next(iter(make_dataset(recordings, config)))[0].numpy()
    # Per window, per channel: mean ~0, std ~1.
    np.testing.assert_allclose(features.mean(axis=1), 0.0, atol=1e-4)
    np.testing.assert_allclose(features.std(axis=1), 1.0, atol=1e-3)
