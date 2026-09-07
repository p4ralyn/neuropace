import numpy as np

from neuropace.config import EEGConfig
from neuropace.dataset import make_windows, split_indices
from neuropace.synthetic import generate_dataset


def test_window_count_and_stride():
    config = EEGConfig(
        sampling_rate=10,
        channels=2,
        recording_duration=5.0,
        window_size=2.0,
        window_overlap=0.5,
    )
    recording = generate_dataset(
        config, n_per_condition=1, rng=np.random.default_rng(0)
    )[0]
    windows = make_windows(recording, config)

    # 50 samples, 20-sample windows, stride 10 -> starts at 0,10,20,30 = 4
    assert windows.shape == (4, 20, 2)
    np.testing.assert_array_equal(windows[1], recording.data[10:30])


def test_window_shorter_than_recording_yields_nothing():
    config = EEGConfig(
        sampling_rate=10, channels=2, recording_duration=1.0, window_size=1.0
    )
    recording = generate_dataset(
        config, n_per_condition=1, rng=np.random.default_rng(0)
    )[0]
    recording.data = recording.data[:5]  # shorter than one window
    assert make_windows(recording, config).shape[0] == 0


def test_splits_are_pairwise_disjoint_and_cover_everything():
    config = EEGConfig(channels=2, recording_duration=2.0, window_size=1.0)
    recordings = generate_dataset(
        config, n_per_condition=10, rng=np.random.default_rng(0)
    )
    train, val, test = split_indices(recordings, seed=42)

    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert sorted(train + val + test) == list(range(len(recordings)))


def test_split_is_stratified_by_condition():
    config = EEGConfig(channels=2, recording_duration=2.0, window_size=1.0)
    recordings = generate_dataset(
        config, n_per_condition=10, rng=np.random.default_rng(0)
    )
    train, _, _ = split_indices(recordings, seed=42)
    assert len({recordings[i].condition_id for i in train}) == 6


def test_ratios_must_sum_to_one():
    config = EEGConfig(channels=2, recording_duration=2.0, window_size=1.0)
    recordings = generate_dataset(
        config, n_per_condition=2, rng=np.random.default_rng(0)
    )
    try:
        split_indices(recordings, train=0.5, val=0.3, test=0.3)
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_too_few_recordings_to_stratify_explains_itself():
    config = EEGConfig(channels=2, recording_duration=2.0, window_size=1.0)
    recordings = generate_dataset(
        config, n_per_condition=2, rng=np.random.default_rng(0)
    )
    try:
        split_indices(recordings)
    except ValueError as exc:
        assert "too few to stratify" in str(exc)
        assert "Generate at least" in str(exc)
    else:
        raise AssertionError("expected ValueError")
