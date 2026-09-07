from neuropace.config import (
    AILMENT_CLASSES,
    MOOD_STATES,
    STRESS_LEVELS,
    EEGConfig,
)


def test_window_geometry():
    config = EEGConfig(sampling_rate=250, window_size=5.0, window_overlap=0.5)
    assert config.samples_per_window == 1250
    assert config.window_step == 625


def test_class_counts_match_label_lists():
    config = EEGConfig()
    assert config.num_ailments == len(AILMENT_CLASSES) == 6
    assert config.num_stress_levels == len(STRESS_LEVELS) == 5
    assert config.num_moods == len(MOOD_STATES) == 10


def test_roundtrips_through_dict():
    config = EEGConfig(channels=8, batch_size=4)
    assert EEGConfig.from_dict(config.to_dict()) == config
