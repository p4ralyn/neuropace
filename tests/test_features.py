import numpy as np

from neuropace.features import FEATURE_NAMES, band_powers, extract_features


def test_band_power_finds_the_injected_frequency():
    """A pure 10 Hz sine must put its power in alpha (8-13 Hz), not elsewhere."""
    fs = 250
    t = np.arange(0, 4, 1 / fs)
    signal = np.sin(2 * np.pi * 10 * t)[:, None]  # one channel

    powers = band_powers(signal, fs)

    assert powers["alpha"] == max(powers.values())
    assert powers["alpha"] > 10 * powers["gamma"]


def test_band_power_tracks_a_moved_frequency():
    """Move the tone to 2 Hz and the dominant band must move to delta."""
    fs = 250
    t = np.arange(0, 4, 1 / fs)
    signal = np.sin(2 * np.pi * 2 * t)[:, None]

    powers = band_powers(signal, fs)

    assert powers["delta"] == max(powers.values())


def test_extract_features_shape_matches_names():
    windows = np.random.default_rng(0).normal(size=(7, 500, 3))
    features = extract_features(windows, 250)

    assert features.shape == (7, len(FEATURE_NAMES))
    assert np.isfinite(features).all()


def test_features_separate_conditions_that_differ_by_band():
    """Deep-sleep-like (delta-heavy) windows must be separable from beta-heavy.

    If the extractor could not tell these apart, the baseline built on it would
    be measuring nothing.
    """
    fs = 250
    t = np.arange(0, 2, 1 / fs)
    slow = np.stack([np.sin(2 * np.pi * 2 * t)] * 4, axis=-1)[None]
    fast = np.stack([np.sin(2 * np.pi * 20 * t)] * 4, axis=-1)[None]

    slow_features = extract_features(slow, fs)[0]
    fast_features = extract_features(fast, fs)[0]

    delta = FEATURE_NAMES.index("delta_power")
    beta = FEATURE_NAMES.index("beta_power")

    assert slow_features[delta] > fast_features[delta]
    assert fast_features[beta] > slow_features[beta]
