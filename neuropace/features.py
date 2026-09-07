"""Spectral features for the classical baseline.

Band edges come from the shared FREQUENCY_BANDS constant rather than being
retyped as literals, which is how the old xgb2.py drifted out of agreement
with the generator that produced its data.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

from .config import FREQUENCY_BANDS

FEATURE_NAMES: tuple[str, ...] = (
    *(f"{band}_power" for band in FREQUENCY_BANDS),
    "mean",
    "std",
    "spectral_entropy",
)


def band_powers(window: np.ndarray, sampling_rate: int) -> dict[str, float]:
    """Mean power spectral density within each band, averaged over channels.

    window is (samples, channels).
    """
    frequencies, psd = welch(
        window, fs=sampling_rate, axis=0, nperseg=min(256, window.shape[0])
    )
    psd = psd.mean(axis=-1)  # average across channels

    powers = {}
    for band, (low, high) in FREQUENCY_BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        powers[band] = float(psd[mask].sum()) if mask.any() else 0.0
    return powers


def extract_features(windows: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Feature matrix for a stack of windows.

    windows is (n_windows, samples, channels); returns (n_windows, n_features)
    in FEATURE_NAMES order.
    """
    rows = []
    for window in windows:
        powers = band_powers(window, sampling_rate)

        frequencies, psd = welch(
            window, fs=sampling_rate, axis=0, nperseg=min(256, window.shape[0])
        )
        mean_psd = psd.mean(axis=-1)
        total = mean_psd.sum()
        spectral_entropy = float(entropy(mean_psd / total)) if total > 0 else 0.0

        rows.append(
            [
                *(powers[band] for band in FREQUENCY_BANDS),
                float(window.mean()),
                float(window.std()),
                spectral_entropy,
            ]
        )

    return np.asarray(rows, dtype=np.float64)
