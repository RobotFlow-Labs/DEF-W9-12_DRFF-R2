from __future__ import annotations

import numpy as np

from anima_drff_r2.features import prepare_iq_window, stft_spectrogram


def test_prepare_iq_window() -> None:
    iq = np.arange(100, dtype=np.float32) + 1j * np.arange(100, dtype=np.float32)
    out = prepare_iq_window(iq, sample_size=32)
    assert out.shape[0] == 32


def test_stft_deterministic() -> None:
    rng = np.random.default_rng(0)
    iq = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)

    a = stft_spectrogram(iq, n_fft=256, hop_length=64, log_power=True)
    b = stft_spectrogram(iq, n_fft=256, hop_length=64, log_power=True)

    assert a.shape == b.shape
    assert np.allclose(a, b)
