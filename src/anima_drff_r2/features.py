from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal


def prepare_iq_window(iq: np.ndarray, sample_size: int) -> np.ndarray:
    if sample_size <= 0:
        return iq
    if iq.size <= sample_size:
        return iq
    return iq[:sample_size]


def stft_spectrogram(
    iq: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 256,
    log_power: bool = True,
) -> np.ndarray:
    noverlap = max(0, n_fft - hop_length)
    _, _, zxx = signal.stft(
        iq,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=None,
        padded=False,
        return_onesided=False,
    )

    magnitude = np.abs(zxx).astype(np.float32)
    if not log_power:
        return magnitude

    eps = np.float32(1e-12)
    return (10.0 * np.log10(magnitude * magnitude + eps)).astype(np.float32)


def save_feature_npz(path: str | Path, spectrogram: np.ndarray, metadata: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, spectrogram=spectrogram, metadata=json.dumps(metadata, ensure_ascii=True))
    return output
