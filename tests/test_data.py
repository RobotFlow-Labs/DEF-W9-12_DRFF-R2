from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from anima_drff_r2.data import build_manifest_entries, parse_filename_metadata, validate_mat_file


def _write_valid_mat(path: Path) -> None:
    n = 2048
    payload = {
        "RF0_I": np.random.randn(n).astype(np.float32),
        "RF0_Q": np.random.randn(n).astype(np.float32),
        "Fs": np.array([100_000_000.0]),
        "CenterFrequence": np.array([5_745_000_000.0]),
        "Gain": np.array([20.0]),
        "State": np.array(["Hover"]),
        "Distance": np.array([30.0]),
        "Height": np.array([10.0]),
        "FlightMode": np.array(["Cruise"]),
    }
    savemat(path, payload)


def test_parse_filename_metadata() -> None:
    meta = parse_filename_metadata("mavic3_1_Ascend_c17_u1_d2.mat")
    assert meta is not None
    assert meta.model == "mavic3"
    assert meta.unit == 1
    assert meta.state == "Ascend"
    assert meta.c_label == 17
    assert meta.usrp == 1
    assert meta.day == 2


def test_validate_and_manifest(tmp_path: Path) -> None:
    file_path = tmp_path / "mavic3_1_Hover_c1_u1_d1.mat"
    _write_valid_mat(file_path)

    result = validate_mat_file(file_path)
    assert result.valid

    entries = build_manifest_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["valid"] is True
    assert entries[0]["label"] == "Hover"
