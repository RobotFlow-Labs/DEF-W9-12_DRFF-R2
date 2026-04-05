from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, whosmat


FILENAME_RE = re.compile(
    r"^(?P<model>.+?)_(?P<unit>\d+)_(?P<state>[A-Za-z]+)_c(?P<c_label>\d+)_u(?P<usrp>\d+)_d(?P<day>\d+)$"
)

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "RF0_I": ("RF0_I", "RF0I", "RF0 I"),
    "RF0_Q": ("RF0_Q", "RF0Q", "RF0 Q"),
    "Fs": ("Fs", "fs", "SamplingRate"),
    "CenterFrequence": ("CenterFrequence", "CenterFrequency", "Fc"),
    "Gain": ("Gain", "gain"),
    "State": ("State", "state"),
    "Distance": ("Distance", "distance"),
    "Height": ("Height", "height", "Altitude"),
    "FlightMode": ("FlightMode", "flight_mode", "Mode"),
}

REQUIRED_FIELDS = tuple(FIELD_ALIASES.keys())


@dataclass(frozen=True)
class FilenameMeta:
    model: str
    unit: int
    state: str
    c_label: int
    usrp: int
    day: int


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    filename_meta: FilenameMeta | None


@dataclass(frozen=True)
class MatRecord:
    path: Path
    iq: np.ndarray
    fs: float | None
    center_frequence: float | None
    gain: float | None
    state: str | None
    distance: float | None
    height: float | None
    flight_mode: str | None


class DataContractError(RuntimeError):
    pass


def parse_filename_metadata(path: str | Path) -> FilenameMeta | None:
    stem = Path(path).stem
    match = FILENAME_RE.match(stem)
    if not match:
        return None

    groups = match.groupdict()
    return FilenameMeta(
        model=groups["model"],
        unit=int(groups["unit"]),
        state=groups["state"],
        c_label=int(groups["c_label"]),
        usrp=int(groups["usrp"]),
        day=int(groups["day"]),
    )


def _find_alias_key(name_set: set[str], canonical: str) -> str | None:
    for alias in FIELD_ALIASES[canonical]:
        if alias in name_set:
            return alias
    return None


def validate_mat_file(path: str | Path) -> ValidationResult:
    mat_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []

    filename_meta = parse_filename_metadata(mat_path)
    if filename_meta is None:
        warnings.append("filename does not match DRFF-R2 naming convention")

    try:
        variables = whosmat(str(mat_path))
        names = {item[0] for item in variables}
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            path=mat_path,
            valid=False,
            errors=(f"unable to read MAT header: {exc}",),
            warnings=tuple(warnings),
            filename_meta=filename_meta,
        )

    for canonical in REQUIRED_FIELDS:
        if _find_alias_key(names, canonical) is None:
            errors.append(f"missing field: {canonical}")

    return ValidationResult(
        path=mat_path,
        valid=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
        filename_meta=filename_meta,
    )


def _resolve_value(data: dict[str, Any], canonical: str) -> Any:
    for alias in FIELD_ALIASES[canonical]:
        if alias in data:
            return data[alias]
    raise DataContractError(f"field not found: {canonical}")


def _to_scalar(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    return float(array.reshape(-1)[0])


def _to_string(value: Any) -> str | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    if array.dtype.kind in {"U", "S"}:
        return str(array.reshape(-1)[0])
    flat = array.reshape(-1)
    if flat.size == 1:
        return str(flat[0])
    return " ".join(str(item) for item in flat)


def load_mat_record(path: str | Path) -> MatRecord:
    mat_path = Path(path)
    data = loadmat(str(mat_path), squeeze_me=True)

    i_raw = np.asarray(_resolve_value(data, "RF0_I")).astype(np.float32).reshape(-1)
    q_raw = np.asarray(_resolve_value(data, "RF0_Q")).astype(np.float32).reshape(-1)

    if i_raw.size == 0 or q_raw.size == 0:
        raise DataContractError(f"empty RF arrays in {mat_path}")

    if i_raw.size != q_raw.size:
        length = min(i_raw.size, q_raw.size)
        i_raw = i_raw[:length]
        q_raw = q_raw[:length]

    iq = i_raw + 1j * q_raw

    return MatRecord(
        path=mat_path,
        iq=iq,
        fs=_to_scalar(_resolve_value(data, "Fs")),
        center_frequence=_to_scalar(_resolve_value(data, "CenterFrequence")),
        gain=_to_scalar(_resolve_value(data, "Gain")),
        state=_to_string(_resolve_value(data, "State")),
        distance=_to_scalar(_resolve_value(data, "Distance")),
        height=_to_scalar(_resolve_value(data, "Height")),
        flight_mode=_to_string(_resolve_value(data, "FlightMode")),
    )


def scan_mat_files(data_root: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(data_root)
    pattern = "**/*.mat" if recursive else "*.mat"
    return sorted(root.glob(pattern))


def build_manifest_entries(data_root: str | Path, recursive: bool = True) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for mat_path in scan_mat_files(data_root, recursive=recursive):
        result = validate_mat_file(mat_path)
        meta = result.filename_meta
        entry = {
            "path": str(mat_path.resolve()),
            "valid": result.valid,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "label": meta.state if meta else None,
            "filename_meta": {
                "model": meta.model,
                "unit": meta.unit,
                "state": meta.state,
                "c_label": meta.c_label,
                "usrp": meta.usrp,
                "day": meta.day,
            }
            if meta
            else None,
        }
        entries.append(entry)
    return entries


def write_manifest_jsonl(entries: list[dict[str, Any]], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=True) + "\n")
    return out


def read_manifest_jsonl(path: str | Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries
