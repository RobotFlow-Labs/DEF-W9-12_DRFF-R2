from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class FeatureConfig:
    n_fft: int = 1024
    hop_length: int = 256
    sample_size: int = 32768
    log_power: bool = True
    image_size: int = 224


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 7
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-3
    arch: str = "smallcnn"
    pretrained: bool = False
    train_split: float = 0.8
    val_split: float = 0.1
    num_workers: int = 0


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"


@dataclass(frozen=True)
class AppConfig:
    features: FeatureConfig = FeatureConfig()
    train: TrainConfig = TrainConfig()
    runtime: RuntimeConfig = RuntimeConfig()


DEFAULTS = AppConfig()


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        return {}
    return value


def load_config(path: Path | str | None) -> AppConfig:
    if path is None:
        return DEFAULTS

    cfg_path = Path(path)
    with cfg_path.open("rb") as fh:
        data = tomllib.load(fh)

    f = _section(data, "features")
    t = _section(data, "train")
    r = _section(data, "runtime")

    return AppConfig(
        features=FeatureConfig(
            n_fft=int(f.get("n_fft", DEFAULTS.features.n_fft)),
            hop_length=int(f.get("hop_length", DEFAULTS.features.hop_length)),
            sample_size=int(f.get("sample_size", DEFAULTS.features.sample_size)),
            log_power=bool(f.get("log_power", DEFAULTS.features.log_power)),
            image_size=int(f.get("image_size", DEFAULTS.features.image_size)),
        ),
        train=TrainConfig(
            seed=int(t.get("seed", DEFAULTS.train.seed)),
            epochs=int(t.get("epochs", DEFAULTS.train.epochs)),
            batch_size=int(t.get("batch_size", DEFAULTS.train.batch_size)),
            learning_rate=float(t.get("learning_rate", DEFAULTS.train.learning_rate)),
            arch=str(t.get("arch", DEFAULTS.train.arch)),
            pretrained=bool(t.get("pretrained", DEFAULTS.train.pretrained)),
            train_split=float(t.get("train_split", DEFAULTS.train.train_split)),
            val_split=float(t.get("val_split", DEFAULTS.train.val_split)),
            num_workers=int(t.get("num_workers", DEFAULTS.train.num_workers)),
        ),
        runtime=RuntimeConfig(device=str(r.get("device", DEFAULTS.runtime.device))),
    )
