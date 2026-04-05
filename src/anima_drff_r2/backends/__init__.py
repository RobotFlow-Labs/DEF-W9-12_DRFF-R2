from __future__ import annotations

from .cuda import cuda_available
from .mlx import mlx_available


def resolve_device(requested: str = "auto") -> str:
    req = requested.lower()
    if req in {"cpu", "cuda"}:
        if req == "cuda" and not cuda_available():
            return "cpu"
        return req

    if cuda_available():
        return "cuda"
    return "cpu"


__all__ = ["resolve_device", "cuda_available", "mlx_available"]
