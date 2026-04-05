from __future__ import annotations


def mlx_available() -> bool:
    try:
        import mlx.core as mx  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True
