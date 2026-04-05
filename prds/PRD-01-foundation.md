# PRD-01 Foundation

## Scope
Project layout, packaging, CLI entrypoint, config presets, and deterministic run defaults.

## Deliverables
- `pyproject.toml`
- package scaffold under `src/anima_drff_r2/`
- `configs/default.toml`, `configs/paper.toml`, `configs/debug.toml`
- smoke-level unit tests

## Done When
`python -m anima_drff_r2.cli --help` works and tests import package successfully.
