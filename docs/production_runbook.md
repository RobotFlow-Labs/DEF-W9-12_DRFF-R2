# Production Runbook (Local then CUDA)

## Local bring-up
1. `uv venv && source .venv/bin/activate`
2. `uv pip install -e '.[dev,train]'`
3. `anima-drff-r2 index --data-root <drff_root>`
4. `anima-drff-r2 train --manifest artifacts/manifest.jsonl --config configs/debug.toml`
5. `anima-drff-r2 eval --manifest artifacts/manifest.jsonl --config configs/debug.toml`

## CUDA migration checklist
- Keep CLI and manifest schema unchanged.
- Replace feature kernel implementation only behind `features.py` boundary.
- Preserve numeric parity checks against CPU outputs.
