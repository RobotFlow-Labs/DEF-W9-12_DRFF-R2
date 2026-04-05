# ANIMA DRFF-R2

Local ANIMA defense module implementation for DRFF-R2.

## What it does
- Index and validate DRFF-R2 MAT files
- Compute STFT/log-spectrogram features
- Run baseline classification training/evaluation
- Emit reproducible artifacts (manifest, checkpoints, metrics, confusion matrix)

## Quick start
```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev,train]'

anima-drff-r2 index --data-root /path/to/drff-r2 --manifest-out artifacts/manifest.jsonl
anima-drff-r2 features --manifest artifacts/manifest.jsonl --out-dir artifacts/features --limit 16
anima-drff-r2 train --manifest artifacts/manifest.jsonl --config configs/debug.toml --out-dir artifacts/train
anima-drff-r2 eval --manifest artifacts/manifest.jsonl --config configs/debug.toml --out-dir artifacts/train
```

## Notes
- Dataset source DOI: `10.57760/sciencedb.36815`
- Defaults follow paper STFT setup (`n_fft=1024`, `hop=256`)
- CUDA optimization is intentionally deferred to handoff phase (`NEXT_STEPS.md`)
