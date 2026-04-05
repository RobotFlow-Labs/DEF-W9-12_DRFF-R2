# ANIMA Defense Module PRD — DRFF-R2

## 1. Overview
This module operationalizes the **DRFF-R2** UAV RF dataset (arXiv:2603.00106, DOI: `10.57760/sciencedb.36815`) into a reproducible ANIMA-ready pipeline for:
- UAV RF data indexing and validation
- Time-frequency feature generation (STFT)
- Baseline flight-state classification (EfficientNet-style workflow)
- Evaluation artifacts and reproducible reports

The immediate target is a robust local implementation that can be reformatted and performance-optimized on a CUDA server without changing public interfaces.

## 2. Objectives
- Convert raw DRFF-R2 MAT files into a validated manifest with traceable metadata.
- Implement paper-aligned STFT preprocessing (`n_fft=1024`, `hop=256`) with deterministic outputs.
- Provide a baseline supervised training/eval path for operational-state recognition.
- Add strict CLI contracts, config files, and tests for local reliability.
- Prepare explicit CUDA handoff boundaries (backend interface + porting notes).

## 3. Non-Objectives
- Reproducing full paper SOTA or exact published confusion matrix values.
- Final CUDA kernel optimization in this phase.
- Full ROS2 runtime integration in this phase.

## 4. Inputs and Dependencies
- Dataset source: Science Data Bank DOI `10.57760/sciencedb.36815`.
- Local data format: MAT files containing I/Q arrays and metadata fields (`RF0_I`, `RF0_Q`, `Fs`, `CenterFrequence`, `Gain`, `State`, `Distance`, `Height`, `FlightMode`).
- Python runtime: `>=3.10`.
- Package manager: `uv`.

## 5. Functional Requirements
1. **Dataset Scanner**
- Recursively discover `.mat` files under configurable roots.
- Parse file-name metadata (`model`, `unit`, `state`, `c_label`, `usrp`, `day`) when available.
- Validate required MAT variables and produce errors/warnings per file.

2. **Feature Pipeline**
- Load complex baseband from I/Q.
- Compute STFT magnitude/log-magnitude spectrograms with configurable window/hop.
- Optionally export feature tensors and sidecar metadata.

3. **Training Pipeline**
- Build train/val/test splits from manifest.
- Support baseline classifier training with deterministic seed.
- Save model checkpoints and metrics JSON.

4. **Evaluation Pipeline**
- Compute core metrics: accuracy, macro-F1, per-class precision/recall/F1.
- Emit confusion matrix (numpy/json friendly format).
- Produce a markdown benchmark report.

5. **CLI + Config**
- Commands: `index`, `features`, `train`, `eval`.
- TOML configuration profiles: default, paper-aligned, debug.

## 6. Non-Functional Requirements
- Reproducibility: seeded random paths, serialized run configs.
- Traceability: every output links to manifest hash and config.
- Portability: CPU-first execution; optional CUDA path via backend abstraction.
- Testability: unit tests for parser, MAT loader, STFT contracts.

## 7. Architecture
- `src/anima_drff_r2/data.py`: MAT IO, schema validation, manifest generation.
- `src/anima_drff_r2/features.py`: STFT + spectrogram conversion.
- `src/anima_drff_r2/model.py`: baseline classifier factory.
- `src/anima_drff_r2/pipeline.py`: train/eval orchestration.
- `src/anima_drff_r2/backends/*`: backend capability wrappers.
- `src/anima_drff_r2/cli.py`: user-facing command interface.

## 8. Acceptance Criteria
- Can index DRFF-R2 MAT files and generate a manifest JSONL with no crashes.
- Can run feature generation for at least one sample and produce spectrogram output.
- Can run a short debug train/eval pass on synthetic or small subset data.
- Tests pass locally for data and feature contracts.
- CUDA handoff notes are explicit and actionable.

## 9. Risks and Mitigations
- **Dataset access friction**: keep code path independent from downloader; accept local mirror.
- **MAT schema drift**: implement alias-based field resolution + strict validation reports.
- **Large-file memory pressure**: add configurable sample window and chunked preprocessing.
- **Backend drift (CPU/CUDA)**: isolate numerical kernels behind backend boundaries.

## 10. Delivery Plan
- PRD-01: Foundation and project scaffolding
- PRD-02: Data contract and indexing
- PRD-03: STFT feature pipeline
- PRD-04: Baseline train/eval
- PRD-05: CUDA handoff package
