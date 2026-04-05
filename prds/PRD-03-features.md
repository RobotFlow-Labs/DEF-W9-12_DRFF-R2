# PRD-03 Feature Pipeline

## Scope
Compute paper-aligned STFT spectrograms from complex I/Q signals.

## Defaults
- `n_fft=1024`
- `hop_length=256`
- Hann window
- optional log-power output

## Deliverables
- STFT transform utility
- sample-windowing to control memory
- feature export utility (`.npz`)

## Done When
Feature command generates deterministic spectrogram arrays for same seed/input.
