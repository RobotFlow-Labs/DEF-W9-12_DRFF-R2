# PRD-02 Data Contract and Indexing

## Scope
Implement MAT schema validation and manifest generation for DRFF-R2 files.

## Required Fields
- I/Q: `RF0_I`, `RF0_Q`
- metadata: `Fs`, `CenterFrequence`, `Gain`, `State`, `Distance`, `Height`, `FlightMode`

## Deliverables
- filename parser for `model_unit_state_c*_u*_d*`
- manifest JSONL writer
- validation report with pass/fail details

## Done When
Indexer processes a directory and emits manifest + validation summary without crash.
