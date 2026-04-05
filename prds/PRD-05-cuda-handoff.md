# PRD-05 CUDA Handoff Package

## Scope
Prepare clear backend seams and migration instructions for CUDA server optimization.

## Deliverables
- backend capability module (`cpu`/`cuda` detection)
- interface boundaries around numerically intensive stages
- `NEXT_STEPS.md` with CUDA porting backlog

## Done When
No public API changes required to swap in CUDA-optimized kernels later.
