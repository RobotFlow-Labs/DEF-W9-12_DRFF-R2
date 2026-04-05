# Next Steps (CUDA Server Handoff)

1. Replace CPU STFT in `src/anima_drff_r2/features.py` with batched GPU STFT.
2. Add pinned-memory DataLoader + async prefetch in `src/anima_drff_r2/pipeline.py`.
3. Add AMP (`torch.cuda.amp`) and benchmark throughput across batch sizes.
4. Introduce fused preprocessing kernels while preserving output parity against CPU path.
5. Add regression tests comparing CPU/CUDA metrics and confusion matrix drift bounds.
