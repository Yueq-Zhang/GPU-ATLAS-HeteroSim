# P12 one-layer GPU Prefill qualification

Date: 2026-08-28

Fixed case: TinyLlama-1.1B FP16, BS=1, Context=16, one layer, GPU-only Model 3 shared 3D-DRAM.

Result:

- 20 device tasks, 0 cross-device routes;
- 378 GPU Parent requests accepted and completed;
- 0 ATLAS Parent requests;
- one Ramulator2 instance and 0 outstanding requests;
- 28 non-overlapping Global PA ranges using 351,615,554 B;
- 100% task coverage and 0 analytical fallback;
- unqualified makespan 36,440,000,000 fs.

Simulation key: `a1a027da4ac1567bb2d9b52aec4f38558ac7360b6573431114078f3d7cf6c6cb`.

Two independent runs produce identical seven-artifact hashes. This stage validates one-layer lifecycle and the GPU-only no-Logic-Die-contention invariant, not calibrated latency.
