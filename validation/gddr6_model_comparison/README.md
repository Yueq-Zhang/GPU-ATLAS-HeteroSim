# GDDR6 model comparison

| Operator | Native RTX 3070 (us) | Accel-Sim internal GDDR6 (us) | Ramulator2 GDDR6 (us) | Ramulator2 vs internal | Internal vs native | Ramulator2 vs native |
|---|---:|---:|---:|---:|---:|---:|
| attention_norm | 100.992 | 51.887 | 49.320 | -4.95% | -48.62% | -51.16% |
| causal_attention | 21.504 | 30.851 | 27.701 | -10.21% | +43.46% | +28.82% |
| down_projection | 96.256 | 91.808 | 336.111 | +266.10% | -4.62% | +249.18% |
| final_norm | 374.784 | 44.147 | 42.846 | -2.95% | -88.22% | -88.57% |
| gate_up_projection | 206.848 | 162.030 | 385.107 | +137.68% | -21.67% | +86.18% |
| lm_head | 351.232 | 337.898 | 871.005 | +157.77% | -3.80% | +147.99% |
| mlp_norm | 168.224 | 51.887 | 49.320 | -4.95% | -69.16% | -70.68% |
| output_projection | 36.864 | 42.234 | 71.086 | +68.31% | +14.57% | +92.83% |
| qkv_projection | 77.440 | 84.056 | 125.606 | +49.43% | +8.54% | +62.20% |
| residual_add | 22.752 | 5.466 | 5.624 | +2.88% | -75.97% | -75.28% |
| rope | 297.056 | 112.274 | 110.238 | -1.81% | -62.20% | -62.89% |
| sampling | 60.416 | 17.673 | 14.498 | -17.96% | -70.75% | -76.00% |
| silu_multiply | 107.520 | 11.799 | 12.399 | +5.08% | -89.03% | -88.47% |
| token_embedding | 10.464 | 6.402 | 5.338 | -16.61% | -38.82% | -48.98% |

## Summary

- Operators: 14
- Ramulator2 vs Accel-Sim internal GDDR6 mean absolute relative difference: 53.34%
- Accel-Sim internal GDDR6 vs native RTX 3070 MARE: 45.67%
- Ramulator2 GDDR6 vs native RTX 3070 MARE: 87.80%
- Ramulator2 vs internal median absolute relative difference: 13.41%
- Within 15% (Ramulator2 vs internal / internal vs native / Ramulator2 vs native): 7/4/0
- Largest Ramulator2-vs-internal difference: down_projection (266.10%)
- Projection operators (5): Ramulator2 vs internal 135.86%; internal vs native 10.64%.
- Other operators (9): Ramulator2 vs internal 7.49%; internal vs native 65.14%.
- Every Ramulator2 point is a deterministic double run with one timing owner, conserved parent/child requests, zero unmapped addresses, and zero outstanding requests.

## Configuration parity

- Capacity: 8 GiB in both modeled memories.
- Organization: 16 channels and 16 data pins per channel.
- Peak bandwidth: 448.064 GB/s (internal) versus 447.552 GB/s (Ramulator2), a -0.114% difference.
- GPU-to-memory adapter: near-zero latency and effectively unlimited bandwidth; this is not a PCIe/CXL experiment.
- Known non-parity: the internal model transfers a 32 B GDDR burst, whereas this Ramulator2 GDDR6 implementation reports a 16 B transaction. The bridge therefore splits each 32 B request.

## Interpretation boundary

The GPU core, L1/L2, and GPU NoC are identical across the two simulated legs. The Ramulator2 leg replaces the DRAM controller, scheduler, timing state machine, transaction granularity, and address mapper. Its external link is configured as a near-zero-cost adapter, not as PCIe/CXL. Because Ramulator2 uses OneLevelInterleave while the RTX 3070 Accel-Sim profile uses its native partition-indexing policy, the measured delta is not a pure timing-only delta.

The native column reuses the P17 CUDA-event median measurements. Those measurements synchronize each Python/PyTorch operator iteration and were not collected with locked clocks or a CUPTI-isolated kernel window. These results are diagnostic; they do not establish a performance-qualified model.
