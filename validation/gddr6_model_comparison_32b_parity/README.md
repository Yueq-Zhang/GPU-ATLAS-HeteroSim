# GDDR6 model comparison

| Operator | Native RTX 3070 (us) | Accel-Sim internal GDDR6 (us) | Ramulator2 GDDR6 (us) | Ramulator2 vs internal | Internal vs native | Ramulator2 vs native |
|---|---:|---:|---:|---:|---:|---:|
| attention_norm | 100.992 | 51.887 | 51.554 | -0.64% | -48.62% | -48.95% |
| causal_attention | 21.504 | 30.851 | 31.332 | +1.56% | +43.46% | +45.70% |
| down_projection | 96.256 | 91.808 | 229.102 | +149.54% | -4.62% | +138.01% |
| final_norm | 374.784 | 44.147 | 44.193 | +0.10% | -88.22% | -88.21% |
| gate_up_projection | 206.848 | 162.030 | 679.811 | +319.56% | -21.67% | +228.65% |
| lm_head | 351.232 | 337.898 | 1246.103 | +268.78% | -3.80% | +254.78% |
| mlp_norm | 168.224 | 51.887 | 51.554 | -0.64% | -69.16% | -69.35% |
| output_projection | 36.864 | 42.234 | 154.297 | +265.34% | +14.57% | +318.56% |
| qkv_projection | 77.440 | 84.056 | 190.690 | +126.86% | +8.54% | +146.24% |
| residual_add | 22.752 | 5.466 | 6.331 | +15.82% | -75.97% | -72.17% |
| rope | 297.056 | 112.274 | 112.577 | +0.27% | -62.20% | -62.10% |
| sampling | 60.416 | 17.673 | 17.975 | +1.71% | -70.75% | -70.25% |
| silu_multiply | 107.520 | 11.799 | 14.198 | +20.33% | -89.03% | -86.80% |
| token_embedding | 10.464 | 6.402 | 6.378 | -0.37% | -38.82% | -39.05% |

## Summary

- Operators: 14
- Ramulator2 vs Accel-Sim internal GDDR6 mean absolute relative difference: 83.68%
- Accel-Sim internal GDDR6 vs native RTX 3070 MARE: 45.67%
- Ramulator2 GDDR6 vs native RTX 3070 MARE: 119.20%
- Ramulator2 vs internal median absolute relative difference: 8.77%
- Within 15% (Ramulator2 vs internal / internal vs native / Ramulator2 vs native): 7/4/0
- Largest Ramulator2-vs-internal difference: gate_up_projection (319.56%)
- Projection operators (5): Ramulator2 vs internal 226.02%; internal vs native 10.64%.
- Other operators (9): Ramulator2 vs internal 4.61%; internal vs native 65.14%.
- Every Ramulator2 point is a deterministic double run with one timing owner, conserved parent/child requests, zero unmapped addresses, and zero outstanding requests.

## Configuration parity

- Unique address capacity: 4 GiB (internal) versus 4 GiB (Ramulator2).
- Ramulator2 organization: 16 channels, 32 data pins per channel, 6993 MT/s.
- Peak bandwidth: 448.064 GB/s (internal) versus 447.552 GB/s (Ramulator2), a -0.114% difference.
- GPU-to-memory adapter: near-zero latency and effectively unlimited bandwidth; this is not a PCIe/CXL experiment.
- Transaction granularity: 32 B in both modeled memories.

## Interpretation boundary

The GPU core, L1/L2, and GPU NoC are identical across the two simulated legs. The Ramulator2 leg replaces the DRAM controller, scheduler, timing state machine, transaction granularity, and address mapper. Its external link is configured as a near-zero-cost adapter, not as PCIe/CXL. Because Ramulator2 uses OneLevelInterleave while the RTX 3070 Accel-Sim profile uses its native partition-indexing policy, the measured delta is not a pure timing-only delta.

The native column reuses the P17 CUDA-event median measurements. Those measurements synchronize each Python/PyTorch operator iteration and were not collected with locked clocks or a CUPTI-isolated kernel window. These results are diagnostic; they do not establish a performance-qualified model.
