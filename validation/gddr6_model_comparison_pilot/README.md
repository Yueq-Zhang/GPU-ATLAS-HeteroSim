# GDDR6 model comparison

| Operator | Native RTX 3070 (us) | Accel-Sim internal GDDR6 (us) | Ramulator2 GDDR6 (us) | Ramulator2 vs internal | Internal vs native | Ramulator2 vs native |
|---|---:|---:|---:|---:|---:|---:|
| token_embedding | 10.464 | 6.402 | 5.338 | -16.61% | -38.82% | -48.98% |

## Summary

- Operators: 1
- Ramulator2 vs Accel-Sim internal GDDR6 mean absolute relative difference: 16.61%
- Accel-Sim internal GDDR6 vs native RTX 3070 MARE: 38.82%
- Ramulator2 GDDR6 vs native RTX 3070 MARE: 48.98%
- Every Ramulator2 point is a deterministic double run with one timing owner, conserved parent/child requests, zero unmapped addresses, and zero outstanding requests.

## Interpretation boundary

The GPU core, L1/L2, and GPU NoC are identical across the two simulated legs. The Ramulator2 leg replaces the DRAM controller, scheduler, timing state machine, transaction granularity, and address mapper. Its external link is configured as a near-zero-cost adapter, not as PCIe/CXL. Because Ramulator2 uses OneLevelInterleave while the RTX 3070 Accel-Sim profile uses its native partition-indexing policy, the measured delta is not a pure timing-only delta.
