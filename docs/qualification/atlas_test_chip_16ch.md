# ATLAS test-chip adapter qualification

Validated on 2026-08-26 in WSL Ubuntu 22.04.

- ATLAS repository: `pku-gsun/ATLAS-MICRO-2026`
- ATLAS commit: `b2787399408e32d327c820daee96d4e6610f551a`
- Chip: `configs/architecture/chip/test_chip_16ch.yaml`, 1000 MHz
- Operator list: `configs/operator_yaml/gemm_comp/gemm.yaml`
- Placement map: `configs/operator_yaml/gemm_comp/gemm_data.yaml`
- Timing contract: `total`; ATLAS owns core, SRAM, NoC, and internal 3D-DRAM; no external memory requests are exported.

Two independent adapter invocations matched exactly:

| Statistic | Baseline | Adapter |
|---|---:|---:|
| E2E cycles | 48,446 | 48,446 |
| DRAM cycles | 48,446 | 48,446 |
| Matrix cycles | 8,192 | 8,192 |
| E2E energy | 0.00581352 J | 0.00581352 J |

The full machine-readable record is generated at
`/opt/gpu-atlas/qualification/atlas_test_chip_16ch/qualification_record.json`
by the `qualify-atlas` command documented in the main README.

This qualifies deterministic adapter equivalence for the exact prepared bundle. It does not qualify that bundle as a compatible performance model for the Tiny LLM node used by the operator-event plumbing probe.
