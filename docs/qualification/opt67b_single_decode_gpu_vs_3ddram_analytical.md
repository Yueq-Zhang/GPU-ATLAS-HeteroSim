# OPT-6.7B single-Decode GPU versus 3D-DRAM analytical comparison

Date: 2026-08-27

## Scope

- Model: OPT-6.7B, 32 layers, FP16
- Workload: BS=1, preloaded KV length 1,024, one `decode_step`
- Graph: zero Prefill passes, one Decode pass, 391 tasks, final KV length 1,025
- Placement: either all tasks on one GPU or all tasks on the ATLAS compute die backed by primary 3D-DRAM
- Data movement: no PCIe/CXL transfer is included because each run is device-local
- Fidelity: analytical Roofline only; `performance_claim_allowed=false`

All three runs execute the same logical graph. It contains 13,838,128,224 FLOPs, 13,838,174,400 read bytes and 4,565,192 write bytes. Of the 391 tasks, 387 non-control tasks are memory-bound under every tested backend.

## Result

The speedup definition is:

```text
3D-DRAM speedup over GPU = GPU latency / ATLAS 3D-DRAM latency
```

| Backend | Compute parameter | Memory-bandwidth parameter | Single Decode latency | 3D-DRAM speedup over this GPU |
|---|---:|---:|---:|---:|
| RTX 3070 Roofline | 81.3 TFLOP/s | 448 GB/s | 30.898972 ms | 0.914286x |
| RTX 4090 Roofline | 330.3 TFLOP/s | 1,008 GB/s | 13.732877 ms | 0.406349x |
| ATLAS + 3D-DRAM reference | 10 TFLOP/s | 409.6 GB/s | 33.795751 ms | baseline |

Therefore, with the currently checked-in parameters:

- RTX 3070 is 1.09375x faster than the ATLAS reference, or the ATLAS latency is 9.375% higher.
- RTX 4090 is 2.46094x faster than the ATLAS reference.
- The 3D-DRAM path does not produce an acceleration in this particular analytical configuration.

## Interpretation and break-even point

The result is controlled by the configured effective bandwidth, not by peak compute. Because the complete Decode graph has approximately one FLOP per byte in this model, increasing ATLAS compute throughput alone does not improve this result.

Under the same ideal Roofline assumptions, the ATLAS effective internal bandwidth must exceed:

- 448 GB/s to outperform the RTX 3070 configuration;
- 1,008 GB/s to outperform the RTX 4090 configuration.

This is a useful DSE boundary, not a hardware conclusion. ATLAS's native simulator models tiling, placement, compute-die execution and Ramulator2 timing, but the repository does not yet contain shape-matched OPT-6.7B ATLAS artifacts or an OPT-6.7B Accel-Sim trace. Consequently this comparison has zero trace coverage, zero artifact coverage and cannot be presented as a cycle-accurate performance claim.

## Reproduction

```bash
python3 -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m8_opt67b_rtx3070_decode_roofline.json \
  --runs-root /opt/gpu-atlas/qualification/opt67b-single-decode-comparison-20260827

python3 -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m8_opt67b_rtx4090_roofline.json \
  --runs-root /opt/gpu-atlas/qualification/opt67b-single-decode-comparison-20260827

python3 -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m8_opt67b_atlas_3ddram_decode_roofline.json \
  --runs-root /opt/gpu-atlas/qualification/opt67b-single-decode-comparison-20260827
```
