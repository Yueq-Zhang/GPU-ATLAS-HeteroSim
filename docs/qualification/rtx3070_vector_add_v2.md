# Accel-Sim 2.0 local RTX 3070 qualification

Date: 2026-08-27

Environment: Ubuntu 22.04 on WSL2, CUDA 11.8, NVIDIA GeForce RTX 3070, driver 591.86

Backend: Accel-Sim v2.0.0 (`64653015...`), pinned compatible GPGPU-Sim dev (`e10018b...`) and public NVBit 1.8

Workload: FP32 vector addition, 4,096 elements, compiled for SM86

NVBit 1.8 captured the workload successfully and the v2 postprocessor generated a zstd-compressed `.tracez` plus `kernelslist.g`. The native baseline and project adapter executions matched exactly:

| Statistic | Native baseline | Adapter |
|---|---:|---:|
| `gpu_tot_sim_cycle` | 5,657 | 5,657 |
| `gpu_tot_sim_insn` | 61,440 | 61,440 |

At the currently configured 1.132 GHz simulation frequency, the adapter reported 4,997,349,824 fs. This frequency is an integration configuration and has not been calibrated against the physical RTX 3070.

This record proves that the previous NVBit 1.7.3/driver 591.86 capture failure is resolved by the v2/NVBit 1.8 toolchain, and that the adapter preserves the pinned simulator result. It does not qualify LLM kernels, cross-DRAM replay safety, microarchitectural correlation, or coupled GPU+ATLAS timing.
