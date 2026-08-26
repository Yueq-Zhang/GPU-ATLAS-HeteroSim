# M5 Step 1 Accel-Sim qualification

Date: 2026-08-26

Environment: Ubuntu 22.04 on WSL2, GCC 11.4, CUDA 11.8

Backend: Accel-Sim v1.3.0 with GPGPU-Sim v4.2.1

Trace: official Tesla V100 Rodinia 2.0 Backprop, input 4096, CUDA 9.1 capture

The official trace bundle SHA-256 is:

```text
aeb7de478785856e4ac834d12be0e71ab0df297f43fc02650e2ca90dea66d8b1
```

The native baseline and adapter executions matched exactly:

| Statistic | Native baseline | Adapter |
|---|---:|---:|
| `gpu_tot_sim_cycle` | 15,329 | 15,329 |
| `gpu_tot_sim_insn` | 10,473,824 | 10,473,824 |

At the configured 1.132 GHz GPU core frequency, the adapter reported a total duration of 13,541,519,435 fs. Accel-Sim owns GPU core, cache, NoC and local DRAM timing in this qualification. External Ramulator2 is disabled.

This result qualifies the total-duration software adapter for the exact pinned QV100 trace/config pair. It does not qualify cross-configuration replay safety, so `replay_safe` remains false. It is not RTX 3070 calibration and is not a coupled GPU+ATLAS result. Reproduce it with the commands in README section 13.
