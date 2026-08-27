# M5 Step 1 Accel-Sim qualification

Date: 2026-08-27

Environment: Ubuntu 22.04 on WSL2, GCC 11.4, CUDA 11.8

Backend: Accel-Sim v2.0.0 (`64653015...`) with the pinned compatible GPGPU-Sim dev revision (`e10018b...`)

Trace: official Tesla V100 Rodinia 2.0 Backprop, input 4096, CUDA 9.1 capture

The official trace bundle SHA-256 is:

```text
aeb7de478785856e4ac834d12be0e71ab0df297f43fc02650e2ca90dea66d8b1
```

The native baseline and adapter executions matched exactly:

| Statistic | Native baseline | Adapter |
|---|---:|---:|
| `gpu_tot_sim_cycle` | 14,731 | 14,731 |
| `gpu_tot_sim_insn` | 10,473,824 | 10,473,824 |

At the configured 1.132 GHz GPU core frequency, the adapter reported a total duration of 13,013,250,884 fs. Accel-Sim owns GPU core, cache, NoC and local DRAM timing in this qualification. External Ramulator2 is disabled.

The same trace produced 15,329 cycles with the former v1.3 integration. The v2 baseline is intentionally recorded separately because simulator-version changes can change timing even when instructions and input traces are unchanged.

This result qualifies the total-duration software adapter for the exact pinned v2 QV100 trace/config pair and proves that v2 still reads the legacy `.traceg` format. It does not qualify cross-configuration replay safety, so `replay_safe` remains false. It is not RTX 3070 calibration and is not a coupled GPU+ATLAS result. Reproduce it with the commands in README section 13.
