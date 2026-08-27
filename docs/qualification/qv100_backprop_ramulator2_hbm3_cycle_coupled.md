# Accel-Sim + Ramulator2 Cycle-Accurate qualification

Date: 2026-08-27

This qualification replaces Accel-Sim's internal DRAM timing path with one
in-process Ramulator2 instance. Accel-Sim continues to own SM, L1, L2 and NoC
timing. Every request that reaches the GPU memory-controller interface is sent
to Ramulator2 with its original address and `mem_fetch` identity. A read is not
returned to the originating memory partition until the Ramulator2 completion
callback fires.

The Logic Die is disabled in this qualification. Therefore there is one request
initiator (GPU) and no GPU/PIM contention.

## Fixed inputs

- Accel-Sim v2.0.0: `64653015f85fb5664c84a10f48527e8897d289d0`
- GPGPU-Sim: `e10018b67a4b668e7b43f89280cf67624f1df4ff`
- Ramulator2: `3996362187d7f8314936e5ad7560d93b66b6a215`
- GPU model and trace: QV100/SM70, official Rodinia Backprop 4096 trace
- Memory candidate: HBM3, 32 channels, `HBM3_4Gb`, `HBM3_2Gbps`, FR-FCFS,
  OpenRowPolicy, RoBaRaCoCh

The HBM3 file is a functional latency/callback candidate configuration. It has
not yet been calibrated to the physical ATLAS 3D-DRAM stack. In addition, its
organization `DQ`, default `channel_width`, burst/tCK path and transaction-size
path do not currently derive one self-consistent peak bandwidth. The result
therefore qualifies the request/completion bridge only; it does not qualify
bandwidth or authorize a target-hardware performance claim.

## Results

Two independent coupled runs matched exactly:

| Statistic | Run 1 | Run 2 |
|---|---:|---:|
| GPU cycles | 14,700 | 14,700 |
| GPU instructions | 10,473,824 | 10,473,824 |
| Ramulator2 cycles | 11,038 | 11,038 |
| Accepted reads | 63 | 63 |
| Accepted writes | 0 | 0 |
| Completed requests | 63 | 63 |
| Outstanding at exit | 0 | 0 |
| Ramulator2 instances | 1 | 1 |

The same trace with Accel-Sim's original internal DRAM model is 14,731 GPU
cycles. The coupled result is 14,700 cycles. This difference, together with the
63 request/completion callbacks, demonstrates that DRAM completion timing feeds
back into GPU progress; it is not a post-hoc Roofline or bulk-byte estimate.

The coupled trace configuration explicitly overrides Accel-Sim's original
`-dram_latency 100` with zero. Otherwise that fixed DRAM delay would be applied
before each Ramulator2 request and count DRAM latency twice. GPU ROP, cache and
NoC timing remain enabled.

The trace reports 575 aggregate L2 cache misses but only 63 requests reach the
external DRAM interface. Those counters have different scopes: L2 sector/write
allocation activity is not identical to completed memory-controller reads.

## Reproduction

```bash
cd /opt/gpu-atlas/GPU-ATLAS-HeteroSim
bash scripts/build_accel_sim_ramulator2.sh

.venv/bin/python -m frontend.hetero.cli qualify-gpu \
  --backend-config configs/hetero/backends/gpu_accelsim_qv100_ramulator2_hbm3.json \
  --trace-manifest configs/hetero/traces/official_qv100_backprop_4096.json \
  --output /opt/gpu-atlas/qualification/accel-sim-v2/qv100-backprop-4096-ramulator2-hbm3-32ch-no-fixed-dram-latency
```

The command fails if no Ramulator2 statistics are emitted, the number of
instances is not one, a nonzero-request trace sends zero requests, accepted and
completed counts differ, or outstanding requests remain at exit.

## Qualification boundary

Qualified now: GPU-only, in-process request/response timing, one shared
Ramulator2 owner, deterministic replay for this exact trace/config pair.

Not qualified: exact OPT/LLM traces, RTX 3070/4090 target calibration, ATLAS
Logic-Die requests, GPU/PIM arbitration, coherence, atomics, MMU/page walks,
PCIe/CXL timing, or cross-memory-configuration trace replay safety.
