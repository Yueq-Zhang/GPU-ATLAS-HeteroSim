# P14 TinyLlama-1.1B BS=1 Context=1024 Prefill deployment

Date: 2026-08-28

## Fixed workload

- model: TinyLlama-1.1B, decoder-only Llama/SwiGLU, 22 layers;
- datatype: FP16;
- batch: 1;
- prompt/context: 1024 tokens;
- output length: 1, so the run performs Prefill and produces the first token without a Decode forward;
- topology: Model 3, GPU uses the shared 3D-DRAM as native memory;
- placement: all operators on GPU, Logic Die compute disabled;
- memory: 16-channel HBDRAM, 64 B transactions, `OneLevelInterleave(channel_lowest_bit=0)`, one live Ramulator2;
- address mode: allocated Global PA, `identity_untranslated`, no MMU/TLB or configurable XOR hashing.

## Deployment result

| Invariant | Result |
|---|---:|
| Device tasks / routes | 272 / 0 |
| Operator classes | 19 |
| Embedding / KV Append / Final Norm / LM Head / Sampling | 1 / 22 / 1 / 1 / 1 |
| Final committed KV length | 1024 |
| Global PA ranges | 448 |
| Allocated / capacity | 3,957,580,290 / 4,294,967,296 B |
| GPU / ATLAS Parent requests | 3,385 / 0 |
| Completed / outstanding | 3,385 / 0 |
| Ramulator2 instances | 1 |
| Cycle-contract coverage / analytical fallback | 100% / 0 |
| Represented logical read+write bytes | 5,922,679,810 B |
| Simulated sampled payload bytes | 216,578 B |
| Recorded deployment makespan | 26,644,550,000,000 fs |

Simulation key: `b209201dc209cba9916e170a27ae33d35f5f8feb3669dd5ab1f6e14f836c35cd`.

Two independent runs are byte-identical. Run-1 hashes include:

- `metrics.json`: `44a76d3792f9e2009d7c789ab6dc7343a78753048c6e39704ff3b1d63c3a38a4`;
- `memory_statistics.json`: `b5ab5777afa49f2d74196f57aa3941279d501e6822d2ffba462b7667a7871baf`;
- `request_cycle_trace.json`: `449a324a10647e41cdc91722e3b93eea52ba4d504f1cde555c6e83ef5c3d24fa`.

Reproduce all P10b-B through P14 checks with:

```bash
bash scripts/qualify_prefill_p10b_to_p14.sh \
  /opt/gpu-atlas/GPU-ATLAS-HeteroSim \
  /opt/gpu-atlas/qualification/prefill-p10b-to-p14-final
```

## Mandatory interpretation

This is a complete **deployment and causal execution** qualification, not a performance qualification. It is not Roofline: tasks, memory parents, completion callbacks, link/gateway clocks, DRAM commands, dependencies and version commits advance on the live cycle timeline. However, compute uses deterministic tiled cycle contracts rather than per-operator Accel-Sim instruction traces or full ATLAS compiled artifacts, and memory uses at most four evenly spaced representatives per value rather than the full access stream.

Therefore `compute_fidelity=tiled_cycle_contract_unqualified`, `memory_fidelity=live_ramulator2_sampled_requests`, `trace_coverage=0`, `extrapolated_fraction=1.0` and `performance_claim_allowed=false` are required. The recorded 26.64455 ms must not be presented as measured or calibrated TinyLlama latency, GPU/ATLAS speedup, or a publication result.
