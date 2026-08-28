# Implementation status

## 2026-08-28 — v0.7.0

- Design contract: v1.5; the four system profiles and the layered external/internal memory path remain frozen.
- Software stage: M0–M8 reference infrastructure operational. P1–P7 cycle-coupled memory-path work is implemented and qualified by focused tests. P8 has one exact TinyLlama operator, not an end-to-end model.
- Qualification stage: local SM86 trace capture, native RTX 3070 Accel-Sim, GPU plus one shared Ramulator2, ATLAS native operator execution, and the dual-initiator memory-port path have passed deterministic checks.

### Completed since v0.6.2

1. **P1 — exact bandwidth contract**
   - External request/response payload bandwidth is independent from internal DRAM bandwidth.
   - Schema validation closes `DQ`, channel width, rate, transfers/clock, `tCK`, burst cycles, prefetch, transaction bytes and peak bandwidth exactly.
   - Current ATLAS-style target: external direct-memory PHY 12.8 GB/s; internal 16 × 512-bit × 400 MT/s = 409.6 GB/s.

2. **P2/P3 — Bridge ABI v2 and Logic-Die Gateway**
   - Parent ID, Global PA, size, byte/sector mask, partition, ordering domain, QoS and initiator are carried explicitly.
   - Parent requests split into aligned 64B children; retries, children and parent completion are conserved.
   - Reads return only after all children and the response link. Writes default to durable completion.
   - GPGPU-Sim 32B sector requests are normalized from the retained 128B-line masks before splitting.

3. **P4/P5 — bidirectional links and clock domains**
   - Request and response directions separately account payload bytes, headers, flits, credits, serialization, propagation and duplex mode.
   - GPU, link, gateway and DRAM clocks advance with integer-femtosecond phases.

4. **P6 — GPU-only layered qualification**
   - External-link-limited case: 346 DRAM/link cycles and 1,038 GPU cycles.
   - Internal-DRAM-limited case: 163 DRAM/link cycles and 489 GPU cycles.
   - Both cases pass parent/child, payload/wire byte, durable completion, one-owner and zero-inflight checks.

5. **P7 — ATLAS internal port and contention**
   - `AtlasHbPort` consumes native `atlasim::ComponentInput` and mirrors `HBFrontend` tile traversal, address alignment, read/write generation and mapper sorting.
   - GPU uses the external port; ATLAS uses the internal Hybrid-Bond port; both share the same Gateway, mapper and Ramulator2.
   - GPU-only / ATLAS-only / concurrent cycles are 163 / 90 / 239. Both initiators observe a longer completion time under concurrency.
   - This qualifies the ATLAS memory-port contract, not concurrent execution of a complete `atlasim.Chip` with Accel-Sim.

6. **P8 — first exact LLM operator**
   - Exact checkpoint: TinyLlama‑1.1B, revision `fe8a4e...`, layer-0 `q_proj`, FP16, BS=1, initial KV=1024, `M=1,K=2048,N=2048`.
   - NVBit 1.8 dynamic kernels 4–5 produce a non-empty 6.1 MiB SM86 trace with WMMA GEMM and Split-K reduction.
   - Native RTX 3070: 36,324 cycles, 15,908,352 instructions, 32.088339 µs.
   - RTX 3070 + layered shared 3D-DRAM: 1,498,113 cycles, 1,323.421378 µs; 262,272 reads completed; one Ramulator2; zero outstanding.
   - ATLAS 16-core N-column-sharded artifact: 24,613 cycles, 24.613 µs; 8,916,992 DRAM request bytes.
   - All three paths require exact double-run equality. Cross-configuration trace replay safety remains false.

### Existing reference infrastructure

- Complete decoder-only logical graphs for Prefill, Decode, KV append, LM head and sampling;
- Llama/SwiGLU and OPT/Dense-GELU model forms;
- Static Ragged, Continuous/Chunked Prefill and device sub-batches;
- runtime memory planning, paged KV lifecycle, residency and four topology lowerings;
- bounded PCIe/CXL reference links, event-modeled shared memory and feedback to the global DAG;
- TraceAddr → TensorID+offset → Global PA → candidate-specific DRAM tuple separation;
- independent Accel-Sim/ATLAS adapters, run provenance, trace/artifact caches and outer-loop DSE;
- four executable system profiles and analytical OPT-6.7B reference configurations.

### Current qualification records

```text
/opt/gpu-atlas/qualification/gpu-only-layered-memory-path-20260828-final/qualification_record.json
/opt/gpu-atlas/qualification/dual-initiator-memory-path-20260828-final/qualification_record.json
/opt/gpu-atlas/qualification/accel-sim-v2/rtx3070-tinyllama11b-qproj-decode-ctx1024/qualification_record.json
/opt/gpu-atlas/qualification/accel-sim-v2/rtx3070-tinyllama11b-qproj-decode-ctx1024-shared-hbdram-v2stats/qualification_record.json
/opt/gpu-atlas/qualification/atlas/tinyllama11b-qproj-decode-bs1-ctx1024-edge16/qualification_record.json
```

### Remaining gaps

1. Run a full `atlasim.Chip` scheduler and Accel-Sim concurrently against the one shared Ramulator2 service; the current P7 qualification stops at the native ATLAS memory-port contract.
2. Capture and lower the remaining Attention, KV and MLP operators, then assemble all 22 TinyLlama layers into Prefill and Decode task graphs.
3. Connect the existing Continuous/Ragged multi-Batch planner to real cycle backends and validate fused/batched kernel shapes instead of event-level reference durations.
4. Implement data residency, copies/fences and versioning for operator placement changes between GPU and Logic Die in the cycle path.
5. Calibrate the RTX 3070 microarchitecture and target 3D-DRAM/link parameters against hardware or published measurements. RTX 4090 remains unqualified.
6. Add longer-running mixed GPU/ATLAS contention workloads, write traffic, fairness/QoS studies and deadlock/liveness stress tests.

### Claim boundary

The new evidence is a qualified, shape-matched **single-operator component comparison**. It is not a measured-hardware result, a full layer, end-to-end TinyLlama latency, multi-Batch throughput, or a proof that a fixed trace is safe for arbitrary memory configurations. Analytical/reference runs continue to set `performance_claim_allowed=false`.
