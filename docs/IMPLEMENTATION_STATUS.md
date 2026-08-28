# Implementation status

## 2026-08-28 — v0.15.0

- Design contract: v1.14. P10b-B through P14 are implemented as a deterministic Prefill deployment path.
- `prefill_cycle` owns one global cycle timeline. GPU external requests, ATLAS internal requests and route acquire probes enter exactly one live Ramulator2; device outputs commit only after sampled reads, explicit tiled compute cycles and sampled durable writes finish.
- P11 materializes all Prefill parameters and tensors and covers 19 operator classes on both device catalogs with no analytical fallback.
- P12 qualifies one-layer GPU-only Prefill; P13 scales it to all 22 TinyLlama layers; P14 deploys TinyLlama-1.1B FP16, BS=1, Context=1024 through first-token sampling and request release.
- P14 has 272 tasks, 448 non-overlapping Global PA ranges, 3,385/3,385 completed GPU parents, 0 ATLAS parents, one Ramulator2 and zero outstanding. Final KV length is 1024.
- The deployment remains performance-unqualified: tiled cycle contracts are not instruction traces and bounded memory samples are not full traffic. `performance_claim_allowed=false` is mandatory.
- Ramulator2 is an active live timing owner. BookSim2 is source-pinned and compiled into the ATLAS library, but the qualified P9a/P9b Chip config has no `architecture.noc` and P14 does not execute the full Chip; BookSim2 therefore remains inactive and `adapter_pending_qualification`.

## 2026-08-28 — v0.11.0

- Design contract: v1.10; the four system profiles, layered external/internal memory path, full-Chip external timing ownership, strict single-placement contract and online Backend launch gate are frozen. Virtual-address translation and configurable DRAM hashing remain deferred, not implemented capabilities.
- Software stage: M0–M8 reference infrastructure operational. P1–P10b-A are implemented. P10b-A starts a real total-duration Backend only after simulated dependencies, resource availability, route completion and latest input-version checks pass. Request-cycle GPU/ATLAS coupling from that plan remains P10b-B.
- Qualification stage: all P9b/P10a qualifications remain valid. P10b-A adds deterministic unit and two-run real-adapter qualification for dependency-gated launch, version availability and exact one-dispatch-per-device-task conservation.

### Completed through v0.11.0

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

7. **P9a — full ATLAS Chip external-memory scheduler**
   - ATLAS patch adds an injected `IExternalDramService`; external mode captures every real Core/Task/Iteration `ComponentInput` and does not instantiate ATLAS's native Ramulator2.
   - Runtime requests use the internal Hybrid-Bond port, per-core 1 MiB Global PA projections, initiator-specific completion dequeue and durable parent completion.
   - TinyLlama Q projection emits 139,456 ATLAS parents / 8,925,184 transaction bytes. ATLAS-only completes at 76,418 global GPU cycles; 4,096 deterministic GPU parents delay it to 81,329 cycles.
   - Parent/child completion, one-owner timing and zero-inflight checks pass. This is full `atlasim.Chip` scheduler coverage with synthetic GPU traffic, not yet the Accel-Sim compute backend.

8. **P9b — real Accel-Sim plus full ATLAS Chip**
   - The full-Chip runtime is loaded by the in-process bridge from a versioned backend config; Chip/operator/placement contents participate in the Simulation Key.
   - Each Accel-Sim GPU cycle polls ATLAS completions, advances the ATLAS clock domain and issues Logic-Die requests. Accel-Sim remains active until both the Chip and its shared-memory traffic finish.
   - Fixed q_proj contention result: 1,541,401 GPU cycles / 262,272 GPU parents; 141,255 ATLAS cycles / 139,456 ATLAS parents; ATLAS finishes at GPU cycle 159,901.
   - One Ramulator2 completes all 401,728 parents with zero outstanding. Initiator-specific queues prevent either backend from consuming the other backend's payload.
   - This deliberately duplicates q_proj on both devices to qualify contention. It is not a single-placement execution or end-to-end schedule.

9. **P10a — strict placement and versioned residency control plane**
   - Placement decisions are rejected unless the node set is exact, unique and targets a supported device.
   - Every device task records versioned input/output values. KV append reads the old K/V versions and writes the next versions.
   - Each cross-device input value creates an independent route with payload size, producer version, dependency and topology-specific lowering actions.
   - Timed modes emit `hetero-residency/v2` events for external input binding, read, write and route completion.
   - The Model 3 reference run conserves 228 logical nodes / 228 device tasks and emits 28 value routes / 571 residency events. This is reference/event qualification, not real multi-operator cycle coupling.
   - `co_resident_atlas` must explicitly declare duplicate-operator contention semantics and is rejected by the normal single-placement operator-event dispatcher.

10. **P10b-A — online real-Backend launch gate**
   - `operator_event` no longer executes Backends while constructing the execution graph. `python.OnlineOperatorRuntime` launches them from the simulated event timeline.
   - A task can launch only when all dependencies and routes have completed, its device resource is available, and every input's latest version is resident on the target device.
   - Route destination versions become visible only at route completion; device output versions become visible only at task completion. Stale or unavailable inputs fail before Backend launch.
   - The Step 2 real-adapter probe conserves 85 logical nodes / 85 device tasks / 85 Backend dispatches, with 12 routes and 117 successful version checks.
   - The selected GPU task runs an official QV100 Accel-Sim trace for 14,731 cycles; the selected ATLAS task runs a test-chip artifact for 48,446 cycles. Their launch times equal their respective maximum dependency-completion times.
   - Two independent output roots produce byte-identical `online_dispatch.json` and `metrics.json`. These bindings remain `surrogate_plumbing_probe`; P10b-A does not qualify request-cycle sharing of one live Ramulator2.

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
/opt/gpu-atlas/qualification/full-chip-scheduler-memory-path-20260828-p9a-final/qualification_record.json
/opt/gpu-atlas/qualification/accel-sim-v2/rtx3070-tinyllama-qproj-full-atlas-chip-shared-memory-p9b/qualification_record.json
/opt/gpu-atlas/qualification/p10b-a-online-dispatch-run1/step2_model1_operator_event_probe/80a6088fc4a6a530cab86c6957a33ff79bedc21746505750cad94889bde4f1bb/online_dispatch.json
/opt/gpu-atlas/qualification/p10b-a-online-dispatch-run2/step2_model1_operator_event_probe/80a6088fc4a6a530cab86c6957a33ff79bedc21746505750cad94889bde4f1bb/online_dispatch.json
/opt/gpu-atlas/qualification/prefill-p10b-to-p14-final/
```

### Remaining gaps

1. Replace the P11 tiled compute contracts and sampled traffic, operator by operator, with shape-matched Accel-Sim instruction traces and complete ATLAS compiler/runtime artifacts; only then calibrate and qualify Prefill latency.
2. Extend the same strict materialized/request-cycle path to one-layer and 22-layer Decode, including full KV traffic and a real single-token loop.
3. Connect Continuous/Ragged multi-Batch scheduling to real fused/batched cycle artifacts and validate admission, padding and shared-kernel shapes.
4. Add longer mixed GPU/ATLAS placement cases, full unsampled read/write traffic, fairness/QoS and deadlock/liveness stress tests.
5. Activate and qualify ATLAS BookSim2 with an explicit NoC topology/configuration, real Packet traffic, cycle/packet/flit conservation, backpressure, deadlock checks and deterministic double runs.
6. Complete Model 2 PCIe DMA and Model 4 CXL.mem cycle paths, then calibrate RTX 3070 and target link/3D-DRAM parameters. Deferred items remain live TraceVA→Global-PA translation, MMU/TLB and configurable/XOR mapping.

### Claim boundary

The evidence includes a qualified shape-matched single-operator comparison, real Accel-Sim/full-ATLAS-Chip contention, strict single placement/versioned residency, and a P10b-B–P14 causal Prefill deployment through one live shared-memory owner. P14 covers a full 22-layer BS=1 Context=1024 graph, but its compute is a tiled contract and its memory traffic is sampled. It is not measured hardware, calibrated TinyLlama latency, instruction-trace coverage for every operator, complete ATLAS compilation, multi-Batch throughput, virtual-memory modeling or proof that a fixed trace is safe for arbitrary memory configurations. `performance_claim_allowed=false` remains mandatory.
