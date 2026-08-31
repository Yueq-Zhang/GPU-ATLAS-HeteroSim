# Implementation status

## 2026-08-31 — v0.25.0 / P17 performance-calibration gate and first native measurements

- Added a machine-readable calibration contract and fail-closed audit for GPU kernels, Copy Engine, runtime control, external Link, Logic-Die Gateway and 3D-DRAM. Configuration artifacts and measurement artifacts are content-hashed; validated status additionally requires accepted evidence classes, required metrics, matched Shape scope and reference errors within tolerance.
- Connected the calibration record to the experiment schema and global runner gate. A performance claim now requires both full component calibration and `performance_eligible=true` for every included device task; host control events may be explicitly excluded but cannot silently qualify device timing.
- Added and ran an RTX 3070 SM86 native CUDA calibration workload with 50 warmups and 500 measured iterations. It records exact-shape Context-16 Embedding and Residual kernels, a 32 KiB local-VRAM D2D copy, an empty-kernel CUDA event and synchronized host launch latency together with source/binary/result hashes.
- The native measurements are deliberately `measured_unvalidated`: they use the local RTX 3070 VRAM path and do not calibrate P16's 12.8 GB/s external Link, Logic-Die Gateway or 409.6 GB/s internal 3D-DRAM. They also lack a matched native-memory Accel-Sim run for the complete 14-kernel set.
- The P17 audit rechecks both P16 legs, confirms identical Simulation Key, makespan and request metrics, validates all four configuration hashes and the native measurement hash, and returns `audit_complete_blocked`. All six required components remain incomplete, so `performance_claim_allowed=false` is mandatory.
- Detailed results and reproduction commands are in `docs/qualification/p17_performance_calibration.md`.

## 2026-08-31 — v0.24.0 / P16 complete for fixed-shape request-cycle causality

- Added an auditable operator capability catalog for all 19 operator types / 20 task instances in the fixed TinyLlama Layer-0 BS=1 Context=16 Prefill graph. The catalog separates implementation, test, request-cycle readiness and performance eligibility and is regression-checked against the materialized graph.
- Added exact model/checkpoint/batch/context gates. Existing request-cycle artifacts now fail closed when their checkpoint revision, Batch or Context changes; the capability gate also covers hidden/intermediate dimensions, attention/KV heads, head dimension, vocabulary and dtype.
- Added shape-locked runtime models for Request Start/Finish and KV Allocate/Append/Release. The three KV tasks now emit exact 64-byte Global PA requests through the external Link into one live Ramulator2 per task; Request Start/Finish are explicit host-control events with no memory traffic and are excluded from the device performance boundary.
- Added shape-locked standalone CUDA implementations for Token Embedding and Residual Add. Both now have non-empty SM86 traces and passed deterministic Range-Rebase double qualification. The catalog therefore has 14 request-cycle-ready operator types covering 15 task instances.
- Double-qualified the P16 20-task timeline with no analytical fallback. Both legs have the same Simulation Key and makespan, 15 ready GPU Trace instances, 517 live KV runtime parents, 87 non-overlapping Global PA ranges, 31 input-version checks and 18 completion-time commits. All request paths conserve Parent/Child/durable completions and exit with zero outstanding work.
- Added a top-of-address-space `external_input_widened_shadow` for the int64 token IDs consumed by the real embedding kernel. It avoids moving previously qualified low-address workspaces and remains an explicit Global PA adaptation, not VA-to-PA translation.
- P16 closes the fixed Layer-0 BS=1 Context=16 implementation/request-cycle causality milestone. Runtime/Copy Engine/GPU/Link/DRAM parameters remain uncalibrated, so all 19 operator types stay performance-ineligible and the 35.450 ms causal makespan is not a publishable latency result.
- Human-readable coverage is in `docs/OPERATOR_MODELING_STATUS.md`; the live qualification boundary is in `docs/qualification/p16_full_task_modeling_status.md`.

## 2026-08-30 — v0.22.0

- Design contract: v1.22. P15h extends caller-owned runtime Global PA bindings to all twelve real GPU operators in the one-layer Context=16 Prefill graph and validates them in one online Accel-Sim timeline.
- P15h local SM86 capture and remote deterministic double qualification are complete for the remaining ten GPU operators. Together with Attention Norm and QKV Projection, twelve request-cycle-ready operators now execute in one Prefill timeline.
- The twelve real GPU backends total 40,060,873 GPU cycles, 6,995,173 parents, 6,998,046 children and 804,512,881 translated addresses. Every operator has zero unmapped and outstanding requests, zero ATLAS requests, and exactly one Ramulator2 owner.
- All DAG dependencies complete before consumer launch and all `gpu0` intervals are non-overlapping. The Global PA map contains 84 non-overlapping ranges, 56 private workspaces, twelve request bindings and 38 semantic bindings derived from graph Values.
- All eighteen graph output versions commit at Backend completion, and every ready operator validates the exact input Value versions observed at launch.
- Eight control, KV-management and residual tasks remain analytical/runtime models. The reported 35,390.378 µs makespan is therefore functional causality evidence, not calibrated end-to-end performance; `performance_eligible=false` remains mandatory. Details are in `docs/qualification/p15h_twelve_operator_prefill_timeline.md`.

## 2026-08-30 — v0.21.0

- Design contract: v1.20. P15f extends allocator capture with the pre-existing CUDA backing segments that contain the target tensors. This closes real Tensor Core transactions into allocator padding while excluding unrelated process segments; missing target coverage remains fail-closed.
- The recaptured QKV Projection manifest has 12 non-overlapping ranges and materializes 33,685,504 bytes of Global PA. Its remote deterministic double run produces 2,168,865 GPU cycles, 34,943,066 instructions, 736,837 translated accesses and zero unmapped accesses in each leg.
- One Ramulator2 completes 375,899 GPU parents and 375,944 internal children (375,854 reads / 45 writes), advances 766,383 DRAM/link/gateway cycles and exits with zero ATLAS requests and zero outstanding.
- Attention Norm and QKV Projection are the two `request_cycle_ready=true` Artifacts at this historical P15f milestone. Their strict range-rebase Catalog conserves 378,075 parents, 378,120 children and 777,807 translated addresses; readiness is not inferred for the other ten operators.
- These are still independent operator qualifications. No request-cycle-ready Accel-Sim process is embedded in the Prefill global scheduler, so `performance_claim_allowed=false` remains mandatory.
- Qualification details are in `docs/qualification/p15f_qkv_range_rebase.md`.

## 2026-08-30 — v0.20.0

- Design contract: v1.19. P15e replaces the monolithic request-cycle JSON payload with a deterministic streaming `jsonl.gz` payload plus a compact manifest. The qualified one-layer Context=16 double run completes 3,462,738 parents, advances 10,401,594 DRAM cycles and exits with zero outstanding; the compressed stream is 94,859,940 bytes with SHA-256 `aa3edd9ca85dd3f600e8a1646d1b3af9bfc84f99d50c81f6b422c4897564795d`, and peak RSS is about 524.6 MiB in both runs.
- The online Accel-Sim bridge now has explicit identity and range-rebase address modes. A recaptured Attention Norm trace uses three known Tensor ranges and three allocator-workspace ranges. Both coupled runs produce 66,697 GPU cycles, 5,290,064 instructions, 40,970 translated accesses, zero unmapped accesses, 2,176 completed parents/children, one Ramulator2 and zero outstanding.
- Only the recaptured Attention Norm range-rebase Artifact is `request_cycle_ready=true`. The legacy coupled Artifacts must remain identity-untranslated, Global-PA-not-ready, replay-unsafe and performance-ineligible; readiness is not inferred by operator similarity.
- The remote LM Head double qualification passed exactly: 23,193,593 GPU cycles, 476,608,000 instructions, 4,096,686 completed parents and 4,097,138 completed children in each leg, with one Ramulator2 and zero outstanding. The sector-mask bridge now normalizes any contiguous selected 32-byte sector span whose byte count equals the request size.
- The strict identity-untranslated coupled catalog now covers 12 real GPU operators and conserves 6,993,530 parents / 6,996,227 children. This remains a set of independent per-operator qualifications, not a Prefill end-to-end timeline.
- Qualification details are in `docs/qualification/p15e_streaming_and_range_rebase.md` and `docs/qualification/p15d_remaining_prefill_ctx16.md`.

## 2026-08-29 — v0.19.0

- Design contract: v1.18. P15d adds shape-locked RTX 3070 SM86 source Artifacts for Output Projection, MLP Norm, Gate/Up Projection, SiLU Multiply, Down Projection, Final Norm, LM Head and Sampling. Together with P15a, the strict source catalog registers all 13 operators selected for full Value traffic.
- Final Norm, LM Head and Sampling explicitly bind overall Context=16 with `q_len=1`; the layer-local operators bind `q_len=16`. Artifact lookup uses `source_q_len` for the overall request context while retaining the true operator Q length.
- A one-layer BS=1/Context=16 run now lowers all Value transactions for 13/20 tasks and bounded samples for 7/20 tasks. Two runs exactly match across eight core files, including the 2,335,336,970-byte request trace.
- One live Ramulator2 completes 3,462,738 parents/children (3,462,673 full, 65 sampled; 3,444,241 reads, 18,497 writes), advances 10,401,594 DRAM cycles and exits with zero outstanding. The global GPU clock is 31,204,782 cycles and the makespan is 26,003,985,000,000 fs.
- Seven new traces have deterministic identity-untranslated instruction-to-memory coupled qualifications: Output Projection, MLP Norm, Gate/Up Projection, SiLU Multiply, Down Projection, Final Norm and Sampling. Together with P15c, the strict catalog covers 11 GPU operators and conserves 2,896,844 parents / 2,899,089 children. The two LM Head qualification legs are running concurrently in isolated directories on the remote validation host.
- The 13-operator Value-traffic timeline still uses the unqualified tiled compute contract and is not the sum of independent Accel-Sim runs. All coupled Artifacts remain `global_pa_binding_ready=false`, `request_cycle_ready=false`, `replay_safe=false` and performance-ineligible.
- Qualification details are in `docs/qualification/p15d_remaining_prefill_ctx16.md`; the deterministic record is `/opt/gpu-atlas/qualification/p15d/thirteen-full-traffic-final/qualification_record.json`.

## 2026-08-28 — v0.18.0

- Design contract: v1.17. P15c identity-untranslated instruction-to-memory coupling now covers all four non-empty first-batch GPU traces: Attention Norm, QKV Projection, RoPE and Causal Attention.
- Deterministic double-run cycles are 66,653 / 2,170,258 / 135,833 / 43,500. The four qualifications collectively accept 383,260 GPU parents and complete 383,286 internal children; every run has exactly one Ramulator2, zero ATLAS requests and zero outstanding.
- QKV exercises real Parent-to-Child expansion: 376,212 parents become 376,238 aligned 64B children. Parent completion, child completion and durable completion are checked independently.
- The strict P15c catalog has 4/4 `compute_memory_coupled=true` coverage and 0/4 `request_cycle_ready` coverage. All paths remain `identity_untranslated`, `global_pa_binding_ready=false`, `replay_safe=false` and performance-ineligible.
- Artifact matching now includes Batch and Context in addition to model/operator/phase/layer/Q/KV/dtype, closing an overly permissive full-traffic selection path.
- The final summary is `/opt/gpu-atlas/qualification/p15c/four-operator-final/qualification_record.json`. These independent coupled runs are still not the one-layer Prefill global timeline and must not be added to P15b Value-traffic time.

## 2026-08-28 — v0.17.0

- Design contract: v1.16. P15c adds the first shape-locked real instruction-to-shared-memory cycle qualification for TinyLlama Prefill RMSNorm.
- The existing Accel-Sim external-memory patch retains each `mem_fetch` until the layered request link, Logic-Die gateway, all internal children, Ramulator2 and the response link complete. Both fixed runs produce 66,653 GPU cycles and 5,290,064 instructions.
- One in-process Ramulator2 accepts and completes 2,176 GPU parents / 2,176 children with zero ATLAS parents and zero outstanding. The two full external-memory statistics objects are identical.
- Artifact readiness is now explicitly split: `compute_memory_coupled=true` records the real stall/resume evidence, while `global_pa_binding_ready=false` records that the online bridge still forwards the identity-untranslated trace address. Consequently `request_cycle_ready=false` and performance eligibility remains false.
- The coupled RMSNorm process is not yet embedded in the one-layer Prefill global scheduler. P15b remains the Value-level full-traffic path and must not be added to the P15c cycle count.

## 2026-08-28 — v0.16.0

- Design contract: v1.15. P15 first-batch artifact production, strict compatibility binding and selective full-value traffic are implemented for one-layer TinyLlama Prefill at BS=1, Context=16.
- The versioned artifact catalog covers `attention_norm`, `qkv_projection`, `rope`, `kv_append` and `causal_attention`. It validates checkpoint/shape/dtype/address semantics and all referenced file hashes. Registration is 5/5; request-cycle trace readiness remains 0/5.
- Four non-empty RTX 3070 SM86 traces pass deterministic Accel-Sim 2.0 double runs: RMSNorm 58,736 cycles; QKV 95,151; RoPE 127,094; Causal Attention 34,923. `replay_safety_qualified=false` for every trace.
- KV Append is correctly classified as a zero-Kernel CUDA state-copy operation and registered as `runtime_state`, not a fabricated GPU compute trace.
- The generated 16-core ATLAS QKV bundle (`M=16,K=2048,N=2560`, tile `8x128x8`) passes native double-run qualification at 150,932 cycles and 42,024,960 memory-access bytes.
- Strict `operator_event` binding executes 4/20 one-layer tasks with exact shape-locked Accel-Sim traces and leaves 16 analytical fallbacks; trace coverage is 20% and performance claims remain disabled.
- Selective full-traffic Prefill lowers all Value reads/writes for the five first-batch tasks into 175,936 real 64B parents. Fifteen remaining tasks contribute 234 sampled parents. One Ramulator2 completes all 176,170 parents with zero outstanding; two output roots are byte-identical for core artifacts.
- This does not yet combine the instruction-level Accel-Sim compute state and live Ramulator2 requests in one per-operator stall/resume loop. Prefill compute in the selective-full-traffic run remains the unqualified tiled contract; Context=1024 and all-operator full traffic are not qualified.

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
/opt/gpu-atlas/qualification/p15b/first-batch-final/qualification_record.json
/opt/gpu-atlas/qualification/p15c/four-operator-final/qualification_record.json
/opt/gpu-atlas/qualification/p15d/thirteen-full-traffic-final/qualification_record.json
```

### Remaining gaps

1. Extend the qualified one-layer Context=16 P15h path to multi-layer execution while validating cross-layer KV lifetime, Global PA capacity, workspace reuse and deterministic long-run behavior. Do not linearly multiply the one-layer timing.
2. Replace or qualify the eight remaining analytical/runtime control, KV-management and residual tasks before making an end-to-end cycle-accurate claim.
3. Calibrate GPU, shared 3D-DRAM and link parameters against measured hardware or another trusted reference before enabling performance claims.
3. Extend the same strict materialized/request-cycle path to one-layer and 22-layer Decode, including full KV traffic and a real single-token loop.
4. Connect Continuous/Ragged multi-Batch scheduling to real fused/batched cycle artifacts and validate admission, padding and shared-kernel shapes.
5. Add longer mixed GPU/ATLAS placement cases, fairness/QoS and deadlock/liveness stress tests; activate and qualify ATLAS BookSim2.
6. Complete Model 2 PCIe DMA and Model 4 CXL.mem cycle paths, then calibrate RTX 3070 and target link/3D-DRAM parameters. Deferred items remain MMU/TLB and configurable/XOR mapping.

### Claim boundary

The evidence includes a qualified shape-matched Decode Q projection, real Accel-Sim/full-ATLAS-Chip contention, strict single placement/versioned residency, a P10b-B–P14 causal Prefill deployment, P15d 13-operator full Value traffic, twelve independent instruction-to-Ramulator2 stall/resume qualifications, and a P15h one-layer timeline in which all twelve recaptured range-rebase GPU operators execute as real Accel-Sim backends with runtime Global PA and version causality. P15h validates 84 non-overlapping Global PA ranges, 56 private workspaces, 12 request bindings and 18 completion-time version commits. Eight control, KV-management and residual tasks remain analytical/runtime models. P14 still uses tiled compute and sampled traffic at Context=1024; P15d uses tiled compute and mixed full/sampled traffic at Context=16. None of these paths is measured hardware or calibrated end-to-end TinyLlama latency. `performance_claim_allowed=false` remains mandatory.
