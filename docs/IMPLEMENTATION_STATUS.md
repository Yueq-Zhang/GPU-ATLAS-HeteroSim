# Implementation status

## 2026-09-10 — v0.42.0 / P32-P34 online Shadow, cycle replay and live schedulers

- P33 independently replays the P31 40-kernel GPU layer twice through Accel-Sim plus one Ramulator2 per leg and replays the complete 338,080-request ATLAS QKV stream twice through one durable Ramulator2 per leg. Cycles, instructions, request/completion digests and external-memory counters are equal; Parent/Child/durable conservation and zero in-flight are mandatory.
- P32 joins the fixed P30 Hugging Face request and the two qualified P31 branches under one femtosecond time owner. The GPU whole-layer Artifact is a reference envelope and the ATLAS QKV Artifact is an alternative candidate, so the barrier uses `max` and never adds the two durations. Dependencies, branch resource occupancy, non-overlapping Global PA, durable completion, version commit and request finish are ordered explicitly.
- P34 observes actual vLLM 0.29.0 `AsyncScheduler` outputs and Paged-KV block tables for a two-request ragged/continuous batch, validates page ownership and binds pages to Global PA. Runtime-random request IDs are canonicalized without discarding the raw evidence.
- P34 also observes a real TensorRT-LLM 1.2.1 LLM object, Optimization Profile and in-process `SimpleScheduler` callbacks in two deterministic runs. This qualification is explicitly limited to the LLM API PyTorch backend; no serialized TensorRT engine, tactic sequence or CUDA Graph is qualified.
- All three stages remain functional/request-cycle qualifications. They do not calibrate RTX 4090, RTX 3070, external-link, Logic-Die, NoC or 3D-DRAM performance, and `performance_claim_allowed=false` remains mandatory.
- Final remote regression: 260 Python tests, the simulator build and all 9 CTest targets pass; changed Python files pass Ruff and format checks, 322 JSON files parse, Python compile and changed shell syntax checks pass, and the design-spec SHA256 matches `dependency_lock.yaml`.
- Qualification details: [P32](qualification/p32_online_shadow_timeline.md), [P33](qualification/p33_cycle_replay.md) and [P34](qualification/p34_live_framework_events.md).

## 2026-09-10 — v0.39.0 / P30-P31 online HF observation and framework-driven Artifacts

- P30 executes fixed-revision TinyLlama Prefill plus one Decode token in the isolated Hugging Face runtime on the remote RTX 4090. Module callbacks record 67 semantic events, 30 modules, Tensor/View/Alias relations, CUDA allocator ranges and real framework results.
- Two P30 runs have identical semantic identity, Simulation Key and result hashes. Their allocator layouts contain 50 and 52 storages, so allocation-specific evidence is intentionally separated from the reusable semantic key instead of treating raw CUDA addresses as stable identity.
- P31 directly controls the NVBit tracer around the actual Layer-0 Decode forward. It captures 40 non-empty framework-selected kernels on SM89, preserves their actual SM86 binary headers, and records SM89 capture device and SM86 replay target as different identities.
- The GPU source Artifact binds four allocator ranges (2,250,244,096 allocated bytes) into non-overlapping 4 GiB Global PA and completes one Accel-Sim plus unique-Ramulator2 address-audit replay: 15,859,267 GPU cycles, 371,846,682 instructions, 2,756,823/2,756,823 completed/durable parent requests, 575,486,980 translated addresses, zero unmapped addresses and zero in-flight requests. This is not double-run request-cycle qualification, so `request_cycle_ready=false` remains mandatory.
- The ATLAS compiler path lowers the matching QKV Projection to a 16-Logic-Core Stage/Tile plan and deterministically materializes 338,080 full memory requests (337,920 reads and 160 writes). Two generated Artifacts and gzip traces are byte-identical. Ramulator2 cycle replay remains pending.
- P30/P31 tests, SASS inspection and Trace capture run only on the remote RTX 4090 from this point forward. The local checkout stores code plus lightweight qualification, Manifest, binding, Artifact and replay-stat JSON records, not the 139 MiB compressed raw Trace.
- P31 does not yet place GPU and ATLAS Artifacts into one online Shadow timeline; vLLM Scheduler/Block Table and TensorRT-LLM Engine callbacks remain pending. `performance_claim_allowed=false` is unchanged.
- Remote regression after P30/P31: 256 Python tests and all 9 CTest targets pass; the 11 framework-integration tests pass independently; P30/P31 Python files pass Ruff check/format verification and all three shell entry points pass Bash syntax validation.
- Qualification details: [P30](qualification/p30_huggingface_online.md) and [P31](qualification/p31_framework_artifacts.md).

## 2026-09-09 — v0.37.0 / P29 isolated live framework runtimes

- P29 installs three independent `uv` environments without changing the project/Accel-Sim CUDA 11.8 stack: Transformers 5.16.1 with PyTorch 2.14.0+cu132, vLLM 0.29.0 with PyTorch 2.13.0+cu132, and TensorRT-LLM 1.2.1 with PyTorch 2.9.1+cu128.
- The local Ubuntu-22.04 WSL environment on RTX 3070/SM86 and remote Ubuntu 22.04 environment on RTX 4090/SM89 both pass package import, CUDA Tensor execution and a pinned TinyLlama revision request. Hugging Face executes a 16-token Prefill and one Decode step; vLLM and TensorRT-LLM each load the real model and produce one token.
- vLLM uses its V1 model-runner fallback only under WSL because the V2 runner's UVA host-buffer path is unavailable there; native Linux SM89 uses the framework default V2 runner. TensorRT-LLM's smoke uses its LLM API with the PyTorch backend. The no-sudo remote host obtains Open MPI through an isolated Ubuntu package extraction selected by the profile launcher.
- Exact package freezes, CUDA probes and live-smoke records are under `validation/p29/framework_runtimes`. `scripts/qualify_p29_framework_runtimes.py` checks both device identities, all three versions, CUDA execution, real request completion, evidence hashes and fail-closed claim fields.
- All six live workloads were executed a second time after installation. `scripts/qualify_p29_framework_live_replay.py` verifies same-host input/output and stable result hashes, cross-host semantic output, vLLM/TensorRT-LLM token parity, and the unchanged fail-closed claim boundary. Initialization and request timing fields are recorded but excluded from the replay pass criterion.
- P29 is a runtime bootstrap qualification, not P28 online integration: actual framework operator/allocation events, vLLM Scheduler/Block Table callbacks, TensorRT-LLM profile/kernel events, automatic NVBit capture and simulation feedback are not connected. `performance_claim_allowed=false` remains mandatory.
- Verification after P29: all 251 Python tests and all 9 CTest targets pass; the P29 Python tools pass Ruff and compile checks, all three shell scripts pass Bash syntax validation, and both the two-host installation record and deterministic live-replay record validate successfully.

## 2026-09-09 — v0.36.0 / P26-P28 lifecycle, real BookSim2 path and framework contracts

- P26 embeds the P24 lifecycle rules in the sealed P23 BS=2/KV17 real-Trace timeline. Three exact epochs double-run deterministically; cancellation is sampled only at an `ALL_REQUESTS_DURABLE` token barrier, waiting cancellation allocates no memory, and active KV is released only after every Parent/Child is durable. A separate 4,096-epoch allocator pressure probe performs 8,192 allocations/releases with 8,190 deterministic retired-range reuses, zero overlap, zero leak and zero inflight work.
- P27 activates the ATLAS-patched BookSim2 through a C ABI with one-cycle Step, Packet completion and explicit Flit/Credit accounting. GPU and ATLAS Parents traverse initiator→Gateway→DRAM and, after durable completion in one Ramulator2 instance, DRAM→Gateway→initiator. The isolated double run conserves 16 Packets, every Flit/Credit and all four Parents, preserves both Initiators and closes with zero inflight work.
- The Ramulator2 bridge adds an explicit memory-side Gateway ingress ABI. It accepts either Initiator after BookSim delivery and returns a durable completion at the Gateway, preventing the bridge's legacy GPU-facing Link from being modelled a second time.
- P28 implements versioned model/request/execution manifests, deterministic Hugging Face config-to-graph/Tensor/Global-PA export, exact GPU Ready-Catalog selection, projection Tensor-IR lowering for ATLAS, vLLM continuous/ragged/Paged-KV event normalization with page-to-Global-PA binding, TensorRT-LLM engine identity binding and immutable shadow-result joining.
- Machine evidence is in `validation/p26/p23_request_controls`, `validation/p27/booksim2_ramulator2_qos` and `validation/p28/framework_shadow`. The external Hugging Face, vLLM and TensorRT-LLM runtimes were not installed or executed in this WSL environment; P28 qualifies offline adapter contracts, not live framework end-to-end execution. P26 cannot extrapolate a single request to KV18+, and none of P26-P28 enables performance claims.
- Verification: 251 Python tests and all 9 CTest targets pass; the BookSim2 and standalone Ramulator2 bridge libraries build successfully; P26, P27 and P28 isolated double-run qualifiers pass; changed Python files pass Ruff and compile checks, and both new build scripts pass shell syntax validation.

## 2026-09-09 — v0.35.0 / P23 sealed fused-BS=2 Decode timeline

- P23 now closes the exact TinyLlama-1.1B Layer-0 FP16, BS=2, Context=16, `q_len=1`, KV=17 functional milestone. All fourteen GPU operator types were compiled/loaded, executed and captured on the remote RTX 4090; fifteen GPU Trace instances plus five runtime/lifecycle tasks form one twenty-task timeline.
- Both isolated timeline legs pass with the same `34,843,748,683,165 fs` makespan, 39,442,237 aggregate GPU cycles, 371,403,956 instructions, 6,862,620 GPU Parents, 6,863,691 Children and 69 runtime-memory Parents per leg. Every request is durable-complete and the run exits with zero outstanding work.
- The seal verifies 97 non-overlapping Global PA allocations, fifteen Trace bindings, five runtime bindings, four per-request K/V slices, nineteen dependency edges, one non-overlapping `gpu0` resource timeline, KV length 16→17, version visibility before Attention and final request/KV retirement.
- Repository evidence includes the remote capture Catalog, fourteen source and fourteen coupled Artifact manifests, fourteen per-operator qualification records, the two lightweight timeline legs, the final timeline qualification and a path/hash-checked Ready Catalog. Raw NVBit/Accel-Sim traces and `backend_runs` remain on the remote evidence host and are intentionally excluded from Git.
- `scripts/validate_p23_sealed_catalog.py` validates the repository seal without pretending that omitted raw traces are locally replayable. The milestone remains functional-only: capture SM89, actual SM80/SM86 library binaries, replay target SM86, persistent DRAM state across separate kernel replay processes and hardware performance calibration remain distinct claims. `performance_claim_allowed=false`.
- Verification after sealing: 238 Python tests and all 9 CTest targets pass in WSL; the P23 offline seal validator passes. The legacy Python 3.10 build directory is stale, while the active Python 3.12 WSL build completes successfully.

## 2026-09-07 — v0.34.0 / P23 remote capture plus P24-P25 functional qualification

- P23 now has a remote-only SASS acquisition pipeline for TinyLlama Layer 0, FP16, BS=2, Context=16, `q_len=1`, KV=17. All 14 GPU operator traces were compiled/loaded, executed and captured on the RTX 4090 host; the local RTX 3070 is not an allowed SASS source for this milestone.
- Capture identity separates the SM89 execution device from each kernel's actual SASS binary version. Accel-Sim 2.0 accepts SM80 and SM86 through its Ampere opcode map, so an explicit SM80/SM86 mixed contract is recorded instead of rewriting those kernels as SM89. Capture completed 14/14; Range-Rebase double qualification is running remotely and the unified Batch timeline remains pending.
- P24 adds replayed EOS, maximum-output termination and explicit cancellation sampled at token-step barriers. KV-capacity-aware admission, retirement release, deterministic first-fit Global PA reuse and allocation/request conservation are wired through the C++ scheduler and Runner. Two isolated one-layer double-run cases pass.
- P25 adds deterministic weighted QoS arbitration across GPU and ATLAS resources, priority-class service, bounded-starvation override, fairness auditing and deadlock/livelock watchdog qualification. The BookSim2 activation gate fails closed because no runtime adapter is installed; scheduler micro-cycles are not hardware performance cycles.
- Verification after integration: 232 Python tests and all 9 CTest targets pass. P24 and P25 qualification records both report `qualification_passed=true` and `performance_claim_allowed=false`.
- Machine evidence: `validation/p24/qualification_record.json`, `validation/p25/qualification_record.json`, and the remote P23 `validation/p23/capture_catalog.json`. No 22-layer validation was run for this milestone.

## 2026-09-07 — v0.33.0 / P22 functional-cycle multi-Batch qualification

- Added explicit multi-request lifecycle states and epoch records for arrival, KV-capacity-aware admission, active membership, token/KV commit, finish and retirement. KV reservations are released before later requests are admitted.
- Added deterministic phase/device sub-batch planning with `homogeneous`, `padding_dense` and `ragged_split` policies. Every selected request has a bijective member mapping, exact Q/KV lengths, padding work, permutation metadata and device assignment.
- Added two fail-closed cycle contracts: `request_cycle_composed` records scheduler-epoch composition for functional causality, while `batched_kernel_cycle` requires an exact sealed catalog entry and rejects missing or mismatched model/shape/device/member lengths.
- Added dynamic KV/Global PA lifecycle auditing: active allocations may not overlap, allocation epochs are unique, request versions commit in order, retirement releases all bytes and freed ranges may be reused safely.
- Qualified five isolated double-run cases: one-layer static homogeneous BS=2, one-layer ragged padding, one-layer ragged split, 22-layer continuous four-request Decode and two-layer mixed Prefill/Decode with GPU and ATLAS device sub-batches. All pass admission/retire, request/token fan-out, resource mutual exclusion, address lifetime and zero-inflight gates.
- Machine evidence is in `validation/p22/qualification_record.json`; the supported functional matrix is in `configs/hetero/operator_capabilities/p22_multi_batch_functional.json`.
- Verification: 213 Python tests and all 9 CTest targets pass. The five P22 cases pass isolated double-run qualification; changed Python files compile successfully. P22-scoped style/whitespace checks are clean.
- P22 does not yet provide real fused/batched Accel-Sim or ATLAS artifacts and does not drive their combined memory streams through live Ramulator2. Scheduler epoch times and derived makespan/Token/s/fairness are regression counters only; `performance_claim_allowed=false` remains mandatory.

## 2026-09-04 — inference-framework integration backlog frozen

- Added a fail-closed F0–F8 roadmap for Hugging Face Transformers, vLLM and later TensorRT-LLM integration. Existing PyTorch/Transformers operator workloads remain capture and calibration programs, not framework adapters.
- The frozen split assigns numerical model execution and token generation to the framework, GPU instruction/memory behavior to compiled NVBit/Accel-Sim artifacts, ATLAS behavior to Tensor-IR lowering, and timing/resource ownership to GPU-ATLAS-HeteroSim shadow simulation.
- Required work now explicitly covers framework manifests, graph/shape export, stable Tensor/KV identity, Global PA binding, exact-shape Trace Catalog lookup, missing-artifact behavior, ATLAS compilation, runtime callbacks, dynamic batching and end-to-end qualification. See `docs/INFERENCE_FRAMEWORK_INTEGRATION_TODO_zh.md`.

## 2026-09-02 — v0.32.0 / P20 multi-token autoregressive Decode qualification

- Added an explicit `decode_loop` execution scope while preserving P19 `decode_step` as an exactly-one-token contract. The first Decode embedding consumes the external token; every later embedding consumes the preceding sampling output.
- Added a v2 Decode KV lifecycle that audits every layer and token step: K/V identities remain stable, append offsets advance by one 512-byte token, attention consumes the committed next version, and final state is KV length 20 / version 4 before request finish and release.
- Qualified TinyLlama-1.1B FP16, BS=1, initial KV=16, four generated tokens in isolated double runs: one layer has 68 tasks / 760 GPU parents / 167,658 uncalibrated GPU cycles; 22 layers have 1,076 tasks / 13,528 GPU parents / 2,249,544 uncalibrated GPU cycles.
- Both scales pass dependency and `gpu0` resource causality, Sampling-to-next-Embedding chaining, Global PA containment, request completion within task intervals, one live Ramulator2, Parent/Child/durable conservation, zero ATLAS parents and zero inflight work.
- P20 remains functional only: compute fidelity is `tiled_cycle_contract_unqualified`, Accel-Sim instruction-trace coverage is 0.0, and `performance_claim_allowed=false` is mandatory.
- Verification: 198 Python tests and all 9 CTest targets pass; both P20 scales also pass isolated double-run qualification, JSON parsing, shell syntax, Python compilation and P20-scoped Ruff checks.

## 2026-09-02 — v0.31.0 / P19 fixed-shape Decode functional qualification

- Generalized the tiled `prefill_cycle` catalog/runtime into a backward-compatible `request_cycle` mode that accepts control, Prefill and Decode phases while retaining one live Ramulator2 and completion-time value commits.
- Added a fail-closed Decode KV lifecycle audit. Every layer must have one append before attention, consume its own K/V, bind the initial and appended byte ranges to non-overlapping Global PA, commit K/V version 1 and release only after request finish.
- Added exact TinyLlama‑1.1B FP16, BS=1, initial Context=16, `q_len=1` configurations and capability catalogs for one layer and all 22 layers. The graphs contain 20 and 272 tasks respectively.
- Both shapes passed independent double runs. The one-layer case completes 190 GPU parents in 42,057 uncalibrated GPU cycles; the 22-layer case completes 3,382 parents in 566,052 uncalibrated GPU cycles. Both have one Ramulator2, zero ATLAS parents, exact Parent/Child/durable conservation and zero outstanding work.
- Added stream-level qualification that proves every issued request remains within its value's Global PA allocation, every completion remains inside the owning task interval, dependencies and `gpu0` resource intervals are causal, and both legs match on cycles, requests, versions and stream hashes.
- P19 is a functional milestone only. GPU compute uses `tiled_cycle_contract_unqualified`, not an Accel-Sim instruction trace; `accel_sim_instruction_trace_coverage=0.0` and `performance_claim_allowed=false` are mandatory.
- Verification: 186 Python tests and all 9 CTest targets pass; the P19 one-command double-run qualification also passes from a clean invocation.

## 2026-09-01 — P18 GPU operator error-triage baseline

- Added a fail-closed P18 triage builder that combines the P17 pairing audit with simulator cycle, instruction and execution-identity counters. It rejects operator-coverage drift and recorded-error inconsistencies instead of silently recomputing a different baseline.
- Generated a 14-operator machine-readable report. Mean absolute relative error is 45.67%, median is 46.04%; eleven simulated latencies are below Native and three are above. The classification contains four within-tolerance, one near-threshold, three material-error and six severe-error operators.
- The report deliberately assigns no causal explanation. It establishes the immutable input hashes and calibration order for subsequent stability, launch-gap, clock, cache and memory sensitivity experiments. `performance_claim_allowed=false` remains unchanged.
- Added three targeted regression tests for the triage calculation, coverage mismatch and checked-in input hashes. The complete WSL regression now passes 174 Python tests; 589 repository JSON records parse successfully and `git diff --check` remains clean.

## 2026-09-01 — v0.30.0 / all fourteen local RTX 3070 execution identities closed

- Completed the remaining twelve TinyLlama Layer-0 BS=1 Context=16 GPU operators on the local RTX 3070. Each operator now has a 50-warmup/500-iteration CUDA Event measurement, real SM86 NVBit 1.8 Trace capture and deterministic native-VRAM Accel-Sim double qualification. Together with Token Embedding and Residual Add, the catalog covers all fourteen operator identities and 63 real kernel launches.
- Extended the execution-identity contract for dynamic PyTorch launches. It seals the Python executable, `torch._C` extension, workload source and Python/PyTorch/Transformers/CUDA versions before combining them with the exact shape launch contract and observed kernel sequence. Native measurement and Trace capture therefore refer to the same launch program without incorrectly claiming that a PyTorch operator is a single ELF.
- Found that NVBit profiler-range capture produced kernels with zero instructions in this environment. The formal runner now uses process-target-only capture, rejects empty instruction traces, remains resumable and stops at the first failed operator while preserving evidence.
- Rebuilt all native, simulator, identity and qualification catalogs. Identity, Artifact and `gpu_local_vram` topology checks pass for 14/14 operators. Down Projection, Output Projection, QKV Projection and LM Head also meet the 15% numerical tolerance; the remaining ten blockers are only relative-error failures. Pairing is therefore 4/14, while six-component performance qualification remains 0/6 and `performance_claim_allowed=false`.
- Refreshed the reproduction manual, operator-status table and fail-closed performance-calibration documentation. Final verification passed 171 Python tests and 9 C++ tests, rebuilt the WSL C++ runtime/binding, parsed 588 repository JSON records, passed shell syntax checks and produced no `git diff --check` errors. The WSL build emitted only mounted-filesystem clock-skew warnings; no compile, link or test failure occurred.

## 2026-09-01 — v0.29.0 / local RTX 3070 same-Binary closure for two operators

- Kept RTX 3070/SM86 as the only P17 performance-calibration target. The new local runner rejects any other physical GPU and compiles one Linux SM86 executable that exposes a single-launch Trace mode and a 50-warmup/500-iteration CUDA Event mode through different arguments.
- Recaptured Token Embedding and Residual Add with NVBit 1.8 on the local RTX 3070, measured both on the same physical GPU and executable, and repeated deterministic native-VRAM Accel-Sim qualification. Executable, launch-contract and kernel-sequence identities now match on both sides; current Artifact hashes and memory topology also match.
- The two operators remain unpaired only because observed relative errors exceed 15%: approximately 38.82% for Token Embedding and 75.97% for Residual Add. The other twelve operators still lack native/Trace same-Binary identity. The audit therefore remains 0/14, 34 blockers, and `performance_claim_allowed=false`.
- Added a reproducible local runner plus native-catalog updater, and preserved raw native measurement JSON, Metadata, Kernel Lists, compressed Trace files, Trace Manifests, double-run qualification records and the merged audit in repository-visible evidence directories.
- Re-ran the fail-closed P16 performance audit (`audit_complete_blocked`, 28 component/run blockers), parsed all 211 configuration/P17 JSON records, passed 170 Python tests and 9 C++ tests, and completed shell/Python syntax checks. These checks validate consistency, not hardware performance.

## 2026-09-01 — v0.28.0 / portable P16 evidence and fail-closed same-Binary identity

- Replaced the two P16 simple-operator external `/opt/...` dependencies with repository-contained Metadata, Kernel Lists, non-empty SM86 traces and Range-Rebase qualification records. Source, Trace and coupled Artifacts now use portable repository-relative locators while retaining raw-file SHA-256 validation.
- Fixed the online address-binding materializer to resolve its TSV before Accel-Sim changes into the per-run output directory. A new remote deployment then completed the P16 twenty-task qualification twice; the synchronized raw run directories independently re-summarize to the same record, including a 35,450,346,739,701 fs causal makespan, 517 runtime-memory parents, one Ramulator2 per runtime task, zero ATLAS requests and zero outstanding work.
- Added `hetero-gpu-execution-identity/v1` and a catalog format that separately seal executable, launch-contract and normalized kernel-sequence SHA-256 values, target SM, launch count, and native/trace observation flags. GPU pairing now fails closed if either side lacks observed identity or if any immutable identity field differs.
- Built a deterministic trace-side identity catalog for the sealed Token Embedding and Residual Add binary. It verifies the recapture record and all Metadata/Kernel List/Trace/Manifest hashes, but deliberately records `native_measurement_observed=false`; the remaining twelve Trace operators also still lack this catalog evidence.
- Regenerated both P17 simulator catalogs and audits. Native-VRAM topology still matches, but pairing remains 0/14: fourteen Native identities and twelve Simulator identities are missing, two current Artifact hashes differ from the old Native catalog, and ten operators exceed the 15% observational tolerance. Performance qualification remains 0/6 and `performance_claim_allowed=false`.

## 2026-09-01 — v0.27.0 / P17 native-VRAM double qualification and sealed SM86 recapture

- Built the Accel-Sim 2.0 native-memory simulator with CUDA 11.8 on the validation host and completed deterministic double qualification for all fourteen fixed TinyLlama Layer-0 BS=1 Context=16 GPU operators. Every record has identical cycle/instruction pairs, Accel-Sim-owned local DRAM timing, no external Ramulator2 owner and total-duration accounting.
- Recovered the previously absent Token Embedding and Residual Add inputs by compiling one sealed SM86 CUDA binary once, verifying that it contains only SM86 cubins, and capturing both kernels with NVBit 1.8. The physical capture host is an RTX 4090, while the traced target binary remains SM86; the record explicitly forbids treating the capture host as the simulated GPU.
- Added resumable operator filters and Trace Manifest overrides to the P17 qualification runner. The native-VRAM catalog now seals the exact Manifest named by each qualification record, rather than silently substituting the contract's older Manifest. Cross-platform calibration hashes may opt into canonical LF UTF-8 hashing while binary evidence remains raw-byte hashed.
- The new same-topology audit has `topology_match=true` but still pairs 0/14 operators: all fourteen native measurements lack exact Trace/binary identity, and ten exceed the 15% observational-error threshold. The four within the numerical threshold are Down Projection, Gate/Up Projection, LM Head and Sampling, but none is qualified while identity is unresolved.
- P17 remains `audit_complete_blocked`: GPU, Copy and Runtime are `measured_unvalidated`; external Link, Logic-Die Gateway and 3D-DRAM remain `specified_only`; six-component performance qualification is 0/6 and `performance_claim_allowed=false`.

## 2026-08-31 — v0.26.0 / P17 exact 14-operator native catalog and topology-safe pairing

- Added a native RTX 3070 benchmark that reuses the exact TinyLlama Layer-0 BS=1 Context=16 operator builders used for NVBit capture. The two P16 shape-locked CUDA operators import their matching CUDA-reference measurements; the other twelve operators execute the same high-level PyTorch targets. All fourteen records bind the exact implementation, shape key and qualified Artifact SHA-256.
- Downloaded and content-hashed the fixed TinyLlama revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`; ran 50 warmups plus 500 per-iteration CUDA Event measurements for every GPU operator. The result, benchmark source, model config/weights, capability catalog, operator builder and imported simple-kernel evidence are sealed by `validation/p17/gpu_operator_pairing/measurement_manifest.json`.
- Added machine-readable simulator catalogs and a fail-closed pairing audit. Pairing now requires exact operator coverage, implementation, shape, Artifact hash, Trace/binary identity, memory topology and relative-error tolerance. The existing qualified cycles were extracted as an `external_shared_3ddram` catalog; the native measurements are `gpu_local_vram`, so the audit correctly reports 0/14 paired operators and leaves performance claims disabled.
- Added a resumable Linux qualification script for fourteen deterministic native-VRAM Accel-Sim double runs and a strict importer that rejects external-memory statistics, non-identical cycles/instructions, incorrect timing ownership or Trace identity. These runs still require the full remote Trace deployment and have not yet produced a native-VRAM simulator catalog.
- P17 remains `audit_complete_blocked`: the GPU component is measured but unvalidated, Copy/Runtime still lack semantically matched references, and the external Link, Logic-Die Gateway and 3D-DRAM lack independent calibration points. Six-component performance qualification remains 0/6.

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
4. Replace P20's four exact Decode-step tiled contracts with qualified Accel-Sim instruction traces for KV lengths 17–20; do not extrapolate from Prefill or P19.
5. P22 now validates Continuous/Ragged scheduling, admission, padding, KV isolation and device sub-batches at functional-cycle fidelity; connect its fail-closed `batched_kernel_cycle` mode to real fused/batched GPU and ATLAS artifacts and live shared-memory request streams.
6. Extend the qualified P26/P27 lifecycle and BookSim data path from the fixed BS=2/KV17 fixture to exact KV18+ and long mixed GPU/ATLAS generation traces; then calibrate NoC/DRAM clocks, router and queue parameters.
7. Complete Model 2 PCIe DMA and Model 4 CXL.mem cycle paths, then calibrate RTX 3070 and target link/3D-DRAM parameters. Deferred items remain MMU/TLB and configurable/XOR mapping.
8. P30/P31 have connected one fixed Hugging Face request to runtime callbacks and framework-selected GPU/ATLAS Artifacts. Next, place both Artifacts into one P32 online Shadow timeline with request/KV/version causality; then connect real vLLM Scheduler/Block Table and TensorRT-LLM Engine events before attempting end-to-end performance qualification.

### Claim boundary

The evidence includes a qualified shape-matched Decode Q projection, real Accel-Sim/full-ATLAS-Chip contention, strict single placement/versioned residency, a P10b-B–P14 causal Prefill deployment, P15d 13-operator full Value traffic, twelve independent instruction-to-Ramulator2 stall/resume qualifications, a P15h one-layer real-operator timeline, P19 single-token Decode, P20 four-token autoregressive Decode, P22 functional-cycle Static/Continuous multi-Batch scheduling, P26 lifecycle control over the exact P23 timeline, P27 BookSim2/Ramulator2 packet-level coupling, P28 offline framework shadow contracts, P29 isolated live framework execution, P30 online Hugging Face operator/allocation observation, and P31 framework-driven GPU/ATLAS Artifact generation. A unified online Shadow timeline, GPU double replay, ATLAS cycle replay, vLLM/TensorRT-LLM callbacks and calibrated performance remain unqualified. `performance_claim_allowed=false` remains mandatory.
