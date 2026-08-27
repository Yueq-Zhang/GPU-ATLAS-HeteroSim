# Implementation status

## 2026-08-27 — v0.6.2

- Design contract: v1.4 frozen.
- Software stage: M0–M4 reference execution complete; M5 independent Accel-Sim and M2 ATLAS adapters operational; M6/M7 reference services and transport protocol implemented; M8 deterministic DSE operational.
- Qualification stage: Accel-Sim v2.0.0 QV100 legacy-trace replay, local RTX 3070/NVBit 1.8 trace capture, and an in-process Accel-Sim + single-owner Ramulator2 request/response path passed. ATLAS test-GEMM equivalence remains passed. Target RTX 4090, exact LLM artifacts, ATLAS Logic-Die memory requests, and dual-initiator contention remain unqualified.

Implemented in this version:

- full-request Prefill/Decode and explicit preloaded-KV `decode_step` scopes;
- Llama/SwiGLU and OPT/Dense-GELU logical operator graphs;
- Static Ragged and Continuous/Chunked scheduling, mixed epochs and Device Sub-Batch plans;
- dynamic KV allocation, release, first-fit reuse, alignment, allocation epochs and peak usage;
- four topology lowering paths and explicit non-coherent Residency transitions;
- bounded PCIe/CXL transaction model with queue depth, credits, serialization, full-duplex directions, propagation latency and backpressure;
- single-owner shared 3D memory reference model with GPU/ATLAS arbitration, transaction splitting, Channel/Bank/Row/Column decode and strict parent/child/byte conservation;
- deterministic task/link/memory feedback loop: link and memory responses extend parent tasks and the global DAG is rerun until the event schedule converges; the internal service remains explicitly `event_modeled`, not `cycle_coupled`;
- versioned JSONL memory bridge that performs TraceAddr → TensorID+offset → Global PA translation before memory timing;
- four `full_runtime` reference experiments and automatic outer-loop DSE;
- OPT-6.7B FP16, BS=1, initial context/KV=1024, one Decode Forward, RTX 4090 theoretical Roofline experiment;
- same-graph OPT-6.7B single-Decode analytical comparison for RTX 3070, RTX 4090 and all-ATLAS 3D-DRAM placement;
- OPT-6.7B FP16, BS=1, Context=1024, Prefill-only RTX 3070 theoretical Roofline experiment;
- exact Ramulator2, BookSim2 and TileLang checkout revisions in `dependency_lock.yaml`.
- Model 3 GPU-only/no-Logic-Die-contention mode with cross-validated placement, disabled ATLAS backend, GPU-only memory initiator allow-list and explicit zero-contention metrics.
- GPU-only minimal Cycle-Accurate coupling: Accel-Sim keeps SM/L1/L2/NoC timing, all memory partitions share exactly one Ramulator2 instance, reads resume through completion callbacks, and finalization drains posted writes. This is a bridge qualification, not yet the final external-link/Logic-Die/internal-transaction hierarchy.

Validation performed for implementation correctness:

- CMake Release build: passed;
- CTest: 9/9 passed;
- Pytest: 65/65 passed;
- four Profile `full_runtime` reference runs covered by regression tests;
- Model 3 request/transaction/byte conservation passed;
- OPT-6.7B decode-only graph: 0 Prefill, 1 Decode Forward, 391 logical tasks, final committed KV length 1025;
- RTX 4090 theoretical Roofline output: approximately 13.733 ms for that one step.
- all-ATLAS 3D-DRAM reference output: approximately 33.796 ms; current 409.6 GB/s reference gives 0.914x and 0.406x speedup versus the configured RTX 3070 and RTX 4090 Rooflines, respectively, so it is not faster in this analytical case.
- RTX 3070 theoretical Prefill Roofline output: approximately 179.558 ms; the local 8 GB card cannot hold the approximately 13.4 GB FP16 weights as a pure-GPU full-model run.
- OPT-6.7B GPU-only shared-3D reference run: 391 GPU tasks, zero cross-device routes, 774 GPU parent memory requests, zero Logic Die requests, and exact conservation for 13,966 child transactions and 13,842,739,592 bytes. Its 1 MiB reference transaction granularity is a scalability device, not a calibrated timing claim.
- Accel-Sim + Ramulator2 QV100 Backprop qualification: two runs both produced 14,700 GPU cycles and 10,473,824 instructions; the one shared Ramulator2 instance produced 11,038 cycles, accepted/completed 63 reads, and exited with zero outstanding requests. The original internal-DRAM baseline is 14,731 GPU cycles; the coupled config disables its fixed `dram_latency` to prevent double timing.

Existing qualified adapter evidence remains:

- Accel-Sim v2 official QV100 Backprop legacy trace, native and adapter: 14,731 cycles and 10,473,824 instructions;
- Accel-Sim v2 local RTX 3070 vector-add `.tracez`, native and adapter: 5,657 cycles and 61,440 instructions;
- ATLAS test GEMM repeated adapter runs: 48,446 cycles and 0.00581352 J.

Qualification boundaries:

- `full_runtime_reference` uses internal reference services and uncalibrated parameters; it is executable architecture infrastructure, not a claim of target-hardware accuracy;
- the JSONL bridge remains the transport-neutral path for event-modeled profiles; the live Accel-Sim read stall/resume path is now qualified for one exact GPU-only trace/config pair, while the ATLAS memory-port and dual-initiator path are still pending;
- the internal shared-memory model is not Ramulator2 and does not claim command-level DRAM accuracy;
- selecting standalone `kind=ramulator2` in `full_runtime` remains rejected because that scheduler path is separate from the new in-process Accel-Sim backend; there is no silent fallback to the reference service;
- the checked-in HBM3 32-channel Ramulator2 file is a functional candidate, not a calibrated ATLAS physical-stack configuration;
- its current HBM3 `DQ`, default `channel_width`, burst and transaction-size paths do not yield one self-consistent peak-bandwidth value; latency callback evidence remains valid, but bandwidth and LLM performance claims are disallowed until P1 configuration validation is complete;
- community SM89/RTX 4090 configs were downloaded in isolation but upstream reports deadlock/assertion failures; they are not active qualified configs;
- exact OPT-6.7B SM89 SASS traces were not found publicly;
- all reference and analytical results retain `performance_claim_allowed=false`.

Next implementation milestone:

1. separate the bidirectional GPU↔Logic-Die external link from internal 3D-DRAM bandwidth;
2. extend the bridge request with parent ID, Global PA, size, byte/sector mask, partition, ordering domain and QoS;
3. add a singleton Logic-Die gateway that splits one external parent into aligned Ramulator2 children and joins all completions;
4. return a GPU read, or a durable write acknowledgement, only after every child and the response link complete;
5. advance GPU, link, Logic Die and DRAM as separate clock domains;
6. qualify external-link-limited, internal-DRAM-limited, GPU-only, ATLAS-only and dual-initiator cases before exact LLM evaluation.
