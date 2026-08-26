# Implementation status

## 2026-08-26

- Design contract: v1.1 frozen.
- Current stage: M1 executable validation complete; M2 operator-event minimum GPU+ATLAS loop operational; M5 Step 1 independent Accel-Sim backend implemented and qualified.
- Implemented: strict resolved configuration and component-ref expansion; full-request Prefill/Decode graph; deterministic manual/rule placement; all four topology lowering decisions; C++ Token-Step Barrier Scheduler; C++ dependency/resource/arrival-aware GlobalEventRuntime; integer-only Roofline and Ideal Link duration estimates; C++ Paged KV allocator; timing ownership conflict registry; fixed-latency round-robin memory service; pybind11 boundary; canonical run artifacts; human reproduction README.
- Golden coverage: Tiny two-layer request, `G-1` Decode rule, exact KV counters, exact Paged KV capacity, frozen continuous-batching Epoch table, shared memory arbitration, resource serialization, delayed request arrival, cycle rejection, four-profile logical-work invariance and Model 1 analytical end-to-end preview.
- M5 Step 1 implemented: pinned Accel-Sim v1.3.0/GPGPU-Sim v4.2.1/NVBit 1.7.3 install and build scripts; CUDA 11.8 SM86 minimal workload; trace-capture preflight; strict Trace Manifest; capture-only TraceAddr to TensorID+offset normalization separated from candidate-specific PhysicalAddress binding; content-addressed Trace Cache; total-duration Accel-Sim adapter; exact native/adapter qualification records.
- M2 operator-event slice implemented: Backend Descriptor and resolved timing contracts; timing-owner conflict rejection; main `run` Backend dispatcher; Accel-Sim trace selection and cache; prepared ATLAS operator/placement adapter and cache; explicit per-task analytical fallback; mixed-fidelity task records; a single global C++ event runtime for GPU, ATLAS, route, and fallback tasks.
- Not implemented: calibrated Roofline parameters, real LLM GPU traces and matching ATLAS operator compilation, dynamic Token-Step Batch fusion, request-cycle bridge, shared Ramulator2 timing, PCIe/CXL bounded queues/credits, dynamic KV release/reuse, GPU L2 miss external-memory bridge.

No request-cycle shared-memory GPU+ATLAS, exact end-to-end LLM, or target-hardware accuracy claim may be made from this scaffold. Qualified cycle counts may only be reported for their exact trace or operator/config pair, with calibration and compatibility limitations attached.

Validated in `/opt/gpu-atlas/GPU-ATLAS-HeteroSim`:

- CMake Release build: passed.
- CTest: 7/7 passed.
- Pytest: 43/43 passed.
- Strict smoke configuration validation: passed.

Four profile scheduler-validation runs completed. They are deliberately marked `performance_claim_allowed=false` because fixed Epoch duration is a semantic-validation clock, not a calibrated device timing model.

The Model 1 analytical preview also completed and is deliberately marked `performance_claim_allowed=false`: its explicit bandwidth/latency/throughput parameters are synthetic examples and its Roofline equations do not model cycle-level cache, NoC, protocol or DRAM behavior.

M5 Step 1 validation:

- Accel-Sim v1.3.0 + GPGPU-Sim v4.2.1 built successfully with explicit CUDA 11.8.
- NVBit 1.7.3 tracer and postprocessor built successfully.
- Native RTX 3070 CUDA vector-add build, execution and result verification passed.
- Local trace capture is intentionally rejected before instrumentation because driver 591.86 is newer than NVBit 1.7.3's supported upper bound 575; an official/imported pre-captured trace is used for independent simulator qualification.
- Official QV100 Backprop qualification passed: native and adapter results both report 15,329 cycles and 10,473,824 instructions; external Ramulator2 is disabled.
- Python regression: 43/43 passed; C++ regression remains 7/7 passed.

M2 operator-event validation:

- One main `run` dispatched the official QV100 trace through Accel-Sim: 15,329 cycles and 10,473,824 instructions.
- The same run dispatched ATLAS's prepared GEMM bundle through real `atlasim.Chip`: 48,446 cycles, including 48,446 DRAM cycles, and 0.00581352 J.
- ATLAS adapter equivalence qualification passed with exact cycle, energy, and complete native-statistics matches across two independent invocations.
- Both backends resolved `total` contracts with `exports=[]`; Accel-Sim owns GPU local HBM and ATLAS owns its internal 3D-DRAM, so no DRAM timing is double counted.
- The combined example is explicitly a surrogate plumbing probe with `performance_claim_allowed=false`; exact LLM operator traces/artifacts remain pending.
