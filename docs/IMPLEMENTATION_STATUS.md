# Implementation status

## 2026-08-26

- Design contract: v1.1 frozen.
- Current stage: M1 executable validation complete; M2 uncalibrated analytical execution preview operational; M5 Step 1 independent Accel-Sim backend implemented and qualified.
- Implemented: strict resolved configuration and component-ref expansion; full-request Prefill/Decode graph; deterministic manual/rule placement; all four topology lowering decisions; C++ Token-Step Barrier Scheduler; C++ dependency/resource/arrival-aware GlobalEventRuntime; integer-only Roofline and Ideal Link duration estimates; C++ Paged KV allocator; timing ownership conflict registry; fixed-latency round-robin memory service; pybind11 boundary; canonical run artifacts; human reproduction README.
- Golden coverage: Tiny two-layer request, `G-1` Decode rule, exact KV counters, exact Paged KV capacity, frozen continuous-batching Epoch table, shared memory arbitration, resource serialization, delayed request arrival, cycle rejection, four-profile logical-work invariance and Model 1 analytical end-to-end preview.
- M5 Step 1 implemented: pinned Accel-Sim v1.3.0/GPGPU-Sim v4.2.1/NVBit 1.7.3 install and build scripts; CUDA 11.8 SM86 minimal workload; trace-capture preflight; strict Trace Manifest; capture-only TraceAddr to TensorID+offset normalization separated from candidate-specific PhysicalAddress binding; content-addressed Trace Cache; total-duration Accel-Sim adapter; exact native/adapter qualification records.
- Not implemented: calibrated Roofline parameters, dynamic Token-Step Batch fusion in analytical execution, ATLAS adapter, request-cycle bridge, Ramulator2 sharing, PCIe/CXL bounded queues/credits, dynamic KV release/reuse, GPU L2 miss external-memory bridge.

No coupled GPU+ATLAS, end-to-end LLM, or target-hardware accuracy claim may be made from this scaffold. A passed M5 Step 1 record permits reporting the pinned Accel-Sim cycle count for that exact trace/config pair, with its qualification and calibration limitations attached.

Validated in `/opt/gpu-atlas/GPU-ATLAS-HeteroSim`:

- CMake Release build: passed.
- CTest: 7/7 passed.
- Pytest: 38/38 passed.
- Strict smoke configuration validation: passed.

Four profile scheduler-validation runs completed. They are deliberately marked `performance_claim_allowed=false` because fixed Epoch duration is a semantic-validation clock, not a calibrated device timing model.

The Model 1 analytical preview also completed and is deliberately marked `performance_claim_allowed=false`: its explicit bandwidth/latency/throughput parameters are synthetic examples and its Roofline equations do not model cycle-level cache, NoC, protocol or DRAM behavior.

M5 Step 1 validation:

- Accel-Sim v1.3.0 + GPGPU-Sim v4.2.1 built successfully with explicit CUDA 11.8.
- NVBit 1.7.3 tracer and postprocessor built successfully.
- Native RTX 3070 CUDA vector-add build, execution and result verification passed.
- Local trace capture is intentionally rejected before instrumentation because driver 591.86 is newer than NVBit 1.7.3's supported upper bound 575; an official/imported pre-captured trace is used for independent simulator qualification.
- Official QV100 Backprop qualification passed: native and adapter results both report 15,329 cycles and 10,473,824 instructions; external Ramulator2 is disabled.
- Python regression: 38/38 passed; C++ regression remains 7/7 passed.
