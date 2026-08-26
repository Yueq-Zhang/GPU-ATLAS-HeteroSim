# Implementation status

## 2026-08-26

- Design contract: v1.1 frozen.
- Current stage: M1 executable validation complete; M2 uncalibrated analytical execution preview operational; M3 infrastructure started.
- Implemented: strict resolved configuration and component-ref expansion; full-request Prefill/Decode graph; deterministic manual/rule placement; all four topology lowering decisions; C++ Token-Step Barrier Scheduler; C++ dependency/resource/arrival-aware GlobalEventRuntime; integer-only Roofline and Ideal Link duration estimates; C++ Paged KV allocator; timing ownership conflict registry; fixed-latency round-robin memory service; pybind11 boundary; canonical run artifacts; human reproduction README.
- Golden coverage: Tiny two-layer request, `G-1` Decode rule, exact KV counters, exact Paged KV capacity, frozen continuous-batching Epoch table, shared memory arbitration, resource serialization, delayed request arrival, cycle rejection, four-profile logical-work invariance and Model 1 analytical end-to-end preview.
- Not implemented: calibrated Roofline parameters, dynamic Token-Step Batch fusion in analytical execution, ATLAS adapter, Accel-Sim adapter, request-cycle bridge, Ramulator2 sharing, PCIe/CXL bounded queues/credits, dynamic KV release/reuse.

No performance or cycle-accuracy claim may be made from this scaffold.

Validated in `/opt/gpu-atlas/GPU-ATLAS-HeteroSim`:

- CMake Release build: passed.
- CTest: 7/7 passed.
- Pytest: 30/30 passed.
- Strict smoke configuration validation: passed.

Four profile scheduler-validation runs completed. They are deliberately marked `performance_claim_allowed=false` because fixed Epoch duration is a semantic-validation clock, not a calibrated device timing model.

The Model 1 analytical preview also completed and is deliberately marked `performance_claim_allowed=false`: its explicit bandwidth/latency/throughput parameters are synthetic examples and its Roofline equations do not model cycle-level cache, NoC, protocol or DRAM behavior.
