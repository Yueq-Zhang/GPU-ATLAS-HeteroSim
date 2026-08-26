# Implementation status

## 2026-08-26

- Design contract: v1.1 frozen.
- Current stage: M1 executable validation complete; M2/M3 infrastructure started.
- Implemented: strict resolved configuration and component-ref expansion; full-request Prefill/Decode graph; deterministic manual/rule placement; all four topology lowering decisions; C++ Token-Step Barrier Scheduler; C++ Paged KV allocator; timing ownership conflict registry; fixed-latency round-robin memory service; Ideal Link formula; pybind11 boundary; canonical run artifacts.
- Golden coverage: Tiny two-layer request, `G-1` Decode rule, exact KV counters, exact Paged KV capacity, frozen continuous-batching Epoch table, shared memory arbitration, four-profile logical-work invariance.
- Not implemented: calibrated Roofline task durations, real GlobalEventRuntime task dispatch, ATLAS adapter, Accel-Sim adapter, request-cycle bridge, Ramulator2 sharing, PCIe/CXL bounded queues/credits, dynamic KV release/reuse.

No performance or cycle-accuracy claim may be made from this scaffold.

Validated in `/opt/gpu-atlas/GPU-ATLAS-HeteroSim`:

- CMake Release build: passed.
- CTest: 6/6 passed.
- Pytest: 24/24 passed.
- Strict smoke configuration validation: passed.

Four profile scheduler-validation runs completed. They are deliberately marked `performance_claim_allowed=false` because fixed Epoch duration is a semantic-validation clock, not a calibrated device timing model.
