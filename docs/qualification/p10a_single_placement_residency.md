# P10a strict single-placement and residency qualification

Date: 2026-08-28
Software version: `0.10.0`
Design contract: v1.9
Reference experiment: `configs/hetero/experiments/m8_model3_full_runtime_reference.json`

## Qualified invariants

- The placement decision set equals the logical node set and contains no duplicates.
- Every logical node materializes as exactly one device task.
- Every read carries a concrete value version and byte size.
- Every cross-device input value is lowered independently.
- Model 3 synchronization preserves `writeback`, `release_fence`, `invalidate`, `acquire_fence` order.
- KV append reads K/V version 0 and writes K/V version 1 in the decode-step golden test.
- Duplicate-operator `co_resident_atlas` contention mode cannot enter normal operator-event dispatch.

## Reproduction

```bash
.venv/bin/python -m pytest tests/hetero -q
ctest --test-dir simulator/build --output-on-failure

.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m8_model3_full_runtime_reference.json \
  --runs-root /tmp/gpu-atlas-p10a
```

Validated results:

| Check | Result |
|---|---:|
| Python tests | 79/79 passed |
| C++ tests | 9/9 passed |
| Logical nodes | 228 |
| Materialized device tasks | 228 |
| `each_logical_node_exactly_once` | `true` |
| Value-granular cross-device routes | 28 |
| Timed residency events | 571 |
| Residency schema | `hetero-residency/v2` |

## Claim boundary

This qualifies the strict placement and versioned residency **control plane** plus the reference event runtime. It does not mean that all 228 operators have real Accel-Sim or ATLAS cycle artifacts, that copy/fence actions are already injected into the live P9b request path, or that end-to-end LLM latency is cycle-accurate. Those items belong to P10b and later operator-coverage stages.
