# WSL build validation

Date: 2026-08-26

Environment:

- Distribution: Ubuntu 22.04 under WSL2
- Project: `/opt/gpu-atlas/GPU-ATLAS-HeteroSim`
- Compiler: GCC 11.4.0
- CMake: 3.22.1
- Python: 3.10.12
- Pytest: 9.1.1 in project-local `.venv`

Commands:

```bash
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --parallel 4
ctest --test-dir simulator/build --output-on-failure
.venv/bin/python -m pytest tests/hetero
PYTHONPATH=. python3 -m frontend.hetero.cli validate \
  --config configs/hetero/experiments/m0_smoke.yaml
```

Validation after the first implementation slice:

- CTest: 6/6 passed.
- Pytest: 24/24 passed.
- Model 1/2/3/4 scheduler-validation runs completed and emitted all nine required run artifacts.
- Tiny golden Paged KV: 18 committed tokens, 8 blocks, 9216 logical bytes, 16384 allocated bytes.
- Continuous-batching dummy workload matched the five frozen Epoch selections exactly.

The system still does not execute ATLAS or Accel-Sim. Fixed Epoch timing is only for semantic validation and makes no performance claim.
