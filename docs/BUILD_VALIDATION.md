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

The scaffold does not yet execute ATLAS or Accel-Sim and makes no performance claim.

