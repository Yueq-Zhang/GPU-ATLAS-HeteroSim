# WSL build validation

Date: 2026-08-27

Environment:

- Distribution: Ubuntu 22.04 under WSL2
- Compiler: GCC 11.4.0
- CMake: 3.22.1
- Python: 3.10.12
- CUDA toolkit: 11.8
- Local GPU: NVIDIA GeForce RTX 3070, driver 591.86

Commands:

```bash
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --parallel 4
ctest --test-dir simulator/build --output-on-failure
.venv/bin/python -m pytest tests/hetero
PYTHONPATH=. python3 -m frontend.hetero.cli validate \
  --config configs/hetero/experiments/m0_smoke.yaml
bash scripts/install_accel_sim.sh
bash scripts/build_accel_sim.sh
bash scripts/build_accel_sim_ramulator2.sh
```

Current validation:

- CMake Release runtime build: passed.
- CTest: 9/9 passed.
- Pytest: 65/65 passed.
- Accel-Sim v2.0.0 simulator, NVBit 1.8 tracer and `.tracez` postprocessor: built successfully.
- Official QV100 legacy trace: native and adapter matched at 14,731 cycles and 10,473,824 instructions.
- Local RTX 3070 vector-add: NVBit 1.8 capture succeeded on driver 591.86; native and adapter matched at 5,657 cycles and 61,440 instructions.
- In-process QV100 + shared Ramulator2 qualification: two runs matched at 14,700 GPU cycles, 10,473,824 instructions, 11,038 Ramulator2 cycles and 63/63 completed reads; one instance and zero outstanding requests. The external-memory config sets the original fixed `dram_latency` to zero to avoid duplicate DRAM timing.
- ATLAS test GEMM qualification remains passed at 48,446 cycles and 0.00581352 J.

These checks qualify build reproducibility, exact adapter equivalence for the listed trace/config pairs, and GPU-only live shared-DRAM callbacks for the exact QV100/Backprop/HBM3 candidate tuple. They do not constitute RTX 3070 microarchitectural calibration, LLM end-to-end validation, ATLAS Logic-Die coupling, dual-initiator contention, or target 3D-DRAM calibration.
