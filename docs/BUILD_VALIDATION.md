# WSL build validation

Date: 2026-08-28

Environment:

- Ubuntu 22.04 under WSL2;
- GCC/G++ 11.4.0, CMake 3.22.1, Python 3.10.12;
- CUDA toolkit 11.8;
- NVIDIA GeForce RTX 3070, driver 591.86;
- Accel-Sim 2.0.0, NVBit 1.8;
- ATLAS-MICRO-2026 commit `b2787399408e32d327c820daee96d4e6610f551a`;
- Ramulator2 commit `3996362187d7f8314936e5ad7560d93b66b6a215`.

Validation commands:

```bash
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --parallel 4
ctest --test-dir simulator/build --output-on-failure
.venv/bin/python -m pytest tests/hetero -q

ACCEL_SIM_BUILD_JOBS=8 bash scripts/build_accel_sim_ramulator2.sh
bash scripts/qualify_gpu_only_memory_path.sh \
  /opt/gpu-atlas/qualification/gpu-only-layered-memory-path-20260828-final
bash scripts/qualify_dual_initiator_memory_path.sh \
  /opt/gpu-atlas/qualification/dual-initiator-memory-path-20260828-final
```

Current validation:

- Main CMake Release runtime build: passed;
- CTest: 9/9 passed;
- Pytest: 73/73 passed;
- clean reconstruction of the patched Accel-Sim/Ramulator2/ATLAS durable-callback tree: passed;
- bridge library, GPU layered smoke, ATLAS internal-port smoke and coupled `accel-sim.out`: built successfully;
- GPU external-link and internal-DRAM bottleneck qualifications: passed;
- GPU-only, ATLAS-only and dual-initiator contention qualification: passed;
- local TinyLlama Q-projection SM86 trace capture: passed;
- native RTX 3070 double run: 36,324 cycles and 15,908,352 instructions in both runs;
- shared 3D-DRAM RTX 3070 double run: 1,498,113 GPU cycles, 529,368 Ramulator2 cycles and 262,272/262,272 completed reads in both runs;
- ATLAS shape-matched Q-projection double run: 24,613 cycles and 0.0002189966288 J in both runs;
- the rebuilt bridge's current Simulation Key exactly matches the shared-3D qualification record.

Warnings from upstream Accel-Sim/GPGPU-Sim sources remain during compilation, including ignored `fread` return values and legacy format specifiers. They do not fail the build. The optional `stubgen` command is absent, so Python type-stub generation is skipped by the upstream target; `accel-sim.out` and the runtime module are produced successfully.

These checks qualify deterministic software execution for the listed fixed inputs. They do not qualify RTX 3070 real-hardware accuracy, arbitrary trace replay, RTX 4090, end-to-end LLM execution, or a complete `atlasim.Chip` running concurrently with Accel-Sim.
