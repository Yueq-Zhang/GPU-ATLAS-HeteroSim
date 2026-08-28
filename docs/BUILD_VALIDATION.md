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
bash scripts/build_atlas_full_chip_runtime.sh
bash scripts/qualify_full_chip_scheduler_memory_path.sh \
  /opt/gpu-atlas/qualification/full-chip-scheduler-memory-path-20260828-p9a-final
bash scripts/qualify_accel_sim_full_chip_concurrency.sh \
  /opt/gpu-atlas/qualification/accel-sim-v2/rtx3070-tinyllama-qproj-full-atlas-chip-shared-memory-p9b

bash scripts/qualify_prefill_p10b_to_p14.sh \
  /opt/gpu-atlas/GPU-ATLAS-HeteroSim \
  /opt/gpu-atlas/qualification/prefill-p10b-to-p14-final

.venv/bin/python -m pytest \
  tests/hetero/test_online_operator_runtime.py \
  tests/hetero/test_operator_event.py -q
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/step2_model1_operator_event_probe.json \
  --runs-root /opt/gpu-atlas/qualification/p10b-a-online-dispatch-run1
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/step2_model1_operator_event_probe.json \
  --runs-root /opt/gpu-atlas/qualification/p10b-a-online-dispatch-run2
```

Current validation:

- Main CMake Release runtime build: passed;
- CTest: 9/9 passed;
- Pytest: 90/90 passed (test count is not treated as a frozen contract);
- P10a strict-placement tests: missing/duplicate decisions rejected, each cross-device input independently routed, Model 3 fence sequence preserved, and KV versions advance from 0 to 1;
- Model 3 P10a reference artifact: 228 logical nodes / 228 device tasks, 28 value routes and 571 timed residency events;
- clean reconstruction of the patched Accel-Sim/Ramulator2/ATLAS durable-callback tree: passed;
- bridge library, GPU layered smoke, ATLAS internal-port smoke and coupled `accel-sim.out`: built successfully;
- GPU external-link and internal-DRAM bottleneck qualifications: passed;
- GPU-only, ATLAS-only and dual-initiator contention qualification: passed;
- local TinyLlama Q-projection SM86 trace capture: passed;
- native RTX 3070 double run: 36,324 cycles and 15,908,352 instructions in both runs;
- shared 3D-DRAM RTX 3070 double run: 1,498,113 GPU cycles, 529,368 Ramulator2 cycles and 262,272/262,272 completed reads in both runs;
- ATLAS shape-matched Q-projection double run: 24,613 cycles and 0.0002189966288 J in both runs;
- patched ATLAS external-DRAM runtime and full-Chip qualification executable: built successfully;
- full `atlasim.Chip` external-memory run: 139,456 ATLAS parents / 8,925,184 transaction bytes, all completed, one Ramulator2 and zero outstanding;
- full-Chip ATLAS-only completion: 76,418 global GPU cycles; with 4,096 deterministic GPU parents: 81,329 cycles, so shared-memory contention is observed;
- real Accel-Sim plus full ATLAS Chip: 1,541,401 GPU cycles / 262,272 GPU parents and 141,255 ATLAS cycles / 139,456 ATLAS parents; ATLAS finishes at GPU cycle 159,901;
- combined one-owner summary: 401,728 parents completed, one Ramulator2, zero outstanding; two independent runs match exactly;
- the rebuilt bridge, ATLAS Chip, operator list and placement map hashes are all covered by the P9b Simulation Key and its qualification record.
- P10b-A online runtime tests: a GPU → Route → ATLAS chain launches the consumer at route completion, and a stale-version consumer is rejected before Backend launch;
- P10b-A Step 2 probe: 85 logical nodes / 85 device tasks / 85 Backend dispatches, 12 routes and 117 successful version checks;
- selected Accel-Sim task: 14,731 cycles and launch at 8,192,002 fs, equal to its maximum dependency completion; selected ATLAS task: 48,446 cycles and launch at 14,248,172,886 fs, also equal to its maximum dependency completion;
- two independent P10b-A roots have identical SHA256 values: `online_dispatch.json` = `70BD9D692BA7C38C2ABF25FD3EA90E518EFCCFAF0275867888348DE160B471EF`, `metrics.json` = `9B1B29F96EA487402B616FD31CF69CF737F4EE2171357533D67B76F0F82C85CC`.
- P10b-B mixed one-layer Prefill: 20 tasks / 4 routes, 347 GPU plus 35 ATLAS parents, all 382 completed by one Ramulator2 with zero outstanding;
- P11 strict graph/catalog checks: 19 operator classes on both devices, correct GQA/SwiGLU/residual/KV tensor contracts, zero analytical fallback;
- P12 one-layer GPU-only: 20 tasks, 378 GPU parents, zero ATLAS parents and zero outstanding;
- P13 22-layer Context=16: 272 tasks, 448 Global PA ranges and 3,382/3,382 completed GPU parents;
- P14 BS=1 Context=1024: 272 tasks, 448 ranges using 3,957,580,290 B of 4 GiB, final KV length 1024 and 3,385/3,385 completed GPU parents;
- all four P10b-B–P14 cases pass independent double-run byte equality for seven core artifacts under `/opt/gpu-atlas/qualification/prefill-p10b-to-p14-final`.

Warnings from upstream Accel-Sim/GPGPU-Sim sources remain during compilation, including ignored `fread` return values and legacy format specifiers. They do not fail the build. The optional `stubgen` command is absent, so Python type-stub generation is skipped by the upstream target; `accel-sim.out` and the runtime module are produced successfully.

These checks qualify deterministic software execution and a complete causal Prefill deployment for the listed fixed inputs. P10b-B qualifies live request-cycle GPU/ATLAS/Route orchestration through one Ramulator2; P14 qualifies lifecycle and address/task completeness for TinyLlama-1.1B BS=1 Context=1024. It does not qualify the reported makespan as performance: compute is an uncalibrated tiled contract, memory is bounded sampling and `trace_coverage=0`. The checks also do not qualify RTX 3070 hardware accuracy, arbitrary trace replay, RTX 4090, MMU/TLB behavior, XOR mapping, Decode or multi-Batch execution.
