# GPU-ATLAS-HeteroSim

GPU、ATLAS Compute Die 与 3D-DRAM 的异构端到端 LLM 联合仿真工程。

当前状态是 M1 可执行验证系统、M2/M3 基础组件已启动，不是完整性能模拟器。已落地：

- Python 离线控制面与 C++ 动态执行面的工程边界；
- `TimeFs = uint64_t`、确定性事件顺序与 C++ Token-Step Barrier Scheduler；
- 四种系统 Profile 的严格配置枚举；
- 完整请求级 Prefill/Decode ModelGraph、Placement 与拓扑 Lowering；
- C++ Paged KV 分配器、时序所有权冲突检查、固定延迟共享内存服务；
- pybind11 Python→C++ 边界和规范化 Run 产物；
- Backend、MemoryService、LinkService 和 Artifact 边界接口；
- 组件 `ref` 展开、严格配置校验和四 Profile 回归。

当前 `run` 使用固定 Epoch 时长，只验证请求、Batch、KV、放置、地址和拓扑语义。输出明确标记 `performance_claim_allowed=false`，不能作为 GPU、ATLAS、PCIe/CXL 或 3D-DRAM 性能结果。

完整设计以 [GPU + ATLAS 异构端到端仿真实现规范](docs/gpu_atlas_heterogeneous_simulation_design_zh.md) 为唯一主规范。

## WSL 构建

规范工程路径：

```text
/opt/gpu-atlas/GPU-ATLAS-HeteroSim
```

```bash
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --parallel
ctest --test-dir simulator/build --output-on-failure
python3 -m pytest tests/hetero
python3 -m frontend.hetero.cli validate \
  --config configs/hetero/experiments/m0_smoke.yaml
python3 -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m1_model3_gpu_native_3ddram.json
```

## 实现纪律

后续修改必须先定位规范中的 M0-M9 阶段和强制不变量。任何拓扑语义、地址语义、Token 语义或时序所有权变更，都应先更新规范及变更记录。
