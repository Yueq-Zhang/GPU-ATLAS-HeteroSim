# GPU-ATLAS-HeteroSim

GPU、ATLAS Compute Die 与 3D-DRAM 的异构端到端 LLM 联合仿真工程。

当前状态是 M0/M1 可构建骨架，不是完整性能模拟器。已冻结并落地的第一批契约包括：

- Python 离线控制面与 C++ 动态执行面的工程边界；
- `TimeFs = uint64_t` 与确定性事件顺序；
- 四种系统 Profile 的严格配置枚举；
- ModelGraph/ExecutionGraph 基础数据类型；
- Backend、MemoryService、LinkService 和 Artifact 边界接口；
- 严格配置校验和 C++/Python 冒烟测试。

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
PYTHONPATH=. python3 -m frontend.hetero.cli validate \
  --config configs/hetero/experiments/m0_smoke.yaml
```

## 实现纪律

后续修改必须先定位规范中的 M0-M9 阶段和强制不变量。任何拓扑语义、地址语义、Token 语义或时序所有权变更，都应先更新规范及变更记录。

