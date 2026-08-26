# GPU-ATLAS-HeteroSim

GPU-ATLAS-HeteroSim 是面向 GPU、ATLAS Compute Die 与 3D-DRAM 的异构端到端 LLM 联合仿真工程。工程把完整 Prefill/Decode 请求图、算子放置、跨设备数据移动、Paged KV Cache 和全局事件调度连接到同一条可复现运行路径。

> 当前版本处于 M2 开发阶段。`scheduler_validation` 用固定 Epoch 验证请求、Batch、KV、放置和拓扑语义；`analytical_preview` 使用未校准 Roofline 与链路参数验证全局任务 DAG。两者都不是周期精确性能结果，产物均明确写入 `performance_claim_allowed=false`。

完整架构约束以 [GPU + ATLAS 异构端到端仿真实现规范](docs/gpu_atlas_heterogeneous_simulation_design_zh.md) 为准，阶段进度见 [实现状态](docs/IMPLEMENTATION_STATUS.md)。

## 1. 当前已实现的能力

- 完整请求级 decoder-only LLM 图：Prefill、逐 Token Decode、KV Append、LM Head 与 Sampling；
- 按 phase、layer、operator group、KV 长度和活动 Batch 进行 GPU/ATLAS 放置；
- 四种系统组织形式的配置与拓扑 Lowering：
  1. ATLAS 原生独立 3D-DRAM，外部 GPU 使用分析模型；
  2. 3D-DRAM 作为主存，GPU 保留独立显存并通过 PCIe 搬运；
  3. 3D-DRAM 直接作为 GPU 显存，GPU 与 Compute Die 共享同一内存；
  4. GPU 保留显存，3D-DRAM 作为 CXL 内存扩展层；
- C++ `GlobalEventRuntime`：按 DAG 依赖、任务到达时间和互斥资源推进确定性事件；
- C++ Token-Step Barrier Scheduler 与多请求连续调度语义验证；
- C++ Paged KV 分配器、固定延迟内存服务和时序所有权冲突检查；
- Python 配置/图控制面与 pybind11 C++ 动态运行时边界；
- 规范化 Run 目录、输入哈希、Git revision、依赖锁和 Fidelity 标签。

## 2. 目录结构

```text
GPU-ATLAS-HeteroSim/
├── configs/hetero/              # 模型、工作负载、放置、地址和实验配置
├── docs/                         # 冻结设计规范、构建记录和实现状态
├── frontend/hetero/              # Python 配置、ModelGraph、Placement、Runner
├── simulator/                    # C++ 运行时、内存服务、pybind11 与单元测试
├── tests/hetero/                 # Python 端到端及配置回归测试
├── dependency_lock.yaml          # 外部依赖版本记录
└── runs/                         # 本地运行产物，默认不提交 Git
```

## 3. 推荐环境

已经验证的参考环境：Windows 11 + WSL2、Ubuntu 22.04、Python 3.10、CMake 3.22、GCC/G++ 11 和 Ubuntu `pybind11-dev` 2.9.1。

本文假定 WSL 工程路径为：

```text
/opt/gpu-atlas/GPU-ATLAS-HeteroSim
```

不要求必须使用该路径，但所有命令都应在同一个工程根目录执行。

## 4. 第一次安装

### 4.1 安装 WSL 依赖

```bash
sudo apt update
sudo apt install -y build-essential cmake python3-venv pybind11-dev rsync
```

### 4.2 准备工程副本

若代码已位于 `C:\Users\yueqi\Desktop\3D_DRAM\OpenSourceWorks\GPU-ATLAS-HeteroSim`，建议同步到 WSL 原生文件系统：

```bash
sudo mkdir -p /opt/gpu-atlas
sudo chown -R "$USER":"$USER" /opt/gpu-atlas
rsync -a \
  --exclude .git \
  --exclude .venv \
  --exclude simulator/build \
  --exclude runs \
  /mnt/c/Users/yueqi/Desktop/3D_DRAM/OpenSourceWorks/GPU-ATLAS-HeteroSim/ \
  /opt/gpu-atlas/GPU-ATLAS-HeteroSim/
cd /opt/gpu-atlas/GPU-ATLAS-HeteroSim
```

若从 GitHub 获取：

```bash
git clone https://github.com/Yueq-Zhang/GPU-ATLAS-HeteroSim.git \
  /opt/gpu-atlas/GPU-ATLAS-HeteroSim
cd /opt/gpu-atlas/GPU-ATLAS-HeteroSim
```

仓库为私有时，需要先在 WSL 中完成 GitHub 身份认证，并确保当前账号有访问权限。

### 4.3 创建 Python 环境

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
```

### 4.4 构建 C++ 运行时

```bash
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --parallel
```

成功后，pybind11 模块会生成在 `frontend/hetero/` 中。Python Runner 找不到该模块时会提示先构建 simulator。

## 5. 构建与测试

每次同步新代码后执行：

```bash
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --parallel
ctest --test-dir simulator/build --output-on-failure
.venv/bin/python -m pytest tests/hetero -q
```

当前基线应通过 7 个 C++ 测试和 30 个 Python 测试。测试数量会随实现推进增加；判断成功应以“0 failed”为准，而不是永久依赖固定数量。

强制重新编译已有目标：

```bash
cmake --build simulator/build --clean-first --parallel
```

## 6. 配置校验

运行前建议先校验配置。该步骤不会启动仿真，也不会产生 Run 目录：

```bash
.venv/bin/python -m frontend.hetero.cli validate \
  --config configs/hetero/experiments/m1_model3_gpu_native_3ddram.json
```

成功时会输出 `simulation_input_key`。它是展开所有 `ref` 后对完整配置计算的 SHA-256；只要有效输入发生变化，Key 就会变化。

只验证运行请求但不执行 C++ 运行时：

```bash
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m1_model3_gpu_native_3ddram.json \
  --dry-run
```

## 7. 运行方法

### 7.1 M1：四种系统 Profile 的语义验证

```bash
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m1_model1_atlas_native.json

.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m1_model2_host_memory_pcie.json

.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m1_model3_gpu_native_3ddram.json

.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m1_model4_cxl_memory_tier.json
```

四个实验使用相同模型、请求和放置策略，只改变系统组织形式。它们验证路由种类、地址空间、KV 分配和调度语义，不计算设备真实执行时长。

### 7.2 M2：GPU + ATLAS 分析预览

```bash
.venv/bin/python -m frontend.hetero.cli validate \
  --config configs/hetero/experiments/m2_model1_analytical_preview.json

.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m2_model1_analytical_preview.json
```

该配置把 Prefill 默认放到 GPU，并把 Decode Attention 放到 ATLAS。Runner 会：

1. 构造完整请求图；
2. 生成设备任务和跨设备传输任务；
3. 用配置中的有效算力、有效内存带宽、链路带宽和固定时延估计持续时间；
4. 交给 C++ 全局事件运行时处理依赖和资源争用；
5. 输出每个任务的 ready/start/completion 时间与 TTFT、TPOT、ITL。

示例参数来源均写为 `illustrative_synthetic_not_calibrated`，只用于验证实现闭环，不能引用为 ATLAS、GPU 或链路真实性能。

### 7.3 指定输出目录

```bash
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m2_model1_analytical_preview.json \
  --runs-root /tmp/gpu-atlas-runs
```

## 8. 运行产物及人工核验

默认目录格式：

```text
runs/<experiment.name>/<simulation_input_key>/
```

| 文件 | 内容 |
|---|---|
| `resolved_config.yaml` | 已展开 `ref` 的完整输入；当前内容使用 JSON 语法，JSON 是 YAML 的合法子集 |
| `dependency_lock.yaml` | 本次运行使用的依赖版本记录 |
| `provenance.json` | Git revision、输入 Key、C++ 运行时和地址分配所有者 |
| `model_graph.json` | 每个请求的完整 Prefill/Decode 图、计数器和放置结果 |
| `execution_graph.json` | 设备任务、传输任务、依赖、参数、持续时间和调度时间 |
| `buffer_bindings.json` | Paged KV 的内存空间、偏移、逻辑/分配字节数 |
| `trace_manifest.json` | Trace 语义和 replay-safe 状态；当前没有周期 Trace |
| `event_log.jsonl` | Scheduler Epoch 或全局 DAG 任务完成记录 |
| `metrics.json` | TTFT、TPOT、ITL、E2E、Fidelity 和结果使用限制 |

从 CLI 输出复制完整 `run_dir` 后查看结果：

```bash
RUN_DIR='runs/m2_model1_analytical_preview/<simulation_input_key>'
.venv/bin/python -m json.tool "$RUN_DIR/metrics.json"
.venv/bin/python -m json.tool "$RUN_DIR/provenance.json"
```

人工核验时至少确认：

- `provenance.json.simulation_input_key` 与 CLI 输出一致；
- `provenance.json.simulator_revision` 是预期 Git commit；
- `metrics.json.run_status` 与选择的模式一致；
- `metrics.json.performance_claim_allowed` 在当前阶段为 `false`；
- `buffer_bindings.json` 中各分配不重叠且总量未超过容量；
- `execution_graph.json` 中消费者依赖对应的 route task；
- 同一 `resource_id` 的任务时间区间不重叠。

若使用第 4.2 节的 `rsync --exclude .git` 工作副本，`simulator_revision` 会显示 WSL 副本自身的 commit 并带 `-dirty`。此时还应单独记录 Windows 源仓库的 `git rev-parse HEAD`；从 GitHub 直接 clone 并在干净工作树运行时，产物会记录精确 commit。

当前 tiny golden case 的 KV 预期值：

```text
final_committed_tokens = 18
allocated_blocks       = 8
bytes_per_block        = 2048
logical_bytes          = 9216
allocated_bytes        = 16384
```

## 9. 修改模型、Batch 和算子放置

### 9.1 修改模型

复制并修改 `configs/hetero/models/tiny_llama_2layer.json`，并在实验配置的 `model.ref` 中指向新文件。模型必须满足：

```text
num_attention_heads × head_dim = hidden_size
```

### 9.2 多 Batch / 多请求

复制 `configs/hetero/workloads/tiny_e2e_single.json`，在 `requests` 中加入多个请求：

```json
{
  "requests": [
    {
      "request_id": "R0",
      "arrival_time_fs": 0,
      "prompt_length": 128,
      "output_length": 16,
      "priority": 0
    },
    {
      "request_id": "R1",
      "arrival_time_fs": 500000000,
      "prompt_length": 64,
      "output_length": 8,
      "priority": 0
    }
  ]
}
```

把实验的 `workload.ref` 指向新文件。`scheduler_validation` 会验证 admission、Prefill chunk 和 Decode priority；`analytical_preview` 会尊重请求到达时间，并让不同请求在相同设备/链路资源上排队。当前 M2 尚未把动态 Token-Step Batch 聚合为共享 Kernel，因此它是“多请求资源争用预览”，不是完整动态批处理性能模型。

### 9.3 决定算子运行在 GPU 还是 ATLAS

复制 `configs/hetero/placements/gpu_prefill_atlas_decode.json` 并编辑 `rules`。规则按顺序 first-match，可匹配 `phase`、`layer_id`、`operator_group`、`kv_len_min/max` 和 `active_batch_min`。

以下规则把 Decode Attention 放到 ATLAS，其余算子放到 GPU：

```json
{
  "mode": "rule_based",
  "unit": "device_subbatch_operator",
  "default_target": "gpu0",
  "rules": [
    {
      "match": {
        "phase": "decode",
        "operator_group": "attention"
      },
      "target": "atlas0.compute"
    }
  ],
  "data": {
    "kv_cache": {
      "home": "primary_3ddram",
      "layout": "paged",
      "page_tokens": 16
    }
  }
}
```

每次修改后应依次执行 `validate`、完整测试和目标实验。跨设备边会根据 Profile 自动 Lowering 为 DMA、同步或 CXL 行为。

## 10. 完整复现实验清单

建议随结果保存：Git commit、`dependency_lock.yaml`、原始配置、`resolved_config.yaml`、输入 Key、CTest/Pytest 记录、整个 Run 目录，以及任何实测/论文参数的来源、单位、换算和校准误差。

推荐复现顺序：

```bash
cd /opt/gpu-atlas/GPU-ATLAS-HeteroSim
git rev-parse HEAD
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --parallel
ctest --test-dir simulator/build --output-on-failure
.venv/bin/python -m pytest tests/hetero -q
.venv/bin/python -m frontend.hetero.cli validate \
  --config configs/hetero/experiments/m2_model1_analytical_preview.json
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m2_model1_analytical_preview.json
```

## 11. 常见问题

### `C++ runtime module is not built`

```bash
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --parallel
```

确认 `frontend/hetero/` 下存在 `_heterosim_runtime*.so`。

### CMake 找不到 pybind11

```bash
sudo apt install -y pybind11-dev
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
```

### 从 Windows 目录直接编译速度慢或权限异常

按第 4.2 节同步到 `/opt/gpu-atlas/`，在 WSL 原生 ext4 文件系统中构建。不要把 Windows 生成的 `.venv` 或 `simulator/build` 复制到 WSL。

### 修改配置后仍看到旧结果

Runner 按完整配置哈希选择目录。核对 CLI 输出的新 `simulation_input_key`，再查看对应路径。相同输入会稳定落到相同目录。

### 为什么分析预览不能用于论文性能结论

当前 Roofline 没有模拟 GPU Cache/NoC、ATLAS 内部流水、3D-DRAM Channel/Bank/Row 时序、真实 PCIe/CXL 协议争用或 Kernel 执行反馈。后续接入 Accel-Sim、ATLAS/Ramulator2 和协议链路模型并完成校准后，才可根据 Fidelity 门槛决定是否允许性能声明。

## 12. 开发纪律

后续修改必须先定位设计规范中的 M0-M9 阶段和强制不变量。任何拓扑、地址、Token、KV 状态或时序所有权变更，都应同步更新规范、测试和实现状态。共享 3D-DRAM 模式必须只有一个时序所有者；Trace 地址必须先稳定映射为 Global PA，再进行候选 DRAM 地址译码。
