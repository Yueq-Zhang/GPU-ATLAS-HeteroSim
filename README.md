# GPU-ATLAS-HeteroSim

GPU-ATLAS-HeteroSim 是面向 GPU、ATLAS Compute Die 与 3D-DRAM 的异构端到端 LLM 联合仿真工程。工程把完整 Prefill/Decode 请求图、算子放置、跨设备数据移动、Paged KV Cache 和全局事件调度连接到同一条可复现运行路径。

> 当前版本为 `0.6.1`。四种Profile已经拥有统一的 `full_runtime` 参考执行入口；Accel-Sim 已迁移到 v2.0.0，并在本机 RTX 3070/驱动 591.86 上完成 NVBit 1.8 `.tracez` 采集与适配器等价验证。参考运行和未校准配置始终输出 `performance_claim_allowed=false`；外部Ramulator2、RTX 4090 Trace和真实LLM Artifact的资格验证留待后续完成。

完整架构约束以 [GPU + ATLAS 异构端到端仿真实现规范](docs/gpu_atlas_heterogeneous_simulation_design_zh.md) 为准，阶段进度见 [实现状态](docs/IMPLEMENTATION_STATUS.md)。

计划与当前实现的逐项差距见 [当前完成情况与计划差距](README_PROGRESS_GAP_zh.md)。

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
- C++ Runtime Memory Planner：多Memory Space、对齐、First-Fit、释放、合并与地址复用；
- C++ Shared3DMemoryModel：GPU/ATLAS双发起方、Channel/Bank译码、父子事务拆分、轮询仲裁、背压与字节守恒；
- C++ BoundedLinkModel：PCIe/CXL队列深度、Credit、全双工序列化、传播延迟与背压；
- Static Ragged、Continuous/Chunked Prefill与按设备拆分的Device Sub-Batch计划；
- TraceAddr → TensorID+offset → Global PA 的JSONL外部内存桥协议；
- 自动DSE笛卡尔搜索、候选缓存运行和带Fidelity的排序报告；
- `full_request` 与已有KV上的独立 `decode_step` 两种工作负载语义；
- Python 配置/图控制面与 pybind11 C++ 动态运行时边界；
- `BackendDescriptor`、`ResolvedTimingContract` 与唯一时序所有者检查；
- Accel-Sim 和 ATLAS `total` 时长适配器、算子/Artifact 选择、显式分析回退与内容缓存；
- 规范化 Run 目录、输入哈希、Git revision、依赖锁和 Fidelity 标签。

## 2. 目录结构

```text
GPU-ATLAS-HeteroSim/
├── configs/hetero/              # 模型、工作负载、放置、地址和实验配置
├── docs/                         # 冻结设计规范、构建记录和实现状态
├── frontend/hetero/              # Python 配置、ModelGraph、Placement、Runner
├── scripts/                      # Accel-Sim 安装、构建、CUDA 工作负载和 Trace 采集
├── simulator/                    # C++ 运行时、内存服务、pybind11 与单元测试
├── tests/hetero/                 # Python 端到端及配置回归测试
├── workloads/cuda/               # 最小可验证 CUDA 工作负载
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
sudo apt install -y build-essential cmake python3-venv pybind11-dev rsync libzstd-dev
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

当前基线通过 9 个 C++ 测试和 63 个 Python 测试。测试数量会随实现推进增加；判断成功应以“0 failed”为准，而不是永久依赖固定数量。

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

### 7.3 第二步：Accel-Sim + ATLAS 算子事件级接线验证

先按第 13 节安装 Accel-Sim，并保证 ATLAS 位于 `/opt/atlas/ATLAS-MICRO-2026`、ATLAS Python 环境位于 `/opt/conda/envs/atlas`。然后运行：

```bash
.venv/bin/python -m frontend.hetero.cli validate \
  --config configs/hetero/experiments/step2_model1_operator_event_probe.json

.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/step2_model1_operator_event_probe.json \
  --runs-root /opt/gpu-atlas/step2-runs
```

该用例在一个完整 Prefill/Decode 图中执行：

- 一个 GPU 算子绑定官方 QV100 Backprop Trace，由 Accel-Sim 返回总周期；
- 一个 ATLAS Decode 算子绑定 ATLAS 自带 GEMM Operator/Placement，由 `atlasim.Chip` 返回总周期和能耗；
- 其他未绑定算子通过 `fallback_kind=analytical` 明确回退；
- 两个周期后端均为 `total` Contract，内部显存或 3D-DRAM 已计时，`exports=[]`，不会重复送入外部 Ramulator2；
- 两个绑定都是 `surrogate_plumbing_probe`，所以 `performance_claim_allowed=false`。

首次运行会生成 `backend_runs/gpu/` 与 `backend_runs/atlas/`；相同输入再次运行时，任务的 `backend_statistics.cache_hit` 应为 `true`。这只验证异构调度和适配器闭环，不代表真实 LLM Operator 已编译或校准。

单独复核 ATLAS 适配器的确定性：

```bash
.venv/bin/python -m frontend.hetero.cli qualify-atlas \
  --backend-config configs/hetero/backends/atlas_test_chip_16ch.json \
  --chip-config /opt/atlas/ATLAS-MICRO-2026/configs/architecture/chip/test_chip_16ch.yaml \
  --operator-list /opt/atlas/ATLAS-MICRO-2026/configs/operator_yaml/gemm_comp/gemm.yaml \
  --placement-map /opt/atlas/ATLAS-MICRO-2026/configs/operator_yaml/gemm_comp/gemm_data.yaml \
  --output /opt/gpu-atlas/qualification/atlas_test_chip_16ch
```

命令连续运行两次相同 ATLAS 输入，要求周期、能耗和全部原生统计完全一致，并生成 `qualification_record.json`。

### 7.4 指定输出目录

```bash
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m2_model1_analytical_preview.json \
  --runs-root /tmp/gpu-atlas-runs
```

### 7.5 四种Profile的完整参考运行

以下四个配置使用同一Tiny模型、三请求Continuous Batch和相同放置规则，只改变物理拓扑：

```bash
for profile in model1 model2 model3 model4; do
  .venv/bin/python -m frontend.hetero.cli run \
    --config "configs/hetero/experiments/m8_${profile}_full_runtime_reference.json"
done
```

`full_runtime` 会额外生成：

- `batch_plan.json`：Ragged序列、Epoch和Device Sub-Batch；
- `memory_lifecycle.json`：KV分配、释放、Allocation Epoch、峰值占用和复用；
- `link_statistics.json`：PCIe/CXL/同步路径事务、Credit、背压与字节守恒；
- `memory_statistics.json`：共享3D内存父子请求、地址译码、Channel分布和完成时间；
- `residency.json`：Copy、Migration、Remote或显式同步后的Owner/Version状态。

这些配置使用 `reference_unqualified` 参数。链路和共享内存响应会反向延长父任务并重新推进全局DAG，直到任务/链路/内存时间表确定性收敛；因此队列与背压会进入端到端延迟，但Fidelity仍是`event_modeled`，参数也不代表目标硬件精度已经验证。

### 7.6 GPU独占3D-DRAM、Logic Die关闭的无竞争基线

该基线表示Model 3中3D-DRAM直接作为GPU显存，但Compute/Logic Die不执行算子，也不能向共享内存服务提交请求。小模型配置适合快速功能回归：

```bash
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m8_model3_gpu_only_no_logic_die_reference.json
```

OPT-6.7B、BS=1、已有1024 Token KV、单步Decode配置用于验证完整391算子图：

```bash
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m8_opt67b_gpu_only_no_logic_die_reference.json \
  --runs-root runs/gpu-only-no-logic-die
```

运行前和运行后均有强制约束：`access_mode=gpu_only`、`backends.atlas.kind=none`、放置规则只允许`gpu0`、`initiator_order=["gpu0"]`；派生执行图中出现非GPU任务或跨设备路由时立即失败。成功结果应满足：

- `execution_graph.json`中全部任务的`device_id`为`gpu0`，且`routes=[]`；
- `memory_statistics.json`中只有`gpu0`发起方；
- `metrics.json`中`logic_die_tasks=0`且`logic_die_memory_requests=0`；
- 父请求、子事务和字节提交/完成数严格守恒。

OPT配置为控制参考模型事件数量，使用1 MiB粗粒度事务；它验证控制流、请求流、时序所有权和守恒，不是命令级DRAM模型，也不是Accel-Sim/Ramulator2周期结果，始终保持`performance_claim_allowed=false`。

### 7.7 OPT-6.7B单步Decode的RTX 4090 Roofline

该配置明确表示“已有1024 Token KV、执行一次Decode Forward”，不会把它误建模为1024 Token Prefill：

```bash
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m8_opt67b_rtx4090_roofline.json
```

当前实现基线输出约 `13.733 ms`。这是使用RTX 4090理论FP16 Tensor吞吐与1008 GB/s带宽得到的未校准Roofline结果，不是Accel-Sim周期结果或4090实测结果。

相同 OPT-6.7B 单步 Decode 图在当前 ATLAS 参考参数（10 TFLOP/s、409.6 GB/s）下为约 `33.796 ms`，而 RTX 3070/4090 Roofline 分别为约 `30.899 ms` 和 `13.733 ms`。因此当前配置下 3D-DRAM 相对两款 GPU 的加速比分别为 `0.914x` 和 `0.406x`，并未获得加速。完整口径、瓶颈与复现命令见 [GPU 与 3D-DRAM 单步 Decode 分析对比](docs/qualification/opt67b_single_decode_gpu_vs_3ddram_analytical.md)。

### 7.8 OPT-6.7B Prefill的RTX 3070 Roofline

以下配置表示FP16、BS=1、Context=1024的一次完整Prefill，不包含Decode：

```bash
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m8_opt67b_rtx3070_prefill_roofline.json
```

RTX 3070参考板参数采用约81.3 TFLOP/s Dense FP16 Tensor吞吐和448 GB/s显存带宽。当前Roofline输出TTFT约`179.558 ms`，计算图为1次Prefill、0次Decode、最终KV长度1024。该结果是未校准理论下界，并非3070实测或Accel-Sim周期结果。OPT-6.7B FP16权重约13.4 GB，超过本机8 GB显存，因此无法作为纯GPU完整FP16模型直接加载；真实运行需量化、CPU Offload或改用更小模型。

本机 CUDA 程序和 Accel-Sim 2.0/NVBit 1.8 Trace 采集均已在 RTX 3070、驱动 591.86 上通过。最小 `vector_add` 产生压缩 `.tracez`，并在 SM86 配置下得到原生与 Adapter 完全一致的 5,657 cycles、61,440 instructions。该结果验证工具链与适配器，不等于 OPT-6.7B 性能或 RTX 3070 微架构校准。

### 7.9 自动DSE

```bash
.venv/bin/python -m frontend.hetero.cli dse \
  --config configs/hetero/experiments/m8_model1_full_runtime_reference.json \
  --search configs/hetero/dse/tiny_roofline_search.json \
  --output-root runs/dse/tiny_roofline
```

搜索轴使用配置点路径，例如 `backends.gpu.effective_memory_bandwidth_Bps`。候选数受 `max_candidates` 限制，结果写入 `dse_report.json`；未经目标Backend资格验证的候选不会自动获得性能声明资格。

### 7.10 外部内存请求桥

`bridge-memory` 接收捕获Trace Manifest、候选Simulation Buffer Bindings和GPU/ATLAS请求JSONL，先完成地址正规化，再交给唯一共享内存服务：

```bash
.venv/bin/python -m frontend.hetero.cli bridge-memory \
  --trace-manifest trace_manifest.json \
  --buffer-bindings simulation_buffer_bindings.json \
  --memory-config shared_memory.json \
  --requests memory_requests.jsonl \
  --responses memory_responses.jsonl
```

当前已实现确定性的离线文件协议。把同一协议接到Accel-Sim L2 Miss的实时暂停/恢复回调，以及把内部参考服务替换为版本锁定的Ramulator2，属于后续资格验证工作。

Ramulator2独立重放适配器可先做两次确定性资格运行；后端配置格式见`configs/hetero/schemas/ramulator2_backend.schema.json`，请求数组中的地址必须已经是Global PA偏移：

```bash
.venv/bin/python -m frontend.hetero.cli qualify-memory \
  --backend-config ramulator2_backend.json \
  --requests physical_memory_requests.json \
  --output qualification/ramulator2
```

`full_runtime`若配置`kind=ramulator2`但在线回调尚未完成资格验证，会明确拒绝运行，不会静默改用内部参考模型。

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
| `trace_manifest.json` | 本次使用的 GPU Trace 与 ATLAS Artifact、兼容性、任务绑定和 replay-safe 状态 |
| `event_log.jsonl` | Scheduler Epoch 或全局 DAG 任务完成记录 |
| `metrics.json` | TTFT、TPOT、ITL、E2E、Fidelity 和结果使用限制 |
| `batch_plan.json` | `full_runtime`的Ragged Epoch和Device Sub-Batch |
| `memory_lifecycle.json` | 动态KV分配、释放、峰值占用和地址复用 |
| `link_statistics.json` | 有界链路事务、Credit、背压和字节计数 |
| `memory_statistics.json` | 共享内存请求、DRAM译码和守恒计数 |
| `residency.json` | Copy/Migration/Remote/Sync后的Residency状态 |

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

## 13. M5 第一步：独立 Accel-Sim 后端

这一阶段只验证 GPU 周期仿真器本身，尚不把 GPU L2 Miss 送入 ATLAS/Ramulator2。时序所有权固定为：

```text
CUDA/TileLang -> GPU Binary -> NVBit Trace -> Accel-Sim
                                             ├─ SM/Core
                                             ├─ L1/L2
                                             ├─ NoC
                                             └─ GPU Local DRAM
```

Accel-Sim 返回完整 Kernel/Trace 的总 Cycle，适配器按配置的 GPU Core 频率换算为整数飞秒。外部 Ramulator2 在本模式下必须关闭；共享 3D-DRAM Bridge 属于后续第三步。

### 13.1 安装并构建固定版本

要求 WSL 中存在 `/usr/local/cuda-11.8/bin/nvcc`，并已安装 `libzstd-dev`。脚本固定到 Accel-Sim v2.0.0（commit `64653015...`）、配套 GPGPU-Sim dev（commit `e10018b...`）和 NVBit 1.8，并对每个下载包校验 SHA-256：

```bash
cd /opt/gpu-atlas/GPU-ATLAS-HeteroSim
bash scripts/install_accel_sim.sh
bash scripts/build_accel_sim.sh
```

默认依赖目录是 `/opt/gpu-atlas/dependencies`。需要改变位置时，在安装、构建和运行阶段统一设置 `ACCEL_SIM_DEPS_ROOT`。

### 13.2 编译并验证最小 CUDA 工作负载

```bash
bash scripts/build_cuda_workload.sh
```

成功标志为：

```text
vector_add verified: 4096 elements
CUDA workload built and verified: .../build/workloads/vector_add
```

### 13.3 采集 Trace

```bash
bash scripts/capture_accel_sim_trace.sh \
  build/workloads/vector_add \
  /opt/gpu-atlas/traces/vector_add_sm86
```

本机 RTX 3070、NVIDIA 驱动 591.86 已通过 NVBit 1.8 采集验证。Tracer 先写入原始 `*.trace.xz`，后处理器默认生成 Accel-Sim 2.0 的 `*.tracez` 和 `kernelslist.g`；项目的 Trace Cache 同时兼容旧 `*.traceg` 与新 `*.tracez`。默认输出清单位于 `<trace output>/traces/kernelslist.g`。

### 13.4 编写 Trace Manifest

每份 Trace 都必须有独立 JSON Manifest。最小结构如下，其中 `replay_safe` 初始必须是 `false`：

```json
{
  "schema_version": "hetero-trace-manifest/v1",
  "trace_id": "vector_add.cuda11_8.sm86",
  "trace_semantics": "functional",
  "replay_safe": false,
  "qualification_record": null,
  "kernels_list": "/opt/gpu-atlas/traces/vector_add_sm86/traces/kernelslist.g",
  "capture": {"tool": "NVBit", "version": "1.8", "gpu": "RTX 3070"},
  "compilation": {"cuda": "11.8", "target_sm": 86},
  "address_ranges": [
    {
      "capture_allocation_id": "cuda.alloc.1",
      "trace_base": "0x7f2000000000",
      "size_bytes": 16384,
      "tensor_id": "input_a",
      "tensor_offset_bytes": 0,
      "capture_epoch": 0,
      "backing_allocation_id": "input_a.storage",
      "view_offset_bytes": 0,
      "alignment_bytes": 256,
      "shape": [4096],
      "layout": "contiguous_fp32"
    }
  ]
}
```

地址层次严格分开：`TraceAddr -> TensorID + Offset -> PhysicalAddress(memory_space_id, offset) -> DRAM Tuple`。Manifest 只保存第一段采集绑定；每个仿真候选通过独立 `SimulationBufferBinding` 决定当前 PhysicalAddress，Channel/Bank/Row/Column 再由具体内存配置在 LLC Miss 后译码。候选分配和 DRAM 译码都不进入 `trace_key`。同一 Trace 只有在资格记录通过且没有执行时序反馈时，才允许跨 DRAM 候选复用。

### 13.5 独立资格验证

```bash
.venv/bin/python -m frontend.hetero.cli qualify-gpu \
  --backend-config configs/hetero/backends/gpu_accelsim_rtx3070.json \
  --trace-manifest configs/hetero/traces/local_rtx3070_vector_add_v2.json \
  --output /opt/gpu-atlas/qualification/accel-sim-v2/rtx3070-vector-add-v2/qualification
```

命令连续执行一遍原生基线和一遍适配器路径，并要求 `gpu_tot_sim_cycle`、`gpu_tot_sim_insn` 完全一致。输出包括：

- `native_baseline/command.json`、日志和 `stats.json`；
- `adapter/command.json`、日志和 `stats.json`；
- `qualification_record.json`；
- `adapter_qualified_trace_manifest.json`，记录适配器资格结果，但仍保持 `replay_safe=false`。

资格验证只证明适配器没有改变固定版本 Accel-Sim 的独立执行结果，不证明跨 DRAM 配置 Replay 安全，不证明 RTX 3070 配置已经完成实机微架构校准，也不证明 GPU+ATLAS 联合仿真已经完成。只有后续显式覆盖时序反馈、同步、Atomics、动态控制流和地址行为的 Replay 资格记录，才能把 `replay_safe` 提升为 `true`。

仓库同时提供 `gpu_accelsim_qv100.json`，专门用于 Accel-Sim 官方 V100 预采集 Trace 的适配器回归。Trace 与硬件配置必须匹配：官方 SM70 Trace 不应套用 RTX 3070/SM86 配置；它能验证软件适配器，但不能替代后续 RTX 3070 Trace 的目标平台校准。

官方回归 Trace 的完整复现命令为：

```bash
bash scripts/download_official_accel_sim_trace.sh
.venv/bin/python -m frontend.hetero.cli qualify-gpu \
  --backend-config configs/hetero/backends/gpu_accelsim_qv100.json \
  --trace-manifest configs/hetero/traces/official_qv100_backprop_4096.json \
  --output /opt/gpu-atlas/qualification/accel-sim-v2/qv100-backprop-4096
```

在 v2.0.0 下，QV100 旧格式 Trace 的原生与 Adapter 结果均为 14,731 cycles、10,473,824 instructions；这与 v1.3.0 的 15,329 cycles 不同，因此升级后必须重新生成仿真基线，不能沿用旧版本周期数。

## 14. GPU + 3D-DRAM Cycle-Accurate 请求/响应模式

本模式不使用 Roofline，也不把 `compute_time` 与一次性汇总的访存字节数相加。执行闭环为：

```text
GPU Warp/Instruction
        ↓
Accel-Sim SM → L1 → L2 → NoC
                         ↓ L2/MC request（地址 + mem_fetch）
                 唯一 Ramulator2 实例
                 Channel/Bank/Row 时序推进
                         ↓ completion callback
              原 memory partition 返回响应
                         ↓
                Cache/Warp 解除等待并继续
```

当前首个资格模式固定为 GPU-only：ATLAS Logic Die 不发请求，因此不含 GPU/PIM 竞争；Accel-Sim 负责 GPU Core/L1/L2/NoC，Ramulator2 是唯一 DRAM 时序所有者。全部 GPU memory partition 连接同一个 Ramulator2，而不是每个 partition 各建一个内存系统。读请求必须等待回调；写回采用 posted-write 语义，但进程退出前会排空 Ramulator2 写队列。

### 14.1 构建

先按 4.2 节把 Windows 工程同步到 WSL，然后运行：

```bash
cd /opt/gpu-atlas/GPU-ATLAS-HeteroSim
bash scripts/build_accel_sim_ramulator2.sh
```

输出包括：

- `build-ramulator2/accel-sim.out`：带外部 DRAM 回调的 Accel-Sim v2；
- `libramulator_gpgpusim_bridge.so`：单实例 Ramulator2 桥；
- `ramulator_bridge_smoke`：4 个 GPU partition 共享一个实例的最小测试。

### 14.2 资格运行

```bash
.venv/bin/python -m frontend.hetero.cli qualify-gpu \
  --backend-config configs/hetero/backends/gpu_accelsim_qv100_ramulator2_hbm3.json \
  --trace-manifest configs/hetero/traces/official_qv100_backprop_4096.json \
  --output /opt/gpu-atlas/qualification/accel-sim-v2/qv100-backprop-4096-ramulator2-hbm3-32ch-no-fixed-dram-latency
```

资格检查要求两次运行的 GPU 周期、指令数和全部桥统计完全一致，并强制检查：`instances=1`、请求数非零、`completed=reads+writes`、`outstanding=0`。任何一项不满足都会失败，不能静默退回 Accel-Sim 内部 DRAM。

当前通过结果为 14,700 GPU cycles、10,473,824 instructions；Ramulator2 为 11,038 cycles、63 reads、63 completed、0 outstanding。相同 Trace 的原生内部 DRAM结果为 14,731 GPU cycles，说明外部 DRAM完成时刻已经反馈到 GPU 周期推进。耦合专用 Trace 配置显式设置 `-dram_latency 0`，避免在 Ramulator2 时序之前再次叠加 Accel-Sim 的固定 DRAM 延迟；GPU ROP/L2/NoC 延迟仍然保留。完整证据见 [Cycle-Accurate 资格记录](docs/qualification/qv100_backprop_ramulator2_hbm3_cycle_coupled.md)。

`ramulator2_hbm3_32ch_gpu_only.yaml` 目前是 HBM3 32 通道功能配置，并非已经按 ATLAS 论文中的 Stack/Logic Die 参数完成校准。RTX 3070 的 4096 元素 `vector_add` Trace 中，全局读被预加载数据命中 L2，未产生外部读请求，因而不用于这项请求闭环资格验证。后续 LLM 评估必须采集能覆盖目标 GEMM/Attention/KV 算子的精确 Trace，不能由该 Backprop 微基准外推。

### 14.3 下一阶段：外部链路与Logic Die内部事务分层

当前Bridge是最小资格原型：一个GPU `mem_fetch`直接对应一个Ramulator2请求。正式Model 3/4周期路径将升级为：

```text
GPU Parent Request
  -> 外部请求Link（带宽/协议/队列/Credit）
  -> Singleton LogicDieMemoryGateway
  -> 按Global PA、Size和Byte/Sector Mask拆分N个内部Child
  -> 唯一Ramulator2完成全部Child
  -> Parent Join
  -> 外部响应Link
  -> GPU完成
```

外部GPU↔Logic Die带宽与Logic Die↔3D-DRAM内部带宽是不同资源，允许且通常满足`B_external < B_internal`。读请求只有在全部Child和响应Link完成后才能解除GPU阻塞；写请求默认采用durable确认，posted write必须显式配置并在退出前排空。

当前HBM3候选配置还没有使`DQ`、默认`channel_width`、Burst、`tCK`和事务大小导出一致的峰值带宽，因此现有Backprop结果只验证请求/回调闭环，不验证有效带宽。实施顺序和验收条件以[设计规范v1.4](docs/gpu_atlas_heterogeneous_simulation_design_zh.md)和[进度/差距表](README_PROGRESS_GAP_zh.md)为准。
