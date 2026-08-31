# GPU-ATLAS-HeteroSim

GPU-ATLAS-HeteroSim 是面向 GPU、ATLAS Compute Die 与 3D-DRAM 的异构端到端 LLM 联合仿真工程。工程把完整 Prefill/Decode 请求图、算子放置、跨设备数据移动、Paged KV Cache 和全局事件调度连接到同一条可复现运行路径。

> 当前版本为 `0.24.0`，P16固定Shape的全任务请求周期建模已完成。20个任务无分析回退地进入同一因果时间线：15个GPU Trace实例和3个KV运行时实例具备请求周期资格，2个主机控制事件被显式排除在设备性能边界之外。硬件参数仍未校准，继续禁止把当前结果作为端到端性能结论。

长时间资格验证可以部署到`192.168.0.197`并把两个确定性Leg绑定到不同CPU并行执行；密码不进入仓库或日志。远端路径、GPT‑5.6 Luna `xhigh`编排约定、单轮入口和完成后严格合并方法见[远端验证规范](docs/REMOTE_VALIDATION.md)。

完整架构约束以 [GPU + ATLAS 异构端到端仿真实现规范](docs/gpu_atlas_heterogeneous_simulation_design_zh.md) 为准，阶段进度见 [实现状态](docs/IMPLEMENTATION_STATUS.md)。

计划与当前实现的逐项差距见 [当前完成情况与计划差距](README_PROGRESS_GAP_zh.md)。

## 1. 当前已实现的能力

- 完整的 decoder-only LLM 请求图：覆盖 Prefill、逐 Token Decode、KV Append、LM Head 和 Sampling，并支持 Static/Ragged Batch、Continuous/Chunked Prefill 与设备级 Sub-Batch；
- 算子级 GPU/ATLAS 放置与四种系统组织：ATLAS 独立模式、GPU 独立显存 + 3D-DRAM 主存、3D-DRAM 直接作为 GPU 显存，以及 CXL 内存扩展；
- 严格的全局事件与多请求调度：按 DAG 依赖、互斥资源、Token-Step Barrier 和请求完成回调推进，校验 read→compute→write、Fence 与版本提交因果；
- 确定性 Global PA 与数据一致性：支持多 Memory Space、对齐/容量/重叠检查、Paged KV、值版本、Residency，以及 Copy、Migration、Remote 和显式 Fence；
- GPU–3D-DRAM 周期级请求链路：外部 PCIe/CXL/Direct Link、Logic-Die Gateway、内部 Hybrid-Bond 和 DRAM 可使用独立时钟/带宽，Parent 按字节掩码拆为 64 B Child，全部 durable 完成后才返回响应；
- 真实 Accel-Sim v2 与完整 `atlasim.Chip` 可同进程、多时钟推进；GPU 与 ATLAS 可争用同一套 Channel/Bank，并由唯一 Ramulator2 负责 3D-DRAM 时序；
- 真实 GPU 算子 Artifact 流程：SM86 Trace 采集、Trace Manifest、Range-Rebase 到 Global PA、Accel-Sim 双遍资格验证与 `request_cycle_ready` Catalog；未合格算子只能显式使用分析回退；
- 可审计的[算子建模与测试状态表](docs/OPERATOR_MODELING_STATUS.md)：机器可读 Catalog 固定模型 Revision、Batch、Context、Q/KV 长度、dtype 与模型维度，并分别记录“已建模、已测试、请求周期 Ready、性能可用”；
- 可复现的实验与 DSE：配置/产物内容哈希、Simulation Key、运行缓存、依赖锁、Fidelity 标签、候选搜索和规范化报告；
- 当前验证边界：P16已双遍验证固定TinyLlama Layer-0、FP16、BS=1、Context=16的20任务时间线；每遍完成7,003,497个GPU Parent和517个KV运行时Parent，依赖、资源、Global PA、唯一Ramulator2、请求守恒和18次版本提交全部通过。运行时与整机参数仍未校准，因此暂不给出端到端性能结论。

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

当前基线通过 9 个 C++ 测试和 90 个 Python 测试。测试数量会随实现推进增加；判断成功应以“0 failed”为准，而不是永久依赖固定数量。

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

从P10b-A开始，Backend不再在执行图构造阶段提前运行。`python.OnlineOperatorRuntime`按照模拟时间启动任务，并生成`online_dispatch.json`；每个Device Task的`backend_launch_time_fs`必须不早于全部依赖的完成时间，`validated_input_versions`记录启动前通过检查的值版本。总时长Backend仍在独立子进程内完成，不能把该模式表述为请求级共享Ramulator2耦合。

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

P10a 后，`execution_graph.json`还包含`placement_contract`与`residency_plan`：前者要求`logical_node_count == materialized_device_task_count`且`each_logical_node_exactly_once=true`；后者为每个Read/Write/Route保留值版本。`residency.json`使用`hetero-residency/v2`，把这些事件绑定到实际任务时间。当前外部输入采用显式记录的`first_consumer_binding`策略；它不是VA→PA翻译，也不替代后续的Simulation Buffer Binding。

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
| `online_dispatch.json` | `operator_event`的Backend启动顺序、模拟启动时间、版本检查次数和最终版本 |
| `request_cycle_trace.json` | `prefill_cycle`的父请求地址、读写类型、代表字节、发出/完成周期与Initiator |
| `global_memory_map.json` | 物化Tensor到Global PA的确定性分配、容量和非重叠证明 |
| `prefill_artifact_coverage.json` | 每个Prefill任务的周期契约覆盖，要求无分析回退 |

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

真实周期 Artifact 不是按算子名称通用复用的。修改 `hidden_size`、`intermediate_size`、Head 数、`head_dim`、`vocab_size`、dtype 或 checkpoint revision 后，即使算子名称不变，也必须生成新的 Shape/Model Contract、重新编译或捕获 Trace、执行 Range-Rebase 双跑资格，并重新验证全局时间线。当前代码会在模型合同不一致时拒绝旧 Artifact。

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

Batch 或 Context 改变会改变 Grid/Block、Tensor Core 指令、内存事务、缓存命中、Workspace、KV 容量和 DRAM 地址分布；Attention 还包含随序列长度增长的二次项。因此禁止直接按 Token 数、Batch 或参数量缩放现有周期。只有 [算子状态表](docs/OPERATOR_MODELING_STATUS.md) 中列出的精确 Shape 可复用当前资格结果，其他 Shape 必须重新捕获和验证。

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
               GPU Parent Request
                         ↓
            外部请求Link（带宽/协议/Credit）
                         ↓
              LogicDieMemoryGateway
              Parent Table + 64B Child拆分
                         ↓
                 唯一 Ramulator2 实例
                 Channel/Bank/Row 时序推进
                         ↓ 全部Child完成
              外部响应Link → Parent Join
                         ↓
                Cache/Warp 解除等待并继续
```

Accel-Sim负责GPU Core/L1/L2/NoC，Ramulator2是唯一DRAM时序所有者。全部GPU Memory Partition连接同一个Gateway和Ramulator2，而不是每个Partition各建一个内存系统。GPU读请求必须等待全部Child和响应Link；写请求默认使用durable确认。ATLAS端口接收原生`atlasim::ComponentInput`，复用ATLAS `HBFrontend`的Tile遍历、地址对齐和读写生成规则，但从内部Hybrid-Bond端口进入Gateway，不经过GPU外部Link。

### 14.1 构建

先按 4.2 节把 Windows 工程同步到 WSL，然后运行：

```bash
cd /opt/gpu-atlas/GPU-ATLAS-HeteroSim
bash scripts/build_accel_sim_ramulator2.sh
```

输出包括：

- `build-ramulator2/accel-sim.out`：带外部 DRAM 回调的 Accel-Sim v2；
- `libramulator_gpgpusim_bridge.so`：单实例 Ramulator2 桥；
- `ramulator_bridge_smoke`：GPU分层请求/响应路径测试；
- `dual_initiator_smoke`：GPU外部端口与ATLAS内部端口共享DRAM的三组黄金对照。

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

### 14.3 已完成：外部链路与Logic Die内部事务分层

当前Bridge ABI v2的数据路径为：

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

配置加载时会同时校验`DQ/channel_width/rate/nBL/tCK/prefetch/transaction_bytes`。当前ATLAS风格候选为16通道、每通道512-bit、400 MT/s、64B事务，对应内部峰值`409.6 GB/s`；GPU外部直连PHY默认`12.8 GB/s`。资格命令：

```bash
bash scripts/qualify_gpu_only_memory_path.sh \
  /opt/gpu-atlas/qualification/gpu-only-layered-memory-path

bash scripts/qualify_dual_initiator_memory_path.sh \
  /opt/gpu-atlas/qualification/dual-initiator-memory-path
```

GPU分层用例分别构造外部Link瓶颈和内部DRAM瓶颈，并检查Parent/Child、Payload/Wire Byte、durable完成和独立时钟比。双发起方用例结果为：GPU-only `163` DRAM cycles，ATLAS-only `90`，并发`239`；并发时GPU与ATLAS完成时间都变长，且始终只有一个Ramulator2实例。该ATLAS端口资格验证的是原生`ComponentInput`访问合同和共享内存竞争，还不是完整`atlasim.Chip`调度器与Accel-Sim同时推进。

### 14.4 TinyLlama真实Q投影的形状匹配资格运行

当前第一个真实LLM Artifact固定为`TinyLlama/TinyLlama-1.1B-Chat-v1.0`、revision `fe8a4e...`、layer-0 `q_proj`、FP16、BS=1、已有KV长度1024、单步Decode，矩阵形状为`M=1, K=2048, N=2048`。GPU侧Trace包含一个CUTLASS WMMA GEMM Kernel和一个Split-K Reduction Kernel；ATLAS侧把N维按16核均分为每核128列，使用`1×512×16` Tile。

重新生成ATLAS Artifact：

```bash
/opt/conda/envs/atlas/bin/python scripts/generate_atlas_qproj_artifact.py \
  --output configs/hetero/atlas/tinyllama11b_qproj_decode_bs1_ctx1024
```

在已有本地Checkpoint上重新采集GPU Trace：

```bash
export ACTIVE_FROM_START=1
export DYNAMIC_KERNEL_RANGE=4-5
bash scripts/capture_accel_sim_trace.sh \
  /opt/conda/envs/qserve-local/bin/python \
  /opt/gpu-atlas/qualification/tinyllama-qproj-decode-sm86-kernels4-5 \
  workloads/python/tinyllama_q_projection.py \
  --model /opt/hf-cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6 \
  --phase decode --context 1024
```

三条资格路径：

```bash
.venv/bin/python -m frontend.hetero.cli qualify-gpu \
  --backend-config configs/hetero/backends/gpu_accelsim_rtx3070.json \
  --trace-manifest configs/hetero/traces/local_rtx3070_tinyllama11b_qproj_decode_v2.json \
  --output /opt/gpu-atlas/qualification/accel-sim-v2/rtx3070-tinyllama11b-qproj-decode-ctx1024

.venv/bin/python -m frontend.hetero.cli qualify-gpu \
  --backend-config configs/hetero/backends/gpu_accelsim_rtx3070_ramulator2_hbdram_edge_16ch.json \
  --trace-manifest configs/hetero/traces/local_rtx3070_tinyllama11b_qproj_decode_v2.json \
  --output /opt/gpu-atlas/qualification/accel-sim-v2/rtx3070-tinyllama11b-qproj-decode-ctx1024-shared-hbdram

/opt/conda/envs/atlas/bin/python -m frontend.hetero.cli qualify-atlas \
  --backend-config configs/hetero/backends/atlas_test_chip_16ch.json \
  --chip-config configs/hetero/atlas/tinyllama_qproj_edge_16core_chip.yaml \
  --operator-list configs/hetero/atlas/tinyllama11b_qproj_decode_bs1_ctx1024/operator_description.yaml \
  --placement-map configs/hetero/atlas/tinyllama11b_qproj_decode_bs1_ctx1024/data_placement.yaml \
  --output /opt/gpu-atlas/qualification/atlas/tinyllama11b-qproj-decode-bs1-ctx1024-edge16
```

已验证结果：RTX 3070原生显存为`36,324 cycles / 32.088 µs`；GPU经12.8 GB/s外部Link访问409.6 GB/s内部3D-DRAM为`1,498,113 cycles / 1,323.421 µs`；ATLAS内部3D-DRAM为`24,613 cycles / 24.613 µs`。这三项是同Checkpoint、同算子和同Shape，但计算微架构不同；只可作为当前配置研究结果，不能外推为整层、端到端模型或实测硬件加速比。详细证据见[TinyLlama Q投影资格对比](docs/qualification/tinyllama11b_qproj_gpu_vs_atlas.md)。

### 14.5 P9a：完整ATLAS Chip实时共享内存

P9a不再用合成`ComponentInput`直接压端口，而是运行完整`atlasim.Chip`。补丁令ATLAS在外部模式下不构造第二个Ramulator2：`Core::pre_simulate()`只捕获各迭代真实DRAM输入，运行时通过外部服务提交、重试、等待完成，同时Matrix/Vector/Buffer继续按ATLAS周期推进。

构建和资格运行：

```bash
bash scripts/build_accel_sim_ramulator2.sh
bash scripts/build_atlas_full_chip_runtime.sh
bash scripts/qualify_full_chip_scheduler_memory_path.sh \
  /opt/gpu-atlas/qualification/full-chip-scheduler-memory-path-20260828-p9a-final
```

固定TinyLlama Q投影结果如下：

- ATLAS-only：完整Chip在`63,681`个ATLAS周期完成，对应`76,418`个1.2 GHz GPU全局推进周期；
- 加入4,096个确定性GPU Parent后：ATLAS完成推迟到`81,329`个GPU周期，完整Chip记录`67,774`个ATLAS周期；
- ATLAS产生139,456个64B Parent（8,925,184 B），GPU产生4,096个128B Parent；二者全部完成，唯一Ramulator2退出时`outstanding=0`；
- 旧ATLAS原生统计为8,916,992 B，其中每核8次32B输出写按逻辑字节计数；实时共享路径按全部对齐64B事务计数，因此两者相差8,192 B，资格守恒以实时Parent/Child事务为准。

此项只证明“完整ATLAS Chip调度器 + 共享Ramulator2 + 确定性GPU内存流量”闭环。`coverage.accel_sim_compute_backend=false`，所以不能表述为完整ATLAS Chip已经与真实Accel‑Sim Kernel并发；该闭环是下一步P9b的基础。详细证据见[完整ATLAS Chip共享内存资格](docs/qualification/tinyllama_qproj_full_atlas_chip_shared_memory.md)。

### 14.6 P9b：真实Accel-Sim与完整ATLAS Chip并发

P9b把完整Chip运行时编译进Accel-Sim使用的共享内存桥。GPU每推进一个Core周期，统一推进器按频率比轮询ATLAS完成、推进Chip并提交新的Logic-Die请求；当GPU Kernel先结束时，`active()`合同仍会保持仿真，直到ATLAS和共享内存均完成。GPU只能取GPU完成，ATLAS只能取ATLAS完成，二者不能误消费对方Payload。

```bash
bash scripts/build_accel_sim_ramulator2.sh
bash scripts/build_atlas_full_chip_runtime.sh
bash scripts/qualify_accel_sim_full_chip_concurrency.sh \
  /opt/gpu-atlas/qualification/accel-sim-v2/rtx3070-tinyllama-qproj-full-atlas-chip-shared-memory-p9b
```

固定TinyLlama layer-0 `q_proj`竞争用例的单次结果为：GPU `1,541,401 cycles / 15,908,352 instructions / 262,272 Parent`；ATLAS完整Chip `141,255 cycles / 139,456 Parent`，在第`159,901`个GPU周期完成；唯一Ramulator2接收`401,728`个Parent和`401,728`个Child，全部完成且`outstanding=0`。ATLAS事务量为`8,925,184 B`，GPU与ATLAS在时间上重叠。

该用例故意让同一Shape同时在两个设备执行，只用于验证计算后端并发和共享DRAM竞争，不能作为算子放置策略、端到端延迟或加速比。正式双次确定性证据见[P9b真实双计算后端资格](docs/qualification/tinyllama_qproj_accelsim_full_atlas_chip_concurrency.md)。

### 14.7 P10a：单放置与版本化Residency控制面

正常执行图先经过`build_single_placement_plan`。放置决策必须与逻辑节点一一对应；每个Read引用确定的值版本；写入生成下一版本；跨设备消费者为每个输入值分别插入路由任务。Model 3路由携带`writeback → release_fence → invalidate → acquire_fence`动作，Model 1/2/4继续按各自拓扑Lowering。

```bash
.venv/bin/python -m pytest tests/hetero/test_single_placement.py -q
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m8_model3_full_runtime_reference.json \
  --runs-root /tmp/gpu-atlas-p10a
```

固定Model 3参考配置包含228个逻辑节点和228个设备任务，`each_logical_node_exactly_once=true`；逐值Lowering产生28条同步路由与571条带时间戳Residency事件。该结果验证控制面、事件级Link/Memory闭环与版本守恒，不表示这些228个算子都已经拥有真实Accel-Sim/ATLAS周期Artifact。P9b使用的`co_resident_atlas`配置现在必须声明`execution_semantics=contention_stress_duplicate_operator`，正常`operator_event` Dispatcher会拒绝它。完整证据与声明边界见[P10a单放置与Residency资格](docs/qualification/p10a_single_placement_residency.md)。

### 14.8 P10b-A：依赖与版本门禁后的真实Backend启动

`operator_event`现在先构造严格执行计划，再由`OnlineOperatorRuntime`按`(time_fs, priority, insertion_sequence)`推进。Route完成时才把指定版本登记到目标设备；Device Task启动前逐个核对`value_id/version/device`；Task完成时才提交输出的新版本。Backend Dispatch数量必须严格等于Device Task数量。

```bash
.venv/bin/python -m pytest tests/hetero/test_online_operator_runtime.py -q
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/step2_model1_operator_event_probe.json \
  --runs-root /opt/gpu-atlas/qualification/p10b-a-online-dispatch-run1
```

Step 2真实适配器用例包含85个逻辑节点、85次Backend Dispatch、12条Route和117次版本检查。一个官方QV100 Trace由Accel-Sim执行14,731 cycles，一个ATLAS GEMM Artifact执行48,446 cycles；GPU与ATLAS启动时刻分别为8,192,002 fs和14,248,172,886 fs，均等于各自最大依赖完成时刻。两个独立输出目录的`online_dispatch.json`和`metrics.json`逐字节一致。两项绑定仍是`surrogate_plumbing_probe`，P10b-A只资格化总时长真实适配器的启动门禁，不资格化跨设备请求级周期交互。详细记录见[P10b-A在线Backend门禁资格](docs/qualification/p10b_a_online_backend_gate.md)。

### 14.9 P10b-B 至 P14：完整Prefill部署

P10b-B把严格计划接到唯一live Ramulator2。GPU请求经过12.8 GB/s外部链路；ATLAS请求从Logic Die内部端口进入；Model 3跨设备Route等待生产者写完成，再执行Fence和消费者Acquire探测。每个算子只有在采样输入请求全部完成后才推进其显式分块计算周期，计算结束后才发出输出写请求。

```bash
bash scripts/qualify_prefill_p10b_to_p14.sh \
  /opt/gpu-atlas/GPU-ATLAS-HeteroSim \
  /opt/gpu-atlas/qualification/prefill-p10b-to-p14-final
```

阶段配置为：

- P10b-B：单层Context=16，`causal_attention`放到ATLAS，其余任务放GPU；20任务、4 Route、GPU/ATLAS Parent为347/35；
- P12：单层Context=16 GPU-only；20任务、378个GPU Parent、ATLAS为0；
- P13：22层Context=16 GPU-only；272任务、448地址区间、3,382个GPU Parent；
- P14：TinyLlama‑1.1B FP16、BS=1、Context=1024、22层GPU-only完整Prefill；272任务、3,385个GPU Parent、最终KV长度1024，Global PA占用3,957,580,290 B / 4 GiB。

四个阶段均运行两次并逐字节比较七类核心产物，所有Parent完成、唯一Ramulator2、`outstanding=0`、周期契约覆盖100%、分析回退为0。P14输出的26,644.55 µs只是当前“分块周期契约 + 有界代表内存请求”部署值；`trace_coverage=0`、`extrapolated_fraction=1.0`、`performance_claim_allowed=false`，不可称为Accel-Sim全指令Trace端到端延迟、完整ATLAS Artifact结果或实机性能。详细记录见[P14完整Prefill部署资格](docs/qualification/p14_prefill_bs1_ctx1024.md)。

### 14.10 P15第一批：真实算子Artifact与选择性完整流量

P15把Artifact身份从实验中的弱Selector提升为带文件哈希的形状锁定合同。兼容键至少包含Checkpoint Revision、模型规格名、Operator、Phase、Layer、Batch、Context、Q/KV长度和Dtype；地址边界固定为`Capture Address → TensorID+offset → 运行期Global PA → 候选DRAM Tuple`。加载时会重新验证元数据、Kernel List、全部Trace、资格记录和ATLAS YAML的SHA-256。绑定到错误Context的`exact_operator`会在Backend启动前失败。

首批固定为TinyLlama‑1.1B、layer 0、FP16、BS=1、Context=16：

| 算子 | GPU/状态Artifact | 独立资格 | ATLAS |
|---|---|---:|---|
| `attention_norm` | 8个SM86 Kernel | 58,736 cycles / 5,290,064 instructions | 尚无 |
| `qkv_projection` | 6个SM86 Kernel | 95,151 cycles / 34,943,066 instructions | 16核`M=16,K=2048,N=2560`，150,932 cycles |
| `rope` | 19个SM86 Kernel | 127,094 cycles / 12,589,812 instructions | 尚无 |
| `kv_append` | `runtime_state`，NVBit Kernel为0 | CUDA D2D状态更新；等待完整写流量Lowering | 尚无 |
| `causal_attention` | 3个SM86 Kernel | 34,923 cycles / 11,962,112 instructions | 尚无 |

重新捕获单个GPU算子：

```bash
scripts/capture_tinyllama_prefill_operator.sh attention_norm 16
scripts/capture_tinyllama_prefill_operator.sh qkv_projection 16
scripts/capture_tinyllama_prefill_operator.sh rope 16
scripts/capture_tinyllama_prefill_operator.sh kv_append 16
scripts/capture_tinyllama_prefill_operator.sh causal_attention 16
```

严格绑定的总时长运行：

```bash
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/p15a_tinyllama_prefill_1layer_ctx16_gpu_operator_artifacts.json \
  --runs-root /opt/gpu-atlas/qualification/p15a/operator-event-run1
```

该运行20个任务中4个使用真实Accel-Sim Trace，Trace Coverage为20%，其余16个仍为分析回退；它只验证依赖门禁和强兼容绑定，不是端到端性能结果。

选择性完整流量运行：

```bash
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/p15b_tinyllama_prefill_1layer_ctx16_first_batch_full_traffic.json \
  --runs-root /opt/gpu-atlas/qualification/p15b/full-traffic-run1
```

双跑结果逐字节一致：5个首批任务使用完整64B Value事务，产生175,936个Full-Traffic Parent；其余15个任务产生234个采样Parent，总计176,170个Parent，全部由唯一Ramulator2完成，`outstanding=0`。DRAM为541,940 cycles，全局GPU时钟推进1,625,820 cycles。该路径仍使用P11分块计算周期，而不是让Accel-Sim Kernel在同一live Ramulator2上暂停/恢复；`request_cycle_coverage_complete=false`和`performance_claim_allowed=false`保持不变。汇总记录位于：

```text
/opt/gpu-atlas/qualification/p15b/first-batch-final/qualification_record.json
```

字段定义、逐项结果、复现入口和声明边界见[P15首批算子资格记录](docs/qualification/p15_first_batch_prefill_ctx16.md)。

### 14.11 P15c四算子真实指令—共享内存闭环

P15c复用P9b已验证的Accel-Sim外部内存补丁：`mem_fetch`从L2/Memory Partition进入外部Link与Logic-Die Gateway，全部内部Child在唯一Ramulator2完成并通过响应Link后，原请求才进入GPU ReturnQ。四个固定Trace均使用RTX 3070配置双跑，完整外部内存统计逐项一致：

| 算子 | GPU cycles | Instructions | GPU Parent / Child | DRAM cycles |
|---|---:|---:|---:|---:|
| RMSNorm | 66,653 | 5,290,064 | 2,176 / 2,176 | 23,552 |
| QKV Projection | 2,170,258 | 34,943,066 | 376,212 / 376,238 | 766,875 |
| RoPE | 135,833 | 12,589,812 | 2,312 / 2,312 | 47,997 |
| Causal Attention | 43,500 | 11,962,112 | 2,560 / 2,560 | 15,371 |

每次资格运行的Ramulator2实例数均为1，全部Parent和Child完成，`outstanding=0`且ATLAS Parent为0。QKV的Parent/Child数量不同是非对齐或跨事务边界Parent发生64B Child拆分，不是请求丢失。

```bash
.venv/bin/python -m frontend.hetero.cli qualify-gpu \
  --backend-config configs/hetero/backends/gpu_accelsim_rtx3070_ramulator2_hbdram_edge_16ch.json \
  --trace-manifest configs/hetero/operator_artifacts/p15a/tinyllama_prefill_bs1_ctx16_attention_norm_sm86_trace.json \
  --output /opt/gpu-atlas/qualification/p15c/accel-sim-rtx3070-attention-norm-shared-hbdram-identity

.venv/bin/python scripts/build_coupled_gpu_operator_artifact.py \
  --source-artifact configs/hetero/operator_artifacts/p15a/tinyllama_prefill_bs1_ctx16_attention_norm_sm86.json \
  --backend-config configs/hetero/backends/gpu_accelsim_rtx3070_ramulator2_hbdram_edge_16ch.json \
  --qualification-record /opt/gpu-atlas/qualification/p15c/accel-sim-rtx3070-attention-norm-shared-hbdram-identity/qualification_record.json \
  --output configs/hetero/operator_artifacts/p15c/tinyllama_prefill_bs1_ctx16_attention_norm_sm86_shared_hbdram_identity.json

.venv/bin/python scripts/summarize_p15c_coupled_gpu_artifacts.py \
  --catalog configs/hetero/operator_artifacts/p15c/tinyllama_prefill_bs1_ctx16_four_gpu_coupled_catalog.json \
  --qualification-root /opt/gpu-atlas/qualification/p15c \
  --output /opt/gpu-atlas/qualification/p15c/four-operator-final/qualification_record.json
```

四类Artifact均明确区分两个门禁：`compute_memory_coupled=true`表示真实指令状态会等待共享内存响应；`global_pa_binding_ready=false`表示当前桥仍直接使用Trace捕获地址。只有完成稳定Global PA重绑定并接入Prefill全局时间线后，才允许把`request_cycle_ready`改为true。汇总资格记录位于`/opt/gpu-atlas/qualification/p15c/four-operator-final/qualification_record.json`。

### 14.12 P15d剩余算子Artifact与13算子完整流量

P15d新增Output Projection、MLP Norm、Gate/Up Projection、SiLU Multiply、Down Projection、Final Norm、LM Head和Sampling八类真实RTX 3070 SM86 Trace。Final Norm、LM Head和Sampling按Prefill最后位置执行，兼容键明确记录`context_length=16`和`q_len=1`；其余新增算子记录`q_len=16`。与P15a五类任务合并后的严格Catalog覆盖13类完整流量算子，KV Append仍保持实测零Kernel的`runtime_state`语义。

```bash
PYTHONPATH=. .venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/p15d_tinyllama_prefill_1layer_ctx16_thirteen_full_traffic.json \
  --runs-root /opt/gpu-atlas/qualification/p15d/full-traffic-run1

PYTHONPATH=. .venv/bin/python scripts/summarize_p15d_thirteen_full_traffic.py \
  --run1 /opt/gpu-atlas/qualification/p15d/full-traffic-run1/p15d_tinyllama_prefill_1layer_ctx16_thirteen_full_traffic/c00d2784ef0dcbed220c33476f94ce18e43786a1c046fce8162a48cb288c7b10 \
  --run2 /opt/gpu-atlas/qualification/p15d/full-traffic-run2/p15d_tinyllama_prefill_1layer_ctx16_thirteen_full_traffic/c00d2784ef0dcbed220c33476f94ce18e43786a1c046fce8162a48cb288c7b10 \
  --output /opt/gpu-atlas/qualification/p15d/thirteen-full-traffic-final/qualification_record.json
```

两次运行的8个核心产物逐字节一致。20/20任务均由周期Artifact覆盖；13个任务产生3,462,673个完整流量Parent，其余7个任务产生65个采样Parent。唯一Ramulator2完成总计3,462,738个Parent/Child，读写分别为3,444,241/18,497，DRAM推进10,401,594 cycles，退出`outstanding=0`。全局GPU时钟为31,204,782 cycles，Makespan为26,003,985,000,000 fs；这些仍是未校准分块计算合同与完整/采样混合Value流量的部署证据，不是端到端性能结果。

Output Projection、MLP Norm、Gate/Up Projection、SiLU Multiply、Down Projection、Final Norm、Sampling和LM Head已经分别完成真实`mem_fetch`—共享Ramulator2双跑资格；加上P15c四类Trace，严格Catalog汇总12个算子、6,993,530个Parent和6,996,227个Child全部完成。LM Head双遍均为23,193,593 GPU cycles、476,608,000条指令、4,096,686个Parent和4,097,138个Child。每次资格运行单独拥有唯一Ramulator2，禁止把各算子周期相加为Prefill延迟。精确统计、复现入口和声明边界见[P15d资格记录](docs/qualification/p15d_remaining_prefill_ctx16.md)。

大型Trace建议始终使用`qualify-gpu --resume-completed-runs`。该选项只复用同时具有`command.json`和`stats.json`、且命令、Backend ID、Simulation Key与频率全部匹配的已完成遍次；残缺输出不会复用并将重新执行，身份不匹配的完整记录会被拒绝，双跑精确一致门槛不变。

### 14.13 P15e流式轨迹与在线Range-Rebase

请求周期轨迹现在逐条写入`request_cycle_trace.jsonl.gz`，摘要JSON只保留索引和统计。Context=16一层Prefill双跑均完成3,462,738个Parent、10,401,594个DRAM cycles并退出零在途；压缩流为94,859,940 B，SHA-256为`aa3edd9ca85dd3f600e8a1646d1b3af9bfc84f99d50c81f6b422c4897564795d`，两遍核心产物完全一致，峰值RSS约524.6 MiB。

在线桥新增`identity`与`range_rebase`显式模式。重新捕获的Attention Norm从真实Allocator事件恢复3个已知Tensor范围和3个不透明Workspace范围，再映射到运行期Global PA。双遍均为66,697 GPU cycles、5,290,064条指令，40,970次地址转换全部命中6个范围，2,176个Parent/Child全部完成，唯一Ramulator2且零在途。该Artifact可以标记`request_cycle_ready=true`；此结论不得外推到其他旧Artifact。详情见[P15e资格记录](docs/qualification/p15e_streaming_and_range_rebase.md)。

### 14.14 P15f QKV Allocator Segment Range-Rebase

QKV Projection的Tensor Core访存会触及CUDA Caching Allocator分配段内、语义Tensor末端之外的合法Padding事务。捕获器因此只选择包含目标Tensor地址的Backing Segment，并与目标执行窗口中新建的Allocator区间合并；它不会把进程中无关的CUDA Segment纳入地址契约。最终Manifest包含12个不重叠范围，占用33,685,504 B Global PA。

远端双遍均为2,168,865 GPU cycles和34,943,066条指令；736,837次地址转换全部命中12个范围，0次漏配。375,899个Parent和375,944个内部Child全部完成，读写Parent为375,854/45，唯一Ramulator2推进766,383 cycles并以零在途退出。Attention Norm与QKV Projection两算子Range-Rebase Catalog合计转换777,807次访问、守恒378,075个Parent与378,120个Child。两个Artifact均可用于后续Prefill请求级全局时间线接入，但当前仍不能把独立周期相加为Prefill延迟。详情见[P15f资格记录](docs/qualification/p15f_qkv_range_rebase.md)。

### 14.15 P15g 两个真实Accel-Sim算子的Prefill全局时间线

P15g不再把两个独立资格周期相加，而是在`OnlineOperatorRuntime`中按DAG依赖启动真实Backend。Attention Norm完成于`58,833,038,873 fs`，QKV Projection恰在同一时刻获得依赖并启动；两者共同占用`gpu0`且区间不重叠。Attention输出与QKV输入绑定到同一Value `TINYLLAMA11B-PREFILL-R0.prefill.s0.l0.norm.attention.out`及同一Global PA `351,485,952`，QKV启动时验证该Value的版本1。

实际运行中Attention Norm为66,599 GPU cycles、2,176 Parent/Child、40,758次地址转换；QKV为2,173,639 GPU cycles、376,690 Parent、376,734 Child和736,827次地址转换。每个算子进程各自仅有一个Ramulator2，全部Parent、Child与durable completion守恒，地址漏配和退出在途均为0。两个真实输出均在Backend完成时提交版本；完整图共提交18个输出版本。其余18个任务仍为分析回退，因此`performance_eligible=false`，本结果只能证明全局因果接入，不能作为TinyLlama端到端延迟或吞吐。复现和精确边界见[P15g资格记录](docs/qualification/p15g_prefill_global_timeline.md)。

P15h使用`scripts/run_p15h_remaining_range_rebase.sh`串行处理RoPE、Causal Attention、Output Projection、MLP Norm、Gate/Up Projection、SiLU Multiply、Down Projection、Final Norm、LM Head和Sampling。推荐在SM86 RTX 3070主机运行`P15H_PHASE=capture`，将捕获结果同步到远端后运行`P15H_PHASE=qualify`完成双遍周期资格。远端可以用`P15H_OPERATORS=rope,causal_attention`选择互不重叠的算子子集并行资格；只有运行完整10算子集合时脚本才生成最终12算子Catalog，避免部分结果被误当成全覆盖。

### 14.16 P15h 十二个真实Accel-Sim算子的Prefill全局时间线

P15h已完成其余10个算子的SM86 Range-Rebase重捕获和远端双遍资格，并与既有Attention Norm、QKV Projection合并为12算子Ready Catalog。统一时间线使用`configs/hetero/experiments/p15h_tinyllama_prefill_1layer_ctx16_twelve_request_cycle_gpu.json`，12个真实GPU算子共执行40,060,873 GPU cycles，完成6,995,173个Parent和6,998,046个Child，804,512,881次地址转换全部命中，所有算子均由唯一Ramulator2服务并以零在途退出。

最终资格验证确认：全部DAG依赖在消费者启动前完成，所有`gpu0`区间互不重叠；Global PA包含84个不重叠区间、56个算子私有Workspace和12条请求周期绑定，38条语义Tensor绑定均由图Value的Global PA派生；18次输出版本提交均发生在对应Backend完成时。时间线报告的35,390.378 µs makespan仍标记`performance_claim_allowed=false`，因为20个任务中只有12个真实GPU算子进入请求周期Backend，其余控制、KV管理和残差任务仍是分析/运行时模型，且尚未完成硬件校准。精确结果和复现边界见[P15h资格记录](docs/qualification/p15h_twelve_operator_prefill_timeline.md)。

### 14.17 P16 全任务显式建模与Shape门禁

P16新增[算子建模与测试状态表](docs/OPERATOR_MODELING_STATUS.md)及其机器可读Catalog。Token Embedding和Residual Add采用Shape锁定的独立CUDA参考实现，分别产生27,648和16,384条NVBit动态指令；Range-Rebase双跑分别稳定为6,691和28,772 GPU cycles，地址漏配、ATLAS请求和退出在途均为0。Residual Add的同一源Trace在一层图中通过不同Value绑定Dispatch两次。

在RTX 3070 / SM86采集主机上可用`P16_PHASE=capture bash scripts/run_p16_simple_operator_qualification.sh`生成两类Trace；将Trace目录和源Artifact同步到资格主机后，使用`P16_PHASE=qualify bash scripts/run_p16_simple_operator_qualification.sh`完成双跑并重建耦合Artifact。脚本支持`P16_CAPTURE_ROOT`、`P16_QUALIFICATION_ROOT`、`P16_ARTIFACT_ROOT`和`HETEROSIM_PYTHON`覆盖，且不会包含远端凭据。

`configs/hetero/experiments/p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu.json`不允许隐式分析回退：14种真实GPU Trace覆盖15个实例；KV Allocate/Append/Release分别生成3、512和2个精确64 B Parent，经外部Link进入各自唯一的live Ramulator2；Request Start/Finish是零内存请求的主机控制边界。两遍20任务运行在同一Simulation Key下得到相同的35,450,346,739,701 fs因果makespan、31次输入版本检查和18次完成时版本提交。该时间不是硬件校准结果，`performance_claim_allowed=false`保持不变。精确边界见[P16资格状态](docs/qualification/p16_full_task_modeling_status.md)。

完整双遍复现入口为：

```bash
HETEROSIM_PYTHON=.venv/bin/python \
  bash scripts/run_p16_full_task_qualification.sh
```

可用`P16_CONFIG`和`P16_RUN_ROOT`覆盖配置及输出根目录。脚本要求每遍返回相同Simulation Key，随后自动执行20任务依赖、资源、地址、请求、版本和两遍一致性资格检查。
