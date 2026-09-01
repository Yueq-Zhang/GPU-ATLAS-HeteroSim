# GPU-ATLAS-HeteroSim 中文手工复现手册

本文面向希望手动构建、运行和核验 GPU-ATLAS-HeteroSim 的使用者。内容对应工程版本 `0.27.0`，记录日期为 2026-09-01。所有命令默认从工程根目录执行。

本手册把“程序成功退出”“请求周期资格通过”和“性能资格通过”视为三个不同结论：

- 程序成功退出：配置、依赖和运行路径可用；
- 请求周期资格通过：双遍周期/请求/地址/版本等合同一致；
- 性能资格通过：仿真结果与身份、拓扑和 Shape 匹配的独立参考点误差合格。

当前工程已经具备前两类能力，但完整性能资格仍未通过。任何输出中只要 `performance_claim_allowed=false`，就不能把其中的 makespan、Token/s 或加速比作为实机性能结论。

## 1. 当前可复现任务总表

| 编号 | 任务 | 当前状态 | 主要环境 | 可以证明什么 |
|---|---|---|---|---|
| R0 | C++与Python基础回归 | 可立即复现 | 本机WSL | 调度、地址、内存服务、后端合同和审计逻辑未回归 |
| R1 | 四种GPU/3D-DRAM组织语义 | 可立即复现 | 本机WSL | Profile、路由、放置、Paged KV与地址空间语义正确 |
| R2 | GPU分层内存与双发起方竞争 | 可复现 | WSL + Ramulator2 Bridge | 外部Link、Gateway、Parent/Child拆分和唯一DRAM时序所有者闭环 |
| R3 | P10a/P10b-A控制面 | 可复现 | 本机WSL；真实适配器需要外部后端 | DAG依赖、单放置、Residency、版本门禁和Backend启动时刻正确 |
| R4 | P10b-B至P14 Prefill部署 | 可复现 | WSL + Ramulator2 Bridge | 从单层混合放置扩展到22层Context=1024 Prefill的确定性部署 |
| R5 | TinyLlama Q投影三后端 | 可复现 | Accel-Sim、ATLAS、Checkpoint和Trace | 同Shape下GPU本地显存、GPU外接3D-DRAM、ATLAS内部执行的独立资格 |
| R6 | P9b GPU与完整ATLAS Chip并发 | 可复现 | Accel-Sim + ATLAS + 唯一Ramulator2 | 两个真实计算后端并发推进并争用共享3D-DRAM |
| R7 | P15h 12真实GPU算子单层Prefill | 已通过，可在完整Artifact环境复现 | Accel-Sim + 12套Range-Rebase Artifact | 12个真实GPU算子进入同一DAG时间线，地址、请求和版本因果闭环 |
| R8 | P16 20任务完整单层Prefill | 历史双遍通过；当前从零复现有外部Artifact缺口 | Accel-Sim + P16归档Artifact | 固定Shape下14种GPU Trace、KV运行时和控制任务的完整单层因果闭环 |
| R9 | P17 14类Native-VRAM单算子双遍 | 已在远端通过，可恢复运行 | Accel-Sim 2.0远端环境 | 14类算子在GPU本地显存拓扑下周期/指令确定性 |
| R10 | P17性能校准审计 | 可立即执行审计 | 本机证据目录 | 审计过程成功；当前预期结果为`audit_complete_blocked`而非`qualified` |

## 2. 冻结软件与实验基线

### 2.1 工程和模型

- 工程版本：`0.27.0`；
- 当前代码基线：运行前使用`git rev-parse HEAD`记录，当前开发基线为`93f52af`加P17工作区更新；
- P15h/P16/P17固定模型：TinyLlama-1.1B；
- Checkpoint revision：`fe8a4ea1ffedaf415f4da2f062534de366a451e6`；
- Dtype：FP16；
- 单层资格Shape：Layer 0、BS=1、Context/Q/KV=16；
- Final Norm、LM Head和Sampling：`q_len=1`；
- Q投影Decode资格Shape：BS=1、已有KV长度1024、`M=1,K=2048,N=2048`。

### 2.2 外部依赖

精确版本以`dependency_lock.yaml`为准。关键版本为：

| 组件 | 固定版本 |
|---|---|
| ATLAS | commit `b2787399408e32d327c820daee96d4e6610f551a` |
| Accel-Sim | v2.0.0，commit `64653015f85fb5664c84a10f48527e8897d289d0` |
| GPGPU-Sim | commit `e10018b67a4b668e7b43f89280cf67624f1df4ff` |
| NVBit | 1.8 |
| CUDA Toolkit | 11.8，用于构建和SM86 Trace路径 |
| Ramulator2 | commit `3996362187d7f8314936e5ad7560d93b66b6a215` |
| BookSim2 | commit `1a8ec21ecc71f26be6907e373034e18c136ee459`，当前不作为已资格路径 |

远端长时间验证主机为`yueqi@192.168.5.2`。认证信息只能交互输入，不写入命令文件、配置、日志或仓库。远端物理GPU为RTX 4090，但现有Trace的模拟目标仍是RTX 3070/SM86；不能把捕获主机型号写成模拟目标。

## 3. 首次安装与构建

### 3.1 推荐目录

短回归可以直接在Windows挂载目录运行：

```text
/mnt/c/Users/yueqi/Desktop/3D_DRAM/OpenSourceWorks/GPU-ATLAS-HeteroSim
```

Accel-Sim、ATLAS和长时间资格推荐同步到WSL原生ext4目录：

```bash
sudo mkdir -p /opt/gpu-atlas
sudo chown -R "$USER":"$USER" /opt/gpu-atlas
rsync -a \
  --exclude .git \
  --exclude .venv \
  --exclude simulator/build \
  --exclude simulator/build-p17-wsl-py312 \
  --exclude runs \
  /mnt/c/Users/yueqi/Desktop/3D_DRAM/OpenSourceWorks/GPU-ATLAS-HeteroSim/ \
  /opt/gpu-atlas/GPU-ATLAS-HeteroSim/
cd /opt/gpu-atlas/GPU-ATLAS-HeteroSim
```

### 3.2 系统与Python环境

```bash
sudo apt update
sudo apt install -y build-essential cmake python3-venv pybind11-dev rsync libzstd-dev

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
```

### 3.3 C++运行时

```bash
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --parallel
ctest --test-dir simulator/build --output-on-failure
```

通过判据：9个CTest全部`Passed`。必须在构建所用的同一个WSL环境中执行CTest和Python测试；不要用Windows Python加载WSL生成的`_heterosim_runtime`。

### 3.4 记录本次复现身份

每次正式复现开始前保存以下信息：

```bash
mkdir -p /opt/gpu-atlas/reproduction-metadata
git rev-parse HEAD | tee /opt/gpu-atlas/reproduction-metadata/git-head.txt
git status --short | tee /opt/gpu-atlas/reproduction-metadata/git-status.txt
.venv/bin/python -m frontend.hetero.cli --version \
  | tee /opt/gpu-atlas/reproduction-metadata/version.txt
sha256sum dependency_lock.yaml \
  | tee /opt/gpu-atlas/reproduction-metadata/dependency-lock.sha256
```

正式比较时必须保留工作区状态。不同源码状态产生的结果不得只按实验名拼接到同一组数据中。

## 4. R0：基础回归

### 4.1 C++测试

```bash
ctest --test-dir simulator/build --output-on-failure
```

覆盖：时间单位、事件队列、Global Event Runtime、后端接口、调度器、内存服务、Paged KV、运行时内存规划和共享服务。

### 4.2 Python自包含回归

当前本机复核结果为163项通过。由于两项P16测试仍引用缺失的旧外部Artifact，当前建议先运行自包含集合：

```bash
.venv/bin/python -m pytest tests/hetero -q \
  -k "not test_p16_simple_gpu_operators_have_deterministic_range_rebase_evidence"
```

预期结果：`163 passed, 2 deselected`。测试数量可能随开发变化，应以`0 failed`为最终判据。

完整集合命令为：

```bash
.venv/bin/python -m pytest tests/hetero -q
```

当前已知的两项失败只应来自以下缺失目录：

```text
/opt/gpu-atlas/qualification/p16-cuda-reference-v3/
```

如果出现其他失败，应视为新回归，不得用上述排除条件掩盖。

## 5. 配置预检

任何实验先执行`validate`和`--dry-run`：

```bash
.venv/bin/python -m frontend.hetero.cli validate \
  --config configs/hetero/experiments/m1_model3_gpu_native_3ddram.json

.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m1_model3_gpu_native_3ddram.json \
  --dry-run
```

`validate`成功时会输出`simulation_input_key`。后续双遍必须使用相同的Key。修改模型、Shape、放置、地址、后端、Trace或DRAM配置后，Key应发生变化。

## 6. R1：四种系统组织语义

### 6.1 配置

四个实验共用：

- 模型：`configs/hetero/models/tiny_llama_2layer.json`；
- 工作负载：`configs/hetero/workloads/tiny_e2e_single.json`；
- 调度器：`configs/hetero/schedulers/token_barrier_smoke.json`；
- 放置：`configs/hetero/placements/gpu_prefill_atlas_decode.json`；
- 地址：`configs/hetero/addresses/paged_kv_tiny.json`。

仅系统Profile不同：

| 组织 | 实验配置 |
|---|---|
| ATLAS独立3D-DRAM | `configs/hetero/experiments/m1_model1_atlas_native.json` |
| GPU显存 + Host/3D-DRAM主存 + PCIe | `configs/hetero/experiments/m1_model2_host_memory_pcie.json` |
| 3D-DRAM直接作为GPU显存 | `configs/hetero/experiments/m1_model3_gpu_native_3ddram.json` |
| CXL扩展3D-DRAM | `configs/hetero/experiments/m1_model4_cxl_memory_tier.json` |

### 6.2 运行

```bash
for config in \
  configs/hetero/experiments/m1_model1_atlas_native.json \
  configs/hetero/experiments/m1_model2_host_memory_pcie.json \
  configs/hetero/experiments/m1_model3_gpu_native_3ddram.json \
  configs/hetero/experiments/m1_model4_cxl_memory_tier.json; do
  .venv/bin/python -m frontend.hetero.cli validate --config "$config"
  .venv/bin/python -m frontend.hetero.cli run \
    --config "$config" \
    --runs-root /opt/gpu-atlas/repro/m1-four-profiles
done
```

通过判据：四个实验均生成Run目录，配置验证通过，路由、Paged KV、地址容量和任务因果检查无异常。这里GPU为Roofline、ATLAS为分析模型，仅证明系统语义，不证明真实执行时长。

## 7. R2：分层内存和双发起方竞争

### 7.1 实验配置摘要

- GPU外部请求/响应Link：默认12.8 GB/s；
- Logic-Die Gateway：Parent按Byte/Sector Mask拆成64 B Child；
- 3D-DRAM：16通道、每通道512 bit、400 MT/s，配置峰值409.6 GB/s；
- 唯一时序所有者：一个Ramulator2实例；
- 写完成：默认durable确认。

### 7.2 运行

```bash
bash scripts/qualify_gpu_only_memory_path.sh \
  /opt/gpu-atlas/repro/gpu-only-layered-memory-path

bash scripts/qualify_dual_initiator_memory_path.sh \
  /opt/gpu-atlas/repro/dual-initiator-memory-path
```

通过判据：Parent/Child、Payload/Wire Byte与durable completion守恒，退出时`outstanding=0`；双发起方实验中GPU和ATLAS均有非零请求，且只实例化一个Ramulator2。

## 8. R3：放置、Residency和Backend启动门禁

### 8.1 P10a单放置

配置：`configs/hetero/experiments/m8_model3_full_runtime_reference.json`。

```bash
.venv/bin/python -m pytest tests/hetero/test_single_placement.py -q
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/m8_model3_full_runtime_reference.json \
  --runs-root /opt/gpu-atlas/repro/p10a
```

通过判据：每个逻辑节点只放置一次；跨设备值生成显式Route；版本、Fence和Residency事件顺序正确。

### 8.2 P10b-A在线Backend门禁

配置：`configs/hetero/experiments/step2_model1_operator_event_probe.json`。

```bash
.venv/bin/python -m pytest tests/hetero/test_online_operator_runtime.py -q
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/step2_model1_operator_event_probe.json \
  --runs-root /opt/gpu-atlas/repro/p10b-a
```

通过判据：Backend只在依赖和输入版本就绪后启动，输出只在Backend完成时提交。该配置中的两个外部后端仍是管线探针，不是完整LLM性能资格。

## 9. R4：P10b-B至P14 Prefill部署

### 9.1 阶段配置

| 阶段 | 配置 | 图规模与用途 |
|---|---|---|
| P10b-B | `p10b_b_tinyllama_prefill_1layer_mixed_live_ramulator2.json` | Context=16单层，Causal Attention放ATLAS，其余GPU |
| P12 | `p12_tinyllama_prefill_1layer_gpu_live_ramulator2.json` | Context=16单层GPU-only |
| P13 | `p13_tinyllama_prefill_22layer_ctx16_gpu_live_ramulator2.json` | 22层Context=16 GPU-only |
| P14 | `p14_tinyllama_prefill_bs1_ctx1024_gpu_live_ramulator2.json` | TinyLlama-1.1B、22层、BS=1、Context=1024完整Prefill |

这些文件均位于`configs/hetero/experiments/`。P14使用：

- `simulation.coupling=request_cycle`；
- GPU `cycle_replay`；
- 一个live Ramulator2；
- 12.8 GB/s外部接口合同；
- 有界代表性内存请求；
- `trace_coverage=0`、`performance_claim_allowed=false`。

### 9.2 双遍资格

```bash
bash scripts/qualify_prefill_p10b_to_p14.sh \
  /opt/gpu-atlas/GPU-ATLAS-HeteroSim \
  /opt/gpu-atlas/repro/prefill-p10b-to-p14
```

脚本会对每个阶段执行两遍，并逐字节比较：

- `metrics.json`；
- `memory_statistics.json`；
- `request_cycle_trace.json`；
- `global_memory_map.json`；
- `prefill_artifact_coverage.json`；
- `execution_graph.json`；
- `residency.json`。

通过判据：两遍字节一致、唯一Ramulator2、全部Parent完成、零在途、地址不重叠、任务合同覆盖100%、分析回退为0。这里的计算仍为未校准分块周期合同，P14输出不能解释为真实TinyLlama Prefill延迟。

## 10. R5：TinyLlama Q投影GPU与ATLAS

### 10.1 固定实验配置

- 模型revision：`fe8a4ea1ffedaf415f4da2f062534de366a451e6`；
- Layer-0 `q_proj`；
- Decode，BS=1，已有KV长度1024；
- FP16，`M=1,K=2048,N=2048`；
- GPU目标：RTX 3070/SM86 Accel-Sim配置；
- ATLAS：16个Logic-Die Core，N维每核128列，Tile=`1×512×16`。

### 10.2 可选：重新生成ATLAS Artifact

```bash
/opt/conda/envs/atlas/bin/python scripts/generate_atlas_qproj_artifact.py \
  --output configs/hetero/atlas/tinyllama11b_qproj_decode_bs1_ctx1024
```

### 10.3 可选：在兼容SM86采集主机重新捕获Trace

```bash
export ACTIVE_FROM_START=1
export DYNAMIC_KERNEL_RANGE=4-5
bash scripts/capture_accel_sim_trace.sh \
  /opt/conda/envs/qserve-local/bin/python \
  /opt/gpu-atlas/repro/tinyllama-qproj-decode-sm86-kernels4-5 \
  workloads/python/tinyllama_q_projection.py \
  --model /opt/hf-cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6 \
  --phase decode --context 1024
```

如果代码、编译器、库、SM、Launch、Shape、Dtype、Layout、融合或Kernel算法改变，必须重新编译和捕获，不能直接复用旧Trace。

### 10.4 三条资格路径

```bash
.venv/bin/python -m frontend.hetero.cli qualify-gpu \
  --backend-config configs/hetero/backends/gpu_accelsim_rtx3070.json \
  --trace-manifest configs/hetero/traces/local_rtx3070_tinyllama11b_qproj_decode_v2.json \
  --output /opt/gpu-atlas/repro/qproj/gpu-native

.venv/bin/python -m frontend.hetero.cli qualify-gpu \
  --backend-config configs/hetero/backends/gpu_accelsim_rtx3070_ramulator2_hbdram_edge_16ch.json \
  --trace-manifest configs/hetero/traces/local_rtx3070_tinyllama11b_qproj_decode_v2.json \
  --output /opt/gpu-atlas/repro/qproj/gpu-shared-3ddram

/opt/conda/envs/atlas/bin/python -m frontend.hetero.cli qualify-atlas \
  --backend-config configs/hetero/backends/atlas_test_chip_16ch.json \
  --chip-config configs/hetero/atlas/tinyllama_qproj_edge_16core_chip.yaml \
  --operator-list configs/hetero/atlas/tinyllama11b_qproj_decode_bs1_ctx1024/operator_description.yaml \
  --placement-map configs/hetero/atlas/tinyllama11b_qproj_decode_bs1_ctx1024/data_placement.yaml \
  --output /opt/gpu-atlas/repro/qproj/atlas-internal
```

每个输出目录应产生`qualification_record.json`，且双遍周期/指令或ATLAS周期一致。该实验只覆盖一个Q投影；三个后端的微架构不同，不能外推整层或端到端加速比。

## 11. R6：P9b真实GPU与完整ATLAS Chip并发

### 11.1 运行

```bash
bash scripts/build_accel_sim_ramulator2.sh
bash scripts/build_atlas_full_chip_runtime.sh
bash scripts/qualify_accel_sim_full_chip_concurrency.sh \
  /opt/gpu-atlas/repro/p9b-qproj-full-chip-concurrency
```

通过判据：GPU与ATLAS都产生非零请求；两者执行窗口重叠；GPU、ATLAS和Ramulator2的提交/完成计数守恒；退出时零在途；共享内存路径只存在一个Ramulator2。

该用例故意把同一Q投影同时放在GPU和ATLAS执行，仅用于验证竞争与完成隔离，不是合法推理图放置，也不能报告为端到端性能。

## 12. R7：P15h十二真实GPU算子单层Prefill

### 12.1 固定配置

- 实验：`configs/hetero/experiments/p15h_tinyllama_prefill_1layer_ctx16_twelve_request_cycle_gpu.json`；
- Ready Catalog：`configs/hetero/operator_artifacts/p15h/tinyllama_prefill_bs1_ctx16_twelve_gpu_range_rebase_catalog.json`；
- 模型：TinyLlama-1.1B Layer 0、FP16、BS=1、Context=16；
- 真实GPU算子：Attention Norm、QKV Projection、RoPE、Causal Attention、Output Projection、MLP Norm、Gate/Up Projection、SiLU Multiply、Down Projection、Final Norm、LM Head、Sampling；
- 地址模式：Range-Rebase到运行期Global PA；
- 后端：RTX 3070 Accel-Sim + 外接16通道3D-DRAM路径。

### 12.2 预检Artifact

```bash
.venv/bin/python -m frontend.hetero.cli validate \
  --config configs/hetero/experiments/p15h_tinyllama_prefill_1layer_ctx16_twelve_request_cycle_gpu.json

.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path(
    "configs/hetero/operator_artifacts/p15h/"
    "tinyllama_prefill_bs1_ctx16_twelve_gpu_range_rebase_catalog.json"
)
catalog = json.loads(path.read_text(encoding="utf-8"))
assert catalog["schema_version"] == "hetero-operator-artifact-catalog/v1"
assert len(catalog["required_operators"]) == 12
assert len(catalog["artifacts"]) == 12
assert catalog["zero_fallback_required"] is True
print("P15h catalog structure passed")
PY
```

若Artifact内部使用绝对路径，必须先把Trace、资格记录和依赖部署到记录路径。不得只编辑路径而不更新哈希和Simulation Key。

### 12.3 运行和汇总

```bash
PYTHONPATH=build:. .venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/p15h_tinyllama_prefill_1layer_ctx16_twelve_request_cycle_gpu.json \
  --runs-root /opt/gpu-atlas/repro/p15h-twelve-operator-timeline
```

记录命令输出中的`simulation_input_key`，然后运行：

```bash
PYTHONPATH=. .venv/bin/python scripts/summarize_p15h_prefill_timeline.py \
  /opt/gpu-atlas/repro/p15h-twelve-operator-timeline/p15h_tinyllama_prefill_1layer_ctx16_twelve_request_cycle_gpu/<simulation-key> \
  --output /opt/gpu-atlas/repro/p15h-twelve-operator-timeline/p15h_qualification.json
```

通过判据：12个真实GPU后端全部启动并完成；DAG消费者晚于生产者；`gpu0`区间不重叠；Global PA不重叠；所有地址转换命中；Parent/Child/durable completion守恒；输出版本只在Backend完成时提交；退出时零在途。

P15h中Token Embedding、两次Residual Add、KV任务和Request边界仍是分析/运行时模型，因此它证明12算子真实时间线接入，不是全任务周期资格。

## 13. R8：P16全任务单层Prefill

### 13.1 固定配置和覆盖

- 实验：`configs/hetero/experiments/p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu.json`；
- 20个任务全部显式建模；
- 14种真实GPU Trace覆盖15个实例；
- Token Embedding和Residual Add使用Shape锁定CUDA实现；
- KV Allocate/Append/Release产生显式64 B请求；
- Request Start/Finish为无内存请求的主机控制边界。

### 13.2 当前复现限制

历史双遍资格已经通过，但当前两个P16 Artifact仍指向缺失的旧目录：

```text
/opt/gpu-atlas/qualification/p16-cuda-reference-v3/
```

因此，在恢复以下算子的原始`operator_metadata.json`、Trace、Kernel List和资格记录之前，不能宣称“从干净checkout一条命令重新跑通P16”：

- Token Embedding；
- Residual Add。

不能把P17 Native-VRAM封存Trace直接替换到P16外接3D-DRAM配置中而仍沿用旧资格；若要迁移，必须生成新的源Artifact、Range-Rebase资格、耦合Artifact和Simulation Key。

### 13.3 外部证据完整后的双遍入口

```bash
HETEROSIM_PYTHON=.venv/bin/python \
P16_RUN_ROOT=/opt/gpu-atlas/repro/p16 \
  bash scripts/run_p16_full_task_qualification.sh
```

脚本自动运行两遍、核对相同Simulation Key，并生成：

```text
/opt/gpu-atlas/repro/p16/p16_full_task_qualification.json
```

通过判据：20任务依赖、资源互斥、Global PA、请求完成、输入版本、输出提交和双遍结果全部一致；每个KV任务只有一个Ramulator2，ATLAS请求为0，退出时零在途。

### 13.4 只复核已归档双遍

如果已经恢复`validation/p16/leg1`与`leg2`完整目录，可执行：

```bash
.venv/bin/python scripts/summarize_p16_full_task_timeline.py \
  validation/p16/leg1/p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu/<simulation-key> \
  validation/p16/leg2/p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu/<simulation-key> \
  --output validation/p16/p16_full_task_qualification.json
```

## 14. R9：P17十四类Native-VRAM Accel-Sim双遍

### 14.1 配置

- 能力Catalog：`configs/hetero/operator_capabilities/tinyllama_prefill_layer0_bs1_ctx16.json`；
- Accel-Sim后端：`configs/hetero/backends/gpu_accelsim_rtx3070.json`；
- Native测量Catalog：`validation/p17/gpu_operator_pairing/native_rtx3070_local_vram.json`；
- 模拟Catalog：`validation/p17/gpu_operator_pairing/simulator_native_vram.json`；
- Trace覆盖文件：`configs/hetero/calibration/p17_native_vram_trace_overrides.json`；
- 默认资格根目录：`/opt/gpu-atlas/qualification/p17-native-vram`。

十四类算子为：Token Embedding、Attention Norm、QKV Projection、RoPE、Causal Attention、Output Projection、MLP Norm、Gate/Up Projection、SiLU Multiply、Down Projection、Residual Add、Final Norm、LM Head和Sampling。

### 14.2 远端运行

```bash
ssh yueqi@192.168.5.2
cd /opt/gpu-atlas/GPU-ATLAS-HeteroSim

export PYTHONPATH=.
export HETEROSIM_PYTHON=/opt/conda/envs/qserve-local/bin/python
export P17_TRACE_MANIFEST_OVERRIDES=configs/hetero/calibration/p17_native_vram_trace_overrides.json
bash scripts/run_p17_native_vram_accelsim_qualification.sh
```

脚本支持断点恢复。只运行部分算子时必须禁止生成全量Catalog：

```bash
P17_OPERATOR_FILTER=rope,causal_attention \
P17_EXPECTED_OPERATOR_COUNT=2 \
P17_FINALIZE_CATALOG=0 \
  bash scripts/run_p17_native_vram_accelsim_qualification.sh
```

通过判据：每个`qualification_record.json`为`status=passed`；两遍GPU cycle和instruction相同；没有`external_memory_stats`；GPU本地DRAM由Accel-Sim拥有；`external_ramulator2=false`；时长使用total模式。

当前14类已经全部满足上述确定性门禁，但这只是单算子Native-VRAM仿真资格，不是单层时间线，也不是硬件性能校准。

## 15. R10：P17性能校准和审计

### 15.1 RTX 3070原生测量

在具有RTX 3070和对应模型权重的Windows环境执行：

```powershell
powershell -ExecutionPolicy Bypass `
  -File scripts/run_p17_tinyllama_native_operator_calibration.ps1 `
  -ModelPath "F:/models/TinyLlama-1.1B-Chat-v1.0-fe8a4e" `
  -Python "F:/study_apps/python/Anaconda/envs/LLM_Design_env/python.exe" `
  -Warmup 50 -Iterations 500
```

`ModelPath`与`Python`应替换为实际位置，但模型revision、Shape、Warmup和Iterations不得在同一资格组中静默改变。

### 15.2 Native-VRAM配对审计

P17资格脚本在14类全部完成后会自动生成：

- `validation/p17/gpu_operator_pairing/simulator_native_vram.json`；
- `validation/p17/gpu_operator_pairing/native_vram_pairing_audit.json`。

当前预期：

- `topology_match=true`；
- `paired_operator_count=0/14`；
- 14类均缺少Native执行/Trace二进制身份闭环；
- 10类误差超过15%；
- `performance_claim_allowed=false`。

### 15.3 P16整层性能门禁审计

在P16双遍目录完整时执行：

```bash
KEY=d5066ff9081332bd31ae5699f4f572736cc7f188ae9f4272cf89a4af0a1d6e3a
.venv/bin/python scripts/audit_p17_performance_calibration.py \
  configs/hetero/calibration/p17_tinyllama_prefill_layer0_ctx16_incomplete.json \
  "validation/p16/leg1/p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu/$KEY" \
  "validation/p16/leg2/p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu/$KEY" \
  --project-root . \
  --output validation/p17/p16_layer0_ctx16/performance_calibration_audit.json
```

当前正确结果是：

```text
status=audit_complete_blocked
qualified_component_count=0
required_component_count=6
performance_claim_allowed=false
```

这是“审计成功且性能门禁保持关闭”，不是仿真失败。只有GPU Kernel、Copy Engine、Runtime、外部Link、Logic-Die Gateway和3D-DRAM六项全部验证后，才允许出现性能资格通过。

## 16. 通用输出检查

一次完整运行通常产生以下文件，具体集合取决于执行模式：

| 文件 | 用途 |
|---|---|
| `metrics.json` | makespan、Fidelity和性能声明门禁 |
| `execution_graph.json` | 任务、Route、依赖和资源区间 |
| `global_memory_map.json` | Global PA范围、容量、对齐和重叠检查 |
| `memory_statistics.json` | Parent/Child、Initiator、完成与在途请求 |
| `request_cycle_trace.json`或流式Manifest | 请求周期事件 |
| `residency.json` | 值版本、设备驻留和跨设备动作 |
| `online_dispatch.json` | 后端启动/完成和版本提交 |
| `qualification_record.json` | 双遍一致性和资格结论 |

正式通过至少检查：

1. 两遍`simulation_input_key`相同；
2. 比较范围内的核心输出逐字节一致；
3. `accepted_parent_ids == observed_completion_ids == completed`；
4. `child_sent == child_completed`；
5. durable完成守恒；
6. `outstanding == 0`；
7. Global PA不重叠且不超过容量；
8. 每个共享3D-DRAM实验只有一个Ramulator2时序所有者；
9. 消费者在生产者完成和正确版本可见后启动；
10. `performance_claim_allowed`按实际校准状态保持关闭或打开，不得手工覆盖。

## 17. 常见故障

### 17.1 `_heterosim_runtime`找不到

原因通常是Windows Python和WSL构建树混用，或Python版本与pybind模块不匹配。解决方法是在同一WSL环境中重新执行：

```bash
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --clean-first --parallel
```

### 17.2 Ramulator2 Bridge不存在

检查配置中的`bridge_library`绝对路径，并确认依赖版本与`dependency_lock.yaml`一致。不要把另一个提交构建出的`.so`复制后沿用旧Simulation Key。

### 17.3 Artifact文件不存在或哈希不匹配

Artifact会校验元数据、Kernel List、Trace、资格记录和配置哈希。应恢复原始证据或重新生成完整Artifact；不要只编辑JSON中的路径或SHA字段。

### 17.4 P17返回`audit_complete_blocked`

这是当前预期结果。它表示审计执行成功，但性能参考点、身份或误差门禁尚未满足。

### 17.5 输出目录已有部分文件

Trace捕获脚本会拒绝把新捕获混入非空目录。为每次捕获使用新目录；资格脚本只有在明确支持`--resume-completed-runs`时才能复用已完成Leg。

## 18. 建议的人工复现顺序

首次人工复现建议严格按以下顺序推进：

1. R0：9个C++测试和Python自包含回归；
2. 配置`validate`与`--dry-run`；
3. R1：四种系统Profile；
4. R2：GPU分层内存和双发起方竞争；
5. R4：P10b-B至P14双遍；
6. R5：单算子Q投影三后端；
7. R6：P9b真实GPU/ATLAS并发；
8. R7：P15h十二真实算子单层时间线；
9. 恢复P16外部Artifact后执行R8；
10. 在远端执行R9，并用R10审计而不是直接报告性能。

若某一级失败，应保留该次输出目录和命令身份，不继续用其结果生成下一级性能结论。
