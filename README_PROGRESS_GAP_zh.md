# GPU-ATLAS-HeteroSim 当前完成情况与计划差距

本文用于回答三个问题：当前工程真正完成了什么、与冻结计划还差什么、下一步应从哪里继续。运行方法见主 [README](README.md)，规范性目标见 [GPU + ATLAS 异构端到端仿真实现规范](docs/gpu_atlas_heterogeneous_simulation_design_zh.md)。

评估日期：2026-08-26

代码评估基线：`d240667` 之后的第二步实现工作树

当前版本：`0.5.0`

状态定义：

- ✅ 已完成：代码、运行入口和对应验收证据均存在；
- 🟡 部分完成：已有结构或语义验证，但缺少计划要求的真实 Backend、精确时序或完整验收；
- ⬜ 未开始：当前仓库中没有可运行实现；
- ⛔ 环境受限：代码路径存在，但当前机器条件不能完成目标实机步骤。

## 1. 一句话结论

工程已经具备异构 LLM 仿真的控制面、请求图、算子放置、全局事件运行时、四种拓扑语义、Paged KV，以及 Accel-Sim 与真实 ATLAS 同时参与一次 `operator_event` 运行的最小闭环；但仍不是请求级共享 3D-DRAM 联合周期仿真器。

当前可以可靠报告：

- Prefill/Decode 图和逻辑工作量；
- GPU/ATLAS 算子放置决策；
- 四种 Profile 的拓扑 Lowering 语义；
- 未校准 Roofline/Link 分析预览；
- 指定官方 Trace 与固定 QV100 配置下的独立 Accel-Sim 周期结果。
- 指定 ATLAS Operator/Placement/Chip 配置下的独立 ATLAS 周期与能耗结果；
- GPU 和 ATLAS `total` 时长任务在同一全局事件运行时中的依赖、资源和传输编排结果。

当前不能报告：

- GPU 与 ATLAS 同时执行时的真实端到端性能；
- GPU 和 Compute Die 争用同一 3D-DRAM 的周期级结果；
- 真实 PCIe/CXL 队列、Credit、背压和迁移性能；
- RTX 3070 目标平台的已校准周期结果；
- 完整动态 Batch、KV 回收和端到端 LLM 周期精确性能。

## 2. 三步 GPU 集成计划对照

三步计划与 M0–M9 里程碑不是同一套顺序。三步计划描述 GPU 仿真器如何逐级接入；M0–M9 描述整个异构系统的功能成熟度。

| 步骤 | 冻结目标 | 当前状态 | 已有证据 | 主要差距 |
|---|---|---|---|---|
| 第一步：独立 Accel-Sim 后端资格 | 固定依赖；编译/采集入口；Trace Manifest；地址正规化；Trace Cache；原生基线与 Adapter 一致 | ✅ 已完成 | QV100 Backprop 两条路径均为 `15,329 cycles`、`10,473,824 instructions`；43/43 Python、7/7 C++ 测试通过 | 本机 RTX 3070 Trace 因 NVBit/Driver 不兼容未采集；该结果只证明 Adapter 等价，`replay_safe=false` |
| 第二步：算子事件级 GPU+ATLAS 集成 | Accel-Sim 总时长进入 ExecutionGraph；实现 ATLAS Adapter；GPU/ATLAS 算子按 Placement 执行；传输和多请求由全局运行时协调 | 🟡 最小闭环完成 | 主 `run` 同时执行 Accel-Sim `15,329 cycles` 与 ATLAS `48,446 cycles`；时序 Contract、缓存、结果回填、显式回退和 C++ 全局事件编排均已运行 | 当前绑定是 `surrogate_plumbing_probe`；仍需生成与 LLM 算子形状/布局完全对应的 GPU Trace 和 ATLAS Artifact，并补多请求真实 Backend 黄金用例 |
| 第三步：请求级共享 3D-DRAM 周期耦合 | GPU L2 Miss 和 ATLAS 请求进入同一 Shared Fabric/唯一 Ramulator2；实现跨时钟、仲裁、背压、响应和守恒 | ⬜ 未开始 | 已有 Timing Ownership 冲突检查和固定延迟内存服务单元测试 | 没有 GPU Memory Bridge、ATLAS Memory Port、Shared Fabric、外部内存请求接口、唯一 Ramulator2 实例、跨时钟事件桥及死锁/带宽验收 |

因此，当前准确含义是：GPU 和 ATLAS 已经参与同一个端到端 `run`，但两者各自返回包含内部内存的总时长，全局只在算子/传输事件粒度协同；共享 3D-DRAM 尚未在请求粒度周期耦合。

## 3. 当前已经完成的模块

### 3.1 控制面与可复现框架

- ✅ 严格实验配置解析、组件 `ref` 展开和输入哈希；
- ✅ 规范化 Run 目录、Resolved Config、Git revision、Dependency Lock、事件日志和 Fidelity 字段；
- ✅ Python 控制面与 C++ 运行时边界；
- 🟡 依赖锁已精确覆盖 ATLAS、Accel-Sim、GPGPU-Sim、NVBit 和 CUDA 11.8，但 Ramulator2、BookSim2、TileLang 仍有 inherited placeholder。

### 3.2 LLM 图、请求和放置

- ✅ Tiny Decoder-only 模型的完整 Prefill、首 Token、逐 Token Decode、KV Append、LM Head 和 Sampling 图；
- ✅ `G` 个输出 Token 对应 `G-1` 次 Decode Forward；
- ✅ 手工/规则化 GPU 与 ATLAS 算子放置；
- ✅ 可按 Phase、Layer、Operator Group、KV 长度和活动 Batch 匹配规则；
- 🟡 Placement 已能驱动 Accel-Sim/ATLAS Artifact 选择；当前只有各一个代理绑定，其余算子显式分析回退。

### 3.3 Batch、KV 与运行时

- ✅ C++ Token-Step Barrier Scheduler；
- ✅ C++ Paged KV 分配和容量检查；
- ✅ 多请求 Arrival、Prefill Chunk、Decode Priority 的确定性语义测试；
- ✅ C++ DAG 依赖、资源互斥和到达时间感知 GlobalEventRuntime；
- 🟡 当前分析执行没有把同一 Token Step 的请求聚合成真实 Batched Kernel；
- 🟡 请求结束后的动态 KV 释放、Block 复用、Mixed Prefill/Decode 真实 Backend 执行尚未实现；
- ⬜ Static Ragged Batch 的真实打包 Kernel 和 Device Sub-Batch 执行尚未实现。

### 3.4 四种物理系统 Profile

- ✅ Model 1：独立 ATLAS 的逻辑拓扑；
- ✅ Model 2：Host 3D-DRAM + GPU HBM + PCIe 的路由语义；
- ✅ Model 3：GPU 与 Compute Die 共享 3D-DRAM、禁止伪 DMA 的拓扑语义；
- ✅ Model 4：GPU HBM + CXL 3D-DRAM 的 Remote/Copy/Migration 路由语义；
- 🟡 四种 Profile 已通过逻辑工作量和 Lowering 测试，但只有 Model 1 有当前分析预览示例；
- ⬜ Model 3 尚无 Shared3DAnalyticalMemoryService，也无请求级共享内存周期耦合；
- ⬜ Model 2/4 尚无有界 PCIe/CXL 队列、Credit 和背压模型。

### 3.5 GPU Accel-Sim 第一步

- ✅ 固定 Accel-Sim v1.3.0、GPGPU-Sim v4.2.1、NVBit 1.7.3；
- ✅ CUDA 11.8 下构建 Accel-Sim、NVBit Tracer 和 Postprocessor；
- ✅ RTX 3070 最小 CUDA Vector Add 编译、原生执行与结果验证；
- ✅ Trace Manifest Schema、Trace Cache 和 Accel-Sim Backend；
- ✅ `TraceAddr -> TensorID+offset` 与候选 `SimulationBufferBinding -> PhysicalAddress` 分离；
- ✅ 原生 Accel-Sim 与 Adapter 的 Cycle/Instruction 精确一致性验证；
- ⛔ 本机 Driver 591.86 超出固定 NVBit 1.7.3 的兼容范围，RTX 3070 本地 Trace 采集被主动拒绝；
- 🟡 官方 QV100 Trace 可用于 Adapter 回归，但不能替代 RTX 3070 目标平台校准；
- 🟡 当前资格范围为 `adapter_equivalence`，不包含跨 DRAM 时序的 Replay Safety，因此 `replay_safe=false`。

## 4. M0–M9 里程碑差距矩阵

| 里程碑 | 计划内容 | 状态 | 当前完成情况 | 尚缺验收 |
|---|---|---|---|---|
| M0 契约和基线 | 四拓扑、时序所有权、单位、依赖锁、黄金工作负载、独立基线 | 🟡 | 规范、四拓扑、单位、黄金用例、Accel-Sim 与 ATLAS Adapter 基线已存在 | Ramulator2/BookSim2/TileLang 的完整版本和资格记录仍需精确锁定 |
| M1 统一模型和两级 IR | Canonical Model、ModelGraph、ExecutionGraph、Prefill/Decode、KV/Request State | ✅ | Tiny 模型、完整请求图、ExecutionGraph、固定生成轨迹和 KV 状态已实现并测试 | 扩展到真实模型只需新增配置/编译路径；不阻塞下一步 |
| M2 基础运行时和 Model 1 | Global Runtime、GPU Roofline、ATLAS Adapter、Link、单请求端到端 | 🟡 | GlobalEventRuntime、Roofline/Link、ATLAS Adapter、Accel-Sim/ATLAS 混合单请求和确定性对比已完成 | Roofline 未校准；真实 LLM 算子 Trace/Artifact 和完整多请求 Backend 验收缺失 |
| M3 地址和四 Profile 事件模式 | MemorySpace、PhysicalAddress、Allocator、Router、传输/迁移/同步、四 Profile | 🟡 | 拓扑 Router、Profile 配置、Paged KV 地址空间和部分配置校验已完成 | 通用 Global Allocator、完整 Residency/Owner、四 Profile operator-event 执行和 Model 3 共享分析内存服务缺失 |
| M4 Multi-Batch | Ragged Batch、Paged KV、Continuous Batch、Chunked Prefill、Mixed、Sub-Batch | 🟡 | Paged KV、Barrier Scheduler、多 Arrival/Chunk 语义和规则放置已完成 | 真实 Batched Kernel、动态 KV 释放复用、Mixed Backend 执行、Device Sub-Batch 恢复与完整 B1–B5 验收缺失 |
| M5 Accel-Sim 资格 | 编译、NVBit Trace、Manifest、地址绑定、Cache、Adapter | 🟡 | 独立 Adapter、官方 Trace 回归和地址分层已完成 | TileLang GPU 编译路径、本机/目标 RTX Trace、真实 Tensor Range 捕获、Replay Safety 资格和 Trace 覆盖策略缺失 |
| M6 Model 3 周期耦合 | GPU Bridge、ATLAS Port、Shared Fabric、唯一内存服务、跨时钟、背压 | ⬜ | 仅有接口原型、时序所有权检查和固定延迟 MemoryService 单测 | M6 全部主体组件与黄金请求守恒验收尚未实现 |
| M7 PCIe/CXL 高精度 | DMA、队列、CXL Root/Switch、Credit、Remote、Migration、Residency | ⬜ | 只有拓扑路由决策和分析链路公式 | 协议状态、队列、Credit、背压、迁移和四路径测试全部缺失 |
| M8 全组合与 DSE | 回归矩阵、缓存清单、真实模型、跨拓扑报告、周期复验 | ⬜ | 已有少量四 Profile 语义回归和输入哈希 | 没有真实模型矩阵、自动 DSE、跨拓扑周期报告和完整缓存复用清单 |
| M9 高级扩展 | 动态 EOS/MoE、MMU、完整 CXL、一致性、多 GPU、gem5、NPU | ⬜ | 未实现 | 全部属于后续扩展，不应提前阻塞 M2–M8 主路径 |

## 5. 与“研究可用”完成定义的差距

冻结规范要求下列十项同时满足。当前不能宣称第一版研究可用。

| # | 研究可用条件 | 当前判断 | 说明 |
|---:|---|---|---|
| 1 | 四 Profile 运行相同端到端 Prefill/Decode | 🟡 | 四 Profile 已运行相同逻辑图，但不是四 Profile 完整时序 Backend |
| 2 | Roofline 覆盖全部 Profile | ⬜ | 当前只有 Model 1 分析预览示例，Model 3 缺共享分析内存服务 |
| 3 | Accel-Sim 完成 GPU 独立路径 | ✅ | 固定 QV100 Trace/Config 的原生与 Adapter 结果精确一致 |
| 4 | Model 3 请求级共享 3D-DRAM 周期耦合 | ⬜ | M6 未开始 |
| 5 | Model 2/4 有界队列、带宽、延迟和背压 | ⬜ | M7 未开始 |
| 6 | Static Ragged、Paged KV、Continuous Batch | 🟡 | Paged KV 和调度语义完成，真实 Ragged/Continuous Backend 执行未完成 |
| 7 | 手工和规则化 GPU/ATLAS 放置 | ✅ | Placement 决策和 Backend Artifact 选择已接入主 `run` |
| 8 | 逻辑工作量、地址、传输和请求守恒 | 🟡 | 已覆盖部分黄金计数与内存服务，尚未覆盖真实 Bridge/PCIe/CXL |
| 9 | Resolved Config、版本、Trace、地址、Seed 可复现 | 🟡 | 框架已存在，部分 ATLAS 系依赖仍是 placeholder，真实 Trace 地址清单不完整 |
| 10 | 明确区分 analytical/event/cycle/extrapolated | ✅ | 当前 Run 产物已有 Fidelity 和禁止性能声明字段 |

## 6. 当前最关键的实现缺口

### P0：把第二步从代理闭环升级为真实 LLM 算子闭环

1. 为 Tiny LLM 的 GPU 算子生成形状和布局一致的 CUDA/TileLang Kernel 与 Trace；
2. 为对应 ATLAS 算子从编译计划生成匹配的 Operator/Placement YAML；
3. 用精确 Artifact 替换两个 `surrogate_plumbing_probe` 绑定；
4. 增加至少两个请求的 GPU/ATLAS 混合执行、资源排队和传输黄金用例；
5. 完成 Trace/Artifact 覆盖率门槛，只有全部关键算子为 `exact_operator` 时才允许进入性能资格评估。

第二步的接线标准已经满足；完成标准仍要求同一个端到端请求中的绑定与真实 LLM 算子完全兼容，全局运行时生成确定性的 TTFT/TPOT/E2E，且没有重复计算设备与传输时间。

### P1：完成第三步的最小 Load/Store 耦合

1. 定义 Accel-Sim 外部内存请求/响应接口；
2. 在 L2 Miss 后执行候选 DRAM Decode；
3. 实现 ATLAS Memory Port、Shared Fabric 和唯一 Shared3DMemoryService；
4. 连接唯一 Ramulator2，并关闭 Accel-Sim 内部 DRAM 时序；
5. 实现跨时钟事件、队列、仲裁、背压和响应唤醒；
6. 从固定延迟服务和最小 Load/Store Trace 开始验证，再切换真实 Ramulator2；
7. 严格检查注入、完成、在途、父/子请求和字节守恒。

第三步完成标准：Model 3 中 GPU 与 ATLAS 请求同时争用同一 3D-DRAM，系统只有一个 DRAM 时序所有者；GPU-only、ATLAS-only 和双发起方结果均通过版本锁定的黄金测试。

### P2：补齐研究可用外围能力

- M4：真实 Ragged/Continuous Batch、KV 释放复用和 Device Sub-Batch；
- M7：PCIe/CXL 有界队列、Credit、Remote/Copy/Migration；
- M8：真实 LLM Workload、自动回归矩阵、DSE 与跨拓扑报告；
- 校准：RTX 3070 Trace Capture、GPU/ATLAS/Ramulator2 参数来源和误差记录。

## 7. 下一次开始实现时的建议入口

下一次应先完成第二步的真实 LLM Artifact，不应直接跳到 Ramulator2 共享内存桥。推荐切片是：

```text
Tiny LLM GPU Kernel/Trace
        + 匹配的 ATLAS Operator/Placement
        -> exact_operator 绑定
        -> 多请求 operator_event 黄金测试
```

上述完成后，再进入第三步的 GPU L2 Miss/ATLAS Memory Port/唯一 Ramulator2 请求级耦合。

## 8. 当前验证证据

- 最新功能提交：`6ba9111`；资格语义修正：`06150ed`；
- Accel-Sim Adapter：官方 QV100 Backprop 原生/适配器均为 `15,329 cycles`、`10,473,824 instructions`；
- ATLAS Adapter：测试 GEMM 两次独立运行均为 `48,446 cycles`、`0.00581352 J`，完整统计精确一致；
- 联合接线：同一次主 `run` 同时产生 Accel-Sim 与 ATLAS 周期任务，两个 Contract 均为 `total` 且 `exports=[]`；
- Python：43/43 passed；
- C++：7/7 passed；
- CUDA：RTX 3070 Vector Add 原生执行与结果验证通过；
- GitHub：`main` 已同步；
- 资格记录：[docs/qualification/qv100_backprop_4096.md](docs/qualification/qv100_backprop_4096.md)；
- 详细状态：[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)。

后续每完成一个切片，应同步更新本文件中的状态、验收证据和剩余缺口；不能仅根据目录或类名将项目标记为完成。
