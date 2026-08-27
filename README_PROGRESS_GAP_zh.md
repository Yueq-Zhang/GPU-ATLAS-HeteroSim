# GPU-ATLAS-HeteroSim 当前完成情况与验证差距

评估日期：2026-08-27
当前版本：`0.6.2`
规范基线：`hetero-sim/v1`、设计合同v1.4

本文区分“功能已经实现”和“目标硬件已经完成资格验证”。前者表示接口、运行入口、状态机和参考模型可执行；后者要求真实Trace、真实Artifact、版本锁定的外部Backend和误差/守恒验收全部通过。

## 1. 当前结论

计划中的软件功能已经形成完整参考闭环：四种系统Profile可运行相同端到端工作负载，支持GPU/ATLAS算子放置、Prefill/Decode、独立Decode Step、多请求Continuous/Ragged Batch、Device Sub-Batch、Paged KV动态释放复用、统一地址与Residency、PCIe/CXL有界链路、共享3D内存请求模型、外部内存桥协议和DSE。

GPU-only 的实时 Accel-Sim 请求/响应路径已经接入唯一 Ramulator2，并完成一个固定 QV100 Backprop Trace/配置的最小闭环资格验证。该闭环目前仍是“一个GPU `mem_fetch`对应一个Ramulator2请求”，尚未建模GPU到Logic Die的外部请求/响应链路、按真实事务大小拆分的内部子事务、Parent/Child完成汇聚和独立Link/Logic/DRAM时钟域。当前HBM3候选配置的`DQ`、`channel_width`、Burst和事务粒度也没有形成自洽的单一带宽口径，因此只证明实时回调可工作，不构成3D-DRAM带宽资格验证。全部新增参考配置继续标记`performance_claim_allowed=false`。

## 2. 三步GPU集成计划对照

| 步骤 | 实现状态 | 已具备功能 | 后续验证 |
|---|---|---|---|
| 第一步：独立Accel-Sim后端 | ✅ v2.0.0 已集成并完成 QV100 与 RTX 3070 Adapter 等价验证 | 固定依赖、CMake构建、NVBit 1.8 `.tracez`采集、Trace Manifest、兼容新旧格式的Trace Cache、周期结果解析 | 真实LLM Trace与RTX 3070/4090微架构校准 |
| 第二步：算子事件级GPU+ATLAS | ✅ 软件闭环完成，真实LLM Artifact待验证 | 同一GlobalEventRuntime调度GPU、ATLAS、传输和分析回退；Artifact选择与缓存 | 用形状/布局完全匹配的LLM GPU Trace和ATLAS YAML替换代理绑定 |
| 第三步：共享3D-DRAM请求级耦合 | 🟡 GPU-only最小实时闭环已通过 | Accel-Sim请求进入唯一Ramulator2并由完成回调解除GPU等待；内部DRAM固定延迟已关闭 | 先实现外部Link→LogicDieGateway→内部Child事务→Ramulator2→响应Link，再接入ATLAS Port和双发起方竞争 |

## 3. M0–M9实现矩阵

| 里程碑 | 软件实现 | 当前证据 | 尚待资格验证 |
|---|---:|---|---|
| M0 契约和依赖 | ✅ | 四拓扑、单位、时序所有权；Ramulator2、BookSim2、TileLang Commit已锁定 | 外部组件逐项资格记录 |
| M1 模型与两级IR | ✅ | Full Request与`decode_step`；Llama/SwiGLU和OPT/Dense-GELU；Prefill/Decode/KV计数 | 更多模型解析器和数值功能验证 |
| M2 基础运行时 | ✅ | C++ DAG Runtime、Roofline、Accel-Sim/ATLAS Adapter、统一Run产物 | 目标硬件校准 |
| M3 地址与四Profile | ✅ | RuntimeMemoryPlanner、Memory Space、PhysicalAddress、Residency、Copy/Migrate/Remote/Sync | 真实地址清单和大容量压力验证 |
| M4 Multi-Batch | ✅ 参考执行 | Continuous/Chunked、Static Ragged、Device Sub-Batch、动态KV释放复用 | 真正融合Batched Kernel与真实Backend Shape覆盖 |
| M5 Accel-Sim资格 | 🟡 | v2.0.0 QV100旧Trace回放通过；RTX 3070/驱动591.86的NVBit 1.8采集及Adapter等价通过 | 精确LLM Trace、RTX 3070微架构校准与RTX 4090合格配置仍未完成 |
| M6 共享3D-DRAM | 🟡 | 单Owner参考模型与离线桥；GPU-only最小实时Accel-Sim + 单Ramulator2闭环已通过固定Backprop资格 | Bridge ABI v2、外部/内部带宽分离、Parent/Child拆分汇聚、durable write、独立时钟域、ATLAS实时端口与目标堆叠校准 |
| M7 PCIe/CXL与外部链路 | 🟡 | 事件级有界事务、Credit、序列化、传播延迟、背压、Residency状态机 | 将有界链路升级为请求/响应双向周期服务，并按direct PHY、PCIe DMA和CXL.mem分别限定访问语义 |
| M8 DSE和报告 | ✅ 参考执行 | 四Profile `full_runtime`配置、自动笛卡尔DSE、OPT-6.7B单步Decode配置 | 大规模矩阵、周期复验和论文级统计 |
| M9高级扩展 | ⬜ 可选 | gem5/NPU接口仍按设计保留 | MMU、完整一致性、多GPU、动态MoE等不属于第一版必需项 |

## 4. 已实现的关键模块

- `RuntimeMemoryPlanner`：对齐、First-Fit、释放、空闲段合并、Allocation Epoch和峰值占用；
- `Shared3DMemoryModel`：64B等可配置事务、GPU/ATLAS轮询、公用Channel/Bank资源、请求拆分和严格守恒；
- Reference Coupling Loop：把Link/Memory响应反馈为父任务Stall，重新推进全局DAG直到确定性收敛；
- `BoundedLinkModel`：PCIe/CXL队列深度、Credit、全双工方向、带宽、延迟和背压；
- `ResidencyManager`：Copy、Migration、Remote、Write和显式非一致同步；
- Batch Planner：Ragged打包、Mixed Prefill/Decode Epoch和按目标设备拆分；
- `bridge-memory`：TraceAddr → TensorID+offset → Global PA → DRAM服务；
- `dse`：点路径搜索轴、候选上限、确定性运行、Fidelity与结果排序；
- OPT-6.7B FP16、BS=1、初始KV=1024、单步Decode的RTX 4090 Roofline配置。
- Model 3 GPU-only无竞争基线：全部算子固定在GPU，Logic Die后端关闭，共享3D-DRAM只接受`gpu0`请求，并输出零Logic Die任务/请求验收计数。
- 最小Cycle-Accurate桥：全部GPU Memory Partition共享一个Ramulator2实例，读请求等待完成回调；它还没有外部链路、Logic Die Parent Table、按Byte/Sector Mask拆分和响应链路，因此不能代表最终M6数据通路。

## 5. 当前运行证据

- CMake Release构建：通过；
- C++：9/9通过；
- Python：65/65通过；
- 四种`full_runtime`参考配置：均由自动测试覆盖；
- Model 3：父请求、子事务和字节数守恒通过；
- OPT-6.7B单步Decode：391个逻辑任务，1次Decode Forward、0次Prefill，最终KV长度1025；RTX 4090理论Roofline约`13.733 ms`；
- 同图全ATLAS 3D-DRAM参考Roofline约`33.796 ms`，RTX 3070 Decode约`30.899 ms`；当前409.6 GB/s参考参数下3D-DRAM相对3070/4090的加速比分别为`0.914x`/`0.406x`，结果明确保持未校准状态；
- OPT-6.7B Prefill：FP16、BS=1、Context=1024，1次Prefill、0次Decode；RTX 3070理论Roofline约`179.558 ms`；
- OPT-6.7B GPU-only共享3D-DRAM功能运行：391个GPU任务、0条跨设备路由、774个GPU父内存请求、0个Logic Die请求，13,966个子事务与13,842,739,592 B流量均守恒；该配置为1 MiB粗粒度未校准参考模型，不用于性能结论；
- Accel-Sim v2 QV100旧Trace：原生与Adapter均为14,731 cycles、10,473,824 instructions；
- Accel-Sim v2 RTX 3070本机`vector_add` `.tracez`：原生与Adapter均为5,657 cycles、61,440 instructions；
- Accel-Sim + 单实例Ramulator2 QV100 Backprop：两次均为14,700 GPU cycles、10,473,824 instructions；Ramulator2均为11,038 cycles、63 reads/63 completed、0 outstanding，原生内部DRAM基线为14,731 GPU cycles；耦合配置关闭原固定`dram_latency`以避免重复计时；
- ATLAS测试GEMM：两次独立运行均为48,446 cycles、0.00581352 J。

## 6. 后续实现计划

| 顺序 | 计划项 | 完成条件 |
|---:|---|---|
| P1 | 固化外部与内部带宽配置 | `external_link`与Ramulator2组织分开；`DQ/channel_width/rate/nBL/tCK/transaction_bytes`自洽，带宽计算器和Schema拒绝矛盾配置 |
| P2 | Bridge ABI v2 | GPU请求携带Parent ID、Global PA、Size、Byte/Sector Mask、Partition、Ordering和QoS；不再默认一个`mem_fetch`等于一个DRAM事务 |
| P3 | Logic Die Memory Gateway | 统一Ingress Queue、Parent Table、确定性拆分、Child重试、全部Child完成汇聚；写请求默认`durable`，`posted`只作为显式可选语义 |
| P4 | 双向Cycle Link | 请求和响应分别建模Payload/Wire Bytes、Header、Flit、Credit、Queue、Serialization、Propagation和Full/Half Duplex |
| P5 | 多时钟推进 | GPU、外部Link、Logic Die和DRAM分别配置频率，以整数fs/相位累加器推进，移除“最后一个Partition代替全系统Tick”的临时约定 |
| P6 | GPU-only分层资格验证 | 分别构造外部带宽瓶颈和内部DRAM瓶颈微基准；Parent/Child/Byte守恒、所有Child完成前GPU不恢复、退出时零在途 |
| P7 | ATLAS实时端口与竞争 | ATLAS请求通过内部Hybrid-Bond Port进入同一Gateway/Ramulator2，完成GPU-only、ATLAS-only和双发起方黄金对照 |
| P8 | 真实LLM Artifact | 获取精确OPT/目标LLM GPU Trace和匹配ATLAS YAML，再进行RTX 3070/4090与3D-DRAM校准和端到端评估 |

Model 3直接显存模式将每个GPU LLC Miss作为外部Parent请求；Model 4使用CXL.mem请求/响应；Model 2的PCIe路径保持DMA/Page Migration语义，不能把每个GPU Load/Store伪装成PCIe事务。完成P1–P7之前，最小Bridge结果不得用于GPU与3D-DRAM有效带宽或LLM加速比结论。

在这些资格记录完成前，目录或接口存在不能被表述为目标硬件已经验证。
