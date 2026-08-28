# GPU-ATLAS-HeteroSim 当前完成情况与计划差距

评估日期：2026-08-28
当前版本：`0.7.0`
规范基线：`hetero-sim/v1`、设计合同v1.5

## 1. 当前结论

P1–P7已经从计划项变为可构建、可运行和有资格记录的实现：GPU请求经过外部请求Link、Logic-Die Gateway、Parent/Child拆分、唯一Ramulator2、Parent汇聚和响应Link；ATLAS原生`ComponentInput`可从内部Hybrid-Bond端口进入同一内存系统；GPU、Link、Gateway和DRAM采用独立时钟。外部12.8 GB/s和内部409.6 GB/s是两个独立、可校验的资源。

P8已经完成第一个真实LLM算子，而不是完整模型：TinyLlama‑1.1B layer-0 Q投影在GPU和ATLAS两侧具有相同Checkpoint、Shape、FP16、Batch和Decode Context。RTX 3070原生、RTX 3070+共享3D‑DRAM、ATLAS内部3D‑DRAM三条路径都通过双次确定性运行。

完整`atlasim.Chip`尚未与Accel-Sim在一个全局周期调度器内同时运行；其余算子、22层端到端Prefill/Decode和真实多Batch周期执行尚未完成。因此当前不能发布端到端Latency、Token/s或目标硬件加速比声明。

## 2. 三步GPU集成计划对照

| 步骤 | 状态 | 当前证据 | 仍需完成 |
|---|---|---|---|
| 第一步：独立Accel-Sim后端 | ✅ | v2.0.0；QV100回归；RTX 3070 NVBit 1.8；真实TinyLlama Q投影36,324 cycles | RTX 3070实机校准；RTX 4090配置与Trace |
| 第二步：算子事件级GPU+ATLAS | ✅ | 同一ModelGraph支持GPU/ATLAS放置、传输和回退；精确Q投影GPU Trace与ATLAS YAML已匹配 | 把全部层算子替换为真实Artifact；真实多Batch Kernel Shape |
| 第三步：共享3D‑DRAM请求级耦合 | 🟡 核心路径完成 | ABI v2、双向Link、Gateway、单Owner Ramulator2、多时钟、ATLAS内部端口和双发起方竞争均通过 | 完整ATLAS Chip与Accel-Sim同时推进；周期级Residency/Fence；长时间竞争与QoS |

## 3. P1–P8执行状态

| 顺序 | 计划项 | 状态 | 验收证据或差距 |
|---:|---|---:|---|
| P1 | 外部/内部带宽合同 | ✅ | `BandwidthContract`精确校验；ATLAS风格16×512-bit×400 MT/s=409.6 GB/s；外部Link独立为12.8 GB/s |
| P2 | Bridge ABI v2 | ✅ | Parent ID、Global PA、Size、Mask、Partition、Ordering、QoS和Initiator均显式传递；SM86 32B Sector Mask已修正 |
| P3 | Logic-Die Gateway | ✅ | 64B Child拆分、Parent Table、重试、全Child汇聚、durable完成和零在途 |
| P4 | 双向Cycle Link | ✅ | 请求/响应Payload、Header、Flit、Wire Byte、Credit、队列、序列化和传播分别建模 |
| P5 | 多时钟推进 | ✅ | GPU 1.2 GHz、Link/Gateway/DRAM 400 MHz黄金用例满足精确3:1周期比 |
| P6 | GPU-only分层资格 | ✅ | 外部瓶颈346 DRAM cycles；内部瓶颈163；父子、字节和完成语义全部通过 |
| P7 | ATLAS端口与竞争 | ✅ 端口级 | GPU-only/ATLAS-only/并发为163/90/239 cycles；两方均观察竞争；完整Chip并发仍待实现 |
| P8 | 真实LLM Artifact | 🟡 单算子完成 | TinyLlama Q投影三路径通过；其余算子、全层、端到端和多Batch周期执行待完成 |

## 4. 当前准确运行证据

### 分层与双发起方

- GPU分层外部瓶颈：72 Parent、144 Child、9,216 Logical/Internal Bytes、1个Ramulator2、0在途；
- GPU分层内部瓶颈：相同请求与字节，完成更快；
- 双发起方：GPU 72 Parent/144 Child，ATLAS 80 Parent/80 Child，总计14,336 B；并发使GPU完成从489增至717 GPU cycles，使ATLAS完成从270增至654 GPU cycles。

### TinyLlama真实Q投影

| 路径 | 周期 | 延迟 | 备注 |
|---|---:|---:|---|
| RTX 3070原生显存 | 36,324 | 32.088339 µs | 15,908,352条模拟指令 |
| RTX 3070经外部Link访问3D‑DRAM | 1,498,113 | 1,323.421378 µs | 262,272次读全部完成；内部529,368 cycles |
| ATLAS 16核内部3D‑DRAM | 24,613 | 24.613 µs | DRAM 24,442 cycles；矩阵5,472 cycles |

当前配置下，共享3D‑DRAM GPU路径/原生GPU延迟为`41.243062×`，原生GPU/ATLAS为`1.303715×`，共享3D‑DRAM GPU/ATLAS为`53.769202×`。这些是配置研究比值，不是实机或端到端结论。

## 5. 与第一版完成定义的差距

| 能力 | 参考/接口 | 周期后端 | 资格状态 |
|---|---:|---:|---|
| 四种系统组织形式 | ✅ | 部分 | Model 3直连路径已落地；Model 2 PCIe DMA与Model 4 CXL.mem仍主要是事件级 |
| 自由算子放置 | ✅ | 单算子 | 周期级跨设备Residency、Copy和Fence待实现 |
| Prefill/Decode完整图 | ✅ | 单Q投影 | 其余Attention/MLP/KV/LM Head/Sampling Artifact待生成 |
| 多Batch Continuous/Ragged | ✅ | ⬜ | 需要真实Batched/Fused Kernel Trace与ATLAS分块Artifact |
| GPU/ATLAS共享DRAM竞争 | ✅ 参考 | ✅ 端口级 | 完整两个计算后端并发调度待实现 |
| 地址层次与映射 | ✅ | ✅ | TraceAddr、Tensor+Offset、Global PA和DRAM Tuple已分层；大规模压力测试待增加 |
| DSE | ✅ | 单候选资格 | Trace `replay_safe=false`，每个时序反馈候选必须单独验证或执行驱动 |

## 6. 下一步顺序

1. 把`AtlasHbPort`接到完整`atlasim.Chip`的实时DRAM请求/完成接口，并与Accel-Sim共用一个全局时间推进器和一个Ramulator2。
2. 完成Q/K/V/O Projection、Attention QK/SV、RMSNorm/RoPE、Gate/Up/Down、KV Append等算子的GPU Trace与ATLAS Artifact生成。
3. 构建一层TinyLlama Decode，再扩展到22层单Token Decode；逐层核对Shape、地址、读写字节和依赖。
4. 增加Prefill和多Batch Continuous/Ragged周期运行，使用真实Batched/Fused Kernel而不是复制单请求延迟。
5. 实现GPU/ATLAS算子切换所需的周期级Residency、Acquire/Release、Copy/Migration和版本检查。
6. 在参数校准后再运行放置、外部带宽、内部组织、地址映射和竞争策略DSE。

## 7. 声明边界

目录、接口或一次微基准存在，不等于端到端系统完成。当前可准确表述为：**分层GPU↔Logic‑Die↔3D‑DRAM路径和双发起方内存端口已经实现；一个真实TinyLlama Q投影已完成形状匹配的周期仿真。** 当前不可表述为：完整TinyLlama/OPT端到端已周期精确运行、RTX 3070或目标3D‑DRAM已经实机校准、RTX 4090已部署，或固定Trace可安全复用于任意内存候选。
