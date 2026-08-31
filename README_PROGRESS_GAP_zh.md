# GPU-ATLAS-HeteroSim 当前完成情况与计划差距

评估日期：2026-08-30
当前版本：`0.22.0`
规范基线：`hetero-sim/v1`、设计合同v1.21

## 1. 当前结论

P1–P7已经从计划项变为可构建、可运行和有资格记录的实现：GPU请求经过外部请求Link、Logic-Die Gateway、Parent/Child拆分、唯一Ramulator2、Parent汇聚和响应Link；ATLAS原生`ComponentInput`可从内部Hybrid-Bond端口进入同一内存系统；GPU、Link、Gateway和DRAM采用独立时钟。外部12.8 GB/s和内部409.6 GB/s是两个独立、可校验的资源。

P8已经完成第一个真实LLM算子，而不是完整模型：TinyLlama‑1.1B layer-0 Q投影在GPU和ATLAS两侧具有相同Checkpoint、Shape、FP16、Batch和Decode Context。RTX 3070原生、RTX 3070+共享3D‑DRAM、ATLAS内部3D‑DRAM三条路径都通过双次确定性运行。

P9b已经把真实Accel-Sim计算后端和完整`atlasim.Chip`接入同一进程与全局时间推进器；双方请求进入唯一Ramulator2，完成队列按Initiator隔离。固定Q投影竞争用例通过请求守恒、执行重叠和零在途验证。

P10a已经完成单放置控制面：每个逻辑节点必须恰好映射到一个设备任务；跨设备消费者按每个输入值分别生成带版本、大小、依赖和Fence动作的路由；KV Append被建模为版本化读改写；带时序运行输出`hetero-residency/v2`。该闭环目前通过参考事件运行验证，尚未把不同真实GPU/ATLAS算子的所有周期请求接入同一执行计划。

P10b-B至P14进一步完成了请求级Prefill部署：任务按读完成、计算周期、写完成和版本提交推进；GPU外部端口、ATLAS内部端口与Route Acquire请求共用唯一Ramulator2。完整TinyLlama‑1.1B FP16、BS=1、Context=1024、22层Prefill已双跑通过，但计算是未校准分块周期契约，内存是有界代表采样，因此不能发布端到端Latency、Token/s或加速比。

P15e已完成Context=16一层Prefill大型请求轨迹的流式gzip双跑、Attention Norm在线Range-Rebase以及LM Head长时双遍资格；P15f完成QKV Projection的Allocator Segment Range-Rebase；P15g首先验证两个Ready算子的全局时间线。P15h现已完成其余10个GPU算子的重捕获、双遍资格与统一接入：12个`request_cycle_ready=true`真实Accel-Sim算子共享同一Prefill时间线，并通过依赖、GPU资源互斥、Global PA、唯一Ramulator2、Parent/Child/durable守恒、零在途和版本提交因果验证。其余8个控制、KV管理与残差任务仍为分析/运行时模型，性能资格保持关闭。

内存与片上网络的当前状态必须分开表述：Ramulator2已经是`prefill_cycle`和既有耦合资格路径的实时内存时序所有者；BookSim2源码已随ATLAS固定版本并编译进`libatlasim-lib.so`，但当前P9a/P9b Chip配置没有`architecture.noc`，P14也未调用ATLAS完整Chip，因此BookSim2尚未在当前主实验中激活，状态保持`adapter_pending_qualification`。

## 2. 三步GPU集成计划对照

| 步骤 | 状态 | 当前证据 | 仍需完成 |
|---|---|---|---|
| 第一步：独立Accel-Sim后端 | ✅ | v2.0.0；QV100回归；RTX 3070 NVBit 1.8；真实TinyLlama Q投影36,324 cycles | RTX 3070实机校准；RTX 4090配置与Trace |
| 第二步：算子事件级GPU+ATLAS | ✅ | 同一ModelGraph支持GPU/ATLAS放置、传输和回退；精确Q投影GPU Trace与ATLAS YAML已匹配 | 把全部层算子替换为真实Artifact；真实多Batch Kernel Shape |
| 第三步：共享3D‑DRAM请求级耦合 | ✅ P10b-B/P14部署闭环 | 严格计划、GPU/ATLAS端口、Route Fence/Acquire、单Owner Ramulator2、22层Prefill全部连通 | 用全算子真实Trace/ATLAS Artifact替换分块周期契约和采样流量；长时间竞争与QoS |

## 3. P1–P15执行状态

| 顺序 | 计划项 | 状态 | 验收证据或差距 |
|---:|---|---:|---|
| P1 | 外部/内部带宽合同 | ✅ | `BandwidthContract`精确校验；ATLAS风格16×512-bit×400 MT/s=409.6 GB/s；外部Link独立为12.8 GB/s |
| P2 | Bridge ABI v2 | ✅ | Parent ID、Global PA、Size、Mask、Partition、Ordering、QoS和Initiator均显式传递；SM86 32B Sector Mask已修正 |
| P3 | Logic-Die Gateway | ✅ | 64B Child拆分、Parent Table、重试、全Child汇聚、durable完成和零在途 |
| P4 | 双向Cycle Link | ✅ | 请求/响应Payload、Header、Flit、Wire Byte、Credit、队列、序列化和传播分别建模 |
| P5 | 多时钟推进 | ✅ | GPU 1.2 GHz、Link/Gateway/DRAM 400 MHz黄金用例满足精确3:1周期比 |
| P6 | GPU-only分层资格 | ✅ | 外部瓶颈346 DRAM cycles；内部瓶颈163；父子、字节和完成语义全部通过 |
| P7 | ATLAS端口与竞争 | ✅ 端口级 | GPU-only/ATLAS-only/并发为163/90/239 cycles；两方均观察竞争；完整Chip并发由P9a/P9b继续验证 |
| P8 | 真实LLM Artifact | 🟡 一层源Trace扩展 | Decode Q投影三路径通过；P15d已有13类完整流量源Artifact；12个非空GPU算子完成独立耦合资格；全层真实Artifact和多Batch待完成 |
| P9a | 完整ATLAS Chip共享内存 | ✅ | 16核真实迭代产生139,456个ATLAS Parent；ATLAS-only 76,418全局GPU周期，并发确定性GPU流量后81,329；唯一Ramulator2、全完成、0在途 |
| P9b | Accel-Sim+完整ATLAS Chip并发 | ✅ | GPU 1,541,401 cycles/262,272 Parent；ATLAS 141,255 cycles/139,456 Parent；唯一Ramulator2共401,728 Parent，全完成、0在途 |
| P10a | 单放置与版本化Residency控制面 | ✅ 参考闭环 | 缺失/重复放置拒绝；逐输入值路由；KV 0→1读改写；Model 3参考图228节点=228任务、28路由、571 Residency事件；真实多算子周期耦合待P10b |
| P10b-A | 在线Backend启动门禁 | ✅ 总时长适配器 | 85节点=85 Dispatch；12 Route；117版本检查；Accel-Sim 14,731 cycles、ATLAS 48,446 cycles；双次启动日志和Metrics完全一致 |
| P10b-B | live请求周期运行时 | ✅ 部署资格 | 混合单层20任务/4 Route；GPU 347、ATLAS 35 Parent；382全部完成；唯一Ramulator2、0在途 |
| P11 | 完整Prefill契约 | ✅ | 显式参数/Tensor、19类算子双设备目录、Global PA、读→算→写阶段；分析回退为0 |
| P12 | 单层GPU Prefill | ✅ | Context=16，20任务、378 GPU Parent、ATLAS=0，双跑一致 |
| P13 | 22层规模扩展 | ✅ | Context=16，272任务、448地址区间、3,382 GPU Parent，双跑一致 |
| P14 | 完整Prefill部署 | ✅ 性能未资格 | TinyLlama‑1.1B FP16、BS=1、Context=1024；最终KV=1024；3,385 Parent全完成；地址占用3,957,580,290 B / 4 GiB |
| P15a | 首批真实算子Artifact | ✅ 独立资格 | 4类GPU Trace双跑；KV为`runtime_state`；ATLAS QKV双跑150,932 cycles；严格绑定运行Trace Coverage=20% |
| P15b | 首批选择性完整流量 | ✅ 部署资格 | 5/20任务完整64B事务；175,936 Full Parent + 234 Sampled Parent；唯一Ramulator2完成176,170 Parent，双跑核心产物一致 |
| P15c | 四算子指令—内存周期闭环 | ✅ 耦合资格；🟡 地址/全局调度待接入 | RMSNorm/QKV/RoPE/Attention双跑全部一致；383,260个GPU Parent、383,286个Child均完成；每次单Ramulator2、0在途；4/4 `compute_memory_coupled=true`，0/4 `request_cycle_ready` |
| P15d | 13算子完整Value流量与新增Trace | ✅ | 13/20任务完整流量、7/20采样；3,462,738 Parent/Child全完成；12个真实GPU算子独立耦合双跑通过，合计6,993,530 Parent和6,996,227 Child |
| P15e | 流式轨迹与在线地址重绑定 | ✅ 单算子资格；🟡 扩展待完成 | gzip双跑94,859,940 B、峰值RSS约524.6 MiB；Attention Norm 40,970次转换零漏配，6个范围、2,176 Parent/Child、双跑66,697 cycles；仅该重捕获Artifact可标记`request_cycle_ready=true` |
| P15f | QKV Allocator Segment Range-Rebase | ✅ 双算子Ready；🟡 全局时间线待接入 | QKV双跑2,168,865 cycles、34,943,066条指令；736,837次转换零漏配，12个范围；375,899 Parent与375,944 Child全部完成；Attention Norm+QKV共2个`request_cycle_ready=true` Artifact |
| P15g | 两个Ready算子嵌入Prefill全局时间线 | ✅ 因果资格；性能未资格 | Attention在58,833,038,873 fs完成，QKV同刻启动；共享Value与Global PA 351,485,952；GPU资源不重叠；请求零在途；QKV验证Attention输出v1；其余任务分析回退 |
| P15h | 12算子Range-Rebase与统一Prefill时间线 | ✅ 因果资格；性能未资格 | 10个新增算子重捕获与双遍资格全部通过；12个Ready算子共40,060,873 GPU cycles、6,995,173 Parent、6,998,046 Child和804,512,881次成功转换；84个Global PA区间、56个Workspace、12条请求绑定与18次版本提交通过因果检查；其余8任务仍为分析/运行时模型 |

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

### 完整ATLAS Chip实时内存

- 16核完整Chip在外部模式下产生139,456个64B ATLAS Parent，共8,925,184事务字节；
- ATLAS-only完成时刻为76,418个1.2 GHz全局GPU周期，完整Chip自身记录63,681个1 GHz周期；
- 加入4,096个确定性GPU Parent后，ATLAS完成推迟至81,329个全局周期，Chip记录67,774个周期；
- 两方请求全部完成，Ramulator2实例数为1，退出时`outstanding=0`；
- 该结果的GPU侧是确定性内存流，不是Accel-Sim计算后端，故只标记P9a通过。

### 真实双计算后端并发

- Accel-Sim执行真实SM86 TinyLlama Q投影Trace：1,541,401 GPU cycles、15,908,352条指令、262,272个GPU Parent；
- 完整ATLAS Chip执行形状匹配Artifact：141,255 ATLAS cycles、139,456个ATLAS Parent，在第159,901个GPU周期完成；
- 唯一Ramulator2接收401,728个Parent，全部完成，实例数1且退出时`outstanding=0`；
- 当前是同一Q投影在两侧重复执行的竞争压力用例，不是有效的算子单放置或端到端调度。

## 5. 与第一版完成定义的差距

| 能力 | 参考/接口 | 周期后端 | 资格状态 |
|---|---:|---:|---|
| 四种系统组织形式 | ✅ | 部分 | Model 3直连路径已落地；Model 2 PCIe DMA与Model 4 CXL.mem仍主要是事件级 |
| 自由算子放置 | ✅ 严格一一对应 | Prefill部署周期 | 首批5类Artifact已注册、GPU严格绑定已验证；真实全算子Artifact仍待生成 |
| Prefill/Decode完整图 | ✅ | Prefill已部署 | 22层Prefill生命周期完成；Decode请求周期部署待完成；当前计算契约/采样流量性能未资格 |
| 多Batch Continuous/Ragged | ✅ | ⬜ | 需要真实Batched/Fused Kernel Trace与ATLAS分块Artifact |
| GPU/ATLAS共享DRAM竞争 | ✅ 参考 | ✅ 真实Accel-Sim+完整ATLAS Chip | 长时间混合读写、公平性和QoS待实现 |
| ATLAS片上BookSim2 NoC | ✅ 源码/构建 | ⬜ 当前实验未激活 | `libnoc.a`和BookSim符号已构建；需增加`architecture.noc`、真实NoC Packet、统计守恒与双跑资格 |
| 地址层次与映射 | ✅ 设计/离线 | 🟡 在线部分 | 离线支持TraceAddr→Tensor+Offset→Global PA；P9b在线GPU仍直接传递`data->get_addr()`，DRAM仅支持OneLevelInterleave |
| DSE | ✅ | 单候选资格 | Trace `replay_safe=false`，每个时序反馈候选必须单独验证或执行驱动 |

## 6. 下一步顺序

1. 将P15h从一层Context=16功能资格扩展到多层模型图，优先验证跨层KV生命周期、Global PA容量、Workspace复用和长时间运行的确定性；不得直接按层数线性外推当前周期。
2. 为仍采用分析/运行时模型的控制、KV Allocate/Append/Release和Residual Add补齐可审计的周期模型或真实Backend，并重新执行端到端依赖、资源、地址、请求和版本门禁。
3. 构建一层TinyLlama Decode，再扩展到22层单Token Decode；逐层核对依赖、KV生命周期、内存容量、任务数和请求守恒。
4. 增加多Batch Continuous/Ragged周期运行，使用真实Batched/Fused Kernel而不是复制单请求延迟。
5. 使用硬件测量或可信参考点校准RTX 3070、外部Link和3D-DRAM，并激活、资格化ATLAS BookSim2；在此之前保持`performance_claim_allowed=false`。
6. 增加长时间混合读写、公平性、QoS和死锁/活锁压力测试，并补齐Model 2 PCIe DMA与Model 4 CXL.mem周期路径。

## 7. 已记录但暂缓：虚拟地址与地址哈希

状态：**需求已冻结，当前阶段不实施。** 该项不阻塞接下来的算子放置、单层Decode和端到端任务图开发，但在进行虚拟内存研究、地址映射DSE或跨捕获地址复用前必须完成。

### 7.1 虚拟地址开发需求

- 在线路径增加`identity`、`range_rebase`和`mmu_tlb`三种显式模式；默认不得把Trace地址无声明地解释为物理地址；
- `range_rebase`使用Trace Manifest把`TraceAddr → TensorID+offset → memory_space_id+Global PA`接入真实Accel-Sim访存路径，并在进入DRAM前检查覆盖、对齐、容量、Alias和生命周期；
- `mmu_tlb`后续建模Page Size、TLB层级、命中/替换、Page Table、Page-table Walk、缺页、迁移和地址翻译延迟；
- 明确地址翻译发生在GPU Cache层次中的位置，并把翻译模式、页表快照和分配结果纳入Simulation Key；
- 当前TinyLlama P9b的GPU地址仍直接来自`data->get_addr()`，不得宣称已模拟VA→PA。

### 7.2 地址哈希开发需求

- 将DRAM Mapper配置化为`OneLevelInterleave`、`RoBaRaCoCh/ChRaBaRoCo`、`MOP4CLXOR`和后续自定义XOR表达式；
- 解除共享ATLAS桥对`OneLevelInterleave`的硬编码，但保持ATLAS请求排序、GPU/ATLAS统一译码和唯一Ramulator2；
- 每次运行记录Global PA到Channel/Bank/Row/Column的Mapper名称、参数和内容哈希，并提供可抽样导出的译码结果；
- 增加容量越界、不同Tensor/Memory Space碰撞、跨Core地址重叠、Channel分布和Row-locality黄金测试；
- Mapper或哈希变化必须生成新的Simulation Key并重新仿真；Trace只有在`replay_safe=true`时才允许跨候选复用。

### 7.3 启动条件

满足以下任一条件时再启动该项：研究问题转向GPU虚拟内存/UVM/CXL共享页；需要比较XOR/Channel/Bank映射；当前Global PA容量不足；或准备发布地址映射DSE结果。在此之前保持现有P9b路径不变，并在报告中标记`virtual_memory_mode=identity_untranslated`、`dram_mapper=OneLevelInterleave`。

## 8. 声明边界

目录、接口或一次运行存在，不等于性能已经资格。当前可准确表述为：**分层GPU↔Logic‑Die↔3D‑DRAM路径、真实Accel-Sim与完整ATLAS Chip并发、严格单放置/版本化Residency、live Ramulator2请求周期运行时、Context=1024的P14部署、Context=16的P15d 13算子完整Value流量、12算子独立耦合资格、流式gzip请求轨迹，以及Attention Norm和QKV两个真实Accel-Sim算子的Prefill全局因果接入已经实现。** 当前不可表述为：全部算子稳定Global PA重绑定已完成、完整Prefill已由真实Accel-Sim算子覆盖、P14/P15是校准端到端性能、Decode或多Batch周期执行已完成、RTX 3070/目标3D‑DRAM已实机校准、VA→PA/MMU/TLB或XOR地址映射已实现，或固定Trace可安全复用于任意内存候选。
