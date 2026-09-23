# 算子建模与测试状态

更新日期：2026-09-10
Prefill权威机器记录：`configs/hetero/operator_capabilities/tinyllama_prefill_layer0_bs1_ctx16.json`
Decode单步权威机器记录：`configs/hetero/operator_capabilities/tinyllama_decode_{1,22}layer_bs1_ctx16_p19.json`
Decode四步权威机器记录：`configs/hetero/operator_capabilities/tinyllama_decode4_{1,22}layer_bs1_ctx16_p20.json`
多Batch权威机器记录：`configs/hetero/operator_capabilities/p22_multi_batch_functional.json`
P23真实BS=2 Decode权威机器记录：`validation/p23/ready_catalog.json`、`validation/p23/timeline/qualification_record.json`
请求控制权威机器记录：`validation/p24/qualification_record.json`
QoS/存活性权威机器记录：`validation/p25/qualification_record.json`
P30在线Hugging Face记录：`validation/p30/huggingface_online/qualification_record.json`
P31框架驱动Artifact记录：`validation/p31/framework_artifacts/qualification_record.json`
P32在线Shadow记录：`validation/p32/qualification_record.json`
P33双侧周期记录：`validation/p33/qualification_record.json`
P34真实框架事件记录：`validation/p34/qualification_record.json`

## P30–P34框架驱动增量状态

- P30：远端RTX 4090真实执行固定TinyLlama Prefill和单步Decode，两遍稳定记录67个语义事件、30个模块、96个Tensor绑定及相同数值结果。CUDA分配器的50/52个Storage差异被保留为观测证据，不进入稳定Simulation Key。
- P31/P33 GPU：真实Layer-0 Decode捕获形成40-Kernel整层源Artifact，四个Allocator Range绑定Global PA；P33双遍均为15,859,267 cycles、371,846,682 instructions、2,756,823个durable Parent和零在途。该固定整层Artifact获得独立请求周期资格，但不等价于下表14类逐算子Ready Catalog，也不是性能资格。
- P31/P33 ATLAS：QKV Projection从Tensor IR生成16核Stage/Tile计划及338,080条完整内存请求；P33双遍完整Ramulator2周期回放均在1,205,772 GPU cycles结束，全部Parent/Child durable且零在途。
- P32：固定HF请求下，GPU整层参考包络与ATLAS QKV候选以非累加分支进入同一Shadow时间线；二者不能相加为混合放置makespan。
- P34：真实vLLM Scheduler/Paged-KV Block Table和TensorRT-LLM Engine/Profile/SimpleScheduler已完成双遍语义资格；TensorRT范围仅为PyTorch backend。

P31固定合同为远端SM89捕获设备、实际SM86 SASS Binary和SM86回放目标。三个身份不得合并；模型、Revision、Batch、Context、Phase、融合边界或框架版本变化均需生成新的Artifact。

## P23–P25增量状态

- P23：TinyLlama Layer 0、FP16、BS=2、Context=16、KV=17的14类GPU算子已全部在远端RTX 4090完成SASS/NVBit捕获和独立Range-Rebase资格；15个GPU实例、KV Append与四个请求/KV生命周期任务组成的20任务统一时间线也已完成双遍功能资格。封存Catalog标记`timeline_integration_ready=true`，但`performance_eligible=false`。
- P24：EOS、最大生成长度、等待/活跃取消、KV容量Admission、Retire释放和Global PA first-fit复用已完成两组单层双遍功能资格。
- P25：GPU/ATLAS确定性加权公平、优先级、饥饿上界和deadlock/livelock Watchdog已完成功能资格；P27已补充隔离BookSim2+Ramulator2真实数据面，但完整ATLAS Chip NoC与性能仍未校准。

P23的SASS获取位置是身份合同的一部分：编译/加载、Kernel执行、SASS读取和NVBit捕获必须共置于远端RTX 4090。本地只封存Artifact Manifest、资格记录及哈希，不得生成替代Trace。捕获设备SM89不等于所有Library Kernel的SASS Binary Version，二者必须分别报告。当前封存仅适用于固定BS=2、Context=16、`q_len=1`、KV=17；KV长度、Batch、Ragged成员、模型或dtype变化均需要新的精确捕获和资格。

## P22 多Batch功能覆盖

P22覆盖5种精确工作负载：静态同形BS=2、静态Ragged Padding BS=2、静态Ragged Split BS=2、22层Continuous四请求Decode，以及两层Continuous四请求混合Prefill/Decode。三种Batch策略、GPU/ATLAS设备Sub-Batch、KV容量Admission/Retire和动态Global PA分配/释放/复用均完成双遍功能资格，资格记录为`validation/p22/qualification_record.json`。

这些记录的`functional_cycle_ready=true`只说明状态、因果、资源和地址生命周期闭环。正式运行使用`request_cycle_composed`和`scheduling.epoch_duration_fs`；`exact_batched_kernel_ready=false`、`performance_eligible=false`。改变成员数、成员Q/KV长度、模型、dtype、算子、设备或融合方式时都需要新的精确Batched/Fused Artifact，不能从BS=1或相邻Batch组合周期。

## P20 多Token Decode功能覆盖

P20为同一模型Revision、FP16、BS=1、初始Context=16建立连续4 Token合同，精确覆盖KV长度17、18、19和20。1层与22层目录分别覆盖68和1,076个任务；每层K/V值保持同一Global PA身份，版本从0逐步提交到4。资格记录为`validation/p20/{one_layer,twenty_two_layer}/qualification_record.json`。

P20目录中的四个KV长度是四个独立精确Shape。它们的`request_cycle_ready=false`与`performance_eligible=false`表示当前后端仍是未校准分块周期合同；只有逐Shape捕获、绑定并资格真实Accel-Sim指令Trace后，才允许提升请求周期Ready或性能状态。

## P19 Decode功能覆盖

P19新增同一模型Revision、FP16、BS=1、初始Context=16、`q_len=1`、最终KV长度17的1层和22层合同。两种合同分别覆盖20和272个任务，所有19种算子类型均有可执行请求周期合同；KV按层拥有独立Global PA范围，并完成`version 0 → 1`提交。资格记录为`validation/p19/{one_layer,twenty_two_layer}/qualification_record.json`。

该Decode目录中的`implementation_status=implemented`只表示功能路径存在，`request_cycle_ready=false`和`performance_eligible=false`是有意设置：计算周期来自未校准分块合同，未捕获Decode精确Shape的Accel-Sim指令Trace。输入Tensor长度、模型维度、Batch、Context或KV长度变化时，必须生成新的Shape Contract并重新运行；不得由本目录外推周期。

## 固定资格边界

当前记录只适用于以下精确合同：

- 模型：TinyLlama-1.1B，checkpoint revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`；
- 精度：FP16；层：Layer 0；Batch：1；Context：16；
- 层内算子：`q_len=16`、`kv_len=16`；Final Norm、LM Head、Sampling：整体 Context=16、实际 `q_len=1`；
- GPU：RTX 3070 / SM86 Trace；内存：共享 3D-DRAM 的 Range-Rebase 请求周期路径。

改变模型、revision、dtype、Batch、Context、Q/KV 长度或模型维度时，旧 Artifact 默认失效，必须重新匹配、捕获并资格验证。

## 当前覆盖

参考图共有 **20 个任务实例、19 种算子类型**。14种GPU算子、共15个任务实例已经完成真实Accel-Sim请求周期双跑资格；3种KV运行时任务已经把精确Global PA请求送入外部Link和唯一live Ramulator2并完成两遍资格；2种主机控制事件只参与因果时间线并排除在设备性能边界之外。17种算子类型为`request_cycle_ready=true`，但全部19种仍为`performance_eligible=false`。

| 算子类型 | 实例数 | 当前建模 | 已完成测试 | 请求周期 Ready | 当前限制 |
|---|---:|---|---|---:|---|
| `request_start` | 1 | 主机控制边界事件 | 双遍图因果、零内存请求 | 否（不适用） | 排除出设备性能；未校准主机开销 |
| `kv_allocate` | 1 | 元数据Global PA读写 + 外部Link + live Ramulator2 | 双遍3个Parent/Child/durable守恒 | 是 | 控制参数未校准 |
| `token_embedding` | 1 | Shape锁定CUDA embedding gather + Accel-Sim | 非空SM86 Trace、Range-Rebase双跑 | 是 | 仅选定CUDA实现；不含框架启动和checkpoint数值影响 |
| `attention_norm` | 1 | Accel-Sim + live Ramulator2 | 双跑、Range-Rebase、全局时间线 | 是 | 仅固定 Shape/Revision |
| `qkv_projection` | 1 | Accel-Sim + live Ramulator2 | 双跑、Range-Rebase、全局时间线 | 是 | 仅固定 Shape/Revision |
| `rope` | 1 | Accel-Sim + live Ramulator2 | 双跑、Range-Rebase、全局时间线 | 是 | 仅固定 Shape/Revision |
| `kv_append` | 1 | Copy Engine精确K/V读写 + 外部Link + live Ramulator2 | 双遍32,768 B/512 Parent/Child/durable守恒 | 是 | Copy Engine参数未校准 |
| `causal_attention` | 1 | Accel-Sim + live Ramulator2 | 双跑、Range-Rebase、全局时间线 | 是 | 仅固定 Shape/Revision |
| `output_projection` | 1 | Accel-Sim + live Ramulator2 | 双跑、Range-Rebase、全局时间线 | 是 | 仅固定 Shape/Revision |
| `residual_add` | 2 | Shape锁定FP16 CUDA elementwise + Accel-Sim | 非空SM86 Trace、Range-Rebase双跑 | 是 | 两实例共享源Trace，但使用独立运行期Value绑定 |
| `mlp_norm` | 1 | Accel-Sim + live Ramulator2 | 双跑、Range-Rebase、全局时间线 | 是 | 仅固定 Shape/Revision |
| `gate_up_projection` | 1 | Accel-Sim + live Ramulator2 | 双跑、Range-Rebase、全局时间线 | 是 | 仅固定 Shape/Revision |
| `silu_multiply` | 1 | Accel-Sim + live Ramulator2 | 双跑、Range-Rebase、全局时间线 | 是 | 仅固定 Shape/Revision |
| `down_projection` | 1 | Accel-Sim + live Ramulator2 | 双跑、Range-Rebase、全局时间线 | 是 | 仅固定 Shape/Revision |
| `final_norm` | 1 | Accel-Sim + live Ramulator2 | 双跑、Range-Rebase、全局时间线 | 是 | 仅固定 Shape/Revision |
| `lm_head` | 1 | Accel-Sim + live Ramulator2 | 双跑、Range-Rebase、全局时间线 | 是 | 仅固定 Shape/Revision |
| `sampling` | 1 | Accel-Sim + live Ramulator2 | 双跑、Range-Rebase、全局时间线 | 是 | 仅固定 Shape/Revision |
| `request_finish` | 1 | 主机控制边界事件 | 双遍图因果、零内存请求 | 否（不适用） | 排除出设备性能；未校准主机开销 |
| `kv_release` | 1 | 元数据Global PA读写 + 外部Link + live Ramulator2 | 双遍2个Parent/Child/durable守恒 | 是 | 控制参数未校准 |

## 状态判定规则

- **已建模**：已有可执行语义、真实 Trace Backend 或显式运行时周期合同。
- **已测试**：只说明列出的单元、因果或双跑测试已经通过，不自动获得性能资格。
- **请求周期 Ready**：GPU Trace必须满足精确Shape/Revision、SM86双跑、地址零漏配、唯一Ramulator2、Parent/Child/durable守恒和零在途；非SM的KV任务必须满足精确Global PA事务、唯一Ramulator2和相同守恒。纯主机控制事件不产生设备请求，因此只获得因果资格。
- **性能可用**：还需整机参数校准和所有关键任务覆盖；当前数量为 0。

P17已经为性能可用增加独立机器门禁。除本表的逐任务`performance_eligible`外，还必须同时通过GPU Operator、Copy Engine、Runtime、外部Link、Logic-Die Gateway和3D-DRAM六项校准。当前14种GPU算子都已具备同Shape的RTX 3070本地显存原生测量、Native/Trace双侧执行程序身份和同为`gpu_local_vram`的Accel-Sim双遍记录；Artifact与拓扑也已匹配。Down Projection、Output Projection、QKV Projection和LM Head通过15%数值误差门禁，严格配对结果为4/14；其余10类仅因误差超阈值阻断。六组件整机校准仍未通过，因此这些数据仍只构成`measured_unvalidated`证据，本表19种算子的性能可用数量仍为0。

## 规模变化的处理原则

算子名称相同不代表周期可复用。Batch、Context、隐藏维度、中间维度、词表、Attention/KV Head、dtype 或 checkpoint 改变，可能同时改变 Kernel 选择与融合、Grid/Block、Tensor Core 指令数、访存事务、缓存命中、Workspace、Global PA 容量和 DRAM Bank/Row 分布。Attention 随序列长度通常包含二次项，GEMM 和 KV 流量也按各自维度变化，因此不得用一个统一比例线性缩放现有周期。

系统采用 fail-closed 策略：不满足精确合同的任务在 Dispatch 前拒绝使用 Ready Artifact。新规模必须生成新的 Shape Contract 和 Artifact，并重新执行捕获、Range-Rebase、双跑请求周期资格及全局时间线验证。
