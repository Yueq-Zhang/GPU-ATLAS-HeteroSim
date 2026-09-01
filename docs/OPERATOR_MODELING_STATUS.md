# 算子建模与测试状态

更新日期：2026-08-31
权威机器记录：`configs/hetero/operator_capabilities/tinyllama_prefill_layer0_bs1_ctx16.json`

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

P17已经为性能可用增加独立机器门禁。除本表的逐任务`performance_eligible`外，还必须同时通过GPU Operator、Copy Engine、Runtime、外部Link、Logic-Die Gateway和3D-DRAM六项校准。当前14种GPU算子都已具备同Shape的RTX 3070本地显存原生测量和同为`gpu_local_vram`的Accel-Sim双遍记录；拓扑已匹配，但新Native执行尚未证明与NVBit捕获Trace二进制一致，且10/14算子的误差超过15%，严格配对结果仍为0/14。因此这些数据只构成`measured_unvalidated`证据，本表19种算子的性能可用数量仍为0。

## 规模变化的处理原则

算子名称相同不代表周期可复用。Batch、Context、隐藏维度、中间维度、词表、Attention/KV Head、dtype 或 checkpoint 改变，可能同时改变 Kernel 选择与融合、Grid/Block、Tensor Core 指令数、访存事务、缓存命中、Workspace、Global PA 容量和 DRAM Bank/Row 分布。Attention 随序列长度通常包含二次项，GEMM 和 KV 流量也按各自维度变化，因此不得用一个统一比例线性缩放现有周期。

系统采用 fail-closed 策略：不满足精确合同的任务在 Dispatch 前拒绝使用 Ready Artifact。新规模必须生成新的 Shape Contract 和 Artifact，并重新执行捕获、Range-Rebase、双跑请求周期资格及全局时间线验证。
