# P15e 流式请求轨迹、在线Range-Rebase与LM Head资格

日期：2026-08-30
版本：GPU-ATLAS-HeteroSim v0.20.0
范围：TinyLlama-1.1B、FP16、Layer 0、BS=1、Context=16。

## 1. 大型请求轨迹流式输出

`PrefillCycleRuntime`不再把全部请求保存在内存并一次性序列化。请求按确定顺序写入`request_cycle_trace.jsonl.gz`，紧凑的`request_cycle_trace.json`和`trace_manifest.json`记录文件、数量、字节和SHA-256。消费端使用流式迭代器校验内容，不能重新物化全部请求。

两次完整运行均满足：

| 项目 | 双跑一致结果 |
|---|---:|
| Parent / Child | 3,462,738 / 3,462,738 |
| DRAM cycles | 10,401,594 |
| GPU global cycles | 31,204,782 |
| Ramulator2 instances / Outstanding | 1 / 0 |
| gzip大小 | 94,859,940 B |
| gzip SHA-256 | `aa3edd9ca85dd3f600e8a1646d1b3af9bfc84f99d50c81f6b422c4897564795d` |
| 峰值RSS | 524,572 / 524,564 KiB |
| 用时 | 约3分54秒 / 3分56秒 |

资格记录：

```text
/opt/gpu-atlas/qualification/p15e-streaming/qualification_record.json
```

该结果关闭的是轨迹可扩展性与确定性门槛，不改变Prefill计算仍来自未校准分块周期合同的事实。

## 2. Attention Norm在线Range-Rebase

重新捕获过程从真实CUDA Allocator事件恢复6个不重叠Capture范围：Attention Norm的Weight、Input和Output，以及3个框架Workspace。三个Workspace为：

| Capture范围 | 大小 |
|---|---:|
| `0xb0813d200..0xb0814d200` | 65,536 B |
| `0xb0815d200..0xb0817d240` | 131,136 B |
| `0xb0817d400..0xb0819d400` | 131,072 B |

在线桥在Cache访问路径之后、进入共享内存桥之前执行`TraceAddr → Range+Offset → Global PA`。两次完整Accel-Sim运行严格一致：

| 项目 | 双跑一致结果 |
|---|---:|
| GPU cycles | 66,697 |
| Instructions | 5,290,064 |
| Address translated / unmapped | 40,970 / 0 |
| Binding ranges | 6 |
| Parent / Child | 2,176 / 2,176 |
| DRAM cycles | 23,567 |
| Ramulator2 instances / Outstanding | 1 / 0 |

资格记录：

```text
/opt/gpu-atlas/qualification/p15e-address/attention-norm-range-rebase-qualified/qualification_record.json
```

在P15e验收时，只有以下重新捕获Artifact可以标记`request_cycle_ready=true`：

```text
configs/hetero/operator_artifacts/p15e/tinyllama_prefill_bs1_ctx16_attention_norm_sm86_shared_hbdram_range_rebase.json
```

P15c/P15d旧Artifact及LM Head仍为`identity_untranslated`、`global_pa_binding_ready=false`和`request_cycle_ready=false`，不得从Attention Norm结果外推。P15f随后独立完成了QKV Projection重新捕获与资格；该新增结论见`p15f_qkv_range_rebase.md`，不改变本记录的P15e历史证据。

## 3. LM Head远端双遍资格

远端两个独立Leg均完整执行4000 CTA并得到完全相同的周期、指令和外部内存统计：

| 项目 | 双跑一致结果 |
|---|---:|
| GPU cycles | 23,193,593 |
| Instructions | 476,608,000 |
| Parent / Child | 4,096,686 / 4,097,138 |
| Read / Write Parent | 4,096,208 / 478 |
| Logical / Internal bytes | 131,137,280 / 262,216,832 |
| DRAM cycles | 8,195,615 |
| Ramulator2 instances / Outstanding | 1 / 0 |

该运行验证了SM86连续Sector区间归一化修复。它仍使用identity-untranslated捕获地址，因此关闭的是计算—内存暂停/恢复和请求守恒门槛，不关闭Global PA门槛。

本地资格记录：

```text
/opt/gpu-atlas/qualification/p15d/coupled/accel-sim-rtx3070-lm-head-shared-hbdram-identity-sector-fix2-remote/qualification_record.json
```

12算子严格Catalog与汇总记录位于：

```text
configs/hetero/operator_artifacts/p15d/tinyllama_prefill_bs1_ctx16_twelve_gpu_coupled_catalog.json
configs/hetero/operator_artifacts/p15d/tinyllama_prefill_bs1_ctx16_twelve_gpu_coupled_summary.json
```

合计6,993,530个Parent和6,996,227个Child，全部完成；该合计仅用于守恒检查，不能作为一层Prefill延迟。

## 4. 后续门槛

1. 逐算子重新捕获其余10个GPU算子的Allocator范围与目标Tensor Backing Segment并完成Range-Rebase双跑；
2. 仅把`request_cycle_ready=true`的算子接入Prefill全局时间线；
3. 保持MMU/TLB/VA→PA时序和可配置XOR DRAM映射为独立后续工作；
4. 在校准前继续保持`performance_claim_allowed=false`。
