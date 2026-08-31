# P15f QKV Projection Allocator Segment Range-Rebase资格

日期：2026-08-30
版本：GPU-ATLAS-HeteroSim v0.21.0
范围：TinyLlama-1.1B、FP16、Layer 0、BS=1、Context=16、QKV Projection。

## 1. 捕获契约修正

最初仅使用语义Tensor精确区间与目标执行窗口中新建的CUDA Allocator区间。该契约在真实Tensor Core GEMM中以未映射地址`0xb03000060`失败；该地址位于Q Weight语义末端之后96 B，但仍处于其CUDA Caching Allocator Backing Segment内。

P15f增加`allocator_segment_ranges_for_addresses`：从内存快照中只选择包含目标输入和参数地址的既存Segment，再与目标窗口中新建区间合并。它覆盖Allocator Padding事务，同时拒绝将无关进程Segment并入地址空间；目标地址找不到Backing Segment时立即失败。

最终QKV Trace Manifest包含12个不重叠范围：8个语义Tensor范围与4个Allocator/Backing范围。Global PA物化结果为：

| 项目 | 结果 |
|---|---:|
| Binding ranges | 12 |
| Allocated bytes | 33,685,504 B |
| Binding table SHA-256 | `f07189731e87159a9541adbba9f1acb939de764ae86025401c569d96fb45f0f9` |

## 2. 远端确定性双遍

两个独立Accel-Sim 2.0进程分别固定到远端CPU 2和CPU 3，使用同一SM86 Trace、同一Range-Rebase Backend与唯一进程内Ramulator2。正式资格记录复用了两个完整Leg，并再次验证命令、Backend ID、Simulation Key与频率身份。

| 项目 | 双跑一致结果 |
|---|---:|
| GPU cycles | 2,168,865 |
| Instructions | 34,943,066 |
| Address translated / unmapped | 736,837 / 0 |
| Binding ranges | 12 |
| Read / Write Parent | 375,854 / 45 |
| Parent completed / durable | 375,899 / 375,899 |
| Child sent / completed | 375,944 / 375,944 |
| Logical / Internal bytes | 12,033,088 / 24,060,416 B |
| DRAM / Link / Gateway cycles | 766,383 / 766,383 / 766,383 |
| Ramulator2 instances / Outstanding | 1 / 0 |
| ATLAS Parent / Child | 0 / 0 |

远端资格记录SHA-256为`114125aca7d516f97e4aef2fcdc4802d58c18b3375a372d1bcda884e74916355`，本地验证副本位于：

```text
validation/p15f/qkv-range-rebase-qualified-v3/qualification_record.json
```

## 3. Artifact与双算子Catalog

QKV耦合Artifact明确记录`compute_memory_coupled=true`、`global_pa_binding_ready=true`和`request_cycle_ready=true`。它与P15e Attention Norm组成严格Range-Rebase Catalog：

```text
configs/hetero/operator_artifacts/p15f/tinyllama_prefill_bs1_ctx16_two_gpu_range_rebase_catalog.json
configs/hetero/operator_artifacts/p15f/tinyllama_prefill_bs1_ctx16_two_gpu_range_rebase_summary.json
```

两算子合计777,807次地址转换、378,075个GPU Parent和378,120个内部Child，全部完成且每次运行只有一个Ramulator2。

## 4. 声明边界

- 只有重新捕获的Attention Norm与QKV Projection可标记`request_cycle_ready=true`；其余10个真实GPU算子不得继承资格。
- 该结果验证算子级真实指令暂停/恢复、在线Global PA重绑定和共享3D-DRAM请求守恒，不包含ATLAS Logic Die竞争。
- 两个独立算子周期不能相加为Prefill延迟；尚未接入Prefill全局时间线。
- Trace仍为`replay_safe=false`，DRAM Mapper仍为`OneLevelInterleave(channel_lowest_bit=0)`；没有模拟MMU/TLB或可配置XOR Hash。
- 在全局调度接入和硬件校准前，`performance_claim_allowed=false`。
