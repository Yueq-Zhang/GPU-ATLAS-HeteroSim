# P15 TinyLlama Prefill首批算子资格记录

## 范围

- 模型：TinyLlama‑1.1B，固定Checkpoint Revision；
- 精度与Shape：FP16、Layer 0、BS=1、Context/Q/KV=16；
- 算子：RMSNorm、QKV Projection、RoPE、KV Append、Causal Attention；
- GPU：SM86 Trace、Accel-Sim 2.0、RTX 3070配置；
- ATLAS：QKV，16核，`M=16,K=2048,N=2560`，分块`8x128x8`；
- 共享内存：P15b选择性完整Value事务、唯一live Ramulator2。

本记录证明首批Artifact生产、严格绑定和完整流量Lowering已经闭环。P15c另外证明RMSNorm、QKV、RoPE和Causal Attention四类真实GPU指令状态都会等待分层Ramulator2响应；这些独立算子耦合尚无稳定Global PA重绑定，也未进入Prefill全局调度器。

## Artifact合同

入口Catalog：`configs/hetero/operator_artifacts/p15a/tinyllama_prefill_bs1_ctx16_first_batch_catalog.json`。

加载时验证Checkpoint、模型、Operator、Phase、Layer、Batch、Context、Q/KV长度、Dtype、Backend类型及全部引用文件SHA-256。地址语义为：

```text
Capture Address -> TensorID + Offset -> Runtime Global PA -> Candidate DRAM Tuple
```

P15a Catalog注册覆盖为5/5。P15c已让四类非空GPU Trace全部具备真实暂停/恢复证据；因Global PA绑定尚未完成，严格Request-cycle Trace Coverage仍为0/4。

## P15c四算子指令—内存耦合资格

四类Trace使用现有Accel-Sim外部内存桥双跑，每个算子的两个运行完全一致：

| 算子 | GPU cycles | Instructions | Parent / Child | Logical / Internal Bytes | DRAM cycles |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSNorm | 66,653 | 5,290,064 | 2,176 / 2,176 | 69,632 / 139,264 | 23,552 |
| QKV Projection | 2,170,258 | 34,943,066 | 376,212 / 376,238 | 12,041,280 / 24,079,232 | 766,875 |
| RoPE | 135,833 | 12,589,812 | 2,312 / 2,312 | 73,984 / 147,968 | 47,997 |
| Causal Attention | 43,500 | 11,962,112 | 2,560 / 2,560 | 81,920 / 163,840 | 15,371 |

每次运行都只有一个Ramulator2，全部Parent、Child和durable completion守恒，ATLAS Parent为0，退出在途为0。总资格记录位于：

```text
/opt/gpu-atlas/qualification/p15c/four-operator-final/qualification_record.json
```

四个严格P15c Artifact均记录`compute_memory_coupled=true`、`supports_stall_resume=true`、`global_pa_binding_ready=false`和`request_cycle_ready=false`。它们证明真实`mem_fetch`等待响应，但没有把Trace捕获地址重定位到运行期Global PA。

## 独立计算资格

| Artifact | Backend | 双跑结果 | 状态 |
| --- | --- | ---: | --- |
| Attention Norm | Accel-Sim | 58,736 cycles | 独立资格通过 |
| QKV Projection | Accel-Sim | 95,151 cycles | 独立资格通过 |
| RoPE | Accel-Sim | 127,094 cycles | 独立资格通过 |
| Causal Attention | Accel-Sim | 34,923 cycles | 独立资格通过 |
| KV Append | Runtime State | 0 NVBit Kernel | 设备状态拷贝，不伪造计算Trace |
| QKV Projection | ATLAS Chip | 150,932 cycles | 原生双跑通过 |

所有GPU Trace均保持`replay_safety_qualified=false`。独立计算周期不能与P15b内存时间相加。

## 严格绑定运行

配置：`configs/hetero/experiments/p15a_tinyllama_prefill_1layer_ctx16_gpu_operator_artifacts.json`。

一层20个任务中4个任务绑定形状锁定Accel-Sim Trace，16个任务使用分析回退；Trace Coverage为20%，Extrapolated Fraction为80%，因此`performance_claim_allowed=false`。

## 选择性完整流量运行

配置：`configs/hetero/experiments/p15b_tinyllama_prefill_1layer_ctx16_first_batch_full_traffic.json`。

| 指标 | 数值 |
| --- | ---: |
| Full-Traffic任务 | 5 |
| Sampled任务 | 15 |
| Full-Traffic Parent | 175,936 |
| Sampled Parent | 234 |
| Parent总数/完成数 | 176,170 / 176,170 |
| Read / Write Parent | 170,201 / 5,969 |
| Logical / Internal Bytes | 11,274,786 / 11,274,880 |
| Ramulator2实例 | 1 |
| 退出在途请求 | 0 |
| DRAM / Global GPU cycles | 541,940 / 1,625,820 |

两个独立输出目录的`memory_statistics.json`、`prefill_artifact_coverage.json`、`metrics.json`和`execution_graph.json` SHA-256逐项一致。最终资格记录位于WSL：

```text
/opt/gpu-atlas/qualification/p15b/first-batch-final/qualification_record.json
```

## 复现命令

在WSL工程根目录执行：

```bash
.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/p15b_tinyllama_prefill_1layer_ctx16_first_batch_full_traffic.json \
  --runs-root /opt/gpu-atlas/qualification/p15b/full-traffic-rerun

.venv/bin/python scripts/summarize_p15_first_batch.py \
  --catalog configs/hetero/operator_artifacts/p15a/tinyllama_prefill_bs1_ctx16_first_batch_catalog.json \
  --gpu-qualification-root /opt/gpu-atlas/qualification/p15a \
  --atlas-qualification /opt/gpu-atlas/qualification/p15a/atlas-tinyllama-qkv-prefill-bs1-ctx16-edge16/qualification_record.json \
  --prefill-run1 /opt/gpu-atlas/qualification/p15b/full-traffic-run1/p15b_tinyllama_prefill_1layer_ctx16_first_batch_full_traffic/c3d5d886e69a7d53509ee17616aa9b999b414574e22eae3f3d806cdb220c077f \
  --prefill-run2 /opt/gpu-atlas/qualification/p15b/full-traffic-run2/p15b_tinyllama_prefill_1layer_ctx16_first_batch_full_traffic/c3d5d886e69a7d53509ee17616aa9b999b414574e22eae3f3d806cdb220c077f \
  --output /opt/gpu-atlas/qualification/p15b/first-batch-final/qualification_record.json
```

## 声明边界与下一门槛

P15c已经为RMSNorm、QKV、RoPE和Causal Attention建立真实Accel-Sim指令状态与分层Ramulator2响应之间的逐请求暂停/恢复；P15b仍是独立的“分块计算+完整Value流量”路径。两条路径尚未在Prefill全局调度器中合并，且P15c地址仍为`identity_untranslated`。Context=1024、全20任务完整流量、多Batch、端到端Decode和性能校准均未通过P15资格。

在地址需求保持冻结时，下一项可执行工作是生成其余一层算子的形状锁定Artifact，并对其中非空GPU Trace执行同等级`identity_untranslated`耦合资格。最终门槛仍是把Trace地址按Manifest转换为TensorID+offset，绑定到Prefill分配器给出的Global PA，并把耦合Backend嵌入全局时间线；只有地址覆盖、请求/完成守恒、唯一Owner、零在途和双跑确定性同时通过后，才能将Artifact标记为Request-cycle Ready。
