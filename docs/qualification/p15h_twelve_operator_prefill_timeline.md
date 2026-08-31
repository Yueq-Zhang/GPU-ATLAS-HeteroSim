# P15h 十二算子 Prefill 全局时间线资格记录

## 结论

P15h已将TinyLlama-1.1B、BS=1、Context=16、一层Prefill图中的12个真实GPU算子全部接入同一条`operator_event`全局时间线。资格验证状态为`passed`，但`performance_eligible=false`、`performance_claim_allowed=false`：本结果证明地址、请求、调度与版本因果闭环，不代表已校准的端到端性能。

## 配置与证据

- 实验配置：`configs/hetero/experiments/p15h_tinyllama_prefill_1layer_ctx16_twelve_request_cycle_gpu.json`
- Ready Catalog：`configs/hetero/operator_artifacts/p15h/tinyllama_prefill_bs1_ctx16_twelve_gpu_range_rebase_catalog.json`
- 本地归档：`validation/p15h/twelve_operator_timeline/`
- 资格报告：`validation/p15h/twelve_operator_timeline/p15h_qualification.json`
- 远端运行键：`e680717b02cce7eabdd76926147da8eb5d760c156e91f8360ccf36301dbe0413`
- 元数据归档SHA-256：`4b1debdabad6349bcd852703cf8144f7eb47f294e041a7f4dd2fc6eb3fa7b99e`

## 算子结果

| 算子 | GPU cycles | 指令 | GPU Parent | GPU Child | 地址转换 |
|---|---:|---:|---:|---:|---:|
| Attention Norm | 66,654 | 5,290,064 | 2,176 | 2,176 | 40,970 |
| QKV Projection | 2,176,858 | 34,943,066 | 376,535 | 376,572 | 736,837 |
| RoPE | 135,809 | 12,589,812 | 2,312 | 2,312 | 30,764 |
| Causal Attention | 43,473 | 11,962,112 | 2,560 | 2,560 | 16,448 |
| Output Projection | 1,749,088 | 20,140,032 | 307,840 | 307,840 | 609,152 |
| MLP Norm | 66,652 | 5,290,064 | 2,176 | 2,176 | 40,976 |
| Gate/Up Projection | 8,316,717 | 86,599,740 | 1,463,410 | 1,465,792 | 2,004,992 |
| SiLU Multiply | 75,477 | 3,020,750 | 11,264 | 11,264 | 33,459 |
| Down Projection | 4,161,563 | 33,361,920 | 727,488 | 727,488 | 933,888 |
| Final Norm | 49,549 | 318,253 | 257 | 257 | 2,617 |
| LM Head | 23,199,917 | 476,608,000 | 4,097,155 | 4,097,609 | 800,060,777 |
| Sampling | 19,116 | 704,205 | 2,000 | 2,000 | 2,001 |
| **合计** | **40,060,873** | — | **6,995,173** | **6,998,046** | **804,512,881** |

所有算子均满足：地址漏配为0、ATLAS Parent/Child/Completed为0、Parent完成与durable完成守恒、Child发送与完成守恒、唯一Ramulator2实例、退出在途为0。

## 全局因果检查

- 12个真实GPU算子均在全部DAG依赖完成后启动。
- 所有GPU及分析/运行时任务在`gpu0`上的时间区间互不重叠。
- Global PA包含84个互不重叠的区间，其中56个为算子私有Workspace。
- 12条请求周期绑定包含38条语义Tensor映射；每条映射的`global_pa_base - value_offset_bytes`均等于对应图Value的分配基址。
- 图共提交18个输出版本；12个真实GPU算子的提交均发生在Backend完成时，输入版本与启动时图Value版本完全一致。
- 最终模拟makespan为35,390.378 µs。

## 复现

```bash
PYTHONPATH=build:. .venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/p15h_tinyllama_prefill_1layer_ctx16_twelve_request_cycle_gpu.json \
  --runs-root /opt/gpu-atlas/qualification/p15h-twelve-operator-timeline

PYTHONPATH=. .venv/bin/python scripts/summarize_p15h_prefill_timeline.py \
  /opt/gpu-atlas/qualification/p15h-twelve-operator-timeline/p15h_tinyllama_prefill_1layer_ctx16_twelve_request_cycle_gpu/e680717b02cce7eabdd76926147da8eb5d760c156e91f8360ccf36301dbe0413
```

## 声明边界

20个图任务中，12个GPU算子使用真实Accel-Sim请求周期Backend；Request Start/Finish、KV Allocate/Append/Release、Token Embedding和两次Residual Add仍为分析或运行时模型。GPU、3D-DRAM与互联参数尚未完成硬件校准。因此不得把35.390 ms解释为实测TinyLlama延迟，也不得据此发布Token/s、加速比或GPU与ATLAS性能对比。
