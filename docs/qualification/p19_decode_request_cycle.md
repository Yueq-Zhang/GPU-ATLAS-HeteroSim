# P19 单Token Decode请求周期功能资格

## 固定范围

- 模型：TinyLlama‑1.1B，revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`；
- 精度与Shape：FP16、BS=1、初始Context/KV=16、`q_len=1`、最终KV=17；
- 系统：Model 3 GPU-only共享3D-DRAM，Logic Die计算关闭；
- 规模：1层20任务，以及22层272任务；
- 内存：有界代表采样、外部Link/Gateway、唯一live Ramulator2、durable写完成。

## 已通过的不变量

两种规模均独立运行两遍，并由`hetero-p19-decode-request-cycle-qualification/v1`逐项审计：

- DAG依赖完成后消费者才启动，全部`gpu0`资源区间不重叠；
- 每层KV Append先于Causal Attention完成，Attention读取已提交的K/V版本1；
- 初始KV有效范围为每个Tensor 8,192 B，追加Token为512 B，所有请求均位于对应Global PA分配内；
- Parent、Child与durable completion守恒；唯一Ramulator2；ATLAS Parent为0；退出时零在途；
- 两遍的任务数、算子计数、GPU/DRAM周期、请求计数、最终版本和流式Trace内容哈希一致。

| 规模 | 任务 | GPU Parent | GPU周期（未校准） | Ramulator2周期 | 因果makespan（未校准） |
|---|---:|---:|---:|---:|---:|
| 1层 | 20 | 190 | 42,057 | 14,019 | 35,047,500,000 fs |
| 22层 | 272 | 3,382 | 566,052 | 188,684 | 471,710,000,000 fs |

## 声明边界

GPU任务使用`request_tiled_cycle_contract`，并未使用Decode精确Shape的Accel-Sim指令Trace。因此表中周期和makespan只用于证明调度、地址、请求和版本因果，不能作为RTX 3070实测预测、Token/s、加速比或论文性能结果。机器记录强制：

```text
compute_fidelity=tiled_cycle_contract_unqualified
accel_sim_instruction_trace_coverage=0.0
performance_claim_allowed=false
```

汇总记录：`validation/p19/qualification_summary.json`。完整复现入口：`scripts/run_p19_decode_qualification.sh`。
