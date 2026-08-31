# P15d TinyLlama Prefill 扩展算子与13算子完整流量资格

日期：2026-08-30
版本：GPU-ATLAS-HeteroSim v0.20.0
范围：TinyLlama-1.1B、checkpoint `fe8a4ea1ffedaf415f4da2f062534de366a451e6`、FP16、Layer 0、BS=1、Context=16。

## 1. 本阶段关闭的范围

P15d在P15a的Attention Norm、QKV、RoPE、KV Append和Causal Attention基础上，新增以下8类形状锁定RTX 3070 SM86 Artifact：

- Output Projection；
- MLP Norm；
- Gate/Up Projection；
- SiLU Multiply；
- Down Projection；
- Final Norm；
- LM Head；
- Sampling。

全部Trace由真实TinyLlama checkpoint算子在RTX 3070上执行并使用NVBit 1.8捕获。Final Norm、LM Head和Sampling只处理Prefill最后位置，因此兼容键固定`context_length=16`、`q_len=1`、`kv_length=16`；其余新增层内算子固定`q_len=16`。

源Catalog为：

```text
configs/hetero/operator_artifacts/p15d/tinyllama_prefill_bs1_ctx16_thirteen_gpu_source_catalog.json
```

它严格注册13类完整流量算子，其中KV Append仍是实测零Kernel的`runtime_state` Artifact。源Artifact注册完成不等于请求周期就绪；所有源Artifact继续保持`request_cycle_ready=false`和`performance_eligible=false`。

## 2. 一层Prefill的13算子完整Value流量

实验配置：

```text
configs/hetero/experiments/p15d_tinyllama_prefill_1layer_ctx16_thirteen_full_traffic.json
```

20个一层Prefill任务中，13个Artifact匹配任务展开全部Value事务；Request Start、Token Embedding、两次Residual Add、Request Finish、KV Allocate和KV Release共7个任务保留有界采样。两次独立运行的8个核心产物逐字节SHA-256一致，包括2,335,336,970 B的`request_cycle_trace.json`。

| 项目 | 双跑一致结果 |
|---|---:|
| Expected / Covered tasks | 20 / 20 |
| Full / Sampled tasks | 13 / 7 |
| Parent | 3,462,738 |
| Full / Sampled Parent | 3,462,673 / 65 |
| Read / Write Parent | 3,444,241 / 18,497 |
| Child | 3,462,738 |
| Logical / Internal bytes | 221,615,138 / 221,615,232 |
| DRAM cycles | 10,401,594 |
| GPU global cycles | 31,204,782 |
| Makespan | 26,003,985,000,000 fs |
| Ramulator2 instances / Outstanding | 1 / 0 |

最终资格记录：

```text
/opt/gpu-atlas/qualification/p15d/thirteen-full-traffic-final/qualification_record.json
```

该运行中的计算仍来自未校准的Prefill分块周期合同；它只资格化13算子的完整Value流量、全局调度与唯一内存时序Owner，不是Accel-Sim Kernel与同一Prefill时间线的合并结果。

## 3. 新增算子的独立指令—共享内存耦合资格

每个已资格算子均执行两次完整Accel-Sim Trace。L2/Memory Partition的真实`mem_fetch`通过外部Link和Logic-Die Gateway进入该次运行唯一的Ramulator2；全部内部Child完成并经过响应Link后，原GPU请求才进入ReturnQ。

| 算子 | GPU cycles | Instructions | GPU Parent / Child | DRAM cycles | 状态 |
|---|---:|---:|---:|---:|---|
| Output Projection | 1,749,102 | 20,140,032 | 307,780 / 307,780 | 618,057 | 双跑通过 |
| MLP Norm | 66,653 | 5,290,064 | 2,176 / 2,176 | 23,552 | 双跑通过 |
| SiLU Multiply | 75,483 | 3,020,750 | 11,264 / 11,264 | 26,672 | 双跑通过 |
| Down Projection | 4,153,383 | 33,361,920 | 727,488 / 727,488 | 1,467,626 | 双跑通过 |
| Final Norm | 49,549 | 318,253 | 257 / 257 | 17,508 | 双跑通过 |
| Sampling | 19,192 | 704,205 | 2,000 / 2,000 | 6,781 | 双跑通过 |
| Gate/Up Projection | 8,309,172 | 86,599,740 | 1,462,619 / 1,464,838 | 2,936,103 | 双跑通过 |
| LM Head | 23,193,593 | 476,608,000 | 4,096,686 / 4,097,138 | 8,195,615 | 双跑通过 |

表中每一行都是独立的双跑资格，不能把各行周期相加为Prefill延迟。每次已通过运行均满足：一个Ramulator2、全部Parent/Child/durable completion守恒、ATLAS Parent/Child为0、退出`outstanding=0`。

12个已通过GPU算子的严格Catalog和汇总记录分别为：

```text
configs/hetero/operator_artifacts/p15d/tinyllama_prefill_bs1_ctx16_twelve_gpu_coupled_catalog.json
configs/hetero/operator_artifacts/p15d/tinyllama_prefill_bs1_ctx16_twelve_gpu_coupled_summary.json
```

它们合计6,993,530个Parent和6,996,227个内部Child，全部完成。LM Head远端资格记录已复制到本地`/opt/gpu-atlas/qualification/p15d/coupled/accel-sim-rtx3070-lm-head-shared-hbdram-identity-sector-fix2-remote/qualification_record.json`。该合计只用于请求守恒，不是端到端时间。

## 4. 声明边界

- 在线GPU桥仍将`data->get_addr()`按`identity_untranslated`语义传递，没有MMU/TLB/VA到PA时序；
- 尚未执行Capture Range到`TensorID+offset`再到运行期Global PA的稳定重绑定；
- DRAM Mapper仍是`OneLevelInterleave(channel_lowest_bit=0)`，未实现可配置XOR Hash；
- 独立Accel-Sim进程尚未嵌入一层Prefill全局调度器；
- 固定Trace仍为`replay_safe=false`，不能把本次时序资格外推到任意DRAM候选；
- `request_cycle_ready=false`和`performance_claim_allowed=false`必须保持。

## 5. 复现

完整Value流量双跑及验收：

```bash
PYTHONPATH=. .venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/p15d_tinyllama_prefill_1layer_ctx16_thirteen_full_traffic.json \
  --runs-root /opt/gpu-atlas/qualification/p15d/full-traffic-run1

PYTHONPATH=. .venv/bin/python scripts/summarize_p15d_thirteen_full_traffic.py \
  --run1 /opt/gpu-atlas/qualification/p15d/full-traffic-run1/p15d_tinyllama_prefill_1layer_ctx16_thirteen_full_traffic/c00d2784ef0dcbed220c33476f94ce18e43786a1c046fce8162a48cb288c7b10 \
  --run2 /opt/gpu-atlas/qualification/p15d/full-traffic-run2/p15d_tinyllama_prefill_1layer_ctx16_thirteen_full_traffic/c00d2784ef0dcbed220c33476f94ce18e43786a1c046fce8162a48cb288c7b10 \
  --output /opt/gpu-atlas/qualification/p15d/thirteen-full-traffic-final/qualification_record.json
```

单算子耦合资格以Output Projection为例：

```bash
PYTHONPATH=. .venv/bin/python -m frontend.hetero.cli qualify-gpu \
  --resume-completed-runs \
  --backend-config configs/hetero/backends/gpu_accelsim_rtx3070_ramulator2_hbdram_edge_16ch.json \
  --trace-manifest configs/hetero/operator_artifacts/p15d/tinyllama_prefill_bs1_ctx16_output_projection_sm86_trace.json \
  --output /opt/gpu-atlas/qualification/p15d/coupled/accel-sim-rtx3070-output-projection-shared-hbdram-identity
```

P15e已用逐条`jsonl.gz`流和紧凑摘要替换单体请求JSON。相同逻辑工作量的双跑压缩流为94,859,940 B，峰值RSS约524.6 MiB，仍保持逐产物确定性哈希；详情见[P15e资格记录](p15e_streaming_and_range_rebase.md)。

`--resume-completed-runs`用于大型双跑的故障恢复：某一遍只有在成功写出命令和统计、且当前命令、Backend ID、Simulation Key与频率完全匹配时才可复用。该机制不会把中断Perf文件解释为成功，也不会放宽两遍统计逐项一致的资格标准。
