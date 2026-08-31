# P15g 两个真实Accel-Sim算子的Prefill全局时间线资格

## 目标

P15g验证两个已经通过Range-Rebase双遍资格的真实GPU算子能否进入同一个一层TinyLlama Prefill全局时间线。它验证的是依赖、资源、Global PA、请求完成和版本提交之间的因果关系，不把两个独立算子周期相加，也不声明端到端性能。

固定实验为TinyLlama‑1.1B、FP16、BS=1、Context=16、一层Prefill。Attention Norm与QKV Projection使用真实Accel-Sim 2.0 + 外部Ramulator2 Backend，其余18个任务使用分析回退。配置文件为：

```text
configs/hetero/experiments/p15g_tinyllama_prefill_1layer_ctx16_two_request_cycle_gpu.json
```

## 运行与汇总

```bash
PYTHONPATH=. .venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/p15g_tinyllama_prefill_1layer_ctx16_two_request_cycle_gpu.json \
  --runs-root /opt/gpu-atlas/qualification/p15g-timeline

PYTHONPATH=. .venv/bin/python scripts/summarize_p15g_prefill_timeline.py \
  --run /opt/gpu-atlas/qualification/p15g-timeline/p15g_tinyllama_prefill_1layer_ctx16_two_request_cycle_gpu/be9f927b7c3846c80f19206f9c0eb966aaf9f14084901971788d8f9d345c22e6 \
  --output /opt/gpu-atlas/qualification/p15g-timeline/p15g_qualification.json
```

本地留存记录为`validation/p15g/p15g_qualification.json`，完整归档为`validation/p15g/p15g-qualified-result.tar.gz`，归档SHA-256为`8491670a5147759bc9c3721a990c2b99101c2de4db831d39f0b047958ededec5`。

## 因果门禁结果

| 门禁 | 验证结果 |
| --- | --- |
| DAG依赖 | Attention Norm在58,833,038,873 fs完成，QKV Projection恰在同一时刻启动 |
| 资源占用 | 两个真实Backend都占用`gpu0`，运行区间严格不重叠 |
| Global PA | Attention输出与QKV输入是同一Value，Global PA base均为351,485,952 |
| 请求完成 | 两个Backend地址漏配均为0；Parent、Child、durable completion守恒；退出在途为0 |
| 版本提交 | Attention输出版本1只在Backend完成时提交；QKV启动时验证该版本；两个真实Backend的输出均在完成时提交 |

全局地址表共38个区间，其中10个为算子私有Workspace；总分配374,995,968 B，低于4 GiB容量，区间不重叠。图Value使用全局分配器地址，Workspace位于图分配之后，两个算子不会自行生成彼此冲突的局部“物理地址”。

## 实际请求统计

| 算子 | GPU cycles | 指令 | 地址转换 | Parent | Child | DRAM/Link/Gateway cycles | 退出在途 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention Norm | 66,599 | 5,290,064 | 40,758 | 2,176 | 2,176 | 23,533 | 0 |
| QKV Projection | 2,173,639 | 34,943,066 | 736,827 | 376,690 | 376,734 | 768,070 | 0 |

Attention为2,176个读Parent、0个写Parent。QKV为376,646个读Parent、44个写Parent；逻辑字节为12,058,304 B，内部Child字节为24,110,976 B。每个算子进程各自创建且只创建一个Ramulator2；因为两个算子在`gpu0`上串行，它们不存在同一时刻双重拥有DRAM时间的问题。

## 声明边界

- 该运行证明两个真实Accel-Sim算子已经进入同一个Prefill调度与版本生命周期，而不是两个独立周期的算术求和。
- 运行时Global PA来自全局分配器；捕获地址只用于Range→Offset查找。当前仍不模拟MMU、TLB、页表或VA→PA延迟。
- 其余18个任务仍为分析回退，故`performance_eligible=false`；不得据此发布TinyLlama端到端Latency、Token/s或加速比。
- P15g运行基于工作树Revision `8c58dbb1a2209829198451cbfd8fb6c95c16c53c-dirty`。最终版本提交必须在完整测试通过后单独完成，不能倒推该资格已对应一个干净Git提交。
- 其余10个GPU算子只有在P15h重新捕获并独立双遍通过后才能替换分析回退；旧`identity_untranslated` Artifact不能提升为Ready。
