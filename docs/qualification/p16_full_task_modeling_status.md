# P16 全任务请求周期建模资格

日期：2026-08-31
状态：固定Shape的功能与请求周期资格通过；性能资格关闭

## 固定范围

- TinyLlama-1.1B，checkpoint revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`；
- FP16，Layer 0，BS=1，Context/Q/KV=16；Final Norm、LM Head和Sampling实际`q_len=1`；
- GPU为RTX 3070 / SM86 Trace；GPU外部请求经过分层Link进入共享3D-DRAM；
- 配置为`configs/hetero/experiments/p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu.json`。

## 已通过的闭环

P16现在精确覆盖19种算子、20个任务实例，没有隐式分析回退：

- 14种GPU算子、15个实例使用`request_cycle_ready=true`的Range-Rebase Accel-Sim Trace；
- `kv_allocate`、`kv_append`、`kv_release`生成显式Global PA请求，经外部Link进入各自唯一的live Ramulator2；
- `request_start`和`request_finish`是无内存请求的主机控制边界事件，参与因果时间线，但明确排除在设备性能边界之外；
- Token ID真实Kernel使用64-bit索引，而物化图仍保留通用元素宽度。为避免改变已资格算子的低地址分配，P16从4 GiB空间顶部保留一个`external_input_widened_shadow`，显式绑定128 B输入，不把它解释为VA翻译。

两遍完整时间线使用相同Simulation Key `d5066ff9081332bd31ae5699f4f572736cc7f188ae9f4272cf89a4af0a1d6e3a`，核心结果完全一致：

| 项目 | 每遍结果 |
|---|---:|
| 任务/算子类型 | 20 / 19 |
| GPU Trace实例 | 15 |
| GPU cycles / instructions | 40,125,102 / 692,663,026 |
| GPU Parent / Child | 7,003,497 / 7,006,370 |
| GPU地址重绑定 | 804,530,289，漏配0 |
| KV运行时Parent | 517，全部完成 |
| KV运行时逻辑字节 | 33,088 B |
| 版本检查/完成时提交 | 31 / 18 |
| Global PA范围/Workspace | 87 / 58 |
| Global PA占用 | 450,296,512 B / 4 GiB |
| 因果makespan | 35,450,346,739,701 fs |

KV运行时请求进一步分解为：

| 任务 | 读请求 | 写请求 | 总请求 | 执行周期 | 公式合同周期 | 逻辑字节 | 退出在途 |
|---|---:|---:|---:|---:|---:|---:|---:|
| KV Allocate | 1 | 2 | 3 | 141 | 19 | 192 B | 0 |
| KV Append | 256 | 256 | 512 | 4,435 | 520 | 32,768 B | 0 |
| KV Release | 1 | 1 | 2 | 113 | 18 | 128 B | 0 |

执行周期按共享GPU外部端口的1.132 GHz时钟统计，包含live链路/DRAM等待和固定控制周期；公式合同周期仍按未校准实现参数单独保存，不再冒充实际执行周期。每个KV任务均满足Ramulator2实例数1、Parent ID唯一、accepted=observed=completed=durable、Child sent=completed、ATLAS请求0和退出在途0。全部任务满足依赖先完成、`gpu0`资源不重叠、Global PA不重叠、输入版本启动时核验以及输出只在Backend完成时提交。

机器可读资格记录为`validation/p16/p16_full_task_qualification.json`，自动检查入口为：

```bash
python scripts/summarize_p16_full_task_timeline.py \
  validation/p16/leg1/p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu/<simulation-key> \
  validation/p16/leg2/p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu/<simulation-key> \
  --output validation/p16/p16_full_task_qualification.json
```

## 声明边界

P16关闭的是固定一层、固定Shape的全任务实现和请求周期因果门槛，不是性能校准门槛：

- 运行时控制、Copy Engine、Link、GPU和3D-DRAM参数尚未与目标硬件测量值统一校准；
- Request Start/Finish的确定性时长不能解释为CUDA Runtime或框架实测开销；
- 当前35.450 ms只用于比较两遍时间线是否一致，不得发布为TinyLlama端到端Latency、Token/s或加速比；
- 模型、revision、dtype、Batch、Context、Q/KV长度或任一模型维度变化时，现有资格默认失效，必须重新捕获/建模并执行完整资格流程。
