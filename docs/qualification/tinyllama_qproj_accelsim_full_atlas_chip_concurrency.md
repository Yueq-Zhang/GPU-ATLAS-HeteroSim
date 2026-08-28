# TinyLlama Q投影：Accel-Sim与完整ATLAS Chip并发资格

日期：2026-08-28
阶段：P9b
资格记录：`/opt/gpu-atlas/qualification/accel-sim-v2/rtx3070-tinyllama-qproj-full-atlas-chip-shared-memory-p9b/qualification_record.json`

## 固定输入

- 模型：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Revision：`fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- 算子：`model.layers.0.self_attn.q_proj`
- Phase：单步Decode，BS=1，初始Context=1024
- Dtype/Shape：FP16，`M=1,K=2048,N=2048`
- GPU：Accel-Sim 2.0，SM86 RTX 3070配置，NVBit 1.8 kernels 4–5 Trace
- Logic Die：完整16核`atlasim.Chip`，N维列分块Artifact
- 内存：16通道HBDRAM，64B事务，内部峰值409.6 GB/s
- GPU外部Link：请求/响应Payload各12.8 GB/s

## 实现路径

Accel-Sim保留SM、Cache和NoC时序；其LLC Miss经外部Link和Logic-Die Gateway进入共享Ramulator2。完整ATLAS Chip通过注入的非阻塞`IExternalDramService`提交真实Core/Task/Iteration内存请求，内部请求不经过GPU外部Link，但与GPU Child争用同一组Channel/Bank和控制器时序。

统一推进器以GPU Core周期为基准，按独立频率相位推进ATLAS、Link、Gateway和DRAM。GPU与ATLAS使用Initiator专属完成出队；ATLAS运行时持有共享Partition句柄，且ATLAS原生Ramulator2不会实例化。

## 双次确定性结果

| 指标 | Run 1 | Run 2 |
|---|---:|---:|
| GPU cycles | 1,541,401 | 1,541,401 |
| GPU instructions | 15,908,352 | 15,908,352 |
| GPU Parent | 262,272 | 262,272 |
| ATLAS Chip cycles | 141,255 | 141,255 |
| ATLAS完成时GPU cycle | 159,901 | 159,901 |
| ATLAS Parent | 139,456 | 139,456 |
| Ramulator2 Parent完成 | 401,728 | 401,728 |
| Ramulator2实例 | 1 | 1 |
| 退出时在途 | 0 | 0 |

ATLAS事务字节为8,925,184 B。两种Initiator均产生非零请求；ATLAS完成时GPU运行尚未结束，证明存在执行窗口重叠。GPU、ATLAS和Ramulator2的提交/完成计数全部守恒。

## 运行方法

```bash
bash scripts/build_accel_sim_ramulator2.sh
bash scripts/build_atlas_full_chip_runtime.sh
bash scripts/qualify_accel_sim_full_chip_concurrency.sh \
  /opt/gpu-atlas/qualification/accel-sim-v2/rtx3070-tinyllama-qproj-full-atlas-chip-shared-memory-p9b
```

## 声明边界

该资格用例故意让相同Q投影在GPU和ATLAS两侧同时执行，以验证两个完整计算后端、完成隔离与共享DRAM竞争。它不表示一个推理图把同一算子执行两次，也不能作为合法放置策略、完整层、端到端LLM延迟、多Batch吞吐或实机加速比。Trace的`replay_safe`仍为false。
