# TinyLlama Q投影完整ATLAS Chip共享内存资格

日期：2026-08-28
阶段：P9a
状态：通过

## 固定输入

- 模型：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- 算子：layer-0 `q_proj`
- 阶段：Decode，BS=1，已有KV长度1024
- 数据类型与形状：FP16，`M=1,K=2048,N=2048`
- ATLAS：16 Core，1 GHz
- 共享内存：16通道、512-bit/通道、400 MT/s、64B事务，内部峰值409.6 GB/s
- 全局推进：GPU 1.2 GHz，Link/Gateway/DRAM 400 MHz

## 实现范围

ATLAS补丁向`Chip -> CoreArray -> Core -> DRAMWrapper`注入`IExternalDramService`。外部模式不创建ATLAS原生Ramulator2；`pre_simulate()`捕获每个真实迭代的`ComponentInput`，运行时提交到内部Hybrid-Bond端口，片上组件继续推进，只有全部外部DRAM Parent完成后才允许迭代结束。

共享完成队列按Initiator出队，避免Accel-Sim把ATLAS Payload解释为`mem_fetch`或反向误取。16个Core的局部地址按每核1 MiB区域投影成互不重叠的Global PA。

## 结果

| 用例 | ATLAS完成时刻（1.2 GHz全局GPU周期） | ATLAS Chip周期 | GPU Parent | ATLAS Parent |
|---|---:|---:|---:|---:|
| ATLAS-only | 76,418 | 63,681 | 0 | 139,456 |
| ATLAS + 确定性GPU流量 | 81,329 | 67,774 | 4,096 | 139,456 |

ATLAS事务字节为8,925,184 B。并发用例中GPU与ATLAS全部完成，Ramulator2实例数为1，退出时`outstanding=0`。ATLAS完成时刻增加4,911个全局GPU周期，证明完整Chip观察到共享DRAM竞争。

旧ATLAS原生组件记录8,916,992 B，其中每核8次32B输出写按逻辑字节统计；实时路径记录所有对齐64B Parent，二者相差8,192 B。事务守恒以实时Parent/Child和Bridge字节为准。

## 复现

```bash
bash scripts/build_accel_sim_ramulator2.sh
bash scripts/build_atlas_full_chip_runtime.sh
bash scripts/qualify_full_chip_scheduler_memory_path.sh \
  /opt/gpu-atlas/qualification/full-chip-scheduler-memory-path-20260828-p9a-final
```

资格记录：

```text
/opt/gpu-atlas/qualification/full-chip-scheduler-memory-path-20260828-p9a-final/qualification_record.json
```

## 声明边界

本记录的`atlas_full_chip_scheduler=true`，但`accel_sim_compute_backend=false`。GPU侧是确定性Parent生成器，因此本结果不能表述为Accel-Sim真实Kernel与完整ATLAS Chip已经并发，也不能外推为完整层、端到端模型或多Batch吞吐。
