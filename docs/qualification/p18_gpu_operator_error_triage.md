# P18 GPU算子误差分解基线

日期：2026-09-01
固定合同：TinyLlama-1.1B、Layer 0、FP16、BS=1、Context=16、RTX 3070/SM86、`gpu_local_vram`
输入：P17本机50+500 CUDA Event测量与Native-VRAM Accel-Sim双遍资格

## 误差定义

有符号误差定义为：

```text
(Accel-Sim延迟 - Native延迟) / Native延迟
```

负值表示Accel-Sim给出的时间短于本机Native时间，正值表示Accel-Sim时间更长。门禁使用绝对相对误差，当前阈值为15%。本表只描述观测差异，不把误差方向直接解释为某个硬件参数或软件开销的因果结论。

## 当前结果

| 算子 | Native/µs | Accel-Sim/µs | 有符号误差 | 绝对误差 | Kernel数 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| Attention Norm | 100.992 | 51.887 | -48.62% | 48.62% | 8 | 超阈值 |
| Causal Attention | 21.504 | 30.851 | +43.46% | 43.46% | 3 | 超阈值 |
| Down Projection | 96.256 | 91.808 | -4.62% | 4.62% | 1 | 通过 |
| Final Norm | 374.784 | 44.147 | -88.22% | 88.22% | 8 | 超阈值 |
| Gate/Up Projection | 206.848 | 162.030 | -21.67% | 21.67% | 3 | 接近阈值 |
| LM Head | 351.232 | 337.898 | -3.80% | 3.80% | 1 | 通过 |
| MLP Norm | 168.224 | 51.887 | -69.16% | 69.16% | 8 | 超阈值 |
| Output Projection | 36.864 | 42.234 | +14.57% | 14.57% | 1 | 通过，接近边界 |
| QKV Projection | 77.440 | 84.056 | +8.54% | 8.54% | 6 | 通过 |
| Residual Add | 22.752 | 5.466 | -75.97% | 75.97% | 1 | 超阈值 |
| RoPE | 297.056 | 112.274 | -62.20% | 62.20% | 19 | 超阈值 |
| Sampling | 60.416 | 17.673 | -70.75% | 70.75% | 1 | 超阈值 |
| SiLU Multiply | 107.520 | 11.799 | -89.03% | 89.03% | 2 | 超阈值 |
| Token Embedding | 10.464 | 6.402 | -38.82% | 38.82% | 1 | 超阈值 |

汇总：4/14通过，10/14阻断；平均绝对误差45.67%，中位绝对误差46.04%。11类为模拟时间低于Native，3类为模拟时间高于Native。身份、Artifact和内存拓扑均已匹配，因此当前误差分组不再包含身份或拓扑阻断。

## 校准优先级

1. 严重误差（>50%）：SiLU Multiply、Final Norm、Residual Add、Sampling、MLP Norm、RoPE；
2. 中等误差（25%–50%）：Attention Norm、Causal Attention、Token Embedding；
3. 接近阈值（15%–25%）：Gate/Up Projection；
4. 已通过但需稳定性复核：Down Projection、LM Head、Output Projection、QKV Projection。

## 下一轮实验

- 对4个已通过算子重复多轮50+500测量，并记录GPU实际时钟、温度与功耗状态，确认误差稳定性；
- 对多Kernel算子分离Kernel执行时间与Kernel之间的运行时空隙，检查Accel-Sim总Kernel周期是否漏掉框架/Launch间隔；
- 对单Kernel小算子单独核对固定Launch成本、核心频率、L2/DRAM事务和指令类别；
- 对Causal Attention、Output Projection和QKV Projection检查模拟时间高于Native是否与微架构配置或融合Kernel行为有关；
- 所有参数调整必须按算子和证据来源记录，禁止以一个全局比例缩放14类结果。

机器可读记录位于`validation/p18/gpu_operator_error_triage.json`，其输入P17 Audit和Simulator Catalog均通过SHA-256封存。当前`performance_claim_allowed=false`保持不变。
