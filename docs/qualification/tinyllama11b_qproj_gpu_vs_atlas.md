# TinyLlama‑1.1B Q投影：GPU与ATLAS形状匹配资格对比

日期：2026-08-28
范围：一个真实Checkpoint算子，不是完整层或端到端推理

## 工作负载合同

| 字段 | 固定值 |
|---|---|
| 模型 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Revision | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |
| 算子 | `model.layers.0.self_attn.q_proj` |
| 阶段 | BS=1、已有KV长度1024、单步Decode |
| 数据类型 | FP16 |
| GEMM | `M=1, K=2048, N=2048` |
| 矩阵工作量 | 4,194,304 MAC / 8,388,608 FLOP |
| 权重 | 8,388,608 B |
| 输入/输出 | 各4,096 B |

GPU Trace来自本机RTX 3070、驱动591.86、NVBit 1.8与Accel‑Sim 2.0。目标动态Kernel范围为4–5，包含一个CUTLASS WMMA GEMM和一个Split-K Reduction。ATLAS Artifact使用同一矩阵形状，把N维按16个Logic-Die Core均分为每核128列，固定Tile为`1×512×16`，每核32次迭代。

## 通过的结果

| 路径 | 周期 | 时钟 | 仿真延迟 | 关键内存条件 |
|---|---:|---:|---:|---|
| RTX 3070原生显存 | 36,324 GPU cycles | 1.132 GHz | 32.088339 µs | Accel-Sim内部GPU内存模型 |
| RTX 3070 → 外部Link → 3D‑DRAM | 1,498,113 GPU cycles | 1.132 GHz | 1,323.421378 µs | 外部12.8 GB/s；内部409.6 GB/s；唯一Ramulator2 |
| ATLAS内部3D‑DRAM | 24,613 ATLAS cycles | 1.0 GHz | 24.613 µs | 16核各25.6 GB/s，总计409.6 GB/s |

对应当前配置：

- 共享3D‑DRAM GPU路径是原生GPU路径延迟的`41.243062×`；
- 原生GPU延迟是ATLAS内部执行的`1.303715×`，即ATLAS在该配置下为`1.303715×`加速；
- 共享3D‑DRAM GPU路径延迟是ATLAS内部执行的`53.769202×`。

这些比值的主导原因不是DRAM内部峰值带宽，而是GPU路径上的12.8 GB/s外部接口。8.39 MiB权重响应必须穿过该接口，因此仅响应序列化下界已经接近1.31 ms。

## GPU共享3D‑DRAM流量证据

- GPU指令数：15,908,352；
- Ramulator2周期：529,368；
- 读请求/完成：262,272 / 262,272；写请求：0；退出时在途：0；
- Ramulator2实例：1；内存Partition：16；
- Logical Bytes：8,392,704 B；
- Internal Transaction Bytes：16,785,408 B；
- Request Wire Bytes：8,392,704 B；
- Response Payload/Wire Bytes：8,392,704 / 16,785,408 B；
- `rejected=37,338,484`表示Gateway/Ramulator背压导致的重试尝试，不是丢失请求。

GPU的32B Sector Parent在Logic Die侧形成完整64B内部事务，所以Internal Bytes为Logical Bytes的两倍。读请求只有在Ramulator2 Child完成、Parent汇聚且响应Link完成后才返回Accel-Sim。

## ATLAS内部统计

- 总周期：24,613；DRAM周期：24,442；矩阵周期：5,472；
- DRAM导致的Matrix Bubble：19,141 cycles；
- ATLAS报告操作数：8,396,800，其中8,388,608为矩阵FLOP，8,192为向量累加；
- DRAM请求字节：8,916,992 B；
- 能量：0.0002189966288 J。

ATLAS DRAM请求量高于唯一逻辑Tensor字节数，是因为固定Tile按照ATLAS Edge前端的访问规则重复读取输入Tile。该Artifact是形状正确、可复现的固定Tiling结果，不代表已经完成所有Tiling候选的最优搜索。

## 证据位置

```text
/opt/gpu-atlas/qualification/accel-sim-v2/
  rtx3070-tinyllama11b-qproj-decode-ctx1024/qualification_record.json
  rtx3070-tinyllama11b-qproj-decode-ctx1024-shared-hbdram-v2stats/qualification_record.json

/opt/gpu-atlas/qualification/atlas/
  tinyllama11b-qproj-decode-bs1-ctx1024-edge16/qualification_record.json
```

统一比较由`scripts/summarize_tinyllama_qproj_qualification.py`生成。脚本会先验证模型Revision、算子、Shape、Dtype、Batch和Context一致，再计算比值。

## 结论边界

1. 三个后端均已完成相同输入的双次确定性资格运行；这证明固定软件和配置可重复，不等同于RTX 3070或目标3D‑DRAM已经用实机校准。
2. GPU Trace的`replay_safe`仍为`false`；不能在没有额外验证时把同一Trace用于任意GPU/DRAM时序候选并声称安全。
3. 12.8 GB/s是当前直连外部PHY研究参数，不是ATLAS论文或某块真实板卡的实测接口结论。
4. ATLAS原生运行采用每核一个本地HBDRAM Partition的原生Edge组织；双发起方资格用例已经验证ATLAS内部端口可与GPU共享唯一Ramulator2，但完整`atlasim.Chip`任务调度尚未与Accel-Sim并发推进。
5. 本页只资格化一个Q投影，不能用本页结果直接推导整模型Latency或Token/s。P14另行完成了22层Prefill部署，但使用未校准分块周期契约和采样内存流量；完整Attention/MLP/KV的真实Accel-Sim Trace、完整ATLAS Artifact、Decode和多Batch性能资格仍未完成。
