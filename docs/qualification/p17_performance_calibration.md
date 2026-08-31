# P17 性能校准状态

日期：2026-08-31
状态：校准基础设施与14类RTX 3070原生算子测量完成；同拓扑/同Trace配对未完成，整机性能资格未通过

## 已实现

P17把请求周期资格与性能资格彻底分开。`hetero-performance-calibration/v1`记录以下六个独立时间所有者：

| 组件 | 当前状态 | 已有证据 | 仍缺少的资格条件 |
|---|---|---|---|
| GPU Operator | `measured_unvalidated` | 14类固定Shape RTX 3070原生测量；14类外接3D-DRAM Accel-Sim周期Catalog | 同拓扑Native-VRAM Accel-Sim双跑以及Native执行/捕获Trace二进制身份核验 |
| Copy Engine | `measured_unvalidated` | RTX 3070本地显存32 KiB D2D Copy | 与外接3D-DRAM KV Copy语义一致的延迟/带宽点 |
| Runtime | `measured_unvalidated` | 空Kernel + Device Synchronize主机计时 | 实际框架Request Start/Finish和Kernel Launch分解 |
| 外部Link | `specified_only` | 12.8 GB/s、10 ns配置合同 | 独立带宽扫描和往返延迟参考 |
| Logic-Die Gateway | `specified_only` | 400 MHz、4请求/周期、256 Parent配置 | Split/Join延迟和饱和吞吐参考 |
| 3D-DRAM | `specified_only` | 16通道、400 MT/s、409.6 GB/s Ramulator2配置 | Row-hit/Row-miss和持续混合流量参考 |

只有组件状态为`validated`、证据来自硬件测量或可信独立模拟器、每个参考点绑定到哈希已核验的源Artifact、必需指标齐全、误差不超过阈值且覆盖精确Shape时，该组件才通过。全局性能门禁还要求所有纳入设备性能的任务都为`performance_eligible=true`。Request Start/Finish可以明确排除，但不能用排除主机事件的方式掩盖GPU、Copy或内存未校准。

## 本机测量

首批CUDA参考基准使用NVIDIA GeForce RTX 3070、驱动591.86和CUDA Runtime 11.6。协议为50次Warmup、500次逐次测量。记录位于`validation/p17/native_rtx3070/native_measurements.json`，SHA-256为`e2b3d0eb098801fafc77e924cc60e5634db999e7aab0c81ded1de4344c3d7a0d`。

| 测量项 | 中位值 |
|---|---:|
| 空Kernel CUDA Event | 8.960 µs |
| 空Kernel + Device Synchronize主机时延 | 8.600 µs |
| Token Embedding，Context=16、Hidden=2048 | 9.216 µs |
| Residual Add，Context=16、Hidden=2048 | 9.216 µs |
| 32 KiB D2D Copy | 9.216 µs |
| 由单次32 KiB Copy换算的有效带宽 | 7.111 GB/s |

该小尺寸Copy结果主要受固定启动/调度开销影响，不能解释为RTX 3070持续显存峰值。所有Kernel和Copy都使用GPU本地显存，与P16的外部Link + Logic Die + 3D-DRAM拓扑不同，因此禁止直接计算P16误差或调节12.8/409.6 GB/s参数来匹配这些数值。

第二批测量固定TinyLlama revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`、Layer 0、FP16、BS=1和Context=16。Embedding/Residual导入与P16相同源码的定形CUDA基准；其余12类使用NVBit捕获脚本的同一高层Target，在Windows PyTorch 2.9.0 + CUDA 13.0上各执行50次Warmup和500次CUDA Event测量。结果位于`validation/p17/gpu_operator_pairing/native_rtx3070_local_vram.json`，SHA-256为`72535255fb2703b272c7cc3d2949cf2dd95ef428def1d1fd4beac72248fb6bb1`。

| 算子 | Native中位延迟 |
|---|---:|
| Token Embedding | 9.216 µs |
| Residual Add | 9.216 µs |
| Attention Norm | 272.576 µs |
| QKV Projection | 147.264 µs |
| RoPE | 655.392 µs |
| Causal Attention | 59.392 µs |
| Output Projection | 84.800 µs |
| MLP Norm | 116.864 µs |
| Gate/Up Projection | 189.312 µs |
| SiLU Multiply | 26.624 µs |
| Down Projection | 89.088 µs |
| Final Norm | 111.616 µs |
| LM Head | 367.616 µs |
| Sampling | 17.408 µs |

这些数值是当前Windows/WDDM原生执行分布的观测值，不是已经校准的算子真值；不同算子的p10/p90范围显示出明显调度波动。测量Manifest同时哈希模型Config、2.20 GB Safetensors、能力Catalog、Benchmark源码、算子构建器、首批CUDA基准和三份结果文件。高层Target相同并不能证明Native执行与WSL/NVBit捕获的指令Trace二进制一致，所以该身份也必须在正式配对前单独核验。

## 14算子配对审计

`simulator_external_shared3d.json`从14份已请求周期资格的Artifact提取确定性周期、核心频率、Trace Manifest和内容哈希；其内存拓扑是`external_shared_3ddram`。配对器逐算子检查：

- 14/14覆盖、Implementation和精确Shape Key；
- Operator Artifact SHA-256；
- Trace Manifest/二进制身份；
- Native与Simulator内存拓扑；
- 允许的相对误差。

当前`pairing_audit.json`的SHA-256为`c1790667cb463b2aa5c8c6f955eee9b4f12f735f62222e1d500db28460ae1365`，结果为`paired_operator_count=0/14`、`topology_match=false`、`performance_claim_allowed=false`。主要阻断是`gpu_local_vram != external_shared_3ddram`以及14类Native新执行均未绑定到已捕获Trace身份。这个结果表示审计正确阻止了不等价数据配对，不表示14类原生测量或既有请求周期资格运行失败。

## 自动审计结果

P17复用了P16最终双遍运行，确认两遍Simulation Key均为`d5066ff9081332bd31ae5699f4f572736cc7f188ae9f4272cf89a4af0a1d6e3a`，makespan均为`35,450,346,739,701 fs`，请求指标一致。四份配置哈希和14算子本机测量哈希全部匹配。

机器记录`validation/p17/p16_layer0_ctx16/performance_calibration_audit.json`返回：

- `deterministic_runs=true`；
- `required_component_count=6`；
- `qualified_component_count=0`；
- `status=audit_complete_blocked`；
- `performance_claim_allowed=false`。

这表示审计过程成功、性能资格未通过，并非仿真失败。

## 后续闭环

1. 在完整Trace主机执行`run_p17_native_vram_accelsim_qualification.sh`，逐算子生成Native-VRAM Accel-Sim双遍资格记录；
2. 在与Trace捕获一致的软件环境计时并记录Kernel Launch序列/Trace身份，避免把高层Target名称等同于二进制相同；
3. 通过配对器核对14类同拓扑数据并生成误差点，失败算子不得从其他算子外推；
4. 分离框架Launch、同步、KV Copy固定时延和持续带宽，避免用一个常数拟合总时间；
5. 为12.8 GB/s外部Link和Logic-Die Gateway建立独立Payload/Queue扫描；
6. 用独立3D-DRAM硬件数据或可信参考模拟器核对Row-hit、Row-miss和持续混合带宽；
7. 六项全部通过后重新生成算子Artifact、运行完整双遍时间线，并由全局门禁自动决定是否允许性能声明。
