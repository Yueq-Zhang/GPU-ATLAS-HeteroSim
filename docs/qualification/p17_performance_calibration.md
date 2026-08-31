# P17 性能校准状态

日期：2026-08-31
状态：校准基础设施与第一批RTX 3070原生测量完成；整机性能资格未通过

## 已实现

P17把请求周期资格与性能资格彻底分开。`hetero-performance-calibration/v1`记录以下六个独立时间所有者：

| 组件 | 当前状态 | 已有证据 | 仍缺少的资格条件 |
|---|---|---|---|
| GPU Kernel | `measured_unvalidated` | RTX 3070原生Embedding/Residual测量；Accel-Sim配置 | 其余Kernel以及同拓扑原生显存Accel-Sim误差对照 |
| Copy Engine | `measured_unvalidated` | RTX 3070本地显存32 KiB D2D Copy | 与外接3D-DRAM KV Copy语义一致的延迟/带宽点 |
| Runtime | `measured_unvalidated` | 空Kernel + Device Synchronize主机计时 | 实际框架Request Start/Finish和Kernel Launch分解 |
| 外部Link | `specified_only` | 12.8 GB/s、10 ns配置合同 | 独立带宽扫描和往返延迟参考 |
| Logic-Die Gateway | `specified_only` | 400 MHz、4请求/周期、256 Parent配置 | Split/Join延迟和饱和吞吐参考 |
| 3D-DRAM | `specified_only` | 16通道、400 MT/s、409.6 GB/s Ramulator2配置 | Row-hit/Row-miss和持续混合流量参考 |

只有组件状态为`validated`、证据来自硬件测量或可信独立模拟器、每个参考点绑定到哈希已核验的源Artifact、必需指标齐全、误差不超过阈值且覆盖精确Shape时，该组件才通过。全局性能门禁还要求所有纳入设备性能的任务都为`performance_eligible=true`。Request Start/Finish可以明确排除，但不能用排除主机事件的方式掩盖GPU、Copy或内存未校准。

## 本机测量

测量设备为NVIDIA GeForce RTX 3070，驱动591.86，CUDA Runtime 11.6。协议为50次Warmup、500次逐次测量。最终记录位于`validation/p17/native_rtx3070/native_measurements.json`，测量内容SHA-256为`e2b3d0eb098801fafc77e924cc60e5634db999e7aab0c81ded1de4344c3d7a0d`。

| 测量项 | 中位值 |
|---|---:|
| 空Kernel CUDA Event | 8.960 µs |
| 空Kernel + Device Synchronize主机时延 | 8.600 µs |
| Token Embedding，Context=16、Hidden=2048 | 9.216 µs |
| Residual Add，Context=16、Hidden=2048 | 9.216 µs |
| 32 KiB D2D Copy | 9.216 µs |
| 由单次32 KiB Copy换算的有效带宽 | 7.111 GB/s |

该小尺寸Copy结果主要受固定启动/调度开销影响，不能解释为RTX 3070持续显存峰值。所有Kernel和Copy都使用GPU本地显存，与P16的外部Link + Logic Die + 3D-DRAM拓扑不同，因此禁止直接计算P16误差或调节12.8/409.6 GB/s参数来匹配这些数值。

## 自动审计结果

P17复用了P16最终双遍运行，确认两遍Simulation Key均为`d5066ff9081332bd31ae5699f4f572736cc7f188ae9f4272cf89a4af0a1d6e3a`，makespan均为`35,450,346,739,701 fs`，请求指标一致。四份配置哈希和本机测量哈希全部匹配。

机器记录`validation/p17/p16_layer0_ctx16/performance_calibration_audit.json`返回：

- `deterministic_runs=true`；
- `required_component_count=6`；
- `qualified_component_count=0`；
- `status=audit_complete_blocked`；
- `performance_claim_allowed=false`。

这表示审计过程成功、性能资格未通过，并非仿真失败。

## 后续闭环

1. 用现有14类Trace在RTX 3070原生显存Accel-Sim配置下逐算子双跑，与相同CUDA实现的原生测量形成误差点；
2. 分离框架Launch、同步、KV Copy固定时延和持续带宽，避免用一个常数拟合总时间；
3. 为12.8 GB/s外部Link和Logic-Die Gateway建立独立Payload/Queue扫描；
4. 用独立3D-DRAM硬件数据或可信参考模拟器核对Row-hit、Row-miss和持续混合带宽；
5. 六项全部通过后重新生成算子Artifact、运行完整双遍时间线，并由全局门禁自动决定是否允许性能声明。
