# P17 性能校准状态

日期：2026-09-01
状态：14类RTX 3070原生测量、Native/Trace执行程序身份和Native-VRAM Accel-Sim双遍全部完成；4/14通过数值误差门禁，整机性能资格未通过

## 已实现

P17把请求周期资格与性能资格彻底分开。`hetero-performance-calibration/v1`记录以下六个独立时间所有者：

| 组件 | 当前状态 | 已有证据 | 仍缺少的资格条件 |
|---|---|---|---|
| GPU Operator | `measured_unvalidated` | 14类固定Shape RTX 3070原生测量、双侧执行程序身份和Native-VRAM Accel-Sim确定性双遍；4/14误差配对通过 | 10个超阈值算子解释与修正；随后重新验证全任务性能资格 |
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

第二批测量固定TinyLlama revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`、Layer 0、FP16、BS=1和Context=16。全部14类都在本机WSL RTX 3070上执行50次Warmup和500次CUDA Event测量；Embedding/Residual使用封存SM86 ELF，其余12类使用封存的Python 3.10、PyTorch 2.3.0+cu121、Transformers 4.57.6和相同工作负载源码。合并结果位于`validation/p17/gpu_operator_pairing/native_rtx3070_local_vram.json`，SHA-256为`efc919b618d9c5d6875787628bc0d45e2675700d3de60a76dfaaba1c7d2970e3`。

| 算子 | Native中位延迟 |
|---|---:|
| Token Embedding | 10.464 µs |
| Residual Add | 22.752 µs |
| Attention Norm | 100.992 µs |
| QKV Projection | 77.440 µs |
| RoPE | 297.056 µs |
| Causal Attention | 21.504 µs |
| Output Projection | 36.864 µs |
| MLP Norm | 168.224 µs |
| Gate/Up Projection | 206.848 µs |
| SiLU Multiply | 107.520 µs |
| Down Projection | 96.256 µs |
| Final Norm | 374.784 µs |
| LM Head | 351.232 µs |
| Sampling | 60.416 µs |

这些数值全部来自本机WSL RTX 3070的同一套50+500 CUDA Event协议，但仍只是观测值，不是已经完成整机校准的算子真值。测量Manifest同时哈希模型Config、2.20 GB Safetensors、能力Catalog、Benchmark源码、算子构建器、执行身份Catalog和结果Catalog。Native计时与NVBit捕获都由同一封存执行程序启动；process范围捕获用于规避当前NVBit profiler-range路径生成零指令Trace的问题。

## 同一Binary身份门禁

`hetero-gpu-execution-identity/v1`把“同一Binary”实现为可核验合同，而不是依赖文件名或算子名称推断。每条记录必须同时保存：

- 可执行文件SHA-256；
- 由Operator、Implementation、精确Shape、dtype和Launch Header规范化得到的Launch Contract SHA-256；
- 排除运行期地址后的有序Kernel Sequence SHA-256；
- Target SM和Kernel Launch数量；
- `native_measurement_observed`与`trace_capture_observed`两个独立布尔标志。

只有两侧身份字段完全相等，并且Native计时侧与Trace捕获侧都明确声明实际观测，配对器才会把`execution_identity_match`置为真。Trace Manifest相同、高层Target相同、Shape相同或把身份记录事后附加到旧测量Catalog，都不能替代双侧实际观测。

当前`validation/p17/sm86_sealed_recapture/execution_identity_catalog.json`采用固定UTF-8/LF字节序列，SHA-256为`bec378caf3fe6e5d38f65ef8c8f891c467f77508bb4b093b020412fdc22c0a82`。14/14记录同时具有Native和Trace实际观测，共覆盖63个Kernel Launch。Embedding/Residual封存同一个SM86 ELF；其余12类把Python解释器、`torch._C`扩展、工作负载源码以及Python/PyTorch/Transformers/CUDA版本规范化为执行程序SHA，再与Launch Contract和Trace Kernel Sequence共同构成身份。这里的“同一Binary”应按机器合同理解为同一封存执行程序，而不是声称PyTorch算子只有一个ELF文件。

## 14算子配对审计

`simulator_external_shared3d.json`从14份已请求周期资格的Artifact提取确定性周期、核心频率、Trace Manifest和内容哈希；其内存拓扑是`external_shared_3ddram`。配对器逐算子检查：

- 14/14覆盖、Implementation和精确Shape Key；
- Operator Artifact SHA-256；
- Trace Manifest/二进制身份；
- Native与Simulator内存拓扑；
- 允许的相对误差。

新增`simulator_native_vram.json`保存14类`gpu_local_vram`双遍周期，当前SHA-256为`cb4f7a6196a717590ee8195259c2b0b5578bbe6f1359dd07810e97c73dae729e`。每类均满足两遍周期/指令相同、GPU本地DRAM由Accel-Sim唯一计时、没有外部Ramulator2且采用总时长统计。全部正式Native/Trace身份均来自本机RTX 3070；远端RTX 4090不参与这组正式身份闭环。

当前`native_vram_pairing_audit.json`的SHA-256为`0db6777f57d94e7c6ef9a74ac2445235b79ce6be74f651831c60518892bfc79d`，结果为`paired_operator_count=4/14`、`topology_match=true`、`performance_claim_allowed=false`。Down Projection、Output Projection、QKV Projection和LM Head的相对误差分别约4.62%、14.57%、8.54%和3.80%，通过15%门禁。其余10条阻断项全部是`relative_error_exceeds_tolerance`；不再存在身份、Artifact或拓扑阻断。

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

1. 对10个超出15%阈值的算子分解框架Launch、同步、核心频率、Cache/DRAM配置和Kernel选择差异，禁止用单一缩放系数修正；
2. 对4个已配对算子执行频率、空Launch和本地显存敏感性实验，确认其误差稳定性而非偶然落入阈值；
3. 分离框架Launch、同步、KV Copy固定时延和持续带宽，避免用一个常数拟合总时间；
4. 为12.8 GB/s外部Link和Logic-Die Gateway建立独立Payload/Queue扫描，并用独立3D-DRAM硬件数据或可信参考模拟器核对Row-hit、Row-miss和持续混合带宽；
5. 六项全部通过后重新生成算子Artifact、运行完整双遍时间线，并由全局门禁自动决定是否允许性能声明。
