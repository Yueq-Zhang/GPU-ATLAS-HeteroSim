# P31框架驱动GPU与ATLAS Artifact资格

## GPU路径

同一个P30请求只在真实Layer-0 Decode区间开启NVBit，生成40个Kernel的完整指令/访存Trace。身份同时记录：

- 捕获设备为远端RTX 4090/SM89；
- Trace头中的实际SASS Binary Version为SM86；
- Accel-Sim回放目标为SM86；
- 四个CUDA分配器Range经Range-Rebase打包到4 GiB Global PA；
- Trace不因捕获主机是SM89而被改写为SM89身份。

单遍Accel-Sim回放完成40个Kernel，共15,859,267个GPU周期、371,846,682条指令；2,756,823个Parent全部完成且durable，2,758,278个Child全部发出并完成，575,486,980次地址转换中漏配为0，退出时在途为0，Ramulator2实例数为1。该回放用于验证地址与请求闭环，不是双遍请求周期资格，因此源Artifact保持`request_cycle_ready=false`。

## ATLAS路径

同一P30模型规格的QKV Projection从Tensor IR Lowering为16个Logic Core的Stage/Tile计划，并显式绑定Activation、Weight和Output的Global PA。两遍均生成338,080条完整请求，其中读337,920条、写160条；未压缩Trace、确定性gzip Trace、Artifact Key和Tensor地址逐字节一致。

当前资格覆盖Tensor IR、Tile、Core、Global PA和完整内存请求生成。Bank/Channel由后续Ramulator2地址映射器决定；ATLAS Trace尚未完成Ramulator2周期回放，不能作为ATLAS性能结果。

## 远端复现与边界

```bash
export P30_P31_PHASE=all
bash scripts/run_p30_p31_remote.sh
```

机器记录位于`validation/p31/framework_artifacts/qualification_record.json`；同目录还归档GPU源Artifact、Trace Manifest、在线地址绑定、单遍回放统计及两份ATLAS摘要。它必须保持：

- `framework_selected_sass_captured=true`；
- `trace_address_to_global_pa_audited=true`；
- `atlas_tensor_ir_to_full_memory_trace=true`；
- `gpu_double_replay_qualified=false`；
- `end_to_end_framework_timeline_qualified=false`；
- `performance_claim_allowed=false`。

P30起所有测试、SASS获取和Trace生成只允许在远端RTX 4090执行；本地仅编辑代码和封存轻量机器记录，不保存大型原始Trace。
