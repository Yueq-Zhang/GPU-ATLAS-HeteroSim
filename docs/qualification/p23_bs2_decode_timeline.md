# P23固定BS=2单层Decode统一时间线资格

## 结论

P23已经完成并封存固定TinyLlama-1.1B、checkpoint `fe8a4ea1ffedaf415f4da2f062534de366a451e6`、Layer 0、FP16、BS=2、Context=16、`q_len=1`、KV=17、单步Decode的功能资格。

SASS编译/加载、CUDA执行、SASS读取和NVBit捕获全部发生在远端RTX 4090。捕获设备为SM89，但每个Library Kernel的实际Binary版本仍按Trace记录为SM80、SM86或显式允许的混合Ampere序列；没有把实际SASS版本改写为SM89。重放目标为SM86共享3D-DRAM配置。

## 时间线范围

- 14类GPU算子，15个真实Trace实例；Residual Add执行两次；
- 1个按R0/R1独立KV子区间写入的KV Append；
- Request Start/Finish与KV Allocate/Release四个生命周期任务；
- 合计20个任务、19条依赖，全部串行占用`gpu0`且区间不重叠；
- 两个成员请求均把KV长度从16提交到17，并在Sampling完成后结束和释放。

## 双遍机器结果

两个隔离Leg完全一致，每遍记录：

| 项目 | 数值 |
|---|---:|
| makespan | 34,843,748,683,165 fs |
| 聚合GPU周期 | 39,442,237 |
| GPU指令 | 371,403,956 |
| GPU Parent | 6,862,620 |
| GPU Child | 6,863,691 |
| 运行时内存Parent | 69 |
| Global PA分配 | 97 |
| Trace / 运行时绑定 | 15 / 5 |
| R0/R1 K/V成员子区间 | 4 |

所有GPU任务均满足唯一Ramulator2、地址零漏配、Parent/Child/durable守恒、零ATLAS请求和零在途。KV Append写入均位于对应成员的K/V子区间；Attention只在Append完成且新版本可见后启动。双遍完整签名相同。

## 仓库封存证据

- 捕获Catalog：`validation/p23/remote_capture_catalog.json`；
- 最终Ready Catalog：`validation/p23/ready_catalog.json`；
- 逐算子资格：`validation/p23/operator_qualifications/`；
- 源与耦合Artifact Manifest：`configs/hetero/operator_artifacts/p23_sm89_decode_bs2{,_coupled}/`；
- 双遍轻量结果：`validation/p23/timeline/leg{1,2}/`；
- 最终资格记录：`validation/p23/timeline/qualification_record.json`。

原始NVBit Trace、Accel-Sim输出、`backend_runs`和运行日志总量约452 MiB，继续保存在远端证据环境，没有提交Git。Catalog以SHA-256绑定仓库内Manifest与资格记录，并保留原始远端文件的路径、大小和哈希。

离线检查命令：

```bash
python3 scripts/validate_p23_sealed_catalog.py
```

预期输出为14类算子、20个任务、双遍签名一致、时间线已资格且性能声明关闭。

## 声明边界

P23证明固定Shape的真实指令/请求时间线能够与KV Append、Global PA、依赖、资源和版本生命周期组成确定性功能闭环。它不证明：

- RTX 4090或RTX 3070的真实性能精度；
- 跨Kernel重放进程之间存在持久DRAM/Cache状态；
- KV=18–20、不同Batch、Ragged Shape、其他层或其他模型可复用；
- Trace可安全跨任意DRAM组织或地址映射候选重放；
- GPU与ATLAS Logic Die竞争、BookSim2 NoC周期或端到端Token/s已经校准。

因此`timeline_integration_ready=true`，但`performance_claim_allowed=false`和逐算子`performance_eligible=false`必须保持不变。
