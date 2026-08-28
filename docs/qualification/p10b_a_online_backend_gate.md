# P10b-A 在线Backend启动门禁资格记录

日期：2026-08-28
软件版本：`0.11.0`
设计合同：`1.10`

## 资格目标

验证正常`operator_event`执行不再在构建执行图时提前调用Backend，而是由在线事件执行器在以下条件全部满足后启动：

1. 模拟时间达到任务Release；
2. 全部前驱Device Task和Route已经完成；
3. 目标设备资源可用；
4. 每个输入的最新版本已经位于目标设备；
5. 每个Device Task恰好触发一次Backend Dispatch。

Route的目标版本只在Route完成时可见，Device Task的输出版本只在任务完成时可见。旧版本或缺失版本必须在Backend启动前失败。

## 固定输入

- Experiment：`configs/hetero/experiments/step2_model1_operator_event_probe.json`
- GPU：Accel-Sim v2.0.0官方QV100 Trace，绑定标记为`surrogate_plumbing_probe`
- ATLAS：ATLAS test-chip Artifact，绑定标记为`surrogate_plumbing_probe`
- 其余节点：分析型回退Backend
- 运行模式：正常严格单放置`operator_event`

上述GPU与ATLAS Artifact用于验证适配器启动门禁，不是同一模型算子的性能资格数据。

## 复现命令

```bash
.venv/bin/python -m pytest \
  tests/hetero/test_online_operator_runtime.py \
  tests/hetero/test_operator_event.py -q

.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/step2_model1_operator_event_probe.json \
  --runs-root /opt/gpu-atlas/qualification/p10b-a-online-dispatch-run1

.venv/bin/python -m frontend.hetero.cli run \
  --config configs/hetero/experiments/step2_model1_operator_event_probe.json \
  --runs-root /opt/gpu-atlas/qualification/p10b-a-online-dispatch-run2
```

## 输出位置

```text
/opt/gpu-atlas/qualification/p10b-a-online-dispatch-run1/step2_model1_operator_event_probe/80a6088fc4a6a530cab86c6957a33ff79bedc21746505750cad94889bde4f1bb
/opt/gpu-atlas/qualification/p10b-a-online-dispatch-run2/step2_model1_operator_event_probe/80a6088fc4a6a530cab86c6957a33ff79bedc21746505750cad94889bde4f1bb
```

## 验证结果

| 项目 | 结果 |
|---|---:|
| 逻辑节点 | 85 |
| Device Task | 85 |
| Backend Dispatch | 85 |
| Route | 12 |
| 成功输入版本检查 | 117 |
| Makespan | 65,594,898,888 fs |

两个真实适配器任务的门禁结果：

| 任务 | Backend | Backend周期 | 启动时刻 | 最大依赖完成时刻 |
|---|---|---:|---:|---:|
| `R0.prefill.s0.l0.attention.projection` | `gpu.accel_sim.sm70_qv100.v2_0_0` | 14,731 | 8,192,002 fs | 8,192,002 fs |
| `R0.decode.s1.l0.attention.projection` | `atlas.atlasim.micro26.test_chip_16ch` | 48,446 | 14,248,172,886 fs | 14,248,172,886 fs |

单元黄金用例还验证了GPU任务完成后，Route在5–15 fs内传递`x@v1`，ATLAS消费者只能在15 fs启动；构造旧版本消费者时，运行时在调用Backend前拒绝执行。

## 确定性

两个独立输出根的关键文件逐字节一致：

| 文件 | SHA256 |
|---|---|
| `online_dispatch.json` | `70BD9D692BA7C38C2ABF25FD3EA90E518EFCCFAF0275867888348DE160B471EF` |
| `metrics.json` | `9B1B29F96EA487402B616FD31CF69CF737F4EE2171357533D67B76F0F82C85CC` |

## 声明边界

P10b-A资格化的是不同真实**总时长适配器**受统一DAG、资源、Route和版本状态门禁后启动。它不表示GPU和ATLAS的内部请求在该用例中同时逐周期推进，也不表示Copy/Fence已向共享Ramulator2提交实际事务。一个实时Ramulator2拥有GPU、ATLAS与Route请求级时序的联合执行属于P10b-B。
