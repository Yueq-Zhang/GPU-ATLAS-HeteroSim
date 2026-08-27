# GPU + ATLAS 异构端到端仿真实现规范

## 文档状态

| 字段 | 内容 |
| --- | --- |
| 状态 | 已冻结的实现基线 |
| 版本 | 1.4 |
| 日期 | 2026-08-27 |
| 适用工程 | ATLAS-MICRO-2026 |
| 当前基线提交 | b2787399408e32d327c820daee96d4e6610f551a |
| 主要目标 | GPU 与 3D-DRAM Compute Die/ATLAS 的端到端 LLM 联合仿真 |

本文档是后续实现、代码评审、配置设计和实验解释的主规范。后续工作如果与本文档冲突，应先修改本文档并记录原因，再修改代码。不能在实现中静默改变本文档已经冻结的拓扑语义、地址语义、计时所有权或端到端 Token 语义。

本文档中的“部署到 3D-DRAM”是简写，准确含义是：

    算子部署到与 3D-DRAM 集成的 ATLAS Compute Die/Core；
    3D-DRAM 本身是存储、带宽和时序资源，不是通用计算设备。

冻结范围包括系统语义、接口、配置主键、状态机、确定性调度规则、实现顺序和验收标准。具体类的私有实现、文件内辅助函数和不会改变外部行为的优化不属于冻结范围。

---

## 1. 目标与范围

### 1.1 第一版必须支持

第一版研究可用系统必须支持：

1. 同一份 LLM 逻辑计算图在 GPU、ATLAS 或二者混合放置。
2. 完整的 Prefill、首 Token、Decode 和请求结束生命周期。
3. Static Ragged Batch、Paged KV Cache 和 Continuous Batching。
4. 四种 GPU/3D-DRAM 物理系统 Profile。
5. GPU Roofline 快速模型和 Accel-Sim 周期级后端。
6. ATLAS 原生算子描述和 atlasim 后端。
7. PCIe、CXL、共享 3D-DRAM 等不同数据路径。
8. 统一的逻辑 Tensor、Memory Space、Global PA 和 DRAM 地址译码。
9. 统一绝对时间上的多设备、链路和内存调度。
10. 可复现的配置、Trace、地址清单、结果和精度标签。

### 1.2 第一版不强制支持

以下内容保留接口，但不作为第一版完成条件：

- 真实数值 Logits 和 Token 生成；
- 动态 EOS 和 Stop Sequence 的功能执行；
- 完整 GPU MMU、TLB 和 Page Table Walk；
- 完整 CXL 协议、PHY 和所有一致性消息；
- 单个 GEMM/Attention 同时跨 GPU 和 ATLAS 拆分；
- 多 GPU、Pipeline Parallel 和完整分布式 Serving；
- 推测解码；
- 输入相关的真实 MoE Expert 路由；
- RTL 或门级精度。

第一版进行架构 DSE 时，默认使用固定请求轨迹和固定生成长度，确保不同拓扑执行完全相同的逻辑工作量。

---

## 2. 当前 ATLAS 基线与扩展边界

当前 ATLAS 主路径为：

    Auto / ATLang / TileLang
        -> tiling 与 placement
        -> operator_description.yaml
        -> data_placement.yaml
        -> atlasim.Chip
        -> CoreArray / Core
        -> Matrix / Vector / Buffer
        -> DRAMWrapper / Ramulator2
        -> 可选 NoC / BookSim2
        -> 统计结果

当前内置 LLM 推理路径主要面向 Decode：

- frontend/model_parser.py 中当前推理形状将 Attention input_length 设置为 1；
- frontend/auto/system.py 中 Cloud 路径使用 DecodeAttention；
- 当前 full-model 结果通常将单层结果乘以 num_layers；
- 现有高层图没有完整显式建模 Embedding、Norm、RoPE、Residual、LM Head、Sampling 和完整 Prefill。

因此，新系统采用“外层异构运行时 + 现有 ATLAS Adapter”的方式扩展：

- 保留当前 ATLAS 路径作为兼容和回归基线；
- 不直接改变现有 Auto/ATLang/TileLang 的语义；
- 新增设备无关的模型图、拓扑相关执行图、全局运行时和 Backend Adapter；
- 真正端到端模式逐层实例化，不能依赖单层延迟简单乘层数；
- 仍可保留 layer_multiplier 快速估计模式，但结果必须标记为 extrapolated。

---

## 3. 规范性术语

### 3.1 计算设备

    gpu0
    atlas0.compute
    host0.compute

GPU 和 ATLAS Compute Die 是计算发起方。

### 3.2 存储空间

    gpu0.hbm
    host0.dram3d
    atlas0.dram3d
    shared0.dram3d
    cxl0.dram3d

计算设备和存储空间必须使用不同对象描述。算子放置不能隐式决定 Tensor 存储位置。

### 3.3 Full-model 与 Full-request

- Full-model：一个 Token 或一个阶段穿过全部 Transformer Layer。
- Full-request：Request Arrival、Prefill、首 Token、后续 Decode、终止和 KV 释放的完整过程。

本文档中的“端到端”默认指 Full-request。

### 3.4 内部统一单位

所有内部字段必须规范化为：

    容量：byte
    带宽：byte/s
    时间：fs
    频率：Hz
    能量：J

配置加载器负责将 GB/s、GiB/s、Gbps、ns、MHz 等单位转换为上述单位。报告层可以再换算成人类可读单位。

---

## 4. 总体架构

    Hugging Face Config + Workload Manifest
                        |
                        v
              CanonicalModelSpec
                        |
                        v
                 ModelGraphIR
       Prefill / Decode / KV / Control Flow
                        |
                        v
       Policy + Compiler Planning + Artifact Catalog
                        |
                        v
             C++ Global Event Runtime (fs)
       Request/KV State + Epoch Scheduler
                        |
                        v
       Graph Instantiator + Placement Evaluator
                        |
                        v
    Runtime Memory Planner + TopologyLowerer(profile)
                        |
                        v
             ExecutionGraphInstance
      Device / Transfer / Migration / Synchronization
              /          |          \
             v           v           v
       GPU Backend  ATLAS Backend  Link/Memory Services
       Roofline      atlasim        PCIe/CXL/Ramulator2
       Accel-Sim
              \          |          /
                        v
        TTFT / TPOT / E2E / Traffic / DRAM Stats

各层责任如下：

- CanonicalModelSpec：模型结构、dtype、量化和逻辑融合边界。
- ModelGraphIR：描述“做什么”，不包含 PCIe/CXL 或物理地址。
- Placement Policy：描述“在哪个计算设备执行”；动态实例由C++ Evaluator确定。
- Compiler Planning：给出布局、Tile、Scratch、Alignment 和融合约束。
- Memory Planner：Python只生成不含地址的`StaticAllocationPlan`约束；C++ RuntimeMemoryPlanner分配全部静态/动态PhysicalAddress并维护Residency、Buffer Binding、Allocation Epoch和Buffer Version。
- Topology Lowering：C++按当前已提交状态和物理Profile决定Copy、Migration、Remote Access或Fence。
- ExecutionGraphIR：模板描述可执行结构；Runtime为每个Epoch生成ExecutionGraphInstance。
- Backend：计算或者协同产生执行时间。
- Link/Memory Service：唯一负责相应链路或物理内存的时序。
- Runtime：在统一绝对时间上调度全部资源。

任何层都不能静默包含其他层已经拥有的时间。

Model 3/4的`request_cycle`数据通路进一步冻结为分层父子事务模型：

    GPU LLC Miss / DMA / CXL.mem Parent Request
                    |
       Bidirectional External Link Service
       Request Packetization / Credit / Serialization
                    |
          Singleton LogicDieMemoryGateway
       Address Normalize / Split / Order / Parent Table
                    |
          N x Internal DRAM Child Request
                    |
              One Ramulator2 Owner
                    |
          N x Child Completion / Join
                    |
       Response Packetization / External Link
                    |
          Original GPU Request Completion

外部GPU接口带宽与Logic Die到3D-DRAM阵列的内部带宽是两个独立资源，通常允许`B_external < B_internal`。二者拥有独立队列、时钟、统计和时序所有权，禁止把它们合并成一个平均带宽或让一个Ramulator2请求同时代表两段传输。

### 4.1 唯一主控与 Python/C++ 边界

第一版冻结为“Python 离线控制面 + C++ 动态执行面”：

- Python 负责配置展开与Schema校验、模型/图模板解析、manual/rule policy编译、auto_dse外层搜索、编译/Trace编排、Artifact Cache和报告；
- C++ 的 `GlobalEventRuntime` 拥有唯一全局事件队列、统一 `global_time_fs`、Request/KV/Residency RuntimeState、EpochScheduler、GraphInstantiator、manual/rule_based Placement Evaluator、Runtime Memory Planner、TopologyLowerer、资源队列和所有完成事件；
- Python 将版本化的 `ModelGraphTemplate + WorkloadManifest + SchedulerPolicy + PlacementPolicy + ArtifactCatalog + ResolvedConfig` 序列化后通过 pybind 提交；
- C++ 在每个Epoch边界基于已提交状态生成`ExecutionGraphInstance`，Python不参与Admission、Batch选择、运行时放置或Lowering决策；
- `analytical`、`operator_event` 和 `request_cycle` 都使用同一个 C++ Runtime；前两者只是提交预计算时长或 Bulk Task，不创建第二套调度器；
- Python 不参与逐 Cycle 或逐 Memory Request 回调；Model 3 中 Accel-Sim、ATLAS Memory Port、Shared Fabric 和 Ramulator2 的请求/响应闭环全部位于 C++；
- 第一次实现允许使用 JSON 作为可审阅的磁盘表示，但 pybind 边界必须先完成 Schema Version 检查再构造 C++ 对象。

动态Shape缺少Artifact时，C++只能在Epoch边界、任何本Epoch任务发射前返回`ArtifactRequest`并冻结模拟时间。Python ArtifactResolver只能为已经确定的Shape/Placement补齐Artifact；不得改变RequestState、调度选择、Placement或`global_time_fs`。补齐后在同一时间恢复。预检能够枚举的Shape应在运行前全部准备，Artifact Miss次数必须写入报告。

PhysicalAddress分配只有一个所有者：Python的`StaticAllocationPlan`只包含Memory Space、容量、Alignment、Lifetime和Layout约束，不产生地址。C++ `RuntimeMemoryPlanner`在time 0实例化Parameter/静态Arena，在Epoch边界分配/释放KV、Activation和Temporary。ATLAS Artifact若依赖最终地址，必须等C++生成`SimulationBufferBindings`后通过ArtifactRequest让Python物化YAML。

唯一运行入口冻结为：

    python -m frontend.hetero.cli run --config <experiment.yaml>

运行目录冻结为：

    runs/<experiment_name>/<simulation_input_key>/
      resolved_config.yaml
      dependency_lock.yaml
      provenance.json
      model_graph.json
      execution_graph.json
      buffer_bindings.json
      trace_manifest.json
      metrics.json
      event_log.jsonl

生产结果不得由另一个 Python 事件循环拼接；Python 只处理Epoch边界Artifact Request，并读取 C++ Runtime 的完成状态和统计快照。

---

## 5. 三个正交的实验维度

### 5.1 物理系统拓扑

    system.profile:
      model1_atlas_native
      model2_host_memory_pcie
      model3_gpu_native_3ddram
      model4_cxl_memory_tier

### 5.2 GPU Backend

    backends.gpu.kind:
      roofline
      accel_sim

### 5.3 联合仿真耦合粒度

    simulation.coupling:
      analytical
      operator_event
      request_cycle

三者必须保持正交。四种 Profile 共用模型解析、计算图、Placement、地址体系、运行时和统计接口。

并不是每种组合都具有相同精度。每项结果必须标记为：

    analytical
    event-modeled
    cycle-coupled
    extrapolated

---

## 6. 四种物理系统 Profile

### 6.1 Model 1：独立 ATLAS

配置名：

    model1_atlas_native

物理结构：

    GPU Roofline/独立 GPU 模型
               |
       分析型外部接口
               |
    ATLAS Compute Die + 3D-DRAM

语义：

- ATLAS 独立运行，最接近当前 ATLAS 原生抽象；
- GPU 默认使用 Roofline；
- GPU 和 ATLAS 使用不同存储空间；
- 跨设备依赖生成分析型 TransferTask；
- 外部链路采用显式带宽、延迟和队列参数；
- ATLAS MemoryService 拥有自身 3D-DRAM 时序。

适用：

- 快速验证；
- ATLAS 原生基线；
- 粗粒度算子卸载；
- 大规模架构筛选。

### 6.2 Model 2：3D-DRAM 是系统主存，GPU 有独立显存

配置名：

    model2_host_memory_pcie

物理结构：

    Host/CPU
       |
    Host 3D-DRAM + ATLAS Compute Die
       |
      PCIe
       |
    GPU + HBM/GDDR

语义：

- Host 3D-DRAM 和 GPU HBM 是不同 Memory Space；
- GPU 不能绕过 PCIe 直接访问 Host 3D-DRAM；
- H2D/D2H 通过 PCIe DMA Engine；
- GPU Backend 可以拥有 GPU 本地 HBM 时序；
- ATLAS MemoryService 拥有 Host 3D-DRAM 时序；
- 两个 Ramulator2 系统可以同时存在，但必须管理不同 memory_space_id；
- 频繁逐算子切换会产生数据乒乓，应优先粗粒度放置。

### 6.3 Model 3：3D-DRAM 是 GPU 原生显存

配置名：

    model3_gpu_native_3ddram

物理结构：

    GPU SM/L1/L2 -- external_gpu_link --\
                                          > LogicDieMemoryGateway
    ATLAS Core ---- internal_hb_port -----/          |
                                             Shared Memory Fabric
                                                    |
                                                 3D-DRAM
                                                 Ramulator2

语义：

- 3D-DRAM 是 GPU 唯一的全局设备内存；
- GPU 保留 Register、Shared Memory、L1、L2 和 NoC；
- GPU LLC Miss 导出到共享 MemoryService；
- GPU LLC Miss先作为外部Parent Request经过请求方向链路进入Logic Die；Ramulator2内部Child全部完成后，再经过响应方向链路返回GPU；
- ATLAS 本地请求通过内部 Hybrid-Bond Port 进入同一 MemoryService；
- GPU 与 ATLAS 竞争相同 Controller、Channel 和 Bank；
- 同一物理 3D-DRAM 只能有一个时序所有者；
- Accel-Sim 内部 DRAM 延迟必须禁用；
- GPU 请求不能再由 Accel-Sim 和 Ramulator2 各计算一次；
- 共享地址不等于自动一致，第一版使用 explicit_noncoherent；
- 设备切换时显式执行 Writeback、Invalidate 和 Fence。

GPU外部请求/响应链路和ATLAS内部端口分别配置带宽、固定延迟、队列深度、请求粒度和仲裁策略。只有明确选择直接混合键合的GPU Logic结构，才允许GPU绕过外部链路并使用内部端口能力；默认Model 3不允许这种绕过。

Model 3 的外部GPU链路、ATLAS内部端口和DRAM内部Fabric是不同资源，禁止合并成一个平均带宽：

    memory_ports:
      - id: gpu_memory_port
        initiator: gpu0
        route: gpu0.external_link -> shared0.logic_die_gateway
      - id: atlas_internal_hb_port
        initiator: atlas0.compute
        route: atlas0.internal_hb -> shared0.logic_die_gateway

两者最终争用同一个 `Shared3DMemoryService` 和同一组 DRAM Controller/Bank。

Model 3的共享内存服务冻结为两个可切换访问模式：

- `access_mode=gpu_only`：无竞争基线。全部ModelGraph节点必须放置到`gpu0`，`backends.atlas.kind=none`，`initiator_order`必须严格等于`["gpu0"]`；派生执行图不得包含非GPU任务或跨设备路由。结果必须显式记录`logic_die_tasks=0`和`logic_die_memory_requests=0`。
- `access_mode=shared_gpu_atlas`：后续竞争实验。允许`gpu0`和`atlas0.compute`通过各自端口提交请求，并由唯一MemoryService仲裁同一组Controller/Channel/Bank。

这两个模式共用Model 3地址空间、MemoryService和时序所有者。GPU-only基线不是另一套GPU DRAM模型，也不得同时启用Accel-Sim内部DRAM计时；因此从无竞争到有竞争时只改变算子放置、允许发起方和端口流量，不改变Global PA语义。

### 6.4 Model 4：GPU HBM + CXL 3D-DRAM

配置名：

    model4_cxl_memory_tier

物理结构：

    GPU + HBM
       |
    CXL Root / Switch
       |
    CXL 3D-DRAM + ATLAS Compute Die

语义：

- GPU HBM 和 CXL 3D-DRAM 是不同 Memory Space；
- GPU 对远端数据可采用 Remote Load/Store、显式 Copy 或 Page Migration；
- 远端 Load/Store 由 DeviceTask 访存经过 CXL 路由，不伪装成整个 Tensor 的 DMA；
- 只有真正复制或迁移才生成 TransferTask/MigrationTask；
- HBM 和 CXL 3D-DRAM 分别拥有自己的时序模型；
- Page Migration 必须更新 Residency、版本、页表状态和 Cache 状态；
- 第一版建模事务、队列、Credit、带宽、延迟和背压；
- 未实现完整协议和 PHY 时，不宣称完整 CXL 协议级精确。
- `access_policy=remote`第一版只允许`simulation.coupling=request_cycle`，GPU Backend必须是`coupled + exports_memory_requests + supports_stall_resume`；
- `operator_event`只允许Model 4的Copy/Migration路径，Remote与其他Contract组合启动失败。

第一版把带有 Compute Die 和内存的 ATLAS 设备固定为“Type-2-like accelerator-with-memory”抽象，而不是声称完整实现某一 CXL 规范设备。拓扑必须显式配置：

    cxl:
      access_initiator: gpu0
      route: direct_p2p | host_root_proxy
      access_policy: remote | copy | migrate
      coherence: explicit_noncoherent

`direct_p2p` 仅表示所研究系统允许 GPU 经建模的 CXL Root/Switch 发起访问，不代表真实平台天然支持该路径；`host_root_proxy` 必须额外经过 Host Root/Proxy 的队列和延迟。

第一版不建模完整 `CXL.cache` 一致性，Model 4 的 GPU/ATLAS 共享远端数据也使用第 16.1 节的 `explicit_noncoherent` Acquire/Release 协议。若未来增加 `modeled_coherent`，必须新增一致性消息、目录、时序所有者和独立精度标签，不能只删除 Flush/Invalidate 时间。

### 6.5 Profile 对比

| Profile | GPU本地显存 | 3D-DRAM身份 | 跨设备数据路径 | 第一版推荐精度 |
| --- | --- | --- | --- | --- |
| Model 1 | 抽象/可选 | 独立ATLAS内存 | 分析型传输 | analytical/operator_event |
| Model 2 | HBM/GDDR | Host主存 | PCIe DMA | operator_event，后续周期化 |
| Model 3 | 无第二套显存 | GPU原生显存 | 共享MemoryService | request_cycle |
| Model 4 | HBM/GDDR | CXL内存层 | Remote/Copy/Migration | Copy/Migration用operator_event；Remote用request_cycle |

---

## 7. 带宽命名和参考配置

新配置禁止继续使用含义模糊的裸 channel_bandwidth。至少区分：

    links[].wire_bandwidth_Bps_per_direction
    links[].latency_fs
    links[].queue_depth_transactions
    memories[].organization.channel_count
    memory_ports[].payload_bandwidth_Bps
    memory_ports[].latency_fs
    memory_ports[].queue_depth_transactions
    memory_ports[].transaction_bytes
    memories[].effective_bandwidth_Bps

ATLAS Edge 参考配置中存在两类不同带宽：

- 外部 LPDDR/系统链路带宽；
- 3D-DRAM 与 Compute Die 之间的内部 Hybrid-Bond 带宽。

它们不能被当作同一个参数，也不能重复计时。

论文 Edge 参考值：

    外部：12.8 GB/s / 3D-Accelerator
    内部：25.6 GB/s / Core
    16 Core：409.6 GB/s / 3D-Accelerator

本地快速测试配置可能使用不同的内部组织，例如 32 GB/s/Core。实验必须记录实际使用的配置文件，不能混用论文值和测试值。

### 7.1 外部/内部带宽层次与完成语义

外部带宽描述GPU或Host到Logic Die边界的协议和PHY；内部带宽描述Logic Die经Hybrid Bond/TSV访问3D-DRAM Channel/Pseudo-Channel/Bank的能力。必须分别配置：

    external_link:
      protocol: direct_memory_phy | pcie_dma | cxl_mem
      request_payload_bandwidth_Bps
      response_payload_bandwidth_Bps
      request_header_bytes
      response_header_bytes
      flit_bytes
      propagation_latency_fs
      queue_depth_transactions
      credits
      duplex_mode
      clock_hz

    logic_die_gateway:
      clock_hz
      ingress_queue_depth
      parent_table_entries
      split_width_per_cycle
      issue_width_per_cycle
      completion_width_per_cycle
      ordering_policy
      write_ack_policy: durable | posted

    internal_dram:
      ramulator2_config
      transaction_bytes: derived_and_validated

`transaction_bytes`必须从Ramulator2实际`channel_width`、prefetch和Burst组织推导并校验，不能只依据配置文件名猜测。`DQ`、`channel_width`、`rate`、`nBL`、`tCK`和事务大小必须导出一致的峰值带宽；不一致时配置加载失败，不允许带着歧义运行性能实验。

一个外部Parent Request可以被拆成多个对齐的内部Child Request。Runtime必须记录Parent ID、Global PA、Logical Bytes、Byte/Sector Mask、Ordering Domain、Child总数和完成数。Ramulator2接收Global PA并负责最终DRAM Tuple译码；Logic Die不得提前编码两套Channel/Bank映射。

读请求的唯一完成点冻结为：全部Child完成且响应方向Link Transaction完成。写请求默认`durable`，同样等待全部Child和写确认链路；`posted`只允许作为显式实验选项，必须分别记录GPU-visible completion与后台durable completion，Run结束前仍需排空。

Model 3的GPU LLC Miss使用`direct_memory_phy` Parent语义；Model 4 Remote使用`cxl_mem`；Model 2通过PCIe DMA/Page Migration生成Bulk Parent，禁止把每个GPU Load/Store直接伪装成PCIe事务。

---

## 8. 两级 IR

### 8.1 ModelGraphIR

ModelGraphIR 是设备无关逻辑图，包含：

#### Value

    Value {
        value_id
        shape_expr
        dtype
        layout
        storage_class
        mutability
        alias_or_view
        lifetime
    }

storage_class 至少支持：

    parameter
    activation
    temporary
    kv_cache
    metadata

#### ComputeOp

    GEMM
    Attention
    Norm
    RoPE
    Activation
    Residual
    Embedding
    LMHead
    Sampling

#### CollectiveOp

    AllReduce
    AllGather
    ReduceScatter
    AllToAll

#### StateOp

    KVAllocate
    KVRead
    KVAppend
    KVRelease
    BlockTableUpdate

#### ControlOp

    RequestStart
    PrefillLoop
    DecodeLoop
    EOSCheck
    RequestFinish

ModelGraphIR 必须保存 data dependency、control dependency、read set 和 write set。

### 8.2 ExecutionGraphIR

完成 Placement、Memory Planning 和 Topology Lowering 后生成：

    DeviceTask
    TransferTask
    MigrationTask
    SynchronizationTask

每个任务至少包含：

    task_id
    template_node_id
    phase
    layer_id
    step_id
    symbol_bindings
    device_id
    backend_id
    input_values
    output_values
    read_memory_spaces
    write_memory_spaces
    dependencies
    resource_requirements
    compiled_artifact_ref
    fidelity

PCIe DMA 和 CXL Migration 只能出现在 ExecutionGraphIR。ModelGraphIR 不包含物理链路。

任务对数据的引用冻结为：

    ValueRef {
        value_id
        version
        offset_bytes
        size_bytes
        access_mode: read | write | read_write
    }

专用任务载荷冻结为：

    TransferTaskPayload {
        source_space
        destination_space
        route_id
        ranges: ValueRef[]
        dma_engine_id
        total_bytes
    }

    MigrationTaskPayload {
        logical_page_ids
        source_space
        destination_space
        old_allocation_epoch
        new_allocation_epoch
        route_id
    }

    SynchronizationTaskPayload {
        scope: range | memory_space | device
        ranges: ValueRef[]
        actions: writeback[] | invalidate[] | fence[]
        fence_semantics: acquire | release | acquire_release
    }

Residency Update 属于 `ResidencyManager` 的 ExecutionGraph Commit 事件，不属于设备无关的 ModelGraph StateOp。部分 KV Append、View、局部迁移和同步都必须携带显式 Byte Range 与 Version。

---

## 9. 端到端 LLM 图

第一版以 Decoder-only Llama 类模型为规范模型。

### 9.1 请求级程序

    Request Arrival
        -> Admission
        -> KV Allocation
        -> PrefillChunkGraph × N
        -> Final Norm
        -> LM Head
        -> Sampling(first token)
        -> DecodeStepGraph × (G - 1)
        -> Request Finish
        -> KV Release

如果生成 G 个 Token：

    Prefill 产生第一个输出 Token；
    Decode Forward 次数为 max(G - 1, 0)。

max_new_tokens=1 时不能额外执行 Decode。

默认端到端时间：

    T_request =
        T_queue
        + T_prefill
        + sum(T_decode_step[1 ... G-1])
        + T_finalize

### 9.2 Transformer Layer 逻辑图

    Hidden State
        -> Norm
        -> QKV Projection
        -> RoPE(Q, K)
        -> KV Append
        -> Causal Attention
        -> Output Projection
        -> Collective
        -> Residual Add
        -> Norm
        -> Gate/Up Projection
        -> Activation and Multiply
        -> Down Projection
        -> Collective
        -> Residual Add
        -> Next Layer

逻辑图保留细粒度节点。Backend 可以根据设备能力融合为 QKV、FlashAttention、Fused MLP 等 Kernel。

融合 Kernel 不能在未解除融合的情况下从内部跨设备拆分。

### 9.3 Prefill

对于请求 i，Prompt 长度为 S_i。Packed Prefill 有效 Token 数为：

    T_p = sum(S_i)

Dense GEMM 的 M 通常等于 T_p。

Chunked Prefill 第 c 个 Chunk：

    chunk_start = c * chunk_size
    q_len = min(chunk_size, prompt_len - chunk_start)
    past_len = chunk_start
    kv_len = past_len + q_len

当前 Chunk 的 Attention 可读取：

- 以前 Chunk 已经提交的全部 KV；
- 当前 Chunk 中不晚于当前位置的 KV。

只有最后一个 Prefill Chunk 默认运行 LM Head 和 Sampling。除非配置要求 Prompt Logprob，否则不为全部 Prompt Token 生成完整词表 Logits。

### 9.4 Decode

第一版要求 `output_length G >= 1`。Prefill 处理 P 个 Prompt Token，随后 LM Head/Sampling 产生 `output[0]`。对第 j 次 Decode Forward，`j` 从 1 到 `G-1`，其输入是 `output[j-1]`，并产生 `output[j]`：

    q_len[i] = 1
    past_kv_len[i] = prompt_len[i] + (j - 1)
    attention_kv_len[i] = committed_kv_len[i] = past_kv_len[i] + 1

执行顺序必须保证：

    QKV -> RoPE -> KV Append -> Attention

当前 Token 产生的 K/V 属于当前自注意力可访问范围。

`generated_length` 表示已经 Sampling 完成的输出 Token 数；`committed_kv_len` 表示已经写入并提交、可供 Attention 读取的 KV Token 数。两者不能混用。生成 G 个输出 Token 后：

    Decode Forward 次数 = G - 1
    LM Head / Sampling 次数 = G
    最终 committed_kv_len = P + G - 1

不能用一个平均 Context Length 替代 `attention_kv_len[]`。

### 9.5 生成模式

第一版支持：

    trace_locked
    fixed_tokens

后续扩展：

    functional_generation
    replayed_eos

跨拓扑公平比较默认使用 trace_locked，并固定：

- 请求到达；
- Prompt 长度；
- 输出长度；
- 随机种子；
- MoE 路由；
- Batch/Scheduler 配置。

### 9.6 已有KV上的独立Decode Step

完整请求语义保持第9.1–9.4节不变。为了进行GPU、ATLAS或内存系统的单步微基准，Workload另外允许显式声明：

    execution_scope: decode_step
    initial_kv_length: K
    output_length: 1

该模式表示KV中已经提交了K个历史Token，当前只执行一次`q_len=1`的Decode Forward、LM Head和Sampling。当前Token的K/V先Append，因此：

    past_kv_len = K
    attention_kv_len = K + 1
    Decode Forward = 1
    Prefill Forward = 0
    final_committed_kv_len = K + 1

它是明确标注的微基准执行范围，不能与`full_request`的TTFT或“`G=1`不执行额外Decode”规则混写。跨系统比较必须保持相同的`execution_scope`和`initial_kv_length`。

---

## 10. KV Cache 状态

KV Cache 是持久化状态，不是普通临时 Tensor。

    KVBlock {
        block_id
        request_id
        layer_id
        kind
        token_begin_inclusive
        token_end_exclusive
        capacity_tokens
        valid_tokens
        dtype
        layout
        residency_record_id
        allocated_bytes
        logical_bytes
        version
        migration_state
        inflight_refcount
    }

kind 为 K 或 V。

逻辑 Value/KV Block 的驻留不能压缩成单个地址或 `resident_devices`。冻结以下副本模型：

    ResidencyRecord {
        logical_object_id
        current_version
        home_memory_space_id
        replicas: ReplicaRecord[]
    }

    ReplicaRecord {
        memory_space_id
        physical_address: PhysicalAddress
        version
        state: valid | dirty | stale | migrating | released
        valid_ranges
        inflight_refcount
    }

`ReplicaRecord` 表示 Memory Space 中有独立分配地址的数据副本；Cache Holder 属于第16.1节的一致性目录，不是Replica。规则冻结为：

- Transfer/Copy 完成后保留源 `valid` Replica，并在目标 Memory Space 新增同 Version 的 `valid` Replica；
- 任一 Replica 上的写提交产生新 Version，并把其他旧 Version Replica 标记为 `stale`；
- Migration 先创建目标 `migrating` Replica，完成后将其变为 `valid`、更新 Home，并释放源 Replica；
- 只有 `valid/dirty` 且 Version 匹配的 Replica 能满足 Acquire；
- 每个 Replica 的 Allocation、地址、容量、Version 和在途引用分别守恒。

单请求每 Token KV 容量：

    Bytes_KV_per_token =
        2 * num_layers * num_kv_heads * head_dim * bytes_per_element

第一版必须支持 Paged KV：

    Logical Token Position
        -> KV Block ID
        -> ResidencyRecord
        -> selected ReplicaRecord
        -> PhysicalAddress
        -> DRAM Tuple

每个请求、每层维护独立 Block Table。

必须验证：

- 每个实际输入 Token、每层只写一次 K 和 V；
- valid_length 单调增加；
- 不同请求不能互相访问 KV；
- 请求结束后 Block 全部释放；
- Page Migration 不产生重叠、悬空或旧版本覆盖；
- Chunked 和非 Chunked Prefill 的逻辑 Attention Pair 数一致。
- `logical_bytes` 只统计有效 KV；`allocated_bytes` 包括 Block Padding，且不得小于 `logical_bytes`。

---

## 11. 算子放置

### 11.1 放置模式

    placement.mode:
      manual
      rule_based
      auto_dse

- manual：用户逐算子或算子组指定。
- rule_based：按 Phase、Layer、Batch、Context Length 和 Residency 匹配。
- auto_dse：分析模型筛选，再用高精度仿真验证。

### 11.2 Capability 检查

每个 Backend 必须通过 BackendCapabilityRegistry 声明：

- 支持的算子语义；
- dtype 和量化格式；
- Shape 和 Layout 限制；
- Matrix/Vector/Attention 原语；
- 本地 SRAM 和内存容量约束；
- 是否支持融合；
- 是否有可靠的 Timing 模型；
- 是否支持导出 Memory Request；
- 是否支持 Stall/Resume。

手工放置不能绕过 Capability 检查。

当前 ATLAS 未支持的端到端算子只能：

1. 放在 GPU；
2. 新增 ATLAS Lowering；
3. 使用标记为 analytical 的临时模型；
4. 配置验证失败。

不能静默当作 ATLAS 周期精确结果。

### 11.3 放置成本

放置决策不能只比较算子计算时间：

    T(op, device) =
        T_compute
        + T_transfer_or_migration
        + T_layout_conversion
        + T_sync
        + T_queue_and_contention

自动放置候选完成时间：

    FinishTime(node, device) =
        max(
            DeviceAvailable(device),
            DependencyReady(node) + DataArrival(node, device)
        )
        + Execution(node, device)
        + Synchronization(node, device)

共享 3D-DRAM 的真实竞争必须由 request_cycle 联合仿真决定；上述公式只用于候选筛选。

Placement、Memory Planning 与 Topology Lowering 之间存在数据驻留反馈，冻结以下求解方式：

- `manual` 和 `rule_based` 使用单遍流程：Placement 决定目标设备，Memory Planner 按显式 home/affinity 规则分配，TopologyLowerer 再生成数据路径；
- `auto_dse` 对每个合法候选创建临时 `CandidateMemoryPlan + CandidateLowering`，计算包含搬运、同步和容量的成本后再选择；
- 候选评估不得修改正式 RuntimeState；选中后只提交一次正式 MemoryPlan；
- 共享 Bank 的动态争用不进入静态 Placement 公式，由少量候选的 `request_cycle` 复验决定。

### 11.4 第一版放置粒度

第一版支持：

- Phase 级；
- Layer 级；
- 连续算子组级；
- Device Sub-Batch 级；
- 整个动态算子实例级。

第一版不支持一个算子实例同时拆到 GPU 和 ATLAS。

推荐算子组：

    Attention Cluster:
      Norm + QKV + RoPE
      KV Append + Attention
      Output Projection

    MLP Cluster:
      Norm
      Gate/Up + Activation
      Down Projection

### 11.5 初始研究策略

    Prefill:
      GPU 执行主要计算

    Decode:
      GPU 执行 QKV、Output Projection、MLP
      ATLAS 执行 KV Append、KV Read、Attention

    KV:
      优先驻留在 3D-DRAM

Norm、RoPE、Activation 和 Residual 等小算子优先与相邻大算子共置，避免为小算子产生跨设备同步和搬运。

### 11.6 KV Affinity

连续批处理必须支持：

    request_id -> kv_home_memory -> preferred_compute_device

建议参数：

    session_affinity: kv_home
    minimum_resident_steps
    switch_gain_threshold
    migration_hysteresis
    hbm_high_watermark
    hbm_low_watermark

避免同一请求每个 Decode Step 在 GPU 和 ATLAS 之间反复迁移。

---

## 12. 多 Batch 与连续批处理

### 12.1 三层 Batch

不能只使用一个 batch_size。

#### Request Batch

    RequestState {
        request_id
        arrival_time_fs
        prompt_length
        prompt_cursor
        output_length
        generated_length
        committed_kv_length
        waiting_epochs
        priority
        deadline_fs
        state
        kv_block_table
        kv_home
    }

状态机：

    WAITING
      -> PREFILL_READY
      -> PREFILL_RUNNING

    PREFILL_RUNNING
      -> PREFILL_READY   if prompt_cursor < prompt_length
      -> DECODE_READY    if prompt完成且generated_length < output_length
      -> FINISHED        if prompt完成且generated_length == output_length

    DECODE_READY
      -> DECODE_RUNNING

    DECODE_RUNNING
      -> DECODE_READY    if generated_length < output_length
      -> FINISHED        if generated_length == output_length

Prefill 最后一个 Chunk 完成后先运行该请求自己的 LM Head/Sampling，再判断 `G=1` 直接结束还是进入 Decode；不能用“整个 Mixed Batch 的最后一个 Chunk”替代逐请求判断。每次状态跳转只发生在 Epoch Commit 阶段。

后续可选：

    PREEMPTED
    SWAPPED
    CANCELLED

#### Phase Batch

    PrefillBatch
    DecodeBatch
    MixedBatch

#### Device Sub-Batch

    DeviceSubBatch {
        target_device
        phase
        operator_group
        token_refs
        q_lens
        kv_lens
        cu_seqlens
        kv_block_tables
        shape_signature
        input_permutation
        output_permutation
    }

其中：

    TokenRef {
        request_id
        absolute_token_position
        packed_index
    }

同一个逻辑 Phase Batch 可以拆成 GPU Sub-Batch 和 ATLAS Sub-Batch。完成后依据显式 `input_permutation/output_permutation` 和 `TokenRef` 恢复逻辑顺序；只按 `request_id` 排序是不充分的。

第一版要求：

    一个 Device Sub-Batch 的一个算子实例只在一个设备执行；
    不同 Sub-Batch 可以在 GPU 和 ATLAS 并行。

### 12.2 Static Ragged Batch

所有请求在 t=0 到达，Prompt 和输出长度可不同。

Packed Prefill：

    total_prefill_tokens = sum(prompt_length[i])

Attention 保留每个请求的边界和因果 Mask。

如果 Backend 使用 Padding Kernel：

- Padding 的指令、流量和时间必须计入；
- 同时报告 effective work 和 padded work。

如果 Backend 使用 Ragged/Packed Kernel：

- 不能虚构 Padding 成本；
- 必须保存 cu_seqlens 或等价边界。

### 12.3 Continuous Batching

第一版采用确定性的 Token-Step Barrier Epoch。

每个 Scheduling Epoch：

1. 提交上一 Epoch 已完成的输出和 KV 更新；
2. 处理新到达请求；
3. 检查 KV 容量并执行 Admission；
4. 选择 Ready Decode 请求；
5. 用剩余 Token Budget 选择 Prefill Chunk；
6. 实例化本 Epoch 的 ModelGraph；
7. 进行 GPU/ATLAS Placement；
8. 形成 Device Sub-Batch；
9. 插入 DMA、CXL、Migration、Fence 或 Layout Conversion；
10. 提交 ExecutionGraph；
11. Sampling 完成后更新 RequestState。

Epoch 的提交屏障冻结为：本 Epoch 的全部 DeviceTask、Transfer/Migration/Sync、KV Commit 和 Residency Commit 均完成后，Epoch 才结束。Epoch 执行期间到达的请求只进入 `pending_arrivals`，在下一边界执行 Admission。

基本预算：

    sum(q_len[i]) <= max_batched_tokens
    active_sequences <= max_num_sequences
    allocated_kv_bytes <= available_kv_capacity

默认策略：

    Decode Priority
    + Chunked Prefill
    + FCFS Admission
    + Prefill Aging
    + Max Batched Tokens

第一版确定性选择规则冻结为：

1. 所有 FCFS/Arrival 比较使用 `(arrival_time_fs, request_id)`；
2. Decode 候选按 `(-priority, deadline_or_inf, arrival_time_fs, request_id)` 排序；
3. Prefill 候选按 `(-priority, -waiting_epochs, arrival_time_fs, request_id)` 排序；
4. 若最老 Prefill 的 `waiting_epochs >= max_prefill_wait_epochs`，先为它保留一个不超过 `prefill_chunk_tokens` 的 Chunk Budget，再按 Decode Priority 分配；否则先分配 Decode；
5. Placement Rule 严格按 YAML 列表顺序 first-match，未匹配项使用 `default_target`；同一优先级不做隐式合并；
6. Device Sub-Batch 按 `(target_device, phase, operator_group, shape_signature)` 分组，组内按 `(request_id, absolute_token_position)` 排序；
7. 所有选择只使用 Epoch 边界的已提交状态，不能读取在途更新。

默认容量策略冻结为：

    kv_reservation_mode: full_request
    max_prefill_wait_epochs: 8

Admission 时按每个请求最终 `prompt_length + output_length - 1` 个 KV Token、全部 Layer、Block 向上取整后预留容量。第一版不允许运行中因 KV 预计容量不足而隐式驱逐已接纳请求。

Epoch 内 GPU 和 ATLAS 可以重叠；下一 Epoch 不能读取尚未提交的 KV。

### 12.4 Mixed Prefill/Decode

一个 Mixed Epoch 可以包含：

    Decode:
      N 个请求 × 1 Token

    Prefill:
      若干 Prompt Chunk

逻辑上属于同一 Epoch，不代表必须融合成同一个 Kernel。

如果 Backend 不支持真正的混合 Ragged Kernel，应生成独立的 Prefill 和 Decode DeviceTask，并在设备队列上调度。

### 12.5 不同 Context Length

Decode Batch 必须保存：

    kv_lens[B]

可按以下签名分桶：

    target_device
    phase
    operator_group
    dtype
    layout
    q_len_bucket
    kv_len_bucket
    kv_residency

阈值必须由测量或 DSE 决定，不能固化为架构结论。

### 12.6 跨设备流水

允许：

    GPU:
      新请求 Prefill
      Decode Projection / MLP

    ATLAS:
      已有请求的长上下文 Decode Attention

已发射的 Sub-Batch 在完成前，其成员、Shape 和设备选择保持不变。新请求只能进入后续 Epoch。

---

## 13. 地址体系

必须保持四层：

    Trace / Device Address
        -> TensorID + ByteOffset
        -> PhysicalAddress {
               memory_space_id,
               offset_bytes,
               allocation_epoch
           }
        -> DRAM Tuple {
               channel,
               pseudo_channel,
               rank,
               bank_group,
               bank,
               row,
               column
           }

### 13.1 规则

1. Trace 地址不是自动等待页表翻译的 GPU VA。
2. 第一版将其视为 Trace/Logical Address，通过 Range Manifest 重定位。
3. TensorID + offset 是稳定的数据身份。
4. `PhysicalAddress` 由当前 Residency 和 Simulation Allocation 决定；它就是本文档唯一的 Global PA 表示，禁止脱离 `memory_space_id` 保存裸 `global_pa`。
5. `PhysicalAddress.offset_bytes` 不编码 Channel/Bank/Row 位。
6. Trace 地址到当前 PhysicalAddress 的绑定必须在 GPU Cache 查询前完成。
7. Candidate-specific DRAM Decode 必须在 LLC Miss 后完成。
8. DRAM DSE 可以改变 DRAM Tuple，但不应改变逻辑 Tensor 身份。
9. 只有研究虚拟内存时才加入 MMU/TLB/Page Table。
10. ATLAS 不同本地内存分区必须使用不同 memory_space_id，即使局部地址都从 0 开始。

### 13.2 Trace Manifest

Trace 捕获地址和仿真物理地址必须分离：

    TraceCaptureBinding {
        capture_allocation_id
        trace_base
        size_bytes
        tensor_id
        tensor_offset_bytes
        capture_epoch
    }

`TraceCaptureBinding` 只用于把原始 Trace 地址正规化为 `TensorID + offset`；它不是仿真 `PhysicalAddress`。每个拓扑/DSE Candidate 再由 `SimulationBufferBinding` 把稳定数据身份绑定到当前 `PhysicalAddress`。

Trace Manifest 至少记录：

    CUDA context
    kernel UID
    trace address range
    TensorID
    trace base
    size
    shape
    layout
    allocation lifetime
    capture allocation epoch
    backing allocation
    view offset
    launch configuration
    trace_semantics
    replay_safe

必须验证：

- Range 覆盖；
- Alignment；
- Capacity；
- Alias；
- 不同 backing allocation 地址区间不重叠；只有声明相同 backing allocation、合法 View Range 和相同 Epoch 的 Alias/View 可以重叠；
- 所有重定位访问合法；
- 旧 allocation_epoch 不可访问。

---

## 14. Backend 与编译接口

### 14.1 两阶段编译

    Compiler.plan()
        -> KernelPlan {
               layout_constraints
               input_output_requirements
               scratch_size
               alignment
               tiling_signature
               fusion_signature
           }

随后按 Backend 分支，不能把 Trace Capture Binding 与 Simulation Binding 混为一体：

    GPU:
      GpuCompiler.materialize_binary(plan)
        -> Binary
      TraceCaptureRunner(binary, TraceCaptureBindings)
        -> RawTrace + RangeManifest
      TraceCanonicalizer(raw_trace, range_manifest)
        -> CanonicalTrace(TensorID + offset)

    Simulation:
      C++ RuntimeMemoryPlanner
        -> SimulationBufferBindings(PhysicalAddress)

    ATLAS:
      AtlasCompiler.materialize(plan, SimulationBufferBindings)
        -> operator_description.yaml
        -> data_placement.yaml

GPU 的位置无关 Binary 不依赖 Simulation Allocation；原始 Trace 依赖捕获时的真实 GPU 地址，但 `CanonicalTrace` 不依赖模拟 Global PA。ATLAS `data_placement.yaml` 依赖 Simulation Allocation。

C++ RuntimeMemoryPlanner 是 Buffer Version、Allocation Epoch 和全部仿真物理地址的唯一所有者。Python Planner、GPU 和 ATLAS Adapter 不能静默重新分配另一套模拟地址。

### 14.2 GPU Roofline

    T = max(
        FLOPs / effective_compute,
        Bytes / effective_bandwidth
    )

适合快速 DSE、调度语义验证和四种 Profile 的早期覆盖。

Roofline 提供两种互斥 Contract：

1. `roofline_total`：直接返回上式的 `total`，不得再导出 Bulk Memory；
2. `roofline_split`：返回

       compute_duration_fs = ceil(FLOPs * 10^15 / effective_compute_FLOPs_per_s)
       bulk_memory_demands = [(memory_space_id, port_id, read_bytes, write_bytes)]

   并由 Runtime 在同一 Task Start 同时启动 Compute Branch 和 Bulk Memory Branch；DeviceTask 在两条 Branch 都完成后结束。因此无争用时仍是 `max(T_compute, T_memory)`，有共享队列时 Bulk Branch 的完成时间由唯一 MemoryService决定。

Model 3 的 M3 分析预览必须使用 `roofline_split`，普通 `roofline_total` 与 Shared3DAnalyticalMemoryService 的组合由 Validator 拒绝。

ATLAS 在该预览阶段使用 `atlas_analytical_split`，不使用已包含DRAM Stall的完整atlasim时长：

    T_compute = max(
        matrix_ops / effective_matrix_ops_per_s,
        vector_ops / effective_vector_ops_per_s,
        local_buffer_bytes / effective_local_buffer_Bps
    )

同时按实际数据流导出3D-DRAM `read_bytes/write_bytes`。Compute/Bulk分支同起点、取较晚完成；该近似不描述算子内部访存相位和细粒度依赖，必须保留`analytical-preview`标签。

### 14.3 GPU Accel-Sim

    CUDA / TileLang
        -> nvcc
        -> GPU Binary
        -> NVBit SASS Trace
        -> Trace Manifest
        -> Accel-Sim

Model 2/4 可以保留 GPU 本地 HBM 时序。

Model 3：

- 保留 GPU Core/Cache/NoC；
- 禁用内部 DRAM Timing；
- 导出 LLC Miss；
- 支持 Stall/Resume；
- 由 Shared3DMemoryService 返回完成事件。

### 14.4 ATLAS Adapter

    ExecutionGraphIR
        -> ATLAS Compiler Planning
        -> Global Buffer Binding
        -> operator_description.yaml
        -> data_placement.yaml
        -> atlasim.Chip

Adapter 必须声明输出时间语义：

    total
    compute_only
    coupled

如果输出 total duration，就不能再导出同一段执行的 DRAM Request 给另一个 MemoryService。

如果输出 coupled，就必须支持请求/响应推进，不能再预先返回完整任务时间。

### 14.5 Backend Timing Contract

每个 Backend 的 `descriptor()` 是实际能力的唯一真源，声明：

    backend_capabilities:
      supported_duration_semantics
      ownable_resource_kinds
      supported_exports
      supports_stall_resume
      supported_trace_semantics
      qualification_records

配置只提出 `requested_timing_mode` 和具体资源绑定。启动前 `ConfigurationValidator + TimingOwnershipRegistry` 取配置请求与Backend能力交集，生成唯一 `ResolvedTimingContract {duration_semantics, owns, exports, supports_stall_resume, trace_semantics, replay_safe}`。请求不是能力子集、配置Owner与Topology Owner不一致或资源冲突时启动失败。Backend运行时行为违反Resolved Contract时立即终止，不允许只记警告。

### 14.6 冻结的 C++ 执行接口

以下是第一版必须满足的语义接口；实现可使用纯虚类、PIMPL 或 Adapter，但不得改变方法输入输出语义：

    ArtifactRequest {
        request_id
        epoch_id
        backend_id
        artifact_kind
        compile_plan_key
        task_signature
        shape_signature
        placement_result_hash
        simulation_buffer_bindings
        expected_artifact_key
    }

    ArtifactResponse {
        request_id
        status
        artifact_ref
        artifact_hash
        manifest_schema_version
        qualification_record_ref
    }

    RuntimeCallbacks
      schedule_wakeup(ComponentId, TimeFs, WakeupToken)
      schedule_task_completion(TaskId, TimeFs, Status)
      schedule_memory_response(MemoryResponse)
      schedule_link_completion(LinkResponse)

    IExecutionBackend
      descriptor() -> BackendDescriptor
      prepare(TaskDescriptor, ArtifactRef) -> PrepareStatus
      can_accept(TaskDescriptor, TimeFs) -> bool
      submit(TaskDescriptor, TimeFs, RuntimeCallbacks) -> SubmitStatus
      on_wakeup(WakeupToken, TimeFs, RuntimeCallbacks) -> void
      on_memory_response(MemoryResponse, TimeFs, RuntimeCallbacks) -> void
      cancel(TaskId) -> CancelStatus
      quiescent() -> bool
      collect_stats() -> BackendStats

    IMemoryService
      descriptor() -> MemoryDescriptor
      bind_runtime(RuntimeCallbacks) -> void
      can_accept(MemoryRequest, TimeFs) -> bool
      try_submit(MemoryRequest, TimeFs) -> Accepted | RetryAt
      on_wakeup(WakeupToken, TimeFs, RuntimeCallbacks) -> void
      advance_to(TimeFs) -> void
      quiescent() -> bool
      collect_stats() -> MemoryStats

    ILinkService
      descriptor() -> LinkDescriptor
      bind_runtime(RuntimeCallbacks) -> void
      can_accept(LinkTransaction, TimeFs) -> bool
      try_submit(LinkTransaction, TimeFs) -> Accepted | RetryAt
      on_wakeup(WakeupToken, TimeFs, RuntimeCallbacks) -> void
      advance_to(TimeFs) -> void
      quiescent() -> bool
      collect_stats() -> LinkStats

    ITopologyModel
      validate(ResolvedConfig, BackendDescriptors) -> ValidationReport
      resolve_route(InitiatorId, MemorySpaceId, AccessKind) -> Route
      lower_acquire(ValueRef, TargetDevice, RuntimeState) -> TaskList
      build_services() -> ServiceRegistry

`advance_to(t)` 只能单调前进，不得执行时间大于 t 的事件。所有完成通知必须通过 `RuntimeCallbacks` 回到唯一的 `GlobalEventRuntime`，Service 之间不能绕过 Runtime 私自推进全局时间。每次成功的`submit/try_submit`必须在返回前安排最终Completion或至少一个未来Wakeup；否则视为接口违规并立即终止，防止“有在途请求但无未来事件”的静默死锁。

Memory Request 冻结为：

    MemoryRequest {
        request_id
        parent_task_id
        initiator_id
        physical_address {
            memory_space_id
            offset_bytes
            allocation_epoch
        }
        value_id
        value_version
        size_bytes
        byte_mask_or_sector_mask
        operation: read | write
        issue_time_fs
        ordering_domain
        sequence_number
        qos_class
        source_partition_id
    }

    MemoryResponse {
        request_id
        parent_task_id
        physical_address
        value_id
        value_version
        completion_time_fs
        status
        completed_bytes
    }

第一版只支持 Read/Write；Atomic、RMW 和隐式一致性操作必须在 Capability/Config 校验时拒绝，不能降级成普通读写。Runtime在提交和响应时都必须验证PhysicalAddress的Allocation Epoch以及Value Version仍与ReplicaRecord一致。LogicDieMemoryGateway按目标端口的`transaction_bytes`、对齐边界和Byte/Sector Mask把父请求确定性拆成子事务；首尾部分事务仍计入完整线上事务字节。全部子事务完成后只表示内部内存服务完成，仍需经过响应方向Link Transaction，之后才产生一次GPU可见的父`MemoryResponse`。统计必须区分logical bytes、external request/response payload和wire bytes、internal transaction bytes及in-flight children。

Link Transaction 冻结为：

    LinkTransaction {
        transaction_id
        parent_task_id
        route_id
        source_id
        destination_id
        payload_bytes
        header_bytes
        wire_bytes
        direction
        ordering_domain
        sequence_number
        issue_time_fs
    }

    LinkResponse {
        transaction_id
        parent_task_id
        route_id
        completion_time_fs
        status
        payload_bytes
        wire_bytes
    }

第一版 Link 配置必须显式给出 `duplex_mode`、双向各自 payload bandwidth、固定延迟、transaction payload、header bytes、DMA engine 数、最大在途事务、Credit 数及 Credit 单位。缺失时 Schema 校验失败，不猜测 PCIe/CXL 代际默认值。Ideal Link 单事务黄金公式为：

    wire_bytes = payload_bytes + header_bytes
    completion_time_fs = issue_time_fs
                       + latency_fs
                       + ceil(wire_bytes * 10^15 / wire_bandwidth_Bps)

多事务还必须经过 DMA Engine、方向队列和 Credit 仲裁；固定延迟每个 Link Transaction 只加一次。报告同时记录Payload Bytes和Wire Bytes，禁止把有效Payload带宽与含Header的Wire序列化公式混用。

C++收到ArtifactResponse后必须核对Request ID、Backend、Artifact Kind、Shape、Placement Result、Simulation Buffer Bindings、Manifest Schema、Artifact Hash和`expected_artifact_key`。任一不一致都保持Epoch暂停并终止本次Run；不得接受“近似相似”的缓存产物。

### 14.7 第一版算子资格基线

下表是首次实现的保守默认值；“需资格验证”表示不能把接口存在等同于周期模型已经验证：

| 逻辑算子/组 | GPU Roofline | GPU Accel-Sim | ATLAS native | 第一版默认 |
| --- | --- | --- | --- | --- |
| Embedding | 支持 | 有对应Trace才支持 | 未资格验证 | GPU |
| Norm/RoPE/Activation/Residual | 支持 | 有对应Trace才支持 | Vector路径需逐算子资格验证 | 与相邻GPU大算子共置 |
| QKV/Projection/MLP GEMM | 支持 | 有对应Trace才支持 | Matrix路径按Shape/Tile资格验证 | GPU，允许规则放置ATLAS |
| Prefill Attention | 支持 | 有对应Trace才支持 | 当前基线未资格验证 | GPU |
| Decode KV Append/Read/Attention | 支持 | 有对应Trace才支持 | 当前`DecodeAttention`目标路径 | 长上下文优先ATLAS |
| LM Head | 支持 | 有对应Trace才支持 | 未作为E2E路径资格验证 | GPU |
| Sampling/EOS控制 | analytical | 可选专用Trace | 不支持 | Host analytical控制任务 |

M2 的最小端到端默认路径使用 GPU Roofline 覆盖全部通用节点，并只在 Capability Test 通过后把 Decode Attention Cluster 放到 ATLAS。任何 `analytical fallback` 必须逐节点记录，不能把完整请求笼统标记为 cycle-accurate。

---

## 15. 全局计时与运行时

全局时间使用：

    using TimeFs = uint64_t
    global_time_fs: TimeFs

各设备本地绝对Cycle根据频率转换为：

    cycle_to_fs(cycle, frequency_Hz)
      = ceil(cycle * 10^15 / frequency_Hz)

C++使用至少128-bit中间乘法；Python/JSON/pybind都以无符号十进制整数无损传递。转换基于绝对Cycle值，禁止逐Cycle累加已舍入的fs。预检发现任一事件可能超过`UINT64_MAX`时拒绝Workload。不能直接相加GPU Cycle、ATLAS Cycle和DRAM Cycle。

### 15.1 analytical

- Backend 直接返回分析时间；
- 不生成周期级内存请求；
- 适合 Model 1 和快速筛选。

### 15.2 operator_event

- 通常由 GPU 和 ATLAS 各自返回完整任务持续时间；
- Runtime 协调依赖、显式传输和资源队列；
- `total` 任务的设备内部内存时间已包含在任务时间内，不得再导出同一流量；
- 也允许 `compute_only + 唯一 BulkMemoryTask`，但必须由 Timing Contract 明确拆分；
- 适合 Model 1、Model 2 和 Model 4 的 Copy/Migration 模式。

Model 3 在 M3 阶段仅允许 `compute_only + Shared3DAnalyticalMemoryService`：GPU/ATLAS 分别提交 Bulk Byte Count，由这个唯一共享服务计时。该结果标记为 `analytical-preview`，不包含 Controller/Bank/Row 级竞争，禁止用来形成共享 3D-DRAM 周期性能结论。M6 的 `request_cycle` 才是 Model 3 的研究结论路径。

### 15.3 request_cycle

- GPU/ATLAS 发出 Memory Request；
- MemoryService 决定完成时间；
- Backend 等待响应并恢复；
- 联合运行自然产生总任务时间；
- Model 3 最终必须使用；
- Model 4 的直接 CXL Remote Load/Store 必须使用。
- 外部Link、LogicDieMemoryGateway和Ramulator2必须在各自时钟域推进；
- GPU请求必须经历`external request link -> parent/child split -> Ramulator2 -> child join -> external response link`，不能由Ramulator2回调直接越过返回链路完成；
- GPU、Link、Logic Die和DRAM周期统一转换为整数fs或由无漂移相位累加器推进，禁止依赖“最后一个GPU Partition每次替全系统Tick一次”的临时调用顺序；
- 运行时内部生成的Child Request可以持久化为审计Trace，但不得预先离线回放替代动态Queue/Credit/Backpressure。

禁止：

    Accel-Sim 完整 Kernel 周期
    + 同一 Kernel Memory Trace 再进入 Ramulator2

禁止：

    ATLAS 完整算子周期
    + 同一算子 DRAM Request 再进入外部 MemoryService

兼容矩阵冻结为：

| coupling | 合法 Backend Contract | Memory 行为 |
| --- | --- | --- |
| analytical | `total` | 不导出周期请求 |
| operator_event | `total` | 内存已包含，不再计时 |
| operator_event | `compute_only` | 必须存在唯一 Bulk Memory Task/Owner |
| request_cycle | `coupled + exports_memory_requests + supports_stall_resume` | MemoryService 响应驱动恢复 |

其他组合由 `ConfigurationValidator` 在启动前拒绝。

### 15.4 事件确定性

全局事件排序键冻结为：

    (time_fs, event_priority, insertion_sequence)

`insertion_sequence` 是 Runtime 分配的严格单调整数。相同时间优先级冻结为：

    0  resource completion / memory response
    1  KV, residency and release commit
    2  request arrival
    3  admission and scheduling
    4  task dispatch
    5  metric snapshot

任何 Backend 不得依赖宿主线程唤醒顺序、Hash Map 遍历顺序或 Python 对象地址决定仿真顺序。

---

## 16. 拓扑 Lowering

跨设备数据依赖先表示为：

    Producer Tensor Version
        -> Data Acquire Requirement
        -> Consumer

根据 Profile 转换：

| 条件 | Lowering |
| --- | --- |
| 数据已在目标设备且版本有效 | 仅依赖 |
| Model 1 跨GPU/ATLAS | 分析型TransferTask |
| Model 2 Host 3D-DRAM与GPU HBM | PCIe DMA |
| Model 3共享3D-DRAM | Writeback/Invalidate/Fence |
| Model 4 CXL直接读取 | Remote Access Policy |
| Model 4复制/迁移 | MigrationTask + Residency Update |

TransferTask 表示源副本保留、目标获得相同版本。

MigrationTask 表示数据驻留改变，并更新：

    Residency
    Version Owner
    Page Mapping
    Cache State
    Allocation Epoch

Epoch规则冻结为：Transfer/Copy不改变源Replica的Allocation Epoch；目标Replica使用目标Allocator签发的当前Epoch。Migration只在Commit点切换一次目标PhysicalAddress/Epoch，不能在Copy开始和结束各更新一次。Release后任何地址区间再次分配都必须由该Memory Space的Allocator产生严格递增的新Epoch；旧Epoch请求和响应均拒绝。

### 16.1 `explicit_noncoherent` 状态机

Model 3，以及 Model 4 中被 GPU 与 ATLAS 同时访问的远端范围，第一版统一使用按 Byte Range 管理的显式非一致协议。Coherence Granule 的状态冻结为：

    UNALLOCATED
    CLEAN_HOME(valid_holders)
    DIRTY(owner)
    MIGRATING(source_space, destination_space, inflight_refcount)
    RELEASED

配置必须提供 `coherence_granule_bytes`。它必须为 2 的幂、不小于所有参与者 Cache Line，并且是相关 Cache Line 和 Memory Transaction 粒度的整数倍。

状态转换冻结为：

1. `AcquireRead(device, range)`：若另一设备是 Dirty Owner，先执行 Owner Writeback/Drain 到 Home；等待其真实内存完成；失效目标设备上该范围的旧副本；完成 Acquire Fence 后把目标加入 `valid_holders`。
2. `AcquireWrite(device, range)`：若另一设备是 Dirty Owner，先 Writeback/Drain；Invalidate 其他所有 Holder；完成 Acquire-Release Fence 后进入 `DIRTY(device)`。
3. 同一设备连续访问同一有效 Version/Range 不插入同步。
4. `CommitWrite` 只更新 Version/Dirty Owner；没有完成 Release 的新版本不得被其他设备 Acquire。
5. `Migrate` 前等待该范围全部 `inflight_refcount == 0`，冲刷 Dirty Owner，失效所有 Cache，复制页面，更新 Page Map 和 `allocation_epoch`，再进入目标 Home 的 `CLEAN_HOME`。
6. `Release` 只能在最后一个 Consumer、在途 Memory Request 和同步任务完成后进入 `RELEASED`。

Model 3 双向顺序固定为：

    GPU -> ATLAS:
      GPU L2 Writeback(range)
      -> Shared3DMemoryService completion
      -> GPU Release Fence
      -> ATLAS stale-state Invalidate
      -> ATLAS Acquire Fence

    ATLAS -> GPU:
      ATLAS write/drain(range)
      -> Shared3DMemoryService completion
      -> ATLAS Release Fence
      -> GPU L2 Invalidate(range)
      -> GPU Acquire Fence

Writeback/Drain 必须在 `request_cycle` 中产生真实 Memory Request；在 `operator_event` 中必须产生由唯一时序所有者计时的 Bulk Memory/Sync Task，不能只增加一个未注明来源的固定延迟。共享地址只消除数据 Copy，不消除 Cache 维护、Fence 和 Bank 竞争。

---

## 17. 配置规范

### 17.1 顶层配置

    schema_version: hetero-sim/v1

    experiment:
      name: llama3_8b_e2e
      seed: 1
      generation_mode: trace_locked

    simulation:
      coupling: request_cycle

    system:
      profile: model3_gpu_native_3ddram
      topology_ref: configs/hetero/topologies/model3_gpu_native_3ddram.yaml

    backends:
      gpu:
        kind: accel_sim
        ref: configs/hetero/backends/gpu_accelsim.yaml
      atlas:
        kind: atlasim
        ref: configs/hetero/backends/atlas_atlasim.yaml
      host:
        kind: none

    model:
      ref: configs/hetero/models/llama3_8b.yaml

    workload:
      ref: configs/hetero/workloads/e2e_fixed.yaml

    scheduling:
      ref: configs/hetero/schedulers/continuous_batching.yaml

    placement:
      ref: configs/hetero/placements/gpu_prefill_atlas_decode.yaml

    address:
      ref: configs/hetero/addresses/paged_kv.yaml

    metrics:
      ref: configs/hetero/metrics/e2e_llm.yaml

唯一规范主键是 `system.profile`、`backends.<device>.kind` 和 `simulation.coupling`。`*_ref/ref` 仅是输入文件展开机制，不是第二个语义来源。加载后必须输出不含任何 `ref`、角色别名或非规范单位的 `resolved_config.yaml`。

缓存键必须根据规范化内容计算，不能根据文件路径或修改时间计算。

### 17.2 放置示例

    placement:
      mode: rule_based
      unit: device_subbatch_operator
      default_target: gpu0

      rules:
        - match:
            phase: prefill
          target: gpu0

        - match:
            phase: decode
            operator_group: attention
            kv_len_min: 1024
          target: atlas0.compute

        - match:
            phase: decode
            operator_group: mlp
            active_batch_min: 16
          target: gpu0

      data:
        kv_cache:
          home: primary_3ddram
          layout: paged
          page_tokens: 16

`primary_3ddram` 是输入配置可用的逻辑角色，Resolved Config 必须按 Profile 展开成：

| Profile | `primary_3ddram` |
| --- | --- |
| Model 1 | `atlas0.dram3d` |
| Model 2 | `host0.dram3d` |
| Model 3 | `shared0.dram3d` |
| Model 4 | `cxl0.dram3d` |

用户也可以直接指定合法的 `memory_space_id`。角色不存在、容量不足或目标不可达时必须报错，不能回退到另一 Memory Space。

### 17.3 Scheduler 示例

    scheduling:
      mode: continuous_batching
      epoch_mode: token_step_barrier
      admission: fcfs
      decode_priority: true
      prefill_aging: true
      max_num_sequences: 64
      max_batched_tokens: 4096
      prefill_chunk_tokens: 512
      max_prefill_wait_epochs: 8
      kv_reservation_mode: full_request

### 17.4 配置职责

- model：模型结构、dtype、量化和逻辑融合边界。
- workload：请求到达、Prompt/输出长度、固定EOS、MoE路由。
- scheduling：Batch、Chunk、Token Budget、优先级和重叠策略。
- placement：phase/layer/op 到设备，以及权重、KV和激活的驻留策略。
- topology：Profile、设备、Memory Space、Link、Route、Coherence和Timing Owner。
- backend：Roofline、Accel-Sim、atlasim和版本配置。
- address：Allocator、Page/Block、Trace Rebase、MMU开关和DRAM Decoder。
- metrics：采集窗口、TTFT、TPOT、E2E、吞吐、链路和DRAM指标。

### 17.5 Schema 冻结规则

配置由版本化 JSON Schema 或等价的严格 Pydantic Schema 校验。未知字段、未知枚举、重复 ID、负值、单位缺失和跨字段冲突全部报错；不得仅警告后继续运行。

顶层核心字段冻结为：

| 字段 | 类型 | 必填 | 默认 | 约束 |
| --- | --- | --- | --- | --- |
| `schema_version` | string | 是 | 无 | 第一版仅接受 `hetero-sim/v1` |
| `experiment.name` | string | 是 | 无 | 非空，作为运行目录名时进行安全转义 |
| `experiment.seed` | uint64 | 否 | `1` | 同时传给调度、采样轨迹和仲裁器 |
| `experiment.generation_mode` | enum | 否 | `trace_locked` | `trace_locked/fixed_tokens/functional_generation/replayed_eos`；第一版生产 DSE 只允许前两项 |
| `simulation.coupling` | enum | 是 | 无 | `analytical/operator_event/request_cycle` |
| `system.profile` | enum | 是 | 无 | 四个冻结 Profile 之一 |
| `system.topology_ref` | path | 输入必填 | 无 | 展开后必须与 `system.profile` 一致 |
| `backends.gpu.kind` | enum | 是 | 无 | `roofline/accel_sim` |
| `backends.atlas.kind` | enum | 是 | `atlasim` | `atlasim/analytical` |
| `backends.host.kind` | enum | 否 | `none` | `none/analytical/gem5`；`gem5` 属于 M9 扩展 |
| `model/workload/scheduling/placement/address/metrics` | object | 是 | 无 | 输入可为内联对象或单个 `ref`，不能二者同时出现 |

Scheduling 核心字段冻结为：

| 字段 | 类型 | 默认 | 约束 |
| --- | --- | --- | --- |
| `mode` | enum | `continuous_batching` | `static_ragged/continuous_batching` |
| `epoch_mode` | enum | `token_step_barrier` | 第一版只接受该值 |
| `admission` | enum | `fcfs` | 第一版只接受该值 |
| `decode_priority` | bool | `true` | 与第 12.3 节排序规则配套 |
| `max_num_sequences` | uint32 | 无 | 必须大于 0 |
| `max_batched_tokens` | uint32 | 无 | 必须大于 0 |
| `prefill_chunk_tokens` | uint32 | `512` | `1..max_batched_tokens` |
| `max_prefill_wait_epochs` | uint32 | `8` | 必须大于 0 |
| `kv_reservation_mode` | enum | `full_request` | 第一版只接受该值 |

Memory Space与Port核心字段冻结为：

| 字段 | 类型 | 默认 | 约束 |
| --- | --- | --- | --- |
| `memory_spaces[].id` | string | 无 | 必填、全局唯一 |
| `memory_spaces[].kind` | enum | 无 | `gpu_hbm/host_3ddram/atlas_3ddram/shared_3ddram/cxl_3ddram` |
| `memory_spaces[].capacity_bytes` | uint64 | 无 | 必须大于0 |
| `memory_spaces[].timing_owner` | string | 无 | 必须引用唯一Service |
| `memory_spaces[].allocation_alignment_bytes` | uint64 | `64` | 2的幂 |
| `memory_ports[].id` | string | 无 | 必填、全局唯一 |
| `memory_ports[].initiator` | string | 无 | 引用Device/Compute ID |
| `memory_ports[].target_memory` | string | 无 | 引用Memory Space ID |
| `memory_ports[].payload_bandwidth_Bps` | uint64 | 无 | 必须大于0 |
| `memory_ports[].latency_fs` | uint64 | `0` | 非负 |
| `memory_ports[].queue_depth_transactions` | uint32 | 无 | 必须大于0 |
| `memory_ports[].transaction_bytes` | uint32 | 无 | 2的幂，且不大于Coherence Granule |
| `memory_ports[].arbitration` | enum | `round_robin` | `round_robin/fcfs/qos_weighted` |

Link核心字段冻结为：

| 字段 | 类型 | 默认 | 约束 |
| --- | --- | --- | --- |
| `links[].id/endpoints` | string/string[2] | 无 | ID唯一，Endpoint必须存在 |
| `links[].duplex_mode` | enum | 无 | `full_duplex/half_duplex` |
| `links[].wire_bandwidth_Bps_per_direction` | map | 无 | 两方向均显式给出且大于0 |
| `links[].latency_fs` | uint64 | 无 | 单个Transaction固定延迟 |
| `links[].transaction_payload_bytes` | uint32 | 无 | 必须大于0 |
| `links[].header_bytes` | uint32 | `0` | 计入wire bytes |
| `links[].dma_engine_count` | uint32 | `1` | 必须大于0 |
| `links[].max_inflight_transactions` | uint32 | 无 | 必须大于0 |
| `links[].credit_count` | uint32 | 无 | 必须大于0 |
| `links[].credit_unit_bytes` | uint32 | 无 | 必须大于0 |
| `links[].ordering` | enum | `per_domain_fifo` | 第一版只接受该值 |

Backend输入配置只允许请求：

| 字段 | 类型 | 默认 | 约束 |
| --- | --- | --- | --- |
| `requested_timing_mode` | enum | 无 | `total/compute_only/coupled`，必须为Descriptor能力子集 |
| `resource_bindings` | map | `{}` | Resource ID到Backend能力槽位，必须与Topology Owner一致 |

以下`ResolvedTimingContract`字段由Validator生成，输入配置不能直接覆盖：

| 字段 | 类型 | 默认 | 约束 |
| --- | --- | --- | --- |
| `duration_semantics` | enum | 无 | `total/compute_only/coupled` |
| `owns` | string[] | `[]` | 每个资源全局唯一Owner |
| `exports` | enum[] | `[]` | `memory_requests/bulk_memory/link_transactions` |
| `supports_stall_resume` | bool | `false` | request_cycle必须为true |
| `trace_semantics` | enum | `none` | `none/performance/functional` |
| `replay_safe` | bool | `false` | true时必须提供Qualification Record |

Topology 中每个 `device`、`memory_space`、`memory_port`、`link`、`route` 和 `service` 都必须有全局唯一字符串 ID。所有容量/带宽/延迟/频率字段在输入中必须带单位或使用以 `_bytes/_Bps/_fs/_Hz` 结尾的规范字段。内部 Resolved Config 只保留规范单位。

跨字段校验至少包括：

- Timing Contract 与第 15 节 Coupling 兼容；
- Model 3 仅存在 `shared0.dram3d` 这一 GPU 全局设备内存，且只有一个时序所有者；
- Model 2 的 GPU Route 到 Host 3D-DRAM 必须经过 PCIe DMA；
- Model 4 的 `route/access_policy/coherence` 组合合法；`remote`严格要求`request_cycle + coupled`，`operator_event`仅可用于`copy/migrate`；
- 每个 Memory Port 的 Initiator、Target Memory、Transaction Size 和 Queue 均存在；
- `output_length >= 1`，Prompt 长度大于 0；
- `coherence_granule_bytes` 满足第 16.1 节；
- `replay_safe` 缺省为 `false`，只能由显式资格验证产物提升为 `true`。

---

## 18. 计划目录

    frontend/hetero/
      __init__.py
      cli.py
      schema/
      ir/
        model_graph.py
        execution_graph.py
        state.py
      model/
      workload/
      scheduling/
      placement/
      lowering/
      address/
      compiler/
      backends/
        gpu/
        atlas/
        host/

    simulator/src/hetero/
      CMakeLists.txt
      runtime/
      topology/
      services/
      links/
        ideal/
        pcie/
        cxl/
      bridges/
        gpu_memory_bridge/
        atlas_memory_port/
      coherence/
      metrics/

    configs/hetero/
      schemas/
      experiments/
      models/
      workloads/
      schedulers/
      placements/
      topologies/
      devices/
      memories/
      links/
      backends/
      addresses/
      metrics/

    tests/hetero/
      graph/
      address/
      topology/
      runtime/
      batching/
      e2e/

    simulator/tests/hetero/
      runtime/
      services/
      bridges/

    tools/hetero/
      prepare_trace/
      run/
      report/

现有 configs/architecture、ATLang、TileLang 和 simulator 目录保持不动，由 Adapter 引用。

C++ 单元测试放在 `simulator/tests/hetero/`，Python/集成测试放在根目录 `tests/hetero/`。构建必须在 `simulator/src/CMakeLists.txt` 中加入 Hetero 子目录，在测试 CMake 中逐项 `add_test`，并确保顶层 `enable_testing()` 生效。固定测试入口为：

    pytest tests/hetero
    ctest --test-dir simulator/build --output-on-failure

---

## 19. Cache Key 与复用

Artifact Cache依赖是DAG：

    ModelGraphKey + WorkloadKey + SchedulerKey
                    + TopologyKey + PlacementKey
                              |
                    ExecutionTemplateKey
                              |
                       CompilePlanKey
                         /         \
                   BinaryKey    Runtime AllocationKey
                       |             |
            TraceCaptureBindingKey   +--> AtlasMaterializationKey
                       |
                  RawTraceKey
                       |
                CanonicalTraceKey

动态运行不能用尚未生成的Execution Instance反向定义运行目录。冻结为：

    SimulationInputKey
      = H(Model/Workload/Scheduler/Placement/Topology,
          AllocatorPolicy/AddressMapper/MemoryTiming,
          Backend/Coupling/Seed,
          ArtifactCatalogManifest,
          SimulatorAndSchemaVersions)

    for each committed epoch n:
      ExecutionInstanceKey[n]
        = H(ExecutionTemplateKey, epoch_id, committed_state_hash,
            symbol_bindings, placement_result,
            allocation_and_residency_snapshot, artifact_bindings)

      SequenceDigest[n+1]
        = H(SequenceDigest[n], ExecutionInstanceKey[n])

    SimulationResultKey
      = H(SimulationInputKey, final_SequenceDigest,
          final_counter_digest, completion_status)

    ReportKey
      = H(SimulationResultKey, metric_formula_version,
          aggregation_and_output_format)

运行目录在仿真前只使用`SimulationInputKey`。每个Epoch把ExecutionInstanceKey和滚动SequenceDigest写入Event Log；仿真结束后才生成`SimulationResultKey`。

### 19.1 Key 内容

- ModelGraphKey：ModelSpec、图构建器版本、dtype、量化和逻辑融合。
- WorkloadKey：请求Trace、Prompt/输出长度、EOS/MoE路由和Seed。
- SchedulerKey：Batch、Chunk、Token Budget和策略。
- TopologyKey：Profile、Device、Memory Space、Link、Route、Coherence和Timing Owner；不含DRAM时序。
- PlacementKey：模型图、Shape Profile、Topology能力和放置规则。
- ExecutionTemplateKey：ModelGraph、Placement、Topology和Lowering版本，不包含最终 Artifact 与 PhysicalAddress。
- CompilePlanKey：任务语义、Shape、dtype、Layout、Fusion、Algorithm、Target ISA/SM、Compiler和Library。
- BinaryKey：CompilePlanKey和编译器全部可执行输入。
- TraceCaptureBindingKey：捕获主机的真实分配范围、Range Manifest、输入数据签名和 Capture Harness 版本。
- RawTraceKey：BinaryKey、TraceCaptureBindingKey、Launch Geometry、Tracer/NVBit版本和原始地址行为。
- CanonicalTraceKey：RawTrace内容、Range Manifest、Canonicalizer版本、trace_semantics和资格验证结果；排除模拟PhysicalAddress、DRAM Mapper和DRAM Timing。
- Runtime AllocationKey：ExecutionTemplate、Epoch Committed State、Lifetime、Memory Space、Capacity、Alignment和Allocator Policy。
- AtlasMaterializationKey：CompilePlanKey、AllocationKey和ATLAS Adapter版本。
- ExecutionInstanceKey：ExecutionTemplate、`epoch_id`、Committed RuntimeState Hash、Symbol Bindings、确定的Placement Result、CanonicalTrace/ATLAS Materialization、Allocation/Residency Snapshot和Artifact Bindings。
- AddressDecodeKey：Allocation、DRAM组织、Interleave和Mapper。
- MemoryTimingKey：DRAM组织、Timing、Controller、Scheduler、Refresh和Ramulator2版本。
- SimulationInputKey：全部预运行确定输入；不包含动态PhysicalAddress或Execution Instance序列。
- ExecutionInstanceSequenceDigest：按Epoch提交顺序累积，顺序不同则Digest不同。
- SimulationResultKey：Input Key、完整Instance序列、最终整数Counter摘要和完成状态。
- ReportKey：Result Key、指标公式、聚合和输出格式。

### 19.2 失效边界

| 改动 | 最早失效位置 |
| --- | --- |
| 报表格式或百分位 | ReportKey |
| tRCD、频率、DRAM Scheduler | MemoryTimingKey |
| Channel/Bank/地址映射 | AddressDecodeKey |
| 模拟PhysicalAddress、KV Block Size | AllocationKey；CanonicalTrace可复用 |
| Trace Capture真实分配 | TraceCaptureBindingKey/RawTraceKey |
| PCIe/CXL/Profile | TopologyKey |
| Prefill Chunk/Batch策略 | SchedulerKey；新Shape可能触发Compile/Trace |
| Kernel、Fusion、Shape、SM | CompilePlanKey/BinaryKey/Trace相关Key |
| 模型结构 | ModelGraphKey及全部下游 |

### 19.3 Decode Trace

Decode Attention 的 kv_len 逐步变化，不能无条件使用一份 Trace。

允许：

1. 每个实际长度采Trace；
2. kv_len分桶；
3. 经过验证的参数化Trace；
4. 对Attention使用请求生成器。

第一版优先使用精确Shape。任何分桶、插值或外推必须记录精度等级和覆盖率。

fixed Trace 只有在 replay_safe=true 时允许复用。

`replay_safe` 默认值是 `false`。只有资格测试明确证明 Trace 的控制流、同步和地址行为不依赖被修改的时序/拓扑维度后，才生成签名的 Qualification Record 将其提升为 `true`。系统不声称可以自动识别全部 Atomics、Spinlock、竞争同步、输入相关分支和动态 MoE 路由风险。

---

## 20. 实现阶段

### M0：冻结契约和基线

交付：

- 四种拓扑定义；
- Timing Ownership表；
- 单位规范；
- ATLAS和Accel-Sim版本锁定；
- `dependency_lock.yaml`与Trace Capture环境锁定；
- 黄金工作负载；
- 结果字段和Fidelity规范。

验收：

- ATLAS独立基线可复现；
- Accel-Sim独立基线可复现；
- 每个Memory Space有明确时序所有者；
- 所有内部单位规范化。

### M1：统一模型和两级IR

交付：

- CanonicalModelSpec；
- ModelGraphIR；
- ExecutionGraphIR；
- Prefill/Decode模板；
- KVCacheState；
- RequestState；
- 固定生成轨迹。

验收：

- 两层小模型图依赖正确；
- KV Append在Attention之前；
- G个输出Token对应G-1次Decode；
- 不同请求不能互访KV。

### M2：基础运行时和Model 1

交付：

- Global Event Runtime；
- GPU Roofline；
- ATLAS Adapter；
- 分析型Link；
- 单请求完整Prefill/Decode。

验收：

- Roofline与公式一致；
- ATLAS包装前后确定性统计一致；
- 能输出TTFT、TPOT和E2E；
- 不重复计算算子和传输时间。

### M3：Memory Space、地址和四种Profile事件模式

交付：

- MemorySpace；
- PhysicalAddress；
- Global Allocator；
- TopologyRouter；
- Transfer/Migration/Sync；
- 四种Profile；
- ConfigurationValidator。

验收：

- Model 1/2/4能用Roofline/operator_event运行；Model 3按第15.2节以`compute_only + Shared3DAnalyticalMemoryService`运行分析预览；
- 非法访问启动前拒绝；
- 地址、容量和链路字节守恒；
- Model 3分析预览不产生伪DMA，并标记`analytical-preview`、禁止报告Bank级争用结论。

### M4：Multi-Batch

交付：

- Static Ragged Batch；
- Paged KV；
- 不同Context Length；
- Token-Step Barrier Continuous Batching；
- Chunked Prefill；
- Mixed Prefill/Decode；
- Device Sub-Batching。

验收：

- B=1与单请求结果一致；
- 多请求逻辑工作量守恒；
- Chunked/Non-Chunked语义一致；
- Sub-Batch不丢失或重复请求；
- 请求结束后KV无泄漏；
- 固定Seed可复现且无饥饿。

### M5：Accel-Sim后端资格验证

交付：

- CUDA/TileLang编译；
- NVBit Trace；
- Trace Manifest；
- Tensor地址绑定；
- Trace Cache；
- Accel-Sim Adapter。

验收：

- 独立GPU结果与原生Accel-Sim基线一致；
- Trace地址先映射到TensorID+offset；
- 对具有有效 Qualification Record 且 `replay_safe=true` 的CanonicalTrace，DRAM Timing变化能复用；
- 未通过资格验证的Trace保持默认`replay_safe=false`并拒绝跨配置复用。

### M6：Model 3共享3D-DRAM周期耦合

交付：

- GPU Memory Bridge；
- Bridge ABI v2：Parent ID、Global PA、Size、Byte/Sector Mask、Partition、Ordering和QoS；
- 双向外部Link Service；
- Singleton LogicDieMemoryGateway、Parent Table和确定性Child Split/Join；
- ATLAS Memory Port；
- Shared Memory Fabric；
- 唯一Shared3DMemoryService；
- 跨时钟事件桥；
- 仲裁和背压。
- durable/posted写完成策略和多时钟域推进。

验收：

- 同一3D-DRAM只有一个时序所有者；
- 外部Link带宽、ATLAS内部端口带宽和Ramulator2内部带宽可以不同，且分别满足独立的Byte/Rate守恒；
- `DQ/channel_width/rate/nBL/tCK/transaction_bytes`必须通过一致性校验；
- 一个Parent拆出的全部Child未完成前不得发起Response Link，Response Link未完成前GPU不得解除阻塞；
- 任意32/64/128B及非对齐请求的Child覆盖范围必须与Byte/Sector Mask完全一致，无丢失、重复或越界；
- GPU-only/ATLAS-only分别与“相同Bridge、Port和MemoryService配置，仅禁用另一发起方”的基线比较，请求PhysicalAddress、Operation、Logical Bytes、Transaction Bytes和完成数完全一致；
- 固定延迟服务的双发起方并发严格通过第22.5节完成顺序；
- 请求注入数等于完成数加在途数，父/子请求均分别守恒；
- 全部Task完成时Event Queue和所有in-flight集合必须同时为空；若存在in-flight但无未来事件立即判定死锁失败；
- 固定测量窗口内`wire/service bytes <= configured_rate * window + one_transaction_bytes`，只允许一个跨窗口未完成事务的边界项；
- 真实Ramulator2延迟只与版本锁定的黄金结果按测试中显式记录的绝对/相对容差回归，不使用“合理争用”作为断言。

### M7：PCIe和CXL高精度化

交付：

- PCIe DMA Engine和队列；
- CXL Root/Switch；
- Credit；
- Remote Access；
- Page Migration；
- Residency/Owner状态机。
- 复用M6的双向Cycle Link接口，但按PCIe DMA和CXL.mem分别约束请求生成语义。

验收：

- Link Byte、Packet和Credit守恒；
- 延迟不低于传播与序列化下界；
- Model 2禁止绕过PCIe；
- Model 4本地、远端和迁移路径分别通过测试。
- Model 2仅允许DMA/Page Migration生成PCIe Parent；Model 4 Remote才允许细粒度CXL.mem Parent。

### M8：全组合验证与DSE

交付：

- 自动化回归矩阵；
- 缓存清单；
- 跨拓扑报告；
- 真实模型Workload；
- Roofline筛选和周期级复验流程。

验收：

- 四种Profile运行相同逻辑Workload；
- 结果包含Resolved Config、版本、Trace、地址和Seed；
- 关键单调性和容量边界测试通过；
- 结果标明Fidelity和复用内容。

### M9：高级扩展

后续可选：

- 动态EOS；
- 真实Logits；
- 动态MoE；
- MMU/TLB；
- 完整CXL一致性；
- 多GPU；
- 单算子跨GPU/ATLAS切分；
- 推测解码。
- Host Backend的gem5控制/CPU时序接入；
- 与`IExecutionBackend`兼容的周期级NPU Adapter。

`dependency_lock.yaml` 至少记录 ATLAS、Accel-Sim/GPGPU-Sim、Ramulator2、BookSim2、pybind11 和 TileLang Commit，以及 CUDA、Driver、NVBit、目标 GPU SM、GCC、CMake、Python、WSL/Ubuntu 版本。还要分别记录 Trace Capture 主机和离线 Simulation 目标配置。Accel-Sim 采用固定 Git Submodule 还是外部只读依赖必须在 M0 选择一种并写入 Lock；不得由运行机器上的任意最新版本解析。

---

## 21. Multi-Batch 子阶段

M4 内部按以下顺序实施：

    B1: Static Ragged Batch + 手工算子组放置
    B2: Paged KV + 不同Context Length
    B3: Barriered Continuous Batching
    B4: Chunked Prefill + Mixed Prefill/Decode
    B5: GPU/ATLAS Device Sub-Batching
    B6: 基于队列、KV驻留和SLO的自动放置
    B7: 完全异步流水、抢占和换出

B1-B5属于第一版。B6-B7可以在基础正确性完成后增加。

---

## 22. 验收用黄金工作负载

### 22.1 最小端到端用例

    模型：2层Tiny-Llama类Decoder
    hidden_size：128
    intermediate_size：256
    num_attention_heads：4
    num_kv_heads：2
    head_dim：32
    vocab_size：256
    MLP：SwiGLU
    Norm：RMSNorm，epsilon=1e-5
    RoPE：base=10000
    dtype：FP16
    Batch：1
    Prompt：16 Tokens
    Output：3 Tokens
    Generation：trace_locked
    KV：Paged，固定小容量

逐节点检查：

- Prefill；
- 首Token；
- 两次Decode Forward；
- 每层KV Append；
- 最终KV长度；
- LM Head和Sampling次数；
- Request Finish和KV Release。

该用例的整数黄金值冻结为：

    Decode Forward = 2
    LM Head = 3
    Sampling = 3
    final committed_kv_len = 18
    KV logical bytes/token/all layers = 512 B
    final KV logical bytes = 9216 B

当 `page_tokens=16`、每层 K/V 独立 Block、无额外对齐时：

    allocated KV blocks = 2 layers * 2 kinds * 2 blocks = 8
    bytes/block = 2048 B
    final allocated KV bytes = 16384 B

总计 18 个实际输入 Token 被写入 KV；按 K/V Pair 计为 36 次 Layer Append，按 K、V 独立 Range Write 计为 72 次。实现必须明确采用哪一个 Counter 名称，不能把二者混写。

### 22.2 Ragged Batch用例

    Prompt Lengths: [8, 16, 31, 64]
    Output Lengths: [1, 2, 3, 4]

检查：

- Packed GEMM有效Token数；
- 每请求因果边界；
- G=1请求无Decode；
- 请求动态退出；
- Device Sub-Batch拆分和恢复顺序。

### 22.3 连续批处理用例

至少三个不同Arrival Time、Prompt和Output Length的请求。

检查：

- Admission；
- Decode Priority；
- Prefill Aging；
- Token Budget；
- KV Capacity；
- 请求无饥饿；
- Static和Continuous逻辑工作量一致。

Scheduler 单元测试另外使用零数值计算、固定 `epoch_duration_fs=1000` 的 Dummy Backend，配置 `max_num_sequences=2`、`max_batched_tokens=4`、`prefill_chunk_tokens=2`：

| Request | Arrival(fs) | Prompt | Output |
| --- | ---: | ---: | ---: |
| R0 | 0 | 4 | 2 |
| R1 | 0 | 2 | 1 |
| R2 | 1500 | 3 | 2 |

逐 Epoch 选择必须精确为：

| Epoch/边界 | 选择内容 |
| --- | --- |
| E0/0 | R0 Prefill[0:2]，R1 Prefill[0:2]；R1产生首Token并结束 |
| E1/1000 | R0 Prefill[2:4]；R0产生首Token |
| E2/2000 | 接纳R2；R0 Decode一步，R2 Prefill[0:2]；R0结束 |
| E3/3000 | R2 Prefill[2:3]；R2产生首Token |
| E4/4000 | R2 Decode一步；R2结束 |

R2 在 E1 执行期间到达，只能在 E2 边界接纳。该测试按 TokenRef 检查不能丢失、重复或重排 Token。

### 22.4 拓扑用例

- Model 1：分析型传输时间符合公式；
- Model 2：H2D/D2H字节守恒；
- Model 3：无GPU/ATLAS伪DMA，且只有一个共享DRAM完成时间；
- Model 4：Remote、Copy和Migration三种路径分别验证。

### 22.5 契约与精确时序用例

必须包含：

- Schema 正例，以及未知字段、非法单位、重复ID、非法Profile/Contract组合等负例；
- Timing Ownership冲突必须在任何仿真事件发生前失败；
- Alias/View合法重叠、非法Backing重叠、越界、旧Epoch和部分Range访问；
- `FixedLatencyMemoryService`：64 B事务、100 fs固定完成延迟、10 fs注入间隔、Round-Robin从GPU开始；t=0同时排队G0、G1、A0时，完成顺序和时间严格为`G0@100, A0@110, G1@120 fs`；
- Model 3两发起方并发时，父请求数、子事务数、响应数和在途数严格守恒；
- Ideal Link序列化结果按第14.6节公式精确到1 fs向上取整；
- Paged KV精确检查Block数、logical bytes、allocated bytes和最终长度；
- 第22.3节逐Epoch黄金表；
- 所有整数Counter和Byte Counter完全相等，不使用容差；浮点功耗/性能回归单独声明相对和绝对容差。

---

## 23. 指标

### 23.1 请求级

    TTFT p50/p95/p99
    TPOT/ITL p50/p95/p99
    E2E latency
    request throughput
    token throughput
    queueing latency
    SLO attainment / goodput

对请求 r，令 `token_ready[r, j]` 是第 j 个输出 Token 可交付给用户的时间：

    TTFT[r] = token_ready[r, 0] - arrival_time[r]
    ITL[r, j] = token_ready[r, j] - token_ready[r, j - 1], j >= 1
    TPOT[r] = mean(ITL[r, 1..G-1]), 仅G>1时定义
    E2E_user[r] = token_ready[r, G-1] - arrival_time[r]
    retire_latency[r] = kv_release_complete[r] - arrival_time[r]

TPOT 平均值和 ITL 百分位必须分开报告；`G=1` 请求的 TPOT 为 `null`，不能记为 0。用户 E2E 与 KV 释放完成时间也必须分开。

吞吐测量窗口必须在 Resolved Config 中固化为 `[measurement_start_fs, measurement_end_fs)`。Warm-up 请求仍参与系统负载但不进入指标分子；仿真必须完成 Drain，并单独报告 drain 时间。若输入使用 `auto`，解析器必须把首个非 Warm-up 请求到达和最后一个被测请求最终 Token 时间写成显式数值后再生成最终 ReportKey。

### 23.2 Batch与调度

    active batch size分布
    tokens per epoch
    prefill/decode token比例
    effective/padded work
    scheduler等待时间
    请求饥饿时间
    preemption/swap次数
    KV admission failure

### 23.3 Placement

    GPU/ATLAS算子数量
    GPU/ATLAS时间占比
    Device Sub-Batch大小
    每请求/每层设备切换次数
    Placement Decision日志
    GPU/ATLAS并行重叠比例

### 23.4 数据与内存

    PCIe/CXL/内部端口字节数
    External Request/Response Payload Bytes与Wire Bytes
    External Link有效带宽、利用率、Credit Stall和Queue Stall
    Parent请求数、Child事务数与Internal Traffic Amplification
    Logic Die Ingress/Split/Issue/Completion Queue占用
    Parent从GPU发出、Logic Die到达、内部完成、响应返回的分段延迟
    bytes per generated token
    DMA/Migration/Fence次数
    KV远端访问和迁移字节
    HBM/3D-DRAM峰值占用
    Paged KV内部/外部碎片
    DRAM Channel/Bank利用率
    Row Hit率
    Bank冲突
    GPU/ATLAS请求排队延迟
    共享DRAM争用减速

### 23.5 能耗

必须分别记录：

    GPU compute/cache/memory
    ATLAS matrix/vector/buffer/DRAM
    PCIe/CXL/link
    Migration
    Idle/active background

没有可靠模型的组件必须标记 estimated 或 unavailable。

每个 Task、请求汇总和全局结果都必须保存多维精度，而不是单个字符串：

    fidelity {
        compute_fidelity
        memory_fidelity
        link_fidelity
        scheduler_fidelity
        extrapolated_fraction
        trace_coverage
    }

各维枚举至少区分 `analytical/event_modeled/cycle_coupled/unavailable`；`extrapolated_fraction` 和 `trace_coverage` 范围为 `[0,1]`。

---

## 24. DSE 搜索空间

### 24.1 软件与调度

- Prefill/Decode设备归属；
- Layer Block边界；
- 算子组边界；
- Batch Size；
- Max Batched Tokens；
- Prefill Chunk Size；
- Decode Batch Size；
- Placement Hysteresis；
- GPU/ATLAS并发优先级；
- Fusion；
- Tile Size。

### 24.2 数据

- KV Home；
- KV Block Size；
- Token-major/Head-major；
- K/V分离或交织；
- Weight复制或分片；
- Copy/Remote/Migration；
- 热冷KV分层；
- Page和Alignment。

### 24.3 硬件

- 3D-DRAM Channel/Bank组织；
- Address Mapping；
- 内部Hybrid-Bond Port；
- GPU Memory Port；
- GPU↔Logic Die请求/响应方向的独立Payload/Wire带宽；
- Logic Die Parent Table、Split/Issue/Completion Width和Clock；
- Ramulator2 Transaction Bytes与内部流量放大率；
- PCIe/CXL带宽和延迟；
- Queue/Credit；
- Scheduler/Arbitration；
- GPU/ATLAS频率；
- SRAM容量和带宽。

### 24.4 搜索策略

    Roofline/解析模型大范围筛选
        -> operator_event联合仿真
        -> 少量候选request_cycle周期仿真

目标可选：

    TTFT
    TPOT p99
    Throughput
    Goodput
    Energy
    EDP
    Link Bytes
    DRAM Contention
    Memory Capacity
    Fairness

---

## 25. 强制不变量

以下内容必须实现为自动断言：

1. 每个物理资源只有一个时间所有者。
2. 每个物理DRAM地址空间只有一个MemoryService。
3. 每个访存请求只接受一次DRAM时序。
4. 每条跨设备数据边只产生一次搬运时间。
5. 预计算总任务时间和请求级Stall不能描述同一段执行。
6. 不同设备Cycle不能直接相加。
7. 四种Profile使用同一ModelGraph和Workload。
8. Topology只改变Route、Migration、Sync和Contention，不改变算子语义。
9. KV Cache是跨Layer、Chunk和Decode Step的显式持久状态。
10. TensorID+offset是稳定数据身份。
11. PhysicalAddress必须包含memory_space_id，且offset_bytes不编码Channel/Bank/Row。
12. Trace Rebase在GPU Cache查询前完成。
13. DRAM Decode在LLC Miss后完成。
14. Model 3不允许GPU和ATLAS各自保留一套3D-DRAM时序。
15. Model 2/4的多个MemoryService必须管理不同Memory Space。
16. 发出内存请求数等于完成数加在途数。
17. 发送链路字节数等于接收数加在途数。
18. Tensor地址区间不重叠且不越过Memory Space容量。
19. 每层、每请求、每Token的K/V只写一次。
20. Prefill产生首Token，G个输出对应G-1次Decode。
21. 不同请求Attention不能交叉读取KV。
22. 固定Workload跨拓扑执行相同Token数和MoE路由。
23. 所有内部带宽、频率和时间统一为B/s、Hz和fs。
24. 每项结果标明Fidelity、Trace复用和外推信息。
25. GPU Trace Capture地址只能通过Manifest正规化，不能作为Simulation PhysicalAddress。
26. 不同Backing Allocation不得重叠；只有显式合法Alias/View可以重叠。
27. `output_length >= 1`，最终KV长度严格等于`P + G - 1`。
28. 同时间事件严格按`(time_fs, event_priority, insertion_sequence)`排序。
29. Epoch只在全部Task、同步和Commit完成后结束。
30. `explicit_noncoherent`跨设备写读必须执行范围化Writeback/Invalidate/Fence状态机。
31. Coupling与Backend Timing Contract不兼容时启动失败。
32. Resolved Config中`system.profile`、`backends.*.kind`和`simulation.coupling`各只有一个真源。
33. Atomic/RMW在第一版必须显式拒绝。
34. `replay_safe`默认false，没有Qualification Record不得跨配置复用。
35. 外部GPU链路和内部Hybrid-Bond/DRAM带宽必须是不同资源、不同配置键和不同统计项。
36. 一个GPU Parent Request可以对应多个内部Child，但Child地址覆盖、Logical Bytes、Transaction Bytes和完成数必须分别守恒。
37. 全部内部Child完成前禁止生成读响应；响应方向Link未完成前禁止向GPU返回原`mem_fetch`。
38. `write_ack_policy=durable`必须等待全部内部写完成；`posted`必须同时记录GPU-visible和durable完成并在退出前排空。
39. Ramulator2的`DQ/channel_width/rate/nBL/tCK/transaction_bytes`不能导出互相矛盾的峰值带宽。
40. Model 2的PCIe只承载DMA/Page Migration Parent；Model 3 LLC Miss和Model 4 CXL.mem Remote使用各自明确的细粒度Parent语义。

---

## 26. 风险与控制

| 风险 | 等级 | 控制 |
| --- | --- | --- |
| Accel-Sim外部内存桥改动大 | 极高 | 先完成独立GPU资格验证；Adapter隔离；从最小Load/Store Trace开始 |
| GPU与ATLAS重复拥有DRAM时间 | 极高 | TimingOwnershipRegistry；冲突直接拒绝启动 |
| Trace地址直接当物理地址 | 极高 | 强制Manifest、TensorID+offset、PhysicalAddress和DRAM Decode |
| 多时钟破坏因果 | 高 | 全局整数fs；稳定事件序号；禁止直接加Cycle |
| Prefill/Decode Token或KV生命周期错误 | 高 | 两层小模型逐Token黄金轨迹 |
| Continuous Batching规模爆炸 | 高 | Roofline先覆盖；周期模式窗口化；Trace Library和Shape选择 |
| CXL精度表述过度 | 高 | 报告明确区分分析、事务/队列周期、完整协议 |
| 搬运和DRAM流量重复计数 | 中高 | Transfer拆分源读、链路、目的写并分别守恒 |
| Trace错误复用 | 中高 | 分层Cache Key和replay_safe |
| 算子频繁切换导致KV抖动 | 中高 | KV Affinity、最小驻留、Hysteresis |
| GB/s、GiB/s、Gbps混用 | 中 | 配置解析统一为B/s |
| 外部/内部带宽被合并或HBM3组织参数不自洽 | 极高 | 分层Schema；启动时交叉计算峰值带宽和事务粒度，不一致直接失败 |
| Parent在部分Child完成后提前返回GPU | 极高 | Parent Table完成屏障；Response Link只接受fully-joined Parent；守恒与非对齐黄金用例 |
| 每个GPU Partition独立注入导致绕过外部带宽 | 高 | 所有Partition连接Singleton LogicDieMemoryGateway并共享Ingress Credit/仲裁 |
| Roofline与周期结果不同 | 中 | 不要求延迟相同；验证逻辑工作量和数据量 |

---

## 27. 第一版完成定义

第一版达到“研究可用”必须同时满足：

1. 四种Profile运行同一端到端Prefill/Decode工作负载；
2. Roofline覆盖全部Profile；
3. Accel-Sim完成GPU独立路径；
4. Model 3达到具有独立外部Link、Logic Die Gateway、内部Child事务、单Ramulator2和响应Link的共享3D-DRAM请求级周期耦合；
5. Model 2/4具有有界队列、带宽、延迟和背压模型；
6. 支持Static Ragged Batch、Paged KV和Continuous Batching；
7. 支持手工和规则化GPU/ATLAS算子放置；
8. 所有逻辑工作量、地址、传输和请求满足守恒；
9. 所有结果由Resolved Config、版本、Trace、地址清单和Seed复现；
10. 报告明确区分analytical、event-modeled、cycle-coupled和extrapolated。

---

## 28. 后续实现工作协议

每次实现任务开始前：

1. 阅读本文档相关章节；
2. 指明当前实现阶段和子阶段；
3. 列出本次涉及的规范性不变量；
4. 检查当前工作树，保留用户已有修改；
5. 先增加最小单元测试或黄金用例；
6. 实现后运行对应阶段验收；
7. 记录Resolved Config、版本和输出；
8. 在结果中标记Fidelity；
9. 如果实现需要改变本文档，先更新本文档和变更记录。

代码评审至少回答：

- 该组件拥有哪部分时间？
- 是否和其他组件重复计算？
- 该Tensor当前版本位于哪个Memory Space？
- 跨设备边被Lowering成什么？
- Trace是否可复用？
- Batch Shape和KV Length是否进入Artifact Key？
- B=1和最小黄金用例是否通过？
- 四种Profile是否保持同一逻辑工作量？

---

## 29. 变更记录

| 版本 | 日期 | 变更 |
| --- | --- | --- |
| 1.0 | 2026-08-26 | 固化四种拓扑、两级IR、端到端Prefill/Decode、算子放置、多Batch、地址体系、Backend契约、实现阶段和验收标准 |
| 1.1 | 2026-08-26 | 冻结Python/C++边界、执行接口、唯一配置Schema、Trace/Simulation地址分离、非一致性状态机、确定性Batch调度、Cache DAG和精确黄金用例 |
| 1.2 | 2026-08-26 | 增加已有KV的显式decode_step微基准语义；固化参考full_runtime、动态地址复用、有界PCIe/CXL、共享3D内存参考服务和外部Memory Bridge与目标Backend资格验证的边界 |
| 1.3 | 2026-08-27 | 固化Model 3 GPU-only无竞争基线：全部算子位于GPU、Logic Die Backend关闭、共享3D-DRAM仅允许GPU请求，并强制派生任务、路由、请求和守恒验收；双发起方竞争作为后续独立模式开启 |
| 1.4 | 2026-08-27 | 固化GPU外部带宽与Logic Die/3D-DRAM内部带宽分离；新增双向外部Link、LogicDieMemoryGateway、Parent/Child拆分汇聚、durable完成、多时钟域、带宽一致性校验和拓扑特定请求语义；明确当前单请求Bridge仅是最小资格原型 |
