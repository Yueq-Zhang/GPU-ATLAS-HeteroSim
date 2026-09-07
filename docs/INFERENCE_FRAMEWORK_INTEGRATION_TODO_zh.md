# 推理框架对接开发清单

状态：**需求已冻结，尚未实现正式框架适配器。**

本文档记录 Hugging Face Transformers、vLLM 与后续 TensorRT-LLM 对接任务。当前仓库中的 PyTorch/Transformers 程序只用于算子实现、原生测量和 NVBit Trace 捕获，不能据此宣称已经完成推理框架接入。

## 1. 冻结的总体语义

推理框架对接由三条路径组成：

1. 框架前端导出模型、请求、算子、动态 Shape、Tensor 生命周期、KV Cache 和调度事件；
2. GPU 算子使用目标 GPU 架构上的实际编译与执行程序捕获 SASS/访存 Trace，并通过 Artifact Catalog 重放；
3. ATLAS 算子从 Tensor IR、分块、布局和地址规划编译出 ATLAS 任务与内存访问 Trace。

框架拥有模型数值语义、权重、Logits 和 Token 生成；GPU-ATLAS-HeteroSim 拥有设备放置、数据移动、Global PA、资源占用和仿真时间。第一版采用 **shadow simulation**：框架完成真实数值执行，仿真器返回时序和资源统计，不把仿真器伪装成 PyTorch 自定义设备，也不由时序模型生成数值结果。

## 2. 分阶段代办事项

### F0：接口与声明边界

- [x] 固化“框架语义、GPU Trace、ATLAS 编译、统一仿真”四段边界；
- [x] 固化首个接入对象为 Hugging Face Transformers，之后接入 vLLM，TensorRT-LLM 放在后续阶段；
- [x] 固化未命中精确 Artifact 时必须失败关闭或显式选择低保真回退；
- [ ] 定义版本化 `FrameworkModelManifest`、`FrameworkRequestManifest` 与 `FrameworkExecutionManifest` Schema；
- [ ] 将框架版本、模型 Revision、Tokenizer Revision、执行参数和适配器版本纳入 Simulation Key。

### F1：Hugging Face Transformers 离线导出器

- [ ] 从 `PretrainedConfig` 和固定 Checkpoint Revision 生成 `ModelSpec`；
- [ ] 从 Prompt、Batch、生成长度和采样参数生成 `RequestSpec`；
- [ ] 记录实际算子顺序、层号、Phase、`q_len`、KV Length、dtype、布局和融合边界；
- [ ] 将框架算子稳定映射为现有 `ModelNode`/`Value`，并验证依赖拓扑与现有规范图一致；
- [ ] 输出可复现的框架执行清单，而不是依赖运行期 Python 对象地址。

验收：固定 TinyLlama-1.1B、FP16、BS=1 用例的模型参数、算子实例数、Shape、依赖和生成步数与框架实际运行一致；两次导出内容哈希一致。

### F2：Tensor、KV Cache 与地址绑定

- [ ] 为 Parameter、Activation、Workspace、KV Cache 和 Metadata 分配稳定 Tensor ID；
- [ ] 记录 View、Alias、In-place、生命周期、版本以及框架分配器产生的 Buffer Range；
- [ ] 建立 `Framework Tensor + offset -> ValueRef -> Global PA` 绑定；
- [ ] 将 vLLM Block Table 或等价 KV Block 元数据映射为 Paged KV 和版本化 Residency；
- [ ] 明确当前 `identity_untranslated`/`range_rebase` 模式，不得把捕获地址无声明地当作最终物理地址；
- [ ] 验证容量、对齐、重叠、越界、释放和复用。

验收：每个 Trace 访存均命中唯一 Value Range；KV Append、Attention 读取、版本提交和释放满足因果关系；最终零未绑定地址和零在途请求。

### F3：GPU 编译、捕获与 Trace Catalog

- [ ] 从框架实际选择的 CUDA Kernel 或融合 Kernel 建立封存执行身份；
- [ ] 在目标 SM 架构上编译/加载并通过 NVBit 捕获非空 SASS 与访存 Trace；
- [ ] Artifact Key 至少包含模型 Revision、算子、精确 Shape、dtype、Kernel/融合实现、编译参数、目标 SM 和执行身份；
- [ ] 把 Batch、Context、`q_len` 和 KV Length 变化纳入 Kernel Dispatch 检测；
- [ ] 只有经验证不改变动态指令、控制流和地址行为时才允许 `replay_safe=true`；
- [ ] 缺失精确 Artifact 时默认拒绝周期级声明；分析回退必须显式配置并携带 Fidelity 标签；
- [ ] 为已缓存 Artifact 提供命中、失配原因和重新捕获建议。

验收：同一 Key 双遍周期、指令、地址绑定和外存统计一致；Key 任一身份字段变化都不会静默复用旧 Trace。

### F4：ATLAS Tensor IR 编译路径

- [ ] 定义框架算子到 ATLAS Tensor IR 的 Lowering 接口；
- [ ] 根据算子 Shape、分块、布局和 Compute-Die 配置生成 StageProgram/任务；
- [ ] 生成可审计的 ATLAS 内部访存请求，并绑定同一 Global PA；
- [ ] 将 Tile Size、Core 映射、Bank/Channel 映射和编译器版本纳入 Artifact Key；
- [ ] 对不支持的算子和 Shape 明确拒绝或显式回退，不使用同名 Artifact 外推。

验收：编译产物的逻辑读写量、任务数、Global PA 范围和 Ramulator2 Parent/Child/durable 计数守恒。

### F5：Shadow Simulation 运行接口

- [ ] 定义框架侧 `request_arrive`、`prefill_begin/end`、`decode_step_begin/end`、`sampling`、`cancel` 和 `request_finish` 回调；
- [ ] 将框架事件转换为仿真 Epoch，并维持唯一时间所有者和确定性事件顺序；
- [ ] 框架产生真实 Token，仿真器产生 TTFT、ITL、吞吐、链路、内存和竞争统计；
- [ ] 区分 Functional Result、Native Measurement 与 Simulated Timing，禁止相互替代；
- [ ] 提供离线清单重放和在线 shadow 两种入口，两者对同一输入生成相同 Simulation Key。

验收：固定输入的 Token 结果来自框架；仿真器不参与数值计算；请求、算子、KV 版本和时间线可一一对应并双遍确定。

### F6：vLLM 调度与多 Batch

- [ ] 接入 Request Scheduler、Continuous Batching、Chunked Prefill 和请求抢占/恢复事件；
- [ ] 对接 PagedAttention 的 Block Table、KV 分配/复用/迁移和容量压力；
- [ ] 捕获真实 Batched/Fused Kernel，禁止复制 BS=1 Trace 或线性缩放延迟；
- [ ] 支持 Ragged Batch、不同 Context/KV Length 和不同终止时刻；
- [ ] 验证跨请求隔离、共享 Kernel 归属、版本提交、取消和资源回收。

验收：多请求总工作量、每请求 Token/KV 状态、共享 Kernel、地址空间与完成回调全部守恒；长时间运行无死锁、活锁和残留请求。

### F7：TensorRT-LLM 与其他后端

- [ ] 封存 Engine、Optimization Profile、Tactic、Plugin 和 CUDA Graph 身份；
- [ ] 把动态 Shape Profile 与实际 Kernel Sequence 纳入 Artifact Key；
- [ ] 对框架融合变化、版本升级和重新构建 Engine 执行强制失效；
- [ ] 在 Hugging Face/vLLM 接口稳定后复用统一 Manifest、Tensor 与事件协议。

### F8：端到端资格与性能门禁

- [ ] 建立至少一个 Hugging Face Prefill+多 Token Decode 黄金用例；
- [ ] 建立至少一个 vLLM Continuous/Ragged Batch 黄金用例；
- [ ] 对四种系统 Profile 分别验证同一逻辑工作量和不同数据移动语义；
- [ ] 对 GPU、ATLAS、Copy Engine、Runtime、外部 Link、Logic-Die Gateway 和 3D-DRAM 分组件校准；
- [ ] 在全部执行任务命中合格 Artifact 且六组件门禁通过前保持 `performance_claim_allowed=false`。

## 3. 推荐实施顺序

`F0 -> F1 -> F2 -> F3/F4 -> F5 -> F6 -> F7 -> F8`。

F3 与 F4 可以并行开发，但都必须依赖 F1/F2 生成的稳定算子和 Tensor 身份。首个可用里程碑是 Hugging Face 离线导出加 shadow simulation；vLLM 必须在该接口稳定后再接入动态 Batch 和 Paged KV。

## 4. 当前不可宣称的能力

- 不能宣称任意 Hugging Face/vLLM/TensorRT-LLM 模型可以直接导入；
- 不能宣称框架执行过程中所有 GPU Trace 会自动捕获和复用；
- 不能宣称任意动态 Shape 可复用现有固定 Shape Artifact；
- 不能宣称仿真器返回真实 Logits 或 Token；
- 不能宣称当前 P20 未校准周期是框架端到端性能。
