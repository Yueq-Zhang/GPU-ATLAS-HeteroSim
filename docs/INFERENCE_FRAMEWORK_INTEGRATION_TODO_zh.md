# 推理框架对接开发清单

状态：**P30/P31已完成固定Hugging Face在线观测与双侧Artifact；P32/P33已完成非累加Shadow时间线和双侧独立周期资格；P34已接入真实vLLM与TensorRT-LLM在线事件。性能校准、真实融合Batch Trace和序列化TensorRT Engine仍待完成。**

本文档记录 Hugging Face Transformers、vLLM 与 TensorRT-LLM 对接任务。P28实现版本化离线合同，P29建立真实框架环境，P30/P31生成固定TinyLlama在线身份与GPU/ATLAS Artifact，P32/P33补齐统一Shadow因果和双侧周期回放，P34观测真实调度器、Block Table和Engine/Profile。当前范围仍是固定用例的Shadow资格，不是任意模型导入或校准性能。

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
- [x] 定义版本化 `FrameworkModelManifest`、`FrameworkRequestManifest` 与 `FrameworkExecutionManifest` Schema；
- [x] 将框架版本、模型 Revision、Tokenizer Revision、执行参数和适配器版本纳入导出身份哈希；P30已为固定HF请求完成正式Simulation Key接线，并把易变分配器观测从稳定语义身份中分离。

### F1：Hugging Face Transformers 离线导出器

- [x] 从 `PretrainedConfig` 字典和固定 Checkpoint Revision 生成 `ModelSpec`，并提供安装Transformers时的真实`AutoConfig`入口；
- [x] 从固定请求清单、Batch、生成长度和采样参数生成 `RequestSpec` Manifest；
- [x] P30为固定TinyLlama请求记录实际模块顺序、层号、Phase、`q_len`、KV Length、dtype、布局和Hook可见融合边界；其他模型/Shape仍需新观测；
- [x] 将离线框架请求稳定映射为现有 `ModelNode`/`Value`，并用P30固定请求的实际模块序列复核；更深层框架融合仍按精确身份处理；
- [x] 输出可复现的框架执行清单，不依赖运行期 Python 对象地址。

验收：固定 TinyLlama-1.1B、FP16、BS=1 用例的模型参数、算子实例数、Shape、依赖和生成步数与框架实际运行一致；两次导出内容哈希一致。

### F2：Tensor、KV Cache 与地址绑定

- [x] 为规范图中的Parameter、Activation、Workspace、KV Cache和Metadata分配稳定Value/Tensor身份；
- [x] P30记录固定HF请求的View、Alias、生命周期和CUDA分配器Buffer Range；分配器Workspace可跨运行变化并作为观测证据单独哈希；
- [x] 建立`Framework Tensor/Value -> Global PA`绑定；P30/P31已把真实分配器`Tensor + offset`/View/Alias接到Range-Rebase；
- [x] 将vLLM真实SchedulerOutput Block Table映射为Paged KV、Owner和Global PA，并双遍验证释放后零Live Page；
- [x] 复用当前`identity_untranslated`/`range_rebase`门禁，不把捕获地址无声明地当作最终物理地址；
- [x] 离线导出验证容量、对齐和重叠；P30验证HF分配器Alias/复用，P34验证真实vLLM Paged-KV Owner、非别名和释放。

验收：每个 Trace 访存均命中唯一 Value Range；KV Append、Attention 读取、版本提交和释放满足因果关系；最终零未绑定地址和零在途请求。

### F3：GPU 编译、捕获与 Trace Catalog

- [x] P31为固定HF Layer-0 Decode实际选择的40个CUDA Kernel建立封存执行身份；
- [x] P31在远端RTX 4090/SM89加载并通过NVBit捕获非空SASS与访存Trace，同时保留实际SM86 Binary Version及SM86回放目标；
- [x] 精确解析既有P23 Ready Catalog中的模型Revision、算子、固定Shape和执行Artifact身份；
- [x] 把Batch、Context、`q_len`和KV Length变化纳入查找门禁，任一失配均拒绝；
- [ ] 只有经验证不改变动态指令、控制流和地址行为时才允许 `replay_safe=true`；
- [x] 缺失精确Artifact时默认拒绝周期级声明；不静默选择分析回退；
- [x] 返回逐算子精确命中，失配明确报告无精确Artifact；P31提供远端固定HF请求的一键实际Kernel捕获入口，其他模型/Shape尚未自动扩展。

验收：同一 Key 双遍周期、指令、地址绑定和外存统计一致；Key 任一身份字段变化都不会静默复用旧 Trace。

### F4：ATLAS Tensor IR 编译路径

- [x] 定义投影类框架算子到ATLAS Tensor IR的Lowering接口；
- [x] P31根据固定QKV Projection Shape、分块、布局和16个Logic Core生成StageProgram/任务；
- [x] P31生成338,080条可审计ATLAS内部访存请求，并绑定Activation/Weight/Output的Global PA；
- [x] 将Tile Size、Core映射和编译器版本纳入正式可执行Artifact Key；P33把完整338,080请求Trace双遍送入唯一Ramulator2，Bank/Channel仍由候选Mapper决定；
- [x] 对不支持的算子、非整除Tile和Shape明确拒绝，不使用同名Artifact外推。

验收：编译产物的逻辑读写量、任务数、Global PA 范围和 Ramulator2 Parent/Child/durable 计数守恒。

### F5：Shadow Simulation 运行接口

- [x] P32定义固定Decode用例的`request_arrive`、`decode_phase_begin`、Artifact begin/durable、Barrier、版本提交和`request_finish`事件；通用Prefill/Sampling/Cancel回调仍待扩展；
- [x] 将固定框架事件转换为统一fs时间线，并维持唯一时间所有者和确定性事件顺序；
- [ ] 框架产生真实 Token，仿真器产生 TTFT、ITL、吞吐、链路、内存和竞争统计；
- [x] 区分 Functional Result、Native Measurement 与 Simulated Timing，禁止相互替代；
- [x] P32以同一Simulation Key连接在线HF请求与双侧周期Artifact，且结果不反馈修改Token或Scheduler。

验收：固定输入的 Token 结果来自框架；仿真器不参与数值计算；请求、算子、KV 版本和时间线可一一对应并双遍确定。

### F6：vLLM 调度与多 Batch

- [x] P34接入真实Request Scheduler并观测两请求Continuous/Ragged Batching；Chunked Prefill、抢占/恢复仍待实现；
- [x] P34接入真实Paged-KV Block Table、Owner、Global PA和释放；迁移与容量压力回调待实现；
- [ ] 捕获真实 Batched/Fused Kernel，禁止复制 BS=1 Trace 或线性缩放延迟；
- [x] P34真实观察Continuous/Ragged Batch和不同Token Count，并以框架输出确定请求完成；
- [x] 验证双请求Block隔离和资源回收；共享Kernel归属、取消和抢占仍待接入。

验收：多请求总工作量、每请求 Token/KV 状态、共享 Kernel、地址空间与完成回调全部守恒；长时间运行无死锁、活锁和残留请求。

### F7：TensorRT-LLM 与其他后端

- [x] 实现Engine、Optimization Profile、Tactic和Plugin离线身份；P34双遍观测真实LLM Engine、Profile和SimpleScheduler；序列化Engine、Tactic和CUDA Graph仍待资格；
- [ ] 把动态 Shape Profile 与实际 Kernel Sequence 纳入 Artifact Key；
- [ ] 对框架融合变化、版本升级和重新构建 Engine 执行强制失效；
- [x] 真实TensorRT-LLM LLM API PyTorch backend已验证并复用统一事件协议；不得外推为序列化TensorRT backend。

### F8：端到端资格与性能门禁

- [ ] 建立至少一个 Hugging Face Prefill+多 Token Decode 黄金用例；
- [ ] 建立至少一个 vLLM Continuous/Ragged Batch 黄金用例；
- [ ] 对四种系统 Profile 分别验证同一逻辑工作量和不同数据移动语义；
- [ ] 对 GPU、ATLAS、Copy Engine、Runtime、外部 Link、Logic-Die Gateway 和 3D-DRAM 分组件校准；
- [x] 在全部执行任务命中合格Artifact且六组件门禁通过前保持`performance_claim_allowed=false`。

## 3. P29运行环境基线

- [x] 三套框架分别使用独立`uv`虚拟环境，不修改Accel-Sim CUDA 11.8构建/捕获环境；
- [x] 本地RTX 3070/SM86与远端RTX 4090/SM89均通过包导入和CUDA Tensor探测；
- [x] 两端均完成Transformers固定16-token Prefill和单步Decode；
- [x] 两端均完成vLLM真实引擎单请求；WSL因UVA限制使用V1 Runner，原生Linux使用默认V2 Runner；
- [x] 两端均完成TensorRT-LLM LLM API单请求；当前为PyTorch backend，不等同于已封存TensorRT序列化Engine/Profile/Tactic；
- [x] 固定版本、Checkpoint Revision、依赖Freeze、设备身份和输出哈希进入`validation/p29/framework_runtimes`；
- [x] P30将固定Hugging Face真实运行时事件转换为版本化Manifest和Simulation Key；P34已接入真实vLLM/TensorRT-LLM在线事件；
- [x] P30–P33建立固定HF在线回调、GPU Trace、ATLAS Lowering、双遍周期回放和统一Shadow时间线。

## 4. P30/P31在线基线

- [x] 所有测试、SASS读取和Trace生成固定在远端RTX 4090/SM89执行，本地仅保存代码和轻量资格记录；
- [x] P30双遍真实HF Prefill+Decode的67个事件、30个模块、96个Tensor绑定、数值结果和Simulation Key一致；
- [x] 允许CUDA分配器的50/52个Storage与Alias组发生合法变化，不把原始地址放入稳定身份；
- [x] P31仅在真实Layer-0 Decode区间开启NVBit，生成40-Kernel完整Trace与四Range Global PA绑定；
- [x] GPU 40-Kernel Trace双遍Accel-Sim+唯一Ramulator2一致：15,859,267 cycles、371,846,682 instructions、2,756,823个durable Parent、零漏配、零在途；
- [x] ATLAS QKV 338,080请求双遍完整Ramulator2周期回放一致，各为1,205,772 GPU cycles，Parent/Child/durable守恒、零GPU请求、零在途；
- [x] P32将GPU整层参考包络与ATLAS QKV候选以非累加分支嵌入同一在线Shadow时间线并验证请求/Global PA/版本因果；该时间线不是混合放置makespan。

## 5. 推荐实施顺序

P30–P34已完成固定HF在线观测、双侧Artifact、独立周期双跑、非累加Shadow时间线及首个vLLM/TensorRT-LLM真实事件接入。下一步进入F8性能校准，并并行补齐真实Batched/Fused Kernel、vLLM抢占/恢复和序列化TensorRT Engine资格。

新增模型、Shape和融合边界时，F3与F4仍须共同依赖F1/F2生成的稳定算子/Tensor身份。vLLM必须复用该接口，不得用BS=1 Trace线性外推动态Batch或Paged KV。

## 6. 当前不可宣称的能力

- 不能宣称任意 Hugging Face/vLLM/TensorRT-LLM 模型可以直接导入；
- 不能把P31固定Layer-0 Decode捕获外推为任意框架请求均会自动捕获和复用；
- 不能宣称任意动态 Shape 可复用现有固定 Shape Artifact；
- 不能宣称仿真器返回真实 Logits 或 Token；
- 不能宣称当前 P20 未校准周期是框架端到端性能。
- 不能把P34的TensorRT-LLM PyTorch backend表述为序列化TensorRT Engine/Tactic/CUDA Graph资格。
- 不能把P33两个独立周期分支相加为GPU+ATLAS混合放置时间；GPU整层Trace已经包含QKV。
