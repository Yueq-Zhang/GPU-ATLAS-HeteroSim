# P28：推理框架离线导出与Shadow资格

P28实现三类版本化Manifest：模型、请求和执行身份。Simulation身份覆盖框架/适配器版本、模型与Tokenizer Revision、生成参数、系统Profile和Placement哈希。固定输入双遍导出同一计算图、执行任务、Tensor身份和Global PA。

GPU侧只允许精确命中P23 Ready Catalog，不允许从Batch、Context、`q_len`、KV Length、模型Revision或算子身份相邻项外推。ATLAS侧当前为投影类Tensor IR到GEMM OperatorDescription、Tile和Compute-Die Core分片的Lowering合同；尚未执行完整ATLAS Bundle或生成经Ramulator2资格的全量内部Trace。

vLLM适配合同归一化Request Arrival、连续/不规则Batch、Decode/Prefill Step、Paged-KV分配释放及页到Global PA的绑定；TensorRT-LLM合同封存Engine、Optimization Profile、Tactic和Plugin身份。Shadow记录按Observation ID一一连接框架事件和仿真结果，并禁止仿真器修改框架Token或调度。

`validation/p28/framework_shadow/qualification_record.json`的离线双遍通过。P29已在两台GPU上独立验证真实Hugging Face、vLLM与TensorRT-LLM请求，但尚未把真实Scheduler、Block Table、分配器或Engine事件输入本合同。因此P28记录中的在线执行字段仍保持`false`，端到端仿真与性能资格均未获得，`performance_claim_allowed=false`。
