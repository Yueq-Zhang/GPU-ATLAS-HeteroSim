# P30 Hugging Face在线观测资格

## 资格对象

- 执行主机：远端RTX 4090，SM89；
- 模型：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`，Revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`；
- 工作负载：FP16、BS=1、16 Token Prefill和1 Token Decode；
- 入口：`scripts/run_p30_huggingface_online.py`；
- 资格器：`scripts/qualify_p30_huggingface_online.py`。

## 已验证内容

真实Transformers请求通过模块前后Hook输出层号、Phase、Shape、dtype、Tensor/View/Alias、CUDA Storage范围、生命周期和数值结果。两遍运行均包含67个事件、30个模块和96个Tensor绑定，Token及Logits哈希一致。CUDA分配器允许两遍出现不同Workspace/Storage组合，因此临时Storage地址、范围和内存图哈希只作为本次观测证据；稳定语义身份排除这些易变字段，并生成相同Simulation Key。

该资格证明P28的离线身份已与一个真实Hugging Face请求接通。它不证明GPU SASS已经生成，不提供vLLM或TensorRT-LLM在线回调，也不允许性能声明。

## 远端复现

```bash
export P30_P31_PHASE=p30
bash scripts/run_p30_p31_remote.sh
```

正式记录位于`validation/p30/huggingface_online/qualification_record.json`。状态必须为`passed`，且`semantic_identity_equal`、`simulation_key_equal`和`semantic_result_equal`均为`true`；`performance_claim_allowed`必须为`false`。
