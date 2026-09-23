# P29：双主机真实推理框架运行时资格

P29只验证外部运行条件，不改变P28离线Shadow资格。三套运行时分别位于独立`uv`环境，避免覆盖主工程和Accel-Sim固定的CUDA 11.8构建/捕获依赖。

| 主机 | GPU | Hugging Face | vLLM | TensorRT-LLM |
|---|---|---|---|---|
| 本地Ubuntu-22.04 WSL | RTX 3070 / SM86 | 16-token Prefill + 1 Decode | 真实单请求，WSL V1 Runner | LLM API真实单请求，PyTorch backend |
| 远端Ubuntu 22.04 | RTX 4090 / SM89 | 16-token Prefill + 1 Decode | 真实单请求，默认V2 Runner | LLM API真实单请求，PyTorch backend |

固定依赖为Transformers 5.16.1、vLLM 0.29.0和TensorRT-LLM 1.2.1，模型为`TinyLlama/TinyLlama-1.1B-Chat-v1.0` Revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`。远端Hugging Face网络不稳定时复用已校验权重缓存；模型权重可共享，但SM86与SM89的SASS、Binary身份和资格记录不可共享。远端缺少系统MPI且无sudo权限，因此使用隔离的Ubuntu Open MPI运行库。

机器证据位于`validation/p29/framework_runtimes`，包含每台主机的三份依赖Freeze、三份导入/CUDA Probe、首轮真实请求和安装后复测记录。运行：

```bash
python3 scripts/qualify_p29_framework_runtimes.py
python3 scripts/qualify_p29_framework_live_replay.py
```

复测结果为两台主机、三套框架共六项全部通过。同一主机两轮的设备/软件/模型/工作负载身份相同；Hugging Face输入、Prefill/Decode Logits哈希和Next Token相同；vLLM与TensorRT-LLM输出Token哈希相同。两种GPU的最终语义输出一致，但SM86与SM89的Logits哈希不要求跨架构逐位相等。

只有安装、CUDA执行、真实框架单请求和同主机确定性复测允许为`true`。在线P28事件适配、自动GPU Trace捕获、框架—仿真器端到端耦合以及性能声明必须保持`false`。记录的冷启动与请求时间仅用于流程观察，不参与通过判据，也不是校准性能结果。
