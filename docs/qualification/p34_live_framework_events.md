# P34 真实在线框架事件资格

P34在远端RTX 4090上使用P29隔离环境完成两套真实框架双跑。

- vLLM 0.29.0：实际加载TinyLlama固定Revision，关闭EngineCore多进程以连接真实`AsyncScheduler`回调；两个不等长请求产生3个Scheduler step，记录Paged-KV Block Table并绑定Global PA。每遍请求ID虽含随机后缀，但规范化后的语义、Block归属、Token结果和释放状态一致，结束时零Live Page。
- TensorRT-LLM 1.2.1：TP=1使用进程内Worker，实际观测LLM对象、`max_batch_size=2`等Optimization Profile和7次`SimpleScheduler`回调；两遍Token和规范化事件一致。

机器记录位于`validation/p34/qualification_record.json`，原始轻量事件位于`validation/p34/vllm/`和`validation/p34/tensorrt_llm/`。模型使用P29已锁定的本地缓存，资格过程不依赖在线Hub。当前TensorRT-LLM后端是PyTorch，序列化TensorRT Engine、Tactic、Plugin Kernel和CUDA Graph均未资格；两套框架结果不会被仿真器反馈修改，性能声明保持关闭。
