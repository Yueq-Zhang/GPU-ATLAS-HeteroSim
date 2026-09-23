# P26：P23真实Trace时间线请求生命周期资格

P26把P24请求终止、容量和地址生命周期嵌入已封存的P23固定TinyLlama Layer-0、FP16、BS=2、Context=16、`q_len=1`、KV=17时间线。每个Epoch均重新校验P23 Seal，不改变其中14类算子、15个GPU Trace实例和KV Append的周期与请求证据。

取消只能在`ALL_REQUESTS_DURABLE` Token Barrier生效。活跃请求先完成该Epoch全部请求和版本提交，再终止并释放KV；等待请求取消不会获得Global PA。Retire按`release-before-allocate`执行，确定性first-fit分配器允许后续请求复用已释放区间。

`validation/p26/p23_request_controls/qualification_record.json`记录三组精确P23 Epoch的双遍结果：8个输入请求中6个Admission、2个等待取消；20,588,067个Parent与20,591,280个Child全部持久完成，最终零在途。附加4096轮容量压力探针完成8192次分配与释放、8190次地址复用，零泄漏、零重叠。

该压力探针只资格生命周期和分配器逻辑。它不含KV=18及以上的真实GPU Trace；同一请求跨多个Decode步仍必须逐Shape重新捕获和资格，`performance_claim_allowed=false`。
