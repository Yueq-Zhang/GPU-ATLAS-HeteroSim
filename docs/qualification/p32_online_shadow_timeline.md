# P32 在线统一Shadow时间线资格

P32以P30稳定Simulation Key连接真实Hugging Face请求、P31 GPU Artifact和P31 ATLAS Artifact，并要求P33双侧周期资格存在。时间线使用唯一fs时间所有者，显式记录请求到达、Decode开始、两分支启动与durable完成、Observation Barrier、版本提交和请求完成。

GPU Artifact是已经包含QKV的整层参考包络；ATLAS Artifact是QKV卸载候选。因此二者是非累加分支，Barrier取两者完成时间的最大值。P32验证依赖时间单调、分支资源占用、GPU/ATLAS Global PA不重叠、所有请求durable后才提交版本，以及版本提交后才结束请求。

机器记录位于`validation/p32/qualification_record.json`。该记录允许声明固定请求的在线Shadow因果闭环，但不允许把两分支周期相加或发布混合放置makespan；后者需要重新捕获排除QKV后的GPU剩余层Trace。
