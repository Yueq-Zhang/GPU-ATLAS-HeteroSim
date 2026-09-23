# P27：BookSim2 + Ramulator2 QoS数据面资格

P27激活隔离构建的ATLAS补丁版BookSim2，并通过稳定C ABI暴露逐周期`step`、Packet提交、完成弹出和Credit统计。网络使用4节点2×2 Mesh：`gpu0`、`atlas0.compute`、`gateway0`和`dram0`。

每个Parent严格经过四段真实NoC Packet：发起方→Gateway、Gateway→DRAM、DRAM持久完成→Gateway、Gateway→发起方。请求抵达DRAM节点后通过`heterosim_ramulator_send_at_gateway_v2`进入唯一Ramulator2实例，避免再次经过桥内GPU外部Link。GPU和ATLAS Initiator身份、Parent/Child计数和持久完成分别保留。

`validation/p27/booksim2_ramulator2_qos/qualification_record.json`双遍哈希一致，两遍均推进95个周期。四个同周期Ready Parent产生16个Packet和56个Flit；Packet、Flit、Credit和Parent全部守恒，网络排队竞争被实际观测，GPU/ATLAS均有请求，唯一Ramulator2关闭时零在途。QoS权重与饥饿上限进入真实Packet注入选择，请求随后竞争同一Ramulator2队列。

该结果证明周期交互和守恒，不证明Mesh、Router、链路、时钟或HBDRAM参数已匹配实体硬件。性能校准完成前必须保持`performance_claim_allowed=false`。
