# P33 双侧完整周期回放资格

P33对P31的两个Artifact分别执行两个隔离Leg。GPU Leg在一个Accel-Sim进程内连续回放40个Kernel并由唯一Ramulator2维护外存状态；ATLAS Leg把完整338,080请求流送入一个ATLAS发起方可见的Ramulator2实例。两个Leg之间不复用模拟器状态。

GPU两遍均为15,859,267 cycles、371,846,682 instructions和14,009,953,180,213 fs；2,756,823个Parent全部完成且durable，2,758,278个Child全部完成，575,486,980次地址转换零漏配，退出时零在途。

ATLAS两遍均处理337,920次读、160次写和21,637,120 B逻辑流量；首请求在cycle 30完成，末请求在cycle 1,205,772完成。338,080个Parent/Child全部完成且durable，GPU发起请求为0，唯一Ramulator2，退出时零在途。两遍Trace、请求流、完成流和全部统计完全一致。

机器记录位于`validation/p33/qualification_record.json`，ATLAS双遍摘要位于`validation/p33/atlas/`。该资格证明请求周期确定性和守恒，不是目标硬件性能校准；GPU与ATLAS也不是在同一实例中并发竞争。
