#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "hetero/services/interfaces.h"

namespace heterosim::services {

class FixedLatencyMemoryService {
public:
    FixedLatencyMemoryService(
        std::vector<std::string> initiator_order,
        TimeFs fixed_latency_fs,
        TimeFs injection_interval_fs);
    void submit(const MemoryRequest& request);
    std::vector<MemoryResponse> drain();
    [[nodiscard]] std::size_t pending() const noexcept;

private:
    std::vector<std::string> initiator_order_;
    std::unordered_map<std::string, std::deque<MemoryRequest>> queues_;
    TimeFs fixed_latency_fs_{};
    TimeFs injection_interval_fs_{};
    TimeFs next_slot_fs_{};
    std::size_t round_robin_index_{};
};

TimeFs ideal_link_completion_fs(
    TimeFs issue_time_fs,
    TimeFs latency_fs,
    std::uint64_t payload_bytes,
    std::uint64_t header_bytes,
    std::uint64_t wire_bandwidth_bytes_per_second);

}  // namespace heterosim::services
