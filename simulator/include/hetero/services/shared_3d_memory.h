#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "hetero/services/interfaces.h"

namespace heterosim::services {

struct DramAddress {
    std::uint64_t channel{};
    std::uint64_t bank{};
    std::uint64_t row{};
    std::uint64_t column{};
};

struct Shared3DMemoryConfig {
    std::string memory_space_id;
    std::vector<std::string> initiator_order;
    std::uint64_t channel_count{};
    std::uint64_t banks_per_channel{};
    std::uint64_t transaction_bytes{64};
    std::uint64_t queue_depth_per_initiator{64};
    TimeFs fixed_latency_fs{};
    TimeFs channel_injection_interval_fs{};
    TimeFs bank_busy_time_fs{};
};

struct Shared3DChildRecord {
    std::uint64_t child_request_id{};
    std::uint64_t parent_request_id{};
    std::string initiator_id;
    PhysicalAddress physical_address;
    DramAddress dram_address;
    std::uint64_t size_bytes{};
    TimeFs issue_time_fs{};
    TimeFs service_start_fs{};
    TimeFs completion_time_fs{};
};

struct Shared3DMemoryResult {
    std::vector<MemoryResponse> parent_responses;
    std::vector<Shared3DChildRecord> child_records;
    std::uint64_t parent_requests_submitted{};
    std::uint64_t parent_requests_completed{};
    std::uint64_t child_requests_submitted{};
    std::uint64_t child_requests_completed{};
    std::uint64_t submitted_bytes{};
    std::uint64_t completed_bytes{};
    std::uint64_t backpressure_events{};
    std::unordered_map<std::string, std::uint64_t> requests_by_initiator;
    std::vector<std::uint64_t> requests_by_channel;
    TimeFs last_completion_fs{};
};

class Shared3DMemoryModel {
public:
    explicit Shared3DMemoryModel(Shared3DMemoryConfig config);
    [[nodiscard]] DramAddress decode(const PhysicalAddress& address) const;
    [[nodiscard]] Shared3DMemoryResult run(
        const std::vector<MemoryRequest>& requests) const;

private:
    Shared3DMemoryConfig config_;
};

}  // namespace heterosim::services
