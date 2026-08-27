#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "hetero/services/interfaces.h"

namespace heterosim::services {

struct BoundedLinkConfig {
    std::string route_id;
    std::uint64_t wire_bandwidth_bytes_per_second{};
    TimeFs propagation_latency_fs{};
    std::uint64_t queue_depth_transactions{};
    std::uint64_t credits{};
    bool full_duplex{true};
};

struct BoundedLinkResult {
    std::vector<LinkResponse> responses;
    std::uint64_t submitted_transactions{};
    std::uint64_t completed_transactions{};
    std::uint64_t payload_bytes{};
    std::uint64_t wire_bytes{};
    std::uint64_t backpressure_events{};
    TimeFs last_completion_fs{};
};

class BoundedLinkModel {
public:
    explicit BoundedLinkModel(BoundedLinkConfig config);
    [[nodiscard]] BoundedLinkResult run(
        const std::vector<LinkTransaction>& transactions) const;

private:
    BoundedLinkConfig config_;
};

}  // namespace heterosim::services
