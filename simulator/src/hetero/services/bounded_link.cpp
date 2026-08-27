#include "hetero/services/bounded_link.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "hetero/services/fixed_latency_memory.h"

namespace heterosim::services {

BoundedLinkModel::BoundedLinkModel(BoundedLinkConfig config)
    : config_(std::move(config)) {
    if (config_.route_id.empty() ||
        config_.wire_bandwidth_bytes_per_second == 0 ||
        config_.queue_depth_transactions == 0 || config_.credits == 0) {
        throw std::invalid_argument("bounded link configuration is incomplete");
    }
}

BoundedLinkResult BoundedLinkModel::run(
    const std::vector<LinkTransaction>& transactions) const {
    std::vector<LinkTransaction> ordered = transactions;
    std::stable_sort(
        ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) {
            return std::tie(lhs.issue_time_fs, lhs.transaction_id) <
                   std::tie(rhs.issue_time_fs, rhs.transaction_id);
        });
    std::unordered_map<std::string, TimeFs> serialization_ready;
    using Completion = std::pair<TimeFs, std::uint64_t>;
    std::priority_queue<Completion, std::vector<Completion>, std::greater<Completion>>
        outstanding;
    BoundedLinkResult result;
    const auto capacity = std::min(
        config_.queue_depth_transactions, config_.credits);

    for (const auto& transaction : ordered) {
        if (transaction.route_id != config_.route_id ||
            transaction.source_id.empty() || transaction.destination_id.empty() ||
            transaction.payload_bytes == 0) {
            throw std::invalid_argument("invalid transaction for bounded link");
        }
        TimeFs admitted = transaction.issue_time_fs;
        while (!outstanding.empty() && outstanding.top().first <= admitted) {
            outstanding.pop();
        }
        if (outstanding.size() >= capacity) {
            admitted = outstanding.top().first;
            outstanding.pop();
            ++result.backpressure_events;
            while (!outstanding.empty() && outstanding.top().first <= admitted) {
                outstanding.pop();
            }
        }
        const auto direction = config_.full_duplex
                                   ? transaction.source_id + "->" + transaction.destination_id
                                   : std::string("shared");
        const auto start = std::max(admitted, serialization_ready[direction]);
        const auto serialization_done = ideal_link_completion_fs(
            start, 0, transaction.payload_bytes, transaction.header_bytes,
            config_.wire_bandwidth_bytes_per_second);
        serialization_ready[direction] = serialization_done;
        if (serialization_done >
            std::numeric_limits<TimeFs>::max() - config_.propagation_latency_fs) {
            throw std::overflow_error("bounded link completion exceeds TimeFs");
        }
        const auto completion = serialization_done + config_.propagation_latency_fs;
        outstanding.emplace(completion, transaction.transaction_id);
        const auto wire_bytes = transaction.payload_bytes + transaction.header_bytes;
        result.responses.push_back(LinkResponse{
            transaction.transaction_id,
            transaction.parent_task_id,
            transaction.route_id,
            completion,
            CompletionStatus::kSuccess,
            transaction.payload_bytes,
            wire_bytes});
        ++result.submitted_transactions;
        ++result.completed_transactions;
        result.payload_bytes += transaction.payload_bytes;
        result.wire_bytes += wire_bytes;
        result.last_completion_fs = std::max(result.last_completion_fs, completion);
    }
    std::sort(result.responses.begin(), result.responses.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.completion_time_fs, lhs.transaction_id) <
               std::tie(rhs.completion_time_fs, rhs.transaction_id);
    });
    return result;
}

}  // namespace heterosim::services
