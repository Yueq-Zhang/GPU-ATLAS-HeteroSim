#include "hetero/services/fixed_latency_memory.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace heterosim::services {

FixedLatencyMemoryService::FixedLatencyMemoryService(
    std::vector<std::string> initiator_order,
    TimeFs fixed_latency_fs,
    TimeFs injection_interval_fs)
    : initiator_order_(std::move(initiator_order)),
      fixed_latency_fs_(fixed_latency_fs),
      injection_interval_fs_(injection_interval_fs) {
    if (initiator_order_.empty() || fixed_latency_fs_ == 0) {
        throw std::invalid_argument("initiators and fixed latency are required");
    }
    for (const auto& initiator : initiator_order_) {
        if (initiator.empty() || queues_.count(initiator) != 0) {
            throw std::invalid_argument("initiator order must be unique and non-empty");
        }
        queues_.emplace(initiator, std::deque<MemoryRequest>{});
    }
}

void FixedLatencyMemoryService::submit(const MemoryRequest& request) {
    const auto iterator = queues_.find(request.initiator_id);
    if (iterator == queues_.end()) {
        throw std::invalid_argument("unknown memory initiator: " + request.initiator_id);
    }
    if (request.size_bytes == 0) {
        throw std::invalid_argument("memory request size must be positive");
    }
    iterator->second.push_back(request);
}

std::vector<MemoryResponse> FixedLatencyMemoryService::drain() {
    std::vector<MemoryResponse> responses;
    while (pending() != 0) {
        bool selected = false;
        for (std::size_t offset = 0; offset < initiator_order_.size(); ++offset) {
            const auto index = (round_robin_index_ + offset) % initiator_order_.size();
            auto& queue = queues_.at(initiator_order_[index]);
            if (queue.empty() || queue.front().issue_time_fs > next_slot_fs_) {
                continue;
            }
            const auto request = queue.front();
            queue.pop_front();
            if (next_slot_fs_ >
                std::numeric_limits<TimeFs>::max() - fixed_latency_fs_) {
                throw std::overflow_error("memory completion exceeds TimeFs");
            }
            responses.push_back(MemoryResponse{
                request.request_id,
                request.parent_task_id,
                request.physical_address,
                request.value_id,
                request.value_version,
                next_slot_fs_ + fixed_latency_fs_,
                CompletionStatus::kSuccess,
                request.size_bytes});
            next_slot_fs_ += injection_interval_fs_;
            round_robin_index_ = (index + 1) % initiator_order_.size();
            selected = true;
            break;
        }
        if (!selected) {
            TimeFs next_issue = std::numeric_limits<TimeFs>::max();
            for (const auto& [initiator, queue] : queues_) {
                static_cast<void>(initiator);
                if (!queue.empty()) {
                    next_issue = std::min(next_issue, queue.front().issue_time_fs);
                }
            }
            if (next_issue == std::numeric_limits<TimeFs>::max()) {
                throw std::logic_error("pending count disagrees with queues");
            }
            next_slot_fs_ = std::max(next_slot_fs_, next_issue);
        }
    }
    return responses;
}

std::size_t FixedLatencyMemoryService::pending() const noexcept {
    std::size_t count = 0;
    for (const auto& [initiator, queue] : queues_) {
        static_cast<void>(initiator);
        count += queue.size();
    }
    return count;
}

TimeFs ideal_link_completion_fs(
    TimeFs issue_time_fs,
    TimeFs latency_fs,
    std::uint64_t payload_bytes,
    std::uint64_t header_bytes,
    std::uint64_t wire_bandwidth_bytes_per_second) {
    if (wire_bandwidth_bytes_per_second == 0) {
        throw std::invalid_argument("wire bandwidth must be positive");
    }
    constexpr unsigned __int128 kFsPerSecond = 1000000000000000ULL;
    const auto wire_bytes = static_cast<unsigned __int128>(payload_bytes) + header_bytes;
    const auto numerator = wire_bytes * kFsPerSecond;
    const auto serialization =
        (numerator + wire_bandwidth_bytes_per_second - 1) /
        wire_bandwidth_bytes_per_second;
    const auto total = static_cast<unsigned __int128>(issue_time_fs) + latency_fs +
                       serialization;
    if (total > std::numeric_limits<TimeFs>::max()) {
        throw std::overflow_error("ideal link completion exceeds TimeFs");
    }
    return static_cast<TimeFs>(total);
}

}  // namespace heterosim::services
