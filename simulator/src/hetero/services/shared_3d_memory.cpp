#include "hetero/services/shared_3d_memory.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace heterosim::services {
namespace {

TimeFs checked_add(TimeFs lhs, TimeFs rhs) {
    if (rhs > std::numeric_limits<TimeFs>::max() - lhs) {
        throw std::overflow_error("shared memory time exceeds TimeFs");
    }
    return lhs + rhs;
}

}  // namespace

Shared3DMemoryModel::Shared3DMemoryModel(Shared3DMemoryConfig config)
    : config_(std::move(config)) {
    if (config_.memory_space_id.empty() || config_.initiator_order.empty() ||
        config_.channel_count == 0 || config_.banks_per_channel == 0 ||
        config_.transaction_bytes == 0 ||
        config_.queue_depth_per_initiator == 0 || config_.fixed_latency_fs == 0) {
        throw std::invalid_argument("shared 3D memory configuration is incomplete");
    }
    auto order = config_.initiator_order;
    std::sort(order.begin(), order.end());
    if (std::adjacent_find(order.begin(), order.end()) != order.end() ||
        std::any_of(order.begin(), order.end(), [](const auto& id) { return id.empty(); })) {
        throw std::invalid_argument("memory initiators must be unique and non-empty");
    }
}

DramAddress Shared3DMemoryModel::decode(const PhysicalAddress& address) const {
    if (address.memory_space_id != config_.memory_space_id) {
        throw std::invalid_argument("physical address belongs to another memory space");
    }
    const auto transaction = address.offset_bytes / config_.transaction_bytes;
    const auto channel = transaction % config_.channel_count;
    const auto bank_linear = transaction / config_.channel_count;
    const auto bank = bank_linear % config_.banks_per_channel;
    const auto column = address.offset_bytes % config_.transaction_bytes;
    const auto row = bank_linear / config_.banks_per_channel;
    return DramAddress{channel, bank, row, column};
}

Shared3DMemoryResult Shared3DMemoryModel::run(
    const std::vector<MemoryRequest>& requests) const {
    std::unordered_map<std::string, std::deque<MemoryRequest>> queues;
    for (const auto& initiator : config_.initiator_order) {
        queues.emplace(initiator, std::deque<MemoryRequest>{});
    }
    Shared3DMemoryResult result;
    result.requests_by_channel.resize(config_.channel_count, 0);
    std::unordered_set<std::uint64_t> request_ids;
    for (const auto& request : requests) {
        if (request.physical_address.memory_space_id != config_.memory_space_id ||
            request.size_bytes == 0 || queues.count(request.initiator_id) == 0) {
            throw std::invalid_argument("invalid shared 3D memory request");
        }
        if (!request_ids.emplace(request.request_id).second) {
            throw std::invalid_argument("shared 3D memory request id must be unique");
        }
        auto& queue = queues.at(request.initiator_id);
        if (queue.size() >= config_.queue_depth_per_initiator) {
            ++result.backpressure_events;
        }
        queue.push_back(request);
        ++result.parent_requests_submitted;
        result.submitted_bytes += request.size_bytes;
        ++result.requests_by_initiator[request.initiator_id];
    }
    for (auto& [initiator, queue] : queues) {
        static_cast<void>(initiator);
        std::stable_sort(queue.begin(), queue.end(), [](const auto& lhs, const auto& rhs) {
            return std::tie(lhs.issue_time_fs, lhs.sequence_number, lhs.request_id) <
                   std::tie(rhs.issue_time_fs, rhs.sequence_number, rhs.request_id);
        });
    }

    std::vector<TimeFs> channel_ready(config_.channel_count, 0);
    std::vector<std::vector<TimeFs>> bank_ready(
        config_.channel_count,
        std::vector<TimeFs>(config_.banks_per_channel, 0));
    std::unordered_map<std::uint64_t, TimeFs> parent_completion;
    std::unordered_map<std::uint64_t, MemoryRequest> parents;
    std::size_t round_robin = 0;
    TimeFs arbitration_time = 0;
    std::uint64_t next_child_id = 1;
    std::size_t remaining = requests.size();

    while (remaining != 0) {
        bool selected = false;
        for (std::size_t offset = 0; offset < config_.initiator_order.size(); ++offset) {
            const auto index = (round_robin + offset) % config_.initiator_order.size();
            auto& queue = queues.at(config_.initiator_order.at(index));
            if (queue.empty() || queue.front().issue_time_fs > arbitration_time) {
                continue;
            }
            const auto parent = queue.front();
            queue.pop_front();
            parents.emplace(parent.request_id, parent);
            std::uint64_t child_offset = 0;
            while (child_offset < parent.size_bytes) {
                const auto size = std::min(
                    config_.transaction_bytes, parent.size_bytes - child_offset);
                const PhysicalAddress child_address{
                    parent.physical_address.memory_space_id,
                    parent.physical_address.offset_bytes + child_offset,
                    parent.physical_address.allocation_epoch};
                const auto dram = decode(child_address);
                const auto start = std::max(
                    {arbitration_time, parent.issue_time_fs,
                     channel_ready.at(dram.channel),
                     bank_ready.at(dram.channel).at(dram.bank)});
                const auto completion = checked_add(start, config_.fixed_latency_fs);
                channel_ready.at(dram.channel) = checked_add(
                    start, config_.channel_injection_interval_fs);
                bank_ready.at(dram.channel).at(dram.bank) = checked_add(
                    start, config_.bank_busy_time_fs);
                result.child_records.push_back(Shared3DChildRecord{
                    next_child_id++, parent.request_id, parent.initiator_id,
                    child_address, dram, size, parent.issue_time_fs, start, completion});
                ++result.child_requests_submitted;
                ++result.child_requests_completed;
                ++result.requests_by_channel.at(dram.channel);
                result.completed_bytes += size;
                parent_completion[parent.request_id] = std::max(
                    parent_completion[parent.request_id], completion);
                result.last_completion_fs = std::max(
                    result.last_completion_fs, completion);
                child_offset += size;
            }
            arbitration_time = checked_add(
                arbitration_time, config_.channel_injection_interval_fs);
            round_robin = (index + 1) % config_.initiator_order.size();
            --remaining;
            selected = true;
            break;
        }
        if (!selected) {
            TimeFs next_issue = std::numeric_limits<TimeFs>::max();
            for (const auto& [initiator, queue] : queues) {
                static_cast<void>(initiator);
                if (!queue.empty()) {
                    next_issue = std::min(next_issue, queue.front().issue_time_fs);
                }
            }
            if (next_issue == std::numeric_limits<TimeFs>::max()) {
                throw std::logic_error("shared memory pending count mismatch");
            }
            arbitration_time = std::max(arbitration_time, next_issue);
        }
    }

    for (const auto& [request_id, parent] : parents) {
        result.parent_responses.push_back(MemoryResponse{
            request_id,
            parent.parent_task_id,
            parent.physical_address,
            parent.value_id,
            parent.value_version,
            parent_completion.at(request_id),
            CompletionStatus::kSuccess,
            parent.size_bytes});
        ++result.parent_requests_completed;
    }
    std::sort(
        result.parent_responses.begin(), result.parent_responses.end(),
        [](const auto& lhs, const auto& rhs) {
            return std::tie(lhs.completion_time_fs, lhs.request_id) <
                   std::tie(rhs.completion_time_fs, rhs.request_id);
        });
    std::sort(
        result.child_records.begin(), result.child_records.end(),
        [](const auto& lhs, const auto& rhs) {
            return std::tie(lhs.service_start_fs, lhs.child_request_id) <
                   std::tie(rhs.service_start_fs, rhs.child_request_id);
        });
    if (result.parent_requests_submitted != result.parent_requests_completed ||
        result.child_requests_submitted != result.child_requests_completed ||
        result.submitted_bytes != result.completed_bytes) {
        throw std::logic_error("shared 3D memory conservation invariant failed");
    }
    return result;
}

}  // namespace heterosim::services
