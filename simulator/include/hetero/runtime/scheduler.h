#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "hetero/types.h"

namespace heterosim::runtime {

enum class RequestPhase : std::uint8_t { kPrefill, kDecode };

struct RequestInput {
    std::string request_id;
    TimeFs arrival_time_fs{};
    std::uint64_t prompt_length{};
    std::uint64_t output_length{};
    std::int64_t priority{};
    bool decode_step{};
    std::uint64_t initial_kv_length{};
};

struct SchedulerConfig {
    std::uint64_t max_num_sequences{};
    std::uint64_t max_batched_tokens{};
    std::uint64_t prefill_chunk_tokens{};
    std::uint64_t max_prefill_wait_epochs{8};
    TimeFs epoch_duration_fs{};
};

struct SelectionItem {
    std::string request_id;
    RequestPhase phase{RequestPhase::kPrefill};
    std::uint64_t token_begin{};
    std::uint64_t token_count{};
};

struct EpochRecord {
    std::uint64_t epoch_id{};
    TimeFs boundary_time_fs{};
    TimeFs completion_time_fs{};
    std::vector<SelectionItem> selections;
};

struct RequestResult {
    std::string request_id;
    std::uint64_t generated_length{};
    std::uint64_t committed_kv_length{};
    std::vector<TimeFs> token_ready_time_fs;
    TimeFs finish_time_fs{};
};

struct SchedulerResult {
    std::vector<EpochRecord> epochs;
    std::vector<RequestResult> requests;
};

SchedulerResult simulate_token_barrier(
    const std::vector<RequestInput>& requests,
    const SchedulerConfig& config);

}  // namespace heterosim::runtime
