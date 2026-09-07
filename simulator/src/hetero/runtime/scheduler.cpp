#include "hetero/runtime/scheduler.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace heterosim::runtime {
namespace {

enum class State : std::uint8_t { kWaiting, kPrefillReady, kDecodeReady, kFinished };

struct MutableRequest {
    RequestInput input;
    State state{State::kWaiting};
    std::uint64_t prompt_cursor{};
    std::uint64_t generated_length{};
    std::uint64_t committed_kv_length{};
    std::uint64_t waiting_epochs{};
    std::vector<TimeFs> token_ready_time_fs;
    TimeFs finish_time_fs{};
    std::string termination_reason{"completed"};
};

std::uint64_t effective_output_limit(const RequestInput& request) {
    auto limit = request.output_length;
    if (request.eos_after_generated_tokens != 0) {
        limit = std::min(limit, request.eos_after_generated_tokens);
    }
    if (request.max_output_tokens != 0) {
        limit = std::min(limit, request.max_output_tokens);
    }
    return limit;
}

std::string natural_termination_reason(const RequestInput& request) {
    const auto limit = effective_output_limit(request);
    if (request.eos_after_generated_tokens != 0 &&
        request.eos_after_generated_tokens == limit) {
        return "eos";
    }
    if (request.max_output_tokens != 0 && request.max_output_tokens == limit) {
        return "max_length";
    }
    return "completed";
}

void validate(const std::vector<RequestInput>& requests, const SchedulerConfig& config) {
    if (requests.empty()) {
        throw std::invalid_argument("requests must not be empty");
    }
    if (config.max_num_sequences == 0 || config.max_batched_tokens == 0 ||
        config.prefill_chunk_tokens == 0 || config.epoch_duration_fs == 0) {
        throw std::invalid_argument("scheduler quantities must be positive");
    }
    if (config.prefill_chunk_tokens > config.max_batched_tokens) {
        throw std::invalid_argument("prefill chunk exceeds token budget");
    }
    std::vector<std::string> ids;
    for (const auto& request : requests) {
        if (request.request_id.empty() || request.prompt_length == 0 ||
            request.output_length == 0 || request.kv_reservation_bytes == 0) {
            throw std::invalid_argument("invalid request input");
        }
        if (request.decode_step && request.initial_kv_length == 0) {
            throw std::invalid_argument("decode-step request requires initial KV length");
        }
        if (request.cancel_time_fs != std::numeric_limits<TimeFs>::max() &&
            request.cancel_time_fs < request.arrival_time_fs) {
            throw std::invalid_argument("request cancellation precedes arrival");
        }
        ids.push_back(request.request_id);
    }
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
        throw std::invalid_argument("duplicate request_id");
    }
}

}  // namespace

SchedulerResult simulate_token_barrier(
    const std::vector<RequestInput>& requests,
    const SchedulerConfig& config) {
    validate(requests, config);
    std::vector<MutableRequest> state;
    state.reserve(requests.size());
    for (const auto& request : requests) {
        state.push_back(MutableRequest{
            request, State::kWaiting, 0, 0,
            request.decode_step ? request.initial_kv_length : 0,
            0, {}, 0, "completed"});
    }

    SchedulerResult result;
    TimeFs boundary = 0;
    std::uint64_t epoch_id = 0;
    std::uint64_t reserved_kv_bytes = 0;
    const std::uint64_t max_epochs = 1000000;

    while (true) {
        const auto finished_count = static_cast<std::size_t>(std::count_if(
            state.begin(), state.end(), [](const MutableRequest& request) {
                return request.state == State::kFinished;
            }));
        if (finished_count == state.size()) {
            break;
        }
        if (epoch_id >= max_epochs) {
            throw std::runtime_error("scheduler exceeded maximum epoch count");
        }

        EpochRecord epoch;
        epoch.epoch_id = epoch_id;
        epoch.boundary_time_fs = boundary;
        if (boundary > std::numeric_limits<TimeFs>::max() - config.epoch_duration_fs) {
            throw std::overflow_error("scheduler time exceeds TimeFs");
        }
        epoch.completion_time_fs = boundary + config.epoch_duration_fs;

        // Cancellation is sampled at token-step barriers.  Work issued in the
        // preceding epoch commits before a cancellation observed here.
        for (auto& request : state) {
            if (request.state == State::kFinished ||
                request.input.cancel_time_fs > boundary) {
                continue;
            }
            if (request.state == State::kPrefillReady ||
                request.state == State::kDecodeReady) {
                if (reserved_kv_bytes < request.input.kv_reservation_bytes) {
                    throw std::runtime_error("KV reservation accounting underflow");
                }
                reserved_kv_bytes -= request.input.kv_reservation_bytes;
            }
            request.state = State::kFinished;
            request.finish_time_fs = boundary;
            request.termination_reason = "cancelled";
            epoch.cancelled_request_ids.push_back(request.input.request_id);
            epoch.retired_request_ids.push_back(request.input.request_id);
        }

        std::size_t active = static_cast<std::size_t>(std::count_if(
            state.begin(), state.end(), [](const MutableRequest& request) {
                return request.state == State::kPrefillReady ||
                       request.state == State::kDecodeReady;
            }));
        std::vector<std::size_t> arrivals;
        for (std::size_t index = 0; index < state.size(); ++index) {
            if (state[index].state == State::kWaiting &&
                state[index].input.arrival_time_fs <= boundary) {
                arrivals.push_back(index);
            }
        }
        std::sort(arrivals.begin(), arrivals.end(), [&](std::size_t lhs, std::size_t rhs) {
            return std::tie(state[lhs].input.arrival_time_fs, state[lhs].input.request_id) <
                   std::tie(state[rhs].input.arrival_time_fs, state[rhs].input.request_id);
        });
        for (const auto index : arrivals) {
            if (active >= config.max_num_sequences) {
                break;
            }
            const auto reservation = state[index].input.kv_reservation_bytes;
            if (config.kv_capacity_bytes != 0 &&
                reservation > config.kv_capacity_bytes - reserved_kv_bytes) {
                continue;
            }
            state[index].state = state[index].input.decode_step
                                     ? State::kDecodeReady
                                     : State::kPrefillReady;
            epoch.admitted_request_ids.push_back(state[index].input.request_id);
            reserved_kv_bytes += reservation;
            ++active;
        }
        for (const auto& request : state) {
            if (request.state == State::kPrefillReady ||
                request.state == State::kDecodeReady) {
                epoch.active_request_ids.push_back(request.input.request_id);
            }
        }
        std::sort(epoch.active_request_ids.begin(), epoch.active_request_ids.end());

        std::vector<std::size_t> decode;
        std::vector<std::size_t> prefill;
        for (std::size_t index = 0; index < state.size(); ++index) {
            if (state[index].state == State::kDecodeReady) {
                decode.push_back(index);
            } else if (state[index].state == State::kPrefillReady) {
                prefill.push_back(index);
            }
        }
        std::sort(decode.begin(), decode.end(), [&](std::size_t lhs, std::size_t rhs) {
            return std::tuple{-state[lhs].input.priority,
                              state[lhs].input.arrival_time_fs,
                              state[lhs].input.request_id} <
                   std::tuple{-state[rhs].input.priority,
                              state[rhs].input.arrival_time_fs,
                              state[rhs].input.request_id};
        });
        std::sort(prefill.begin(), prefill.end(), [&](std::size_t lhs, std::size_t rhs) {
            return std::tuple{-state[lhs].input.priority,
                              -static_cast<std::int64_t>(state[lhs].waiting_epochs),
                              state[lhs].input.arrival_time_fs,
                              state[lhs].input.request_id} <
                   std::tuple{-state[rhs].input.priority,
                              -static_cast<std::int64_t>(state[rhs].waiting_epochs),
                              state[rhs].input.arrival_time_fs,
                              state[rhs].input.request_id};
        });

        std::uint64_t budget = config.max_batched_tokens;

        std::size_t reserved_prefill = std::numeric_limits<std::size_t>::max();
        if (!prefill.empty() &&
            state[prefill.front()].waiting_epochs >= config.max_prefill_wait_epochs) {
            reserved_prefill = prefill.front();
            const auto& request = state[reserved_prefill];
            const auto remaining = request.input.prompt_length - request.prompt_cursor;
            const auto count = std::min({remaining, config.prefill_chunk_tokens, budget});
            epoch.selections.push_back(SelectionItem{
                request.input.request_id, RequestPhase::kPrefill,
                request.prompt_cursor, count});
            budget -= count;
        }

        for (const auto index : decode) {
            if (budget == 0) {
                break;
            }
            epoch.selections.push_back(SelectionItem{
                state[index].input.request_id, RequestPhase::kDecode,
                state[index].committed_kv_length, 1});
            --budget;
        }
        for (const auto index : prefill) {
            if (budget == 0) {
                break;
            }
            if (index == reserved_prefill) {
                continue;
            }
            const auto remaining = state[index].input.prompt_length - state[index].prompt_cursor;
            const auto count = std::min({remaining, config.prefill_chunk_tokens, budget});
            epoch.selections.push_back(SelectionItem{
                state[index].input.request_id, RequestPhase::kPrefill,
                state[index].prompt_cursor, count});
            budget -= count;
        }

        if (epoch.selections.empty()) {
            if (!epoch.retired_request_ids.empty()) {
                result.epochs.push_back(std::move(epoch));
                boundary += config.epoch_duration_fs;
                ++epoch_id;
                continue;
            }
            const auto has_arrived_waiting = std::any_of(
                state.begin(), state.end(), [&](const MutableRequest& request) {
                    return request.state == State::kWaiting &&
                           request.input.arrival_time_fs <= boundary;
                });
            if (active == 0 && has_arrived_waiting &&
                config.kv_capacity_bytes != 0) {
                throw std::runtime_error(
                    "no arrived request fits the KV admission capacity");
            }
            const auto next_arrival = std::min_element(
                state.begin(), state.end(), [](const MutableRequest& lhs, const MutableRequest& rhs) {
                    const auto lhs_time = lhs.state == State::kWaiting
                                              ? lhs.input.arrival_time_fs
                                              : std::numeric_limits<TimeFs>::max();
                    const auto rhs_time = rhs.state == State::kWaiting
                                              ? rhs.input.arrival_time_fs
                                              : std::numeric_limits<TimeFs>::max();
                    return lhs_time < rhs_time;
                });
            if (next_arrival == state.end() || next_arrival->state != State::kWaiting) {
                throw std::runtime_error("no runnable request and no future arrival");
            }
            boundary += config.epoch_duration_fs;
            ++epoch_id;
            continue;
        }

        for (const auto& selection : epoch.selections) {
            auto request = std::find_if(
                state.begin(), state.end(), [&](const MutableRequest& candidate) {
                    return candidate.input.request_id == selection.request_id;
                });
            if (selection.phase == RequestPhase::kPrefill) {
                request->prompt_cursor += selection.token_count;
                request->committed_kv_length += selection.token_count;
                request->waiting_epochs = 0;
                if (request->prompt_cursor == request->input.prompt_length) {
                    request->generated_length = 1;
                    request->token_ready_time_fs.push_back(epoch.completion_time_fs);
                    if (request->generated_length ==
                        effective_output_limit(request->input)) {
                        request->state = State::kFinished;
                        request->finish_time_fs = epoch.completion_time_fs;
                        request->termination_reason =
                            natural_termination_reason(request->input);
                        epoch.retired_request_ids.push_back(request->input.request_id);
                        reserved_kv_bytes -= request->input.kv_reservation_bytes;
                    } else {
                        request->state = State::kDecodeReady;
                    }
                }
            } else {
                ++request->generated_length;
                ++request->committed_kv_length;
                request->token_ready_time_fs.push_back(epoch.completion_time_fs);
                if (request->generated_length ==
                    effective_output_limit(request->input)) {
                    request->state = State::kFinished;
                    request->finish_time_fs = epoch.completion_time_fs;
                    request->termination_reason =
                        natural_termination_reason(request->input);
                    epoch.retired_request_ids.push_back(request->input.request_id);
                    reserved_kv_bytes -= request->input.kv_reservation_bytes;
                } else {
                    request->state = State::kDecodeReady;
                }
            }
        }
        for (auto& request : state) {
            if (request.state == State::kPrefillReady &&
                std::none_of(epoch.selections.begin(), epoch.selections.end(),
                             [&](const SelectionItem& item) {
                                 return item.request_id == request.input.request_id;
                             })) {
                ++request.waiting_epochs;
            }
        }

        result.epochs.push_back(std::move(epoch));
        boundary += config.epoch_duration_fs;
        ++epoch_id;
    }

    for (const auto& request : state) {
        result.requests.push_back(RequestResult{
            request.input.request_id,
            request.generated_length,
            request.committed_kv_length,
            request.token_ready_time_fs,
            request.finish_time_fs,
            request.termination_reason});
    }
    std::sort(result.requests.begin(), result.requests.end(),
              [](const RequestResult& lhs, const RequestResult& rhs) {
                  return lhs.request_id < rhs.request_id;
              });
    return result;
}

}  // namespace heterosim::runtime
