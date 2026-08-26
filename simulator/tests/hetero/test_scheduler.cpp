#include <cassert>
#include <string>
#include <vector>

#include "hetero/runtime/scheduler.h"

using heterosim::runtime::RequestInput;
using heterosim::runtime::RequestPhase;
using heterosim::runtime::SchedulerConfig;
using heterosim::runtime::simulate_token_barrier;

int main() {
    const std::vector<RequestInput> requests{
        {"R0", 0, 4, 2, 0},
        {"R1", 0, 2, 1, 0},
        {"R2", 1500, 3, 2, 0},
    };
    const SchedulerConfig config{2, 4, 2, 8, 1000};
    const auto result = simulate_token_barrier(requests, config);
    assert(result.epochs.size() == 5);

    const auto& e0 = result.epochs.at(0);
    assert(e0.boundary_time_fs == 0);
    assert(e0.selections.size() == 2);
    assert(e0.selections.at(0).request_id == "R0");
    assert(e0.selections.at(0).phase == RequestPhase::kPrefill);
    assert(e0.selections.at(0).token_begin == 0);
    assert(e0.selections.at(0).token_count == 2);
    assert(e0.selections.at(1).request_id == "R1");

    const auto& e1 = result.epochs.at(1);
    assert(e1.selections.size() == 1);
    assert(e1.selections.at(0).request_id == "R0");
    assert(e1.selections.at(0).token_begin == 2);

    const auto& e2 = result.epochs.at(2);
    assert(e2.selections.size() == 2);
    assert(e2.selections.at(0).request_id == "R0");
    assert(e2.selections.at(0).phase == RequestPhase::kDecode);
    assert(e2.selections.at(1).request_id == "R2");
    assert(e2.selections.at(1).phase == RequestPhase::kPrefill);

    assert(result.epochs.at(3).selections.at(0).request_id == "R2");
    assert(result.epochs.at(3).selections.at(0).phase == RequestPhase::kPrefill);
    assert(result.epochs.at(4).selections.at(0).request_id == "R2");
    assert(result.epochs.at(4).selections.at(0).phase == RequestPhase::kDecode);

    assert(result.requests.at(0).committed_kv_length == 5);
    assert(result.requests.at(1).committed_kv_length == 2);
    assert(result.requests.at(2).committed_kv_length == 4);
    return 0;
}
