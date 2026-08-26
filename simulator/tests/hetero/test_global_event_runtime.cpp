#include <cassert>
#include <stdexcept>
#include <vector>

#include "hetero/runtime/global_event_runtime.h"

using heterosim::runtime::GlobalEventRuntime;
using heterosim::runtime::RuntimeTask;

int main() {
    const GlobalEventRuntime runtime;
    const std::vector<RuntimeTask> tasks{
        {"gpu.first", "gpu0.compute", {}, 0, 10},
        {"atlas.first", "atlas0.compute", {}, 0, 20},
        {"gpu.second", "gpu0.compute", {}, 0, 7},
        {"gpu.join", "gpu0.compute", {"gpu.first", "atlas.first"}, 0, 5},
    };
    const auto result = runtime.run(tasks);
    assert(result.tasks.size() == 4);
    assert(result.tasks.at(0).start_time_fs == 0);
    assert(result.tasks.at(0).completion_time_fs == 10);
    assert(result.tasks.at(1).completion_time_fs == 20);
    assert(result.tasks.at(2).start_time_fs == 10);
    assert(result.tasks.at(2).completion_time_fs == 17);
    assert(result.tasks.at(3).ready_time_fs == 20);
    assert(result.tasks.at(3).start_time_fs == 20);
    assert(result.tasks.at(3).completion_time_fs == 25);
    assert(result.makespan_fs == 25);

    const auto arrivals = runtime.run({
        {"future", "gpu0.compute", {}, 100, 10},
        {"early", "gpu0.compute", {}, 0, 5},
    });
    assert(arrivals.tasks.at(1).start_time_fs == 0);
    assert(arrivals.tasks.at(1).completion_time_fs == 5);
    assert(arrivals.tasks.at(0).start_time_fs == 100);
    assert(arrivals.makespan_fs == 110);

    bool rejected_cycle = false;
    try {
        const auto unused = runtime.run({
            {"a", "gpu0.compute", {"b"}, 0, 1},
            {"b", "atlas0.compute", {"a"}, 0, 1},
        });
        static_cast<void>(unused);
    } catch (const std::invalid_argument&) {
        rejected_cycle = true;
    }
    assert(rejected_cycle);
    return 0;
}
