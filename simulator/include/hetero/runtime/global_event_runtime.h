#pragma once

#include <string>
#include <vector>

#include "hetero/types.h"

namespace heterosim::runtime {

struct RuntimeTask {
    std::string task_id;
    std::string resource_id;
    std::vector<std::string> dependencies;
    TimeFs release_time_fs{};
    TimeFs duration_fs{};
};

struct TaskTiming {
    std::string task_id;
    std::string resource_id;
    TimeFs ready_time_fs{};
    TimeFs start_time_fs{};
    TimeFs completion_time_fs{};
};

struct RuntimeResult {
    std::vector<TaskTiming> tasks;
    TimeFs makespan_fs{};
};

class GlobalEventRuntime {
public:
    [[nodiscard]] RuntimeResult run(const std::vector<RuntimeTask>& tasks) const;
};

}  // namespace heterosim::runtime
