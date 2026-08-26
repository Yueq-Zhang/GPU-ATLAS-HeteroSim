#include "hetero/runtime/global_event_runtime.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hetero/runtime/event_queue.h"

namespace heterosim::runtime {
namespace {

TimeFs checked_add(TimeFs lhs, TimeFs rhs) {
    if (rhs > std::numeric_limits<TimeFs>::max() - lhs) {
        throw std::overflow_error("task completion exceeds TimeFs");
    }
    return lhs + rhs;
}

}  // namespace

RuntimeResult GlobalEventRuntime::run(const std::vector<RuntimeTask>& tasks) const {
    RuntimeResult result;
    result.tasks.resize(tasks.size());
    if (tasks.empty()) {
        return result;
    }

    std::unordered_map<std::string, std::size_t> task_indices;
    task_indices.reserve(tasks.size());
    for (std::size_t index = 0; index < tasks.size(); ++index) {
        const auto& task = tasks.at(index);
        if (task.task_id.empty() || task.resource_id.empty()) {
            throw std::invalid_argument("task_id and resource_id must not be empty");
        }
        if (task.duration_fs == 0) {
            throw std::invalid_argument("task duration must be positive");
        }
        if (!task_indices.emplace(task.task_id, index).second) {
            throw std::invalid_argument("duplicate task_id: " + task.task_id);
        }
        result.tasks.at(index).task_id = task.task_id;
        result.tasks.at(index).resource_id = task.resource_id;
    }

    std::vector<std::size_t> remaining_dependencies(tasks.size(), 0);
    std::vector<std::vector<std::size_t>> dependents(tasks.size());
    std::vector<TimeFs> dependency_ready_time(tasks.size(), 0);
    for (std::size_t index = 0; index < tasks.size(); ++index) {
        dependency_ready_time.at(index) = tasks.at(index).release_time_fs;
        std::unordered_set<std::string> unique_dependencies;
        for (const auto& dependency_id : tasks.at(index).dependencies) {
            const auto dependency = task_indices.find(dependency_id);
            if (dependency == task_indices.end()) {
                throw std::invalid_argument(
                    "unknown dependency " + dependency_id + " for " +
                    tasks.at(index).task_id);
            }
            if (dependency->second == index) {
                throw std::invalid_argument("task cannot depend on itself");
            }
            if (!unique_dependencies.insert(dependency_id).second) {
                throw std::invalid_argument("duplicate dependency: " + dependency_id);
            }
            ++remaining_dependencies.at(index);
            dependents.at(dependency->second).push_back(index);
        }
    }

    EventQueue event_queue;
    std::unordered_map<std::string, TimeFs> resource_available;
    std::vector<bool> dispatched(tasks.size(), false);
    std::vector<bool> completed(tasks.size(), false);

    const auto dispatch = [&](std::size_t index) {
        if (dispatched.at(index) || remaining_dependencies.at(index) != 0) {
            return;
        }
        const auto& task = tasks.at(index);
        const auto available = resource_available.find(task.resource_id);
        const TimeFs resource_ready =
            available == resource_available.end() ? 0 : available->second;
        auto& timing = result.tasks.at(index);
        timing.ready_time_fs = dependency_ready_time.at(index);
        timing.start_time_fs = std::max(timing.ready_time_fs, resource_ready);
        timing.completion_time_fs = checked_add(timing.start_time_fs, task.duration_fs);
        resource_available[task.resource_id] = timing.completion_time_fs;
        dispatched.at(index) = true;
        event_queue.schedule(
            timing.completion_time_fs,
            EventPriority::kResourceCompletion,
            static_cast<std::uint64_t>(index));
    };

    for (std::size_t index = 0; index < tasks.size(); ++index) {
        if (remaining_dependencies.at(index) == 0) {
            event_queue.schedule(
                dependency_ready_time.at(index),
                EventPriority::kTaskDispatch,
                static_cast<std::uint64_t>(index));
        }
    }

    std::size_t completed_count = 0;
    while (!event_queue.empty()) {
        const auto event = event_queue.pop();
        const auto index = static_cast<std::size_t>(event.token);
        if (index >= tasks.size()) {
            throw std::logic_error("invalid event token");
        }
        if (event.priority == EventPriority::kTaskDispatch) {
            dispatch(index);
            continue;
        }
        if (event.priority != EventPriority::kResourceCompletion || completed.at(index)) {
            throw std::logic_error("invalid or duplicate completion event");
        }
        completed.at(index) = true;
        ++completed_count;
        result.makespan_fs = std::max(result.makespan_fs, event.time_fs);

        auto ready_dependents = dependents.at(index);
        std::sort(ready_dependents.begin(), ready_dependents.end());
        for (const auto dependent : ready_dependents) {
            dependency_ready_time.at(dependent) = std::max(
                dependency_ready_time.at(dependent), event.time_fs);
            if (remaining_dependencies.at(dependent) == 0) {
                throw std::logic_error("dependency counter underflow");
            }
            --remaining_dependencies.at(dependent);
            if (remaining_dependencies.at(dependent) == 0) {
                event_queue.schedule(
                    dependency_ready_time.at(dependent),
                    EventPriority::kTaskDispatch,
                    static_cast<std::uint64_t>(dependent));
            }
        }
    }

    if (completed_count != tasks.size()) {
        throw std::invalid_argument("task graph contains a dependency cycle");
    }
    return result;
}

}  // namespace heterosim::runtime
