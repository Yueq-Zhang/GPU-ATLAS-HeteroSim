#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <vector>

#include "hetero/types.h"

namespace heterosim::runtime {

enum class EventPriority : std::uint8_t {
    kResourceCompletion = 0,
    kStateCommit = 1,
    kRequestArrival = 2,
    kAdmissionAndScheduling = 3,
    kTaskDispatch = 4,
    kMetricSnapshot = 5,
};

struct Event {
    TimeFs time_fs{};
    EventPriority priority{EventPriority::kTaskDispatch};
    std::uint64_t insertion_sequence{};
    std::uint64_t token{};
};

struct EventLater {
    bool operator()(const Event& lhs, const Event& rhs) const noexcept;
};

class EventQueue {
public:
    std::uint64_t schedule(TimeFs time_fs, EventPriority priority, std::uint64_t token);
    Event pop();
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    std::uint64_t next_sequence_{0};
    std::priority_queue<Event, std::vector<Event>, EventLater> events_;
};

}  // namespace heterosim::runtime

