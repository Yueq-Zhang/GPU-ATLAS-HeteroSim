#include "hetero/runtime/event_queue.h"

#include <stdexcept>
#include <tuple>

namespace heterosim::runtime {

bool EventLater::operator()(const Event& lhs, const Event& rhs) const noexcept {
    return std::tie(lhs.time_fs, lhs.priority, lhs.insertion_sequence) >
           std::tie(rhs.time_fs, rhs.priority, rhs.insertion_sequence);
}

std::uint64_t EventQueue::schedule(
    TimeFs time_fs,
    EventPriority priority,
    std::uint64_t token) {
    const auto sequence = next_sequence_++;
    events_.push(Event{time_fs, priority, sequence, token});
    return sequence;
}

Event EventQueue::pop() {
    if (events_.empty()) {
        throw std::underflow_error("cannot pop an empty event queue");
    }
    const Event event = events_.top();
    events_.pop();
    return event;
}

bool EventQueue::empty() const noexcept {
    return events_.empty();
}

std::size_t EventQueue::size() const noexcept {
    return events_.size();
}

}  // namespace heterosim::runtime

