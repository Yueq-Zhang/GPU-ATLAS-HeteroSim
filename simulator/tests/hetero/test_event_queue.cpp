#include <array>
#include <cstdint>
#include <iostream>

#include "hetero/runtime/event_queue.h"

int main() {
    using heterosim::runtime::EventPriority;
    using heterosim::runtime::EventQueue;

    EventQueue queue;
    queue.schedule(100, EventPriority::kTaskDispatch, 40);
    queue.schedule(100, EventPriority::kResourceCompletion, 10);
    queue.schedule(50, EventPriority::kMetricSnapshot, 5);
    queue.schedule(100, EventPriority::kResourceCompletion, 11);

    constexpr std::array<std::uint64_t, 4> expected{5, 10, 11, 40};
    for (const auto token : expected) {
        const auto event = queue.pop();
        if (event.token != token) {
            std::cerr << "expected token " << token << ", got " << event.token << '\n';
            return 1;
        }
    }
    return queue.empty() ? 0 : 1;
}

