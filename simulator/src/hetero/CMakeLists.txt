add_library(heterosim_runtime STATIC
    runtime/event_queue.cpp
    runtime/global_event_runtime.cpp
    runtime/scheduler.cpp
    runtime/timing_ownership.cpp
    memory/paged_kv.cpp
    memory/runtime_memory_planner.cpp
    services/bounded_link.cpp
    services/fixed_latency_memory.cpp
    services/shared_3d_memory.cpp
)

set_target_properties(heterosim_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(heterosim_runtime
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(heterosim_runtime
    PRIVATE
        -Wall
        -Wextra
        -Werror
)
