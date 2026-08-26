add_library(heterosim_runtime STATIC
    runtime/event_queue.cpp
)

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

