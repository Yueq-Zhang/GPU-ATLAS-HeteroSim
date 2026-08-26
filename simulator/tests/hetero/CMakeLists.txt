add_executable(test_time test_time.cpp)
target_link_libraries(test_time PRIVATE heterosim_runtime)
add_test(NAME hetero_time COMMAND test_time)

add_executable(test_event_queue test_event_queue.cpp)
target_link_libraries(test_event_queue PRIVATE heterosim_runtime)
add_test(NAME hetero_event_queue COMMAND test_event_queue)

add_executable(test_interfaces test_interfaces.cpp)
target_link_libraries(test_interfaces PRIVATE heterosim_runtime)
add_test(NAME hetero_interfaces COMMAND test_interfaces)

