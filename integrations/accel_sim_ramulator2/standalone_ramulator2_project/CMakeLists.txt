cmake_minimum_required(VERSION 3.14)
project(heterosim_standalone_ramulator2 LANGUAGES CXX)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT HETEROSIM_SOURCE_ROOT)
  message(FATAL_ERROR "HETEROSIM_SOURCE_ROOT is required")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(yaml-cpp_SOURCE_DIR "${HETEROSIM_SOURCE_ROOT}/3rd/yaml-cpp")
include_directories("${HETEROSIM_SOURCE_ROOT}/src")
add_subdirectory("${yaml-cpp_SOURCE_DIR}" yaml-cpp)
add_subdirectory(
  "${HETEROSIM_SOURCE_ROOT}/src/dram/ramulator2"
  ramulator2
)
