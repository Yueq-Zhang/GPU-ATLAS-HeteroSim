cmake_minimum_required(VERSION 3.16)
project(heterosim_booksim2_adapter LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED BOOKSIM2_SOURCE_DIR)
  message(FATAL_ERROR "BOOKSIM2_SOURCE_DIR must point to the patched ATLAS BookSim2 checkout")
endif()
if(NOT DEFINED ATLAS_SIMULATOR_SOURCE_DIR)
  message(FATAL_ERROR "ATLAS_SIMULATOR_SOURCE_DIR must point to ATLAS simulator")
endif()

file(GLOB_RECURSE BOOKSIM2_CPP CONFIGURE_DEPENDS
  "${BOOKSIM2_SOURCE_DIR}/src/*.cpp"
)
list(FILTER BOOKSIM2_CPP EXCLUDE REGEX "/noc_main\\.cpp$")
list(FILTER BOOKSIM2_CPP EXCLUDE REGEX "/main\\.cpp$")

add_library(heterosim_booksim2_adapter SHARED
  booksim2_cycle_adapter.cpp
  "${ATLAS_SIMULATOR_SOURCE_DIR}/src/noc/noc_wrapper.cpp"
  ${BOOKSIM2_CPP}
)

target_compile_definitions(heterosim_booksim2_adapter PRIVATE SPATIAL)
target_include_directories(heterosim_booksim2_adapter PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${ATLAS_SIMULATOR_SOURCE_DIR}/src"
  "${BOOKSIM2_SOURCE_DIR}/src"
  "${BOOKSIM2_SOURCE_DIR}/src/include"
  "${BOOKSIM2_SOURCE_DIR}/src/allocators"
  "${BOOKSIM2_SOURCE_DIR}/src/arbiters"
  "${BOOKSIM2_SOURCE_DIR}/src/networks"
  "${BOOKSIM2_SOURCE_DIR}/src/power"
  "${BOOKSIM2_SOURCE_DIR}/src/routers"
)

find_library(BOOKSIM2_PARSER_LIBRARY
  NAMES parser
  PATHS "${BOOKSIM2_SOURCE_DIR}/src/parser"
  NO_DEFAULT_PATH
)
if(NOT BOOKSIM2_PARSER_LIBRARY)
  message(FATAL_ERROR "patched BookSim2 parser library was not found")
endif()
target_link_libraries(heterosim_booksim2_adapter PRIVATE ${BOOKSIM2_PARSER_LIBRARY})

set_target_properties(heterosim_booksim2_adapter PROPERTIES
  OUTPUT_NAME "heterosim_booksim2_adapter"
)
