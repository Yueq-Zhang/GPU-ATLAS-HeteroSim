find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_heterosim_runtime MODULE python_module.cpp)
target_link_libraries(_heterosim_runtime PRIVATE heterosim_runtime)
target_include_directories(_heterosim_runtime PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_options(_heterosim_runtime PRIVATE -Wall -Wextra -Werror)

set_target_properties(_heterosim_runtime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../frontend/hetero"
)
