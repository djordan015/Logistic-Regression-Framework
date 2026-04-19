cmake_minimum_required(VERSION 3.12)
project(LogisticRegressionFramework)

set(CMAKE_CXX_STANDARD 14)

# 1. Pull in pybind11
add_subdirectory(pybind11)

# 2. Define your module
pybind11_add_module(logistic_regression_cpu src/bindings.cc)

# 3. Include directories 
target_include_directories(logistic_regression_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 4. Automatically place the .so file in the python/bindings directory
set_target_properties(logistic_regression_cpu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings"
)