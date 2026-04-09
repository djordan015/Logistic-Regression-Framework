cmake_minimum_required(VERSION 3.12)
project(LogisticRegressionFramework)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE Release)

add_subdirectory(pybind11)

find_package(OpenMP)

pybind11_add_module(logistic_regression_cpu src/bindings.cc)

# Expose src/ so that angle-bracket includes (e.g. #include <Types.h> in
# Optimizer.h) resolve correctly when compiling the pybind11 module.
target_include_directories(logistic_regression_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)


if (OpenMP_CXX_FOUND)
  include_directories(OpenMP_CXX_INCLUDE_DIRS)
  target_link_libraries(LogisticRegressionFramework PUBLIC OpenMP::OpenMP_CXX)
endif()

