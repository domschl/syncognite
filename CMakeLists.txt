cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

# None Debug Release RelWithDebInfo MinSizeRel
set(CMAKE_BUILD_TYPE Release)

project(cpneural)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory (cpneural)
add_subdirectory (cptest)

set_property(TARGET cpneural PROPERTY CXX_STANDARD 11)
set_property(TARGET testneural PROPERTY CXX_STANDARD 11)
