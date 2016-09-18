cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

# None Debug Release RelWithDebInfo MinSizeRel
# CXX=/usr/bin/clang++ CXXFLAGS="-O2" cmake ..

# Build type needs to be set as parameter to CMAKE: -DCMAKE_BUILD_TYPE=xxxx
# set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS_DEBUG  "-g")
set(CMAKE_CXX_FLAGS_DEBUG  "-ggdb")

project(cpneural)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(ViennaCL)
if (ViennaCL_FOUND)
    if (DARWIN)
        include_directories("/usr/local/include")
    endif()
endif()

find_package(OpenCL)
if (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})
    link_directories(${OpenCL_LIBRARY})
endif()

find_package(CUDA)
if (CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIR})
    link_directories(${CUDA_LIBRARY})
    include_directories("/opt/cuda/include")
endif()

add_subdirectory (cpneural)
include_directories("cpneural")
add_subdirectory (cptest)
add_subdirectory (cpmnist)
add_subdirectory (cpcifar10)
add_subdirectory (bench)

set_property(TARGET cpneural PROPERTY CXX_STANDARD 11)
set_property(TARGET testneural PROPERTY CXX_STANDARD 11)
set_property(TARGET mnisttest PROPERTY CXX_STANDARD 11)
set_property(TARGET cifar10test PROPERTY CXX_STANDARD 11)
set_property(TARGET bench PROPERTY CXX_STANDARD 11)
