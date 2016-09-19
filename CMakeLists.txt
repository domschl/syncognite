cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

# None Debug Release RelWithDebInfo MinSizeRel
# CXX=/usr/bin/clang++ CXXFLAGS="-O2" cmake ..

# Build type needs to be set as parameter to CMAKE: -DCMAKE_BUILD_TYPE=xxxx
# set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS_DEBUG  "-g")
set(CMAKE_CXX_FLAGS_DEBUG  "-ggdb")

project(cpneural)

option(WITH_FLOAT "Use float 32-bit precision (OFF=double)" ON)
option(WITH_OPENMP "Use OpenMP libraries" ON)
option(WITH_VIENNACL "Use ViennaCL library" ON)
option(WITH_CUDA "Use Nvidia CUDA libraries" OFF)
option(WITH_OPENCL "Use OpenCL libraries" ON)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (WITH_FLOAT)
    add_definitions(-DUSE_FLOAT)
else()
    add_definitions(-DUSE_DOUBLE)
endif()

if (WITH_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DUSE_OPENMP)
    endif()
endif()

if (WITH_VIENNACL)
    find_package(ViennaCL)
    if (ViennaCL_FOUND)
        add_definitions(-DUSE_VIENNACL)
        if (DARWIN)
            include_directories("/usr/local/include")
        endif()
    endif()

    if (WITH_OPENCL)
        find_package(OpenCL)
        if (OpenCL_FOUND)
            include_directories(${OpenCL_INCLUDE_DIRS})
            link_directories(${OpenCL_LIBRARY})
            add_definitions(-DUSE_OPENCL)
        endif()
    endif()

    if (WITH_CUDA)
        find_package(CUDA)
        if (CUDA_FOUND)
            include_directories(${CUDA_INCLUDE_DIR})
            link_directories(${CUDA_LIBRARY})
            include_directories("/opt/cuda/include")
            add_definitions(-DUSE_CUDA)
        endif()
    endif()
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
