cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

# Build type needs to be set as parameter to CMAKE: -DCMAKE_BUILD_TYPE=xxxx
# set(CMAKE_BUILD_TYPE Release)
# If the user specifies -DCMAKE_BUILD_TYPE on the command line, take their definition
# and dump it in the cache along with proper documentation, otherwise set CMAKE_BUILD_TYPE
# to Debug prior to calling PROJECT()
IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of `build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE()
   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG  "-g")
set(CMAKE_CXX_FLAGS_DEBUG  "-ggdb")

project(cpneural)

option(WITH_FLOAT "Use float 32-bit precision (OFF=double)" ON)
# option(WITH_OPENMP "Use OpenMP libraries" ON)
option(USE_SYSTEM_EIGEN "Use system version of Eigen3 instead of project's version" OFF)
if (APPLE)
    message(STATUS "Apple detected, using Apple's BLAS in Eigen3")
    include_directories("/usr/local/include")
    include_directories("/opt/homebrew/include")
    SET(BLA_VENDOR "Apple" CACHE STRING "Set BLAS vendor (see FindBLAS) e.g. OpenBlas, Apple, Generic")
    option(USE_SYSTEM_BLAS "Configure eigen3 to use BLAS" ON)
else()
    option(USE_SYSTEM_BLAS "Configure eigen3 to use BLAS" OFF)
    SET(BLA_VENDOR "" CACHE STRING "Set BLAS vendor (see FindBLAS) e.g. OpenBlas, Apple, Generic")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

if(USE_SYSTEM_BLAS)
    add_definitions(-DEIGEN_USE_BLAS)
    find_package(BLAS REQUIRED)
    message(STATUS "Eigen3 will use system BLAS libraries")
endif()

if (USE_SYSTEM_EIGEN)
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    include_directories("${PROJECT_SOURCE_DIR}/cpneural/eigen3")
endif()

find_package(HDF5 COMPONENTS CXX C HL REQUIRED)
if (HDF5_FOUND)
  include_directories(${HDF5_INCLUDE_DIRS})
  link_directories(${HDF5_LIB_DIRS})
endif (HDF5_FOUND)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (WITH_FLOAT)
    add_definitions(-DUSE_FLOAT)
else()
    add_definitions(-DUSE_DOUBLE)
endif()


set(gcc_like_compilers GNU Clang AppleClang Intel)
set(intel_archs x86_64 i386 i686)

# Setup some options to allow a user to enable SSE and AVX instruction use.
if ((";${gcc_like_compilers};" MATCHES ";${CMAKE_CXX_COMPILER_ID};") AND
    (";${intel_archs};"        MATCHES ";${CMAKE_SYSTEM_PROCESSOR};"))
    option(USE_SSE2_INSTRUCTIONS "Compile your program with SSE2 instructions" ON)
    option(USE_SSE4_INSTRUCTIONS "Compile your program with SSE4 instructions" ON)
    option(USE_AVX_INSTRUCTIONS  "Compile your program with AVX & AVX2 instructions"  ON)
    option(USE_FMA_INSTRUCTIONS  "Compile your program with FMA instructions"  ON)
    option(USE_FAST_MATH  "Compile your program with fast-math optimizations" ON)
    if(USE_AVX_INSTRUCTIONS)
        add_definitions(-march=native)
        add_definitions(-mavx -mavx2)
        add_definitions(-DUSE_AVX)
        message(STATUS "Enabling AVX & AVX2 instructions")
    elseif (USE_SSE4_INSTRUCTIONS)
        add_definitions(-msse4)
        add_definitions(-DUSE_SSE4)
        message(STATUS "Enabling SSE4 instructions")
    elseif(USE_SSE2_INSTRUCTIONS)
        add_definitions(-msse2)
        add_definitions(-DUSE_SSE2)
        message(STATUS "Enabling SSE2 instructions")
    endif()
    if(USE_FMA_INSTRUCTIONS)
        add_definitions(-mfma)
        add_definitions(-DUSE_FMA)
        message(STATUS "Enabling FMA (fused multiple add, Haswell onwards) instructions")
    endif()
    if(USE_FAST_MATH)
        add_definitions(-ffast-math)
        add_definitions(-DUSE_FAST_MATH)
        message(STATUS "Enabling fast-math optimizations")
    endif()
endif()

set(arm_archs aarch64 arm64)

if (";${arm_archs};" MATCHES ";${CMAKE_SYSTEM_PROCESSOR};")
    option(USE_ARM_OPT "Enable ARM64 optimization" ON)
    if (USE_ARM_OPT)
        add_definitions(-Ofast)
        add_definitions(-DEIGEN_MAX_ALIGN_BYTES=16)
        message("ARM64 opt enabled")
    endif()
endif()

add_subdirectory (cpneural)
include_directories("cpneural")
add_subdirectory (cptest)
add_subdirectory (minitests)
add_subdirectory (cpmnist)
add_subdirectory (cpcifar10)
add_subdirectory (bench)
add_subdirectory (rnnreader)
add_subdirectory (minimal)

set_property(TARGET cpneural PROPERTY CXX_STANDARD 11)
set_property(TARGET testneural PROPERTY CXX_STANDARD 11)
set_property(TARGET minitests PROPERTY CXX_STANDARD 11)
set_property(TARGET mnisttest PROPERTY CXX_STANDARD 11)
set_property(TARGET cifar10test PROPERTY CXX_STANDARD 11)
set_property(TARGET bench PROPERTY CXX_STANDARD 11)
set_property(TARGET rnnreader PROPERTY CXX_STANDARD 11)
set_property(TARGET minimal PROPERTY CXX_STANDARD 11)
