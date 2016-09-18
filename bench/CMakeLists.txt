add_executable(bench cp-bench.cpp)
target_link_libraries(bench LINK_PUBLIC Threads::Threads ${OpenCL_LIBRARY} ${CUDA_LIBRARY})
