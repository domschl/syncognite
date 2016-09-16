add_executable(bench cp-bench.cpp)
target_link_libraries(bench LINK_PUBLIC cpneural Threads::Threads OpenCL)
