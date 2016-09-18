add_executable(testneural testneural.cpp)
target_link_libraries(testneural LINK_PUBLIC Threads::Threads ${OpenCL_LIBRARY} ${CUDA_LIBRARY})
