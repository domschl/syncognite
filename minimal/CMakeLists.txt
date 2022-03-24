add_executable(minimal minimal.cpp)
target_link_libraries(minimal LINK_PUBLIC Threads::Threads ${HDF5_LIBRARIES} ${BLAS_LIBRARIES})
