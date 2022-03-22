add_executable(testneural testneural.cpp)
target_link_libraries(testneural LINK_PUBLIC Threads::Threads ${HDF5_LIBRARIES} ${BLAS_LIBRARIES})
