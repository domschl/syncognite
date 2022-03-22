add_executable(minitests minitests.cpp)
target_link_libraries(minitests LINK_PUBLIC Threads::Threads ${HDF5_LIBRARIES} ${BLAS_LIBRARIES})
