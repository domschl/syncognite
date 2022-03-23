add_executable(mnisttest mnisttest.cpp)
target_link_libraries(mnisttest LINK_PUBLIC Threads::Threads ${HDF5_LIBRARIES} ${BLAS_LIBRARIES})
