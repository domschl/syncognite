add_executable(rnnreader rnnreader.cpp)
target_link_libraries(rnnreader LINK_PUBLIC Threads::Threads ${HDF5_LIBRARIES} ${BLAS_LIBRARIES})
