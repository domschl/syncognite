add_executable(cifar10test cifar10test.cpp)
target_link_libraries(cifar10test LINK_PUBLIC Threads::Threads ${HDF5_LIBRARIES} ${BLAS_LIBRARIES})
