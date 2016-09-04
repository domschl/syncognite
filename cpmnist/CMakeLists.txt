add_executable(mnisttest mnisttest.cpp)
target_link_libraries(mnisttest LINK_PUBLIC cpneural hdf5 hdf5_cpp Threads::Threads)
