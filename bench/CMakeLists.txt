add_executable(bench cp-bench.cpp)

# SET(CURSES_NEED_NCURSES TRUE)
find_package(Curses REQUIRED)
if (Curses_FOUND)
    include_directories(${CURSES_INCLUDE_DIR})
    link_directories(${CURSES_LIB_DIRS})
endif(Curses_FOUND)

if (OpenCL_FOUND)
    target_link_libraries(bench LINK_PUBLIC Threads::Threads ${HDF5_LIBRARIES} ${OpenCL_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CURSES_LIBRARIES})
else()
    target_link_libraries(bench LINK_PUBLIC Threads::Threads ${HDF5_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CURSES_LIBRARIES})
endif()
