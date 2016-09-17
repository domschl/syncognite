add_library(cpneural STATIC cp-random.cpp cp-math.cpp cp-neural.cpp)
target_include_directories(cpneural PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
