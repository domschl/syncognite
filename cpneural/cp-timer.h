#ifndef _CP_TIMER_H
#define _CP_TIMER_H

#include <chrono>
#include <ctime>

class Timer {
    std::clock_t c_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
public:
     void startCpu();
     double stopCpuMicro();
     void startWall();
     double stopWallMicro();
};

#endif
