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

void Timer::startCpu() {
        c_start = std::clock();
    }
double Timer::stopCpuMicro() {
        std::clock_t c_end = std::clock();
        return 1000000.0 * (double)(c_end-c_start) / (double)CLOCKS_PER_SEC;
    }
void Timer::startWall() {
        t_start = std::chrono::high_resolution_clock::now();
    }
double Timer::stopWallMicro() {
        auto t_end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t_end-t_start).count();
    }

#endif
