#include <chrono>
#include <ctime>
#include "cp-timer.h"

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

/*
#include <iostream>
int main()
{
    Timer t;
    t.startWall();
    t.startCpu();
    int s=0;
    for (auto i=0; i<1000000; i++) {
        s += i*i*i;
    };
    std::cout << std::endl << std::endl;
    double t1=t.stopCpuMicro();
    double t2=t.stopWallMicro();
    std::cout << s << std::endl << t1 << " " << t2 << " ∂:" << t2-t1 << std::endl;
}
*/
