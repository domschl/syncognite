#ifndef _CP_RANDOM_H
#define _CP_RANDOM_H

#include <iostream>
#include <random>

class Random {
    // std::mt19937 rde; // Mersenne twister engine
    std::default_random_engine rde;
public:
    float floatrange(float a, float b);
    int intrange(int a, int b);
    float normal(float mean, float std);
};

#endif
