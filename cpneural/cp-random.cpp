#include "cp-random.h"

float Random::floatrange(float a=0.0, float b=1.0) { // [a,b(
        std::uniform_real_distribution<float> dist(a, b);
        return dist(rde);
}

int Random::intrange(int a=0, int b=10) { // [a,b]
        std::uniform_int_distribution<int> disti(a, b);
        return disti(rde);
}

float Random::normal(float mean=0.0, float std=1.0) {
        std::normal_distribution<float> distn(mean, std);
        return distn(rde);
}
