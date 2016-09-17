
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
