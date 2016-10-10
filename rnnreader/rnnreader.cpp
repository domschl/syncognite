#include <iostream>
#include <string>

#include "cp-neural.h"


int main(int argc, char *argv[]) {
     if (argc!=2) {
         cout << "rnnreader <path-to-text-file>" << endl;
         exit(-1);
     }


     cpInitCompute("Rnnreader");
     registerLayers();
     cpExitCompute();
 }
