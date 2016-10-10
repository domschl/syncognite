#include <iostream>
#include <string>

#include "cp-neural.h"


int main(int argc, char *argv[]) {
    bool allOk=true;
    if (argc!=2) {
        cout << "rnnreader <path-to-text-file>" << endl;
        exit(-1);
    }

    string inputfile=argv[1];
    std::ifstream txtfile(inputfile);
    if (!txtfile.is_open()) {
        cout << "Cannot read from: " << inputfile << endl;
        exit(-1);
    }
    string text;
    txtfile.seekg(0, std::ios::end);
    text.reserve(txtfile.tellg());
    txtfile.seekg(0, std::ios::beg);
    text.assign((std::istreambuf_iterator<char>(txtfile)),
                std::istreambuf_iterator<char>());
    txtfile.close();

    cout << inputfile << ": " << text.size() << endl;

    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    cpInitCompute("Rnnreader");
    registerLayers();

    LayerBlock lb("{name='rnnreader'}");

    lb.addLayer("RNN","rnn1","{inputShape=[10];T=10;N=10}",{"input"});
    //lb.addLayer("Relu","rl1","",{"af1"});
    lb.addLayer("Affine","af1","{hidden=10}",{"rnn1"});
    lb.addLayer("Softmax","sm1","",{"af1"});
    if (!lb.checkTopology(true)) {
        allOk=false;
        cout << red << "Topology-check for LayerBlock: ERROR." << def << endl;
    } else {
        cout << green << "Topology-check for LayerBlock: ok." << def << endl;
    }




    cpExitCompute();
}
