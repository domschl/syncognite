#include <iostream>
#include <fstream>
//#include <codecvt>
//#include <cstddef>
#include <locale>
#include <string>
#include <iomanip>

#include "cp-neural.h"

using std::wcout; using std::wstring;

bool readDataFile(string inputfile, wstring& text) {
    std::wifstream txtfile(inputfile, std::ios::binary);
    if (!txtfile.is_open()) {
        return false;
    }
    txtfile.imbue(std::locale("en_US.UTF8"));
    wstring ttext((std::istreambuf_iterator<wchar_t>(txtfile)),
                 std::istreambuf_iterator<wchar_t>());
    txtfile.close();
    text=ttext;
    return true;
}

int main(int argc, char *argv[]) {
    std::setlocale (LC_ALL, "");
    wcout << L"Rnn-Readär" << std::endl;

    bool allOk=true;
    if (argc!=2) {
        wcout << L"rnnreader <path-to-text-file>" << endl;
        exit(-1);
    }
    std::string inputfile(argv[1]);
    wstring text;
    if (!readDataFile(inputfile, text)) {
        cout << "Cannot read from: " << inputfile << endl;
        exit(-1);
    }

    cout << inputfile << ": " << text.size() << endl;

    std::map<wchar_t,int> freq;

    for (auto wc : text) {
        freq[wc]++;
    }
    cout << "Freq-size:" << freq.size() << std::endl;
    for (auto f : freq) {
        int c=(int)f.first;
        wstring wc(1,f.first);
        wcout << wc << "|" <<  wchar_t(f.first) << L"(0x" << std::hex << c << L")" ": " << std::dec <<  f.second << endl;
    }

    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    cpInitCompute("Rnnreader");
    registerLayers();

    LayerBlock lb("{name='rnnreader'}");

    lb.addLayer("RNN","rnn1","{inputShape=[128];T=64;N=128}",{"input"});
    lb.addLayer("Affine","af1","{hidden=10}",{"rnn1"});
    lb.addLayer("Softmax","sm1","",{"af1"});
    if (!lb.checkTopology(true)) {
        allOk=false;
        cout << red << "Topology-check for LayerBlock: ERROR." << def << endl;
    } else {
        cout << green << "Topology-check for LayerBlock: ok." << def << endl;
    }

    wstring chunk;
    chunk = text.substr(512,128);

    wcout << chunk << endl;

    cpExitCompute();
}
