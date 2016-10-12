#include <iostream>
#include <fstream>
//#include <codecvt>
//#include <cstddef>
#include <locale>
#include <string>
#include <iomanip>

#include "cp-neural.h"

using std::wcout; using std::wstring;
using std::cerr;

class Text {
private:
    bool isinit=false;
    bool readDataFile(string inputfile) {
        std::wifstream txtfile(inputfile, std::ios::binary);
        if (!txtfile.is_open()) {
            return false;
        }
        txtfile.imbue(std::locale("en_US.UTF8"));
        wstring ttext((std::istreambuf_iterator<wchar_t>(txtfile)),
                     std::istreambuf_iterator<wchar_t>());
        txtfile.close();
        text=ttext;
        filename=inputfile;
        return true;
    }
public:
    wstring text;
    string filename;
    std::map<wchar_t,int> freq;
    std::map<wchar_t,int> w2v;
    std::map<int,wchar_t> v2w;
    Text(string filename) {
        if (readDataFile(filename)) {
            for (auto wc : text) {
                freq[wc]++;
            }
            int it=0;
            for (auto wc : freq) {
                w2v[wc.first]=it;
                v2w[it]=wc.first;
                ++it;
            }
            isinit=true;
        }
    }
    ~Text() {

    }
    bool isInit() {
        return isinit;
    }
    int vocsize() {
        return v2w.size();
    }
};

int main(int argc, char *argv[]) {
    std::setlocale (LC_ALL, "");
    wcout << L"Rnn-Readär" << std::endl;

    bool allOk=true;
    if (argc!=2) {
        cerr << "rnnreader <path-to-text-file>" << endl;
        exit(-1);
    }
    Text txt(argv[1]);
    if (!txt.isInit()) {
        cerr << "Cannot initialize Text object from inputfile: " << argv[1] << endl;
        exit(-1);
    }

    wcout << L"Text size: " << txt.text.size() << endl;

    wcout << L"Size of vocabulary:" << txt.freq.size() << std::endl;
/*    for (auto f : txt.freq) {
        int c=(int)f.first;
        wstring wc(1,f.first);
        wcout << wc << "|" <<  wchar_t(f.first) << L"(0x" << std::hex << c << L")" ": " << std::dec <<  f.second << endl;
    }
*/
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    cpInitCompute("Rnnreader");
    registerLayers();

    LayerBlock lb("{name='rnnreader'}");

    int VS=txt.vocsize();
    int H=128;
    int T=64;
    int BS=128;

    CpParams cp1;
    cp1.setPar("inputShape",vector<int>{VS});
    cp1.setPar("T",T);
    cp1.setPar("N",BS);
    cp1.setPar("hidden",H);
    lb.addLayer("RNN","rnn1",cp1,{"input"});

    CpParams cp2;
    cp2.setPar("inputShape",vector<int>{H});
    cp2.setPar("hidden",VS);
    lb.addLayer("Affine","af1",cp2,{"rnn1"});

    CpParams cp3;
    cp3.setPar("inputShape",vector<int>{VS});
    lb.addLayer("Softmax","sm1",cp3,{"af1"});
    if (!lb.checkTopology(true)) {
        allOk=false;
        cerr << red << "Topology-check for LayerBlock: ERROR." << def << endl;
    } else {
        cerr << green << "Topology-check for LayerBlock: ok." << def << endl;
    }

    wstring chunk;
    chunk = txt.text.substr(512,128);
    wcout << chunk << endl;

    cpExitCompute();
}
