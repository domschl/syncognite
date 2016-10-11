#include <iostream>
#include <codecvt>
#include <cstddef>
#include <locale>
#include <string>
#include <iomanip>

#include "cp-neural.h"


int main(int argc, char *argv[]) {
    bool allOk=true;
    if (argc!=2) {
        cout << "rnnreader <path-to-text-file>" << endl;
        exit(-1);
    }

    string inputfile=argv[1];
    std::wifstream txtfile(inputfile);
    if (!txtfile.is_open()) {
        cout << "Cannot read from: " << inputfile << endl;
        exit(-1);
    }

    //std::wstring text;
    //txtfile.imbue(std::locale(std::locale::empty(), new std::codecvt_utf8<wchar_t,0x10ffff, std::consume_header>));
    txtfile.imbue(std::locale("en_US.UTF8"));
    std::wstring text((std::istreambuf_iterator<wchar_t>(txtfile)),
                 std::istreambuf_iterator<wchar_t>());
/*    std::wstringstream wss;
    cout << txtfile.rdbuf();
*/    //wss >> text;
/*    std::wstring text((std::istreambuf_iterator<wchar_t>(txtfile)),
                std::istreambuf_iterator<wchar_t>());
*/

    txtfile.close();

    cout << inputfile << ": " << text.size() << endl;

    /*std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    std::u32string utf32str = conv.from_bytes(text);
*/
    std::map<wchar_t,int> freq;

    for (auto wc : text) {
//        freq[text.substr(i,1)]++;
        freq[wc]++;
    }
    setenv("LANG","en_US.utf8",1);
    setlocale(LC_ALL,"en_US.utf8");
    //std::wcout.imbue(std::locale("en_US.UTF8"));
    std::locale::global(std::locale("en_US.utf8"));
    std::wcout.imbue(std::locale("en_US.utf8"));
    std::wstring test(L"Tibetan: སེམས་ཉིད་རྒྱུད, or german ä.");
    std::wcout << test << endl;
    cout << "Freq-size:" << freq.size() << endl;
    for (auto f : freq) {
        int c=(int)f.first;
        std::wstring wc(1,f.first);
        std::wcout << wchar_t(f.first) << L"(0x" << std::hex << c << L")" ": " << std::dec <<  f.second << endl;
    }

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
