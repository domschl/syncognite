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
            cerr << "Freq:" << freq.size() << ", w2v:" << w2v.size() << ", v2w:" << v2w.size() << endl;
        }
    }
    ~Text() {

    }
    bool isInit() {
        return isinit;
    }
    int vocsize() {
        if (!isinit) return 0;
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
        wcout << wc << L"|" <<  wchar_t(f.first) << L"(0x" << std::hex << c << L")" ": " << std::dec <<  f.second << endl;
    }
*/
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    int T=64;
    int N=txt.text.size()-T+1;

    MatrixN Xr(N,T);
    MatrixN yr(N,T);

    wstring chunk,chunky;
    for (int i=0; i<N-1; i++) {
        chunk=txt.text.substr(i,T);
        chunky=txt.text.substr(i+1,T);
        for (int t=0; t<T; t++) {
            Xr(i,t)=txt.w2v[chunk[t]];
            yr(i,t)=txt.w2v[chunky[t]];
        }
    }

    MatrixN X(10000,T);
    MatrixN y(10000,T);
    MatrixN Xv(1000,T);
    MatrixN yv(1000,T);
    MatrixN Xt(1000,T);
    MatrixN yt(1000,T);

    X=Xr.block(0,0,10000,T);
    y=yr.block(0,0,10000,T);
    Xv=Xr.block(10000,0,1000,T);
    yv=yr.block(10000,0,1000,T);
    Xt=Xr.block(11000,0,1000,T);
    yt=yr.block(11000,0,1000,T);

    cpInitCompute("Rnnreader");
    registerLayers();

    LayerBlock lb("{name='rnnreader';init='orthonormal'}");

    int VS=txt.vocsize();
    int H=256;
    int D=128;
    int BS=32;

    CpParams cp0;
    cp0.setPar("inputShape",vector<int>{T});
    cp0.setPar("V",VS);
    cp0.setPar("D",D);
    lb.addLayer("WordEmbedding","WE0",cp0,{"input"});

    CpParams cp1;
    cp1.setPar("inputShape",vector<int>{D,T});
    //cp1.setPar("T",T);
    cp1.setPar("N",BS);
    cp1.setPar("H",H);
    lb.addLayer("RNN","rnn1",cp1,{"WE0"});

    CpParams cp2;
    cp2.setPar("inputShape",vector<int>{H});
    cp2.setPar("T",T);
    cp2.setPar("D",H);
    cp2.setPar("M",VS);
    lb.addLayer("TemporalAffine","af1",cp2,{"rnn1"});

    CpParams cp3;
    cp3.setPar("inputShape",vector<int>{VS});
    lb.addLayer("Softmax","sm1",cp3,{"af1"});
    if (!lb.checkTopology(true)) {
        allOk=false;
        cerr << red << "Topology-check for LayerBlock: ERROR." << def << endl;
    } else {
        cerr << green << "Topology-check for LayerBlock: ok." << def << endl;
    }

/*    wstring chunk;
    chunk = txt.text.substr(512,128);
    wcout << chunk << endl;
*/
    CpParams cpo("{verbose=true;epsion=1e-8}");
    cpo.setPar("learning_rate", (floatN)2e-2); //2.2e-2);
    cpo.setPar("lr_decay", (floatN)1.0);
    cpo.setPar("regularization", (floatN)1e-5);

    cpo.setPar("epochs",(floatN)40.0);
    cpo.setPar("batch_size",BS);

    floatN cAcc=lb.train(X, y, Xv, yv, "Adam", cpo);

    floatN train_err, val_err, test_err;
    bool evalFinal=true;
    if (evalFinal) {
        train_err=lb.test(X, y, cpo.getPar("batch_size", 50));
        val_err=lb.test(Xv, yv, cpo.getPar("batch_size", 50));
        test_err=lb.test(Xt, yt, cpo.getPar("batch_size", 50));

        cerr << "Final results on RnnReader after " << cpo.getPar("epochs",(floatN)0.0) << " epochs:" << endl;
        cerr << "      Train-error: " << train_err << " train-acc: " << 1.0-train_err << endl;
        cerr << " Validation-error: " << val_err <<   "   val-acc: " << 1.0-val_err << endl;
        cerr << "       Test-error: " << test_err <<  "  test-acc: " << 1.0-test_err << endl;
    }
    return cAcc;

    cpExitCompute();
}
