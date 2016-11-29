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

    int T=1; //32;
    int N=txt.text.size()-T+1;

    MatrixN Xr(N,T);
    MatrixN yr(N,T);

    wstring chunk,chunky;
    for (int i=0; i<N-T; i++) {
        for (int t=0; t<T; t++) {
            chunk=txt.text.substr(i+t,T);
            chunky=txt.text.substr(i+t+1,T);
            Xr(i,t)=txt.w2v[chunk[t]];
            yr(i,t)=txt.w2v[chunky[t]];
        }
    }

    MatrixN X(1000000,T);
    MatrixN y(1000000,T);
    MatrixN Xv(100000,T);
    MatrixN yv(100000,T);
    MatrixN Xt(100000,T);
    MatrixN yt(100000,T);

/*    X=Xr.block(0,0,1000000,T);
    y=yr.block(0,0,1000000,T);
    Xv=Xr.block(1000000,0,100000,T);
    yv=yr.block(1000000,0,100000,T);
    Xt=Xr.block(1100000,0,100000,T);
    yt=yr.block(1100000,0,100000,T);
*/
    X=Xr.block(0,0,10000,T);
    y=yr.block(0,0,10000,T);
    Xv=Xr.block(11000,0,1000,T);
    yv=yr.block(12000,0,1000,T);
    Xt=Xr.block(12000,0,1000,T);
    yt=yr.block(12000,0,1000,T);

    cpInitCompute("Rnnreader");
    registerLayers();

    LayerBlock lb("{name='rnnreader';init='normal'}");
    int VS=txt.vocsize();
    int H=1024;
    int D=16;
    int BS=128;

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
/*
    CpParams cp2;
    cp2.setPar("inputShape",vector<int>{H,T});
    //cp1.setPar("T",T);
    cp2.setPar("N",BS);
    cp2.setPar("H",H);
    lb.addLayer("RNN","rnn2",cp2,{"rnn1"});

    CpParams cp3;
    cp3.setPar("inputShape",vector<int>{H,T});
    //cp1.setPar("T",T);
    cp3.setPar("N",BS);
    cp3.setPar("H",H);
    lb.addLayer("RNN","rnn3",cp3,{"rnn2"});
*/
    CpParams cp10;
    cp10.setPar("inputShape",vector<int>{H,T});
    //cp10.setPar("T",T);
    //cp10.setPar("D",H);
    cp10.setPar("M",VS);
    lb.addLayer("TemporalAffine","af1",cp10,{"rnn1"});

    CpParams cp11;
    cp11.setPar("inputShape",vector<int>{VS,T});
    lb.addLayer("TemporalSoftmax","sm1",cp11,{"af1"});

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

    cpo.setPar("epochs",(floatN)1.0);
    cpo.setPar("batch_size",BS);

    MatrixN xg(1,1);
    wstring sg;
    for (int i=0; i<100; i++) {
        /*floatN cAcc=*/lb.train(X, y, Xv, yv, "Adam", cpo);
        wstring instr=L"Er sagte";
        for (auto wc : instr) {
            sg[0]=wc;
            xg(0,0)=txt.w2v[sg[0]];
            MatrixN z(0,0);
            MatrixN yg=lb.forward(xg,z,nullptr);
        }
        for (int g=0; g<100; g++) {
            xg(0,0)=txt.w2v[sg[0]];
            MatrixN z(0,0);
            MatrixN yg=lb.forward(xg,z,nullptr);
            float mx=-1000.0;
            int ind=-1;
            for (int j=0; j<yg.cols(); j++) {
                if (yg(0,j)>mx) {
                    mx=yg(0,j);
                    ind=j;
                }
            }
            if (ind==-1) {
                cerr << "Unexpected ind:" << ind << endl;
                exit(-1);
            }
            wchar_t cw=txt.v2w[ind];
            wcout << cw;
            sg[0]=cw;
        }
        wcout << endl;
    }

    /*
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
    //return cAcc;
    */
    cpExitCompute();
}
