#include <iostream>
#include <fstream>
//#include <codecvt>
//#include <cstddef>
#include <locale>
#include <string>
#include <iomanip>
#include <ctime>

#include "cp-neural.h"

using std::wcout; using std::wstring;
using std::cerr; using std::vector;

class Text {
private:
    bool isinit=false;
    bool readDataFile(string inputfile) {
        // XXX: srand should be something more centrally chosen.
        std::srand(std::time(0));
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

    wstring sample(int len) {
        int p=std::rand()%(text.size()-len);
        return text.substr(p,len);
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

    int T=20;
    int N=txt.text.size() / (T+1);

    MatrixN Xr(N,T);
    MatrixN yr(N,T);

    wstring chunk,chunky;
    int n=0;
    int isf=10;
    for (int i=0; i<N; i++) {
        wstring smp = txt.sample(T+1);
        chunk=smp.substr(0,T);
        chunky=smp.substr(1,T);
        if (isf>0) {
            --isf;
            wcout << "1:" << chunk << endl;
            wcout << "y:" << chunky << endl;
        }
        for (int t=0; t<T; t++) {
            Xr(i,t)=txt.w2v[chunk[t]];
            yr(i,t)=txt.w2v[chunky[t]];
            ++n;
        }
    }

    int maxN = 50000;
    if (n>maxN) n=maxN;
    int n1=n*0.9;
    int dn=(n-n1)/2;

    cerr << n1 << " datasets" << endl;

/*    MatrixN X(n1,T);
    MatrixN y(n1,T);
    MatrixN Xv(dn,T);
    MatrixN yv(dn,T);
    MatrixN Xt(dn,T);
    MatrixN yt(dn,T);
*/
    MatrixN X=Xr.block(0,0,n1,T);
    MatrixN y=yr.block(0,0,n1,T);
    MatrixN Xv=Xr.block(n1,0,dn,T);
    MatrixN yv=yr.block(n1,0,dn,T);
    MatrixN Xt=Xr.block(n1+dn,0,dn,T);
    MatrixN yt=yr.block(n1+dn,0,dn,T);

    cpInitCompute("Rnnreader");
    registerLayers();

    LayerBlock lb("{name='rnnreader';init='normal';initfactor=0.5}");
    int VS=txt.vocsize();
    int H=48;
    int D=16;
    int BS=32;
    float clip=5.0;

    CpParams cp0;
    cp0.setPar("inputShape",vector<int>{T});
    cp0.setPar("V",VS);
    cp0.setPar("D",D);
    cp0.setPar("clip",clip);
    cp0.setPar("init",(string)"orthonormal");
    lb.addLayer("WordEmbedding","WE0",cp0,{"input"});

    CpParams cp1;
    cp1.setPar("inputShape",vector<int>{D,T});
    cp1.setPar("N",BS);
    cp1.setPar("H",H);
    cp1.setPar("clip",clip);
    cp1.setPar("initfactor","0.01");
    lb.addLayer("RNN","rnn1",cp1,{"WE0"});

    CpParams cp2;
    cp2.setPar("inputShape",vector<int>{H,T});
    cp2.setPar("N",BS);
    cp2.setPar("H",H);
    cp2.setPar("clip",clip);
//    lb.addLayer("RNN","rnn2",cp2,{"rnn1"});

    CpParams cp3;
    cp3.setPar("inputShape",vector<int>{H,T});
    cp3.setPar("N",BS);
    cp3.setPar("H",H);
    cp3.setPar("clip",clip);
//    lb.addLayer("RNN","rnn3",cp3,{"rnn2"});

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
    // preseverstates not necessary for training!
    CpParams cpo("{verbose=true;shuffle=false;preservestates=false;epsilon=1e-8}");
    // CpParams cpo("{verbose=false;epsilon=1e-8}");
    cpo.setPar("learning_rate", (floatN)4e-3); //2.2e-2);
    cpo.setPar("lr_decay", (floatN)0.95);
    //cpo.setPar("regularization", (floatN)1.);

    floatN dep=20.0;
    floatN sep=0.0;
    cpo.setPar("epochs",(floatN)dep);
    cpo.setPar("batch_size",BS);

    for (int i=0; i<10000; i++) {

        cpo.setPar("startepoch", (floatN)sep);
        //cpo.setPar("maxthreads", (int)1);  // XXX: can be increased now!
        t_cppl states;
        t_cppl statesv;
        states["y"] = new MatrixN(y);
        statesv["y"] = new MatrixN(yv);
        floatN cAcc=lb.train(X, &states, Xv, &statesv, "Adam", cpo);
        cppl_delete(&states);
        cppl_delete(&statesv);

        sep+=dep;

        int pos=rand() % 1000 + 5000;
        wstring instr=txt.text.substr(pos,T);

        MatrixN xg(1,T);
        for (int t=0; t<T; t++) {
            xg(0,t) = txt.w2v[' '];
        }

        Layer* prnn=lb.layerMap["rnn1"];
        t_cppl states;
        prnn->getZeroStates(&states, 1);

        for (int g=0; g<200; g++) {
            t_cppl cache{};

            MatrixN probst=lb.forward(xg,&cache, &states);
            MatrixN probsd=MatrixN(N*TT,VS);
            for (int n=0; n<1; n++) {
                for (int t=0; t<TT; t++) {
                    for (int d=0; d<VS; d++) {
                        probsd(n*TT+t,d)=probst(n,t*VS+d);
                    }
                }
            }

            for (int t=0; t<TT; t++) {
                vector<floatN> probs(VS);
                vector<floatN> index(VS);
                for (int d=0; d<VS; d++) {
                    probs[d]=probsd(0*TT+t,d);
                    index[d]=d;
                }
                int ind=(int)index[randomChoice(index, probs)];
*//*                    float mx=-1000.0;
                int ind=-1;
                for (int d=0; d<VS; d++) {
                    floatN p=yg(0,t*D+d);
                    floatN pr=p*((floatN)(rand()%100)/5000.0+0.98);
                    if (pr>mx) {
                        mx=pr; // yg(0,t*D+d);
                        ind=d;
                    }
                }
*//*
                    wchar_t cw=txt.v2w[ind];
                    //if (t==0) wcout << L"[" << cw << L"<";
                    //wcout << L"<" << cw << L">";
                    if (t==0) wcout << cw;  //  << L"(" << ind << L")";
                    // if (ind==0) cerr << "probs: " << probs << endl;
                    xg2(0,t)=ind;
                }
                //wcout << L"<" << endl;

                //for (int t=T-1; t>0; t--) xg(0,t)=xg(0,t-1);
                //for (int t=0; t< T-1; t++) xg(0,t)=xg(0,t+1);
                xg(0,0)=xg2(0,0);

                rnn1_ho=*(cache["rnn1-ho"]);
                //rnn2_ho=*(cache["rnn2-ho"]);
                //rnn3_ho=*(cache["rnn3-ho"]);
                //xg=xg2;
                cppl_delete(&cache);
                cppl_delete(&states);
            }
            //cache.clear();
            wcout << endl;
        }
        cerr << "setting eliminated T-Steps param" << endl;
        lb.cp.setPar("T-Steps",T);


*/
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
