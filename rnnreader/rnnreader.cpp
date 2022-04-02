#include <iostream>
#include <fstream>
#include <codecvt>
//#include <cstddef>
#include <locale>
#include <clocale>
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
        std::wifstream txtfile(inputfile, std::ios::binary);
        if (!txtfile.is_open()) {
            return false;
        }
        txtfile.imbue(std::locale("en_US.UTF-8"));
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
        return (int)v2w.size();
    }

    wstring sample(int len) {
        int p=std::rand()%(text.size()-len);
        return text.substr(p,len);
    }
};

void currentDateTime(wstring& timestr) {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    timestr=std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(buf);
}

int main(int argc, char *argv[]) {
#ifndef __APPLE__
    std::setlocale(LC_ALL, "");
#else
    setlocale(LC_ALL, "");
    std::wcout.imbue(std::locale("en_US.UTF-8"));
#endif
    wcout << L"Rnn-Readär" << std::endl;

    bool allOk=true;
    if (argc!=2) {
        cerr << L"rnnreader <path-to-text-file>" << endl;
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

    int timeSteps=64;

    int N=(int)txt.text.size() / (timeSteps+1);
    cerr << N << " Max datasets" << endl;
    MatrixN Xr(N,timeSteps);
    MatrixN yr(N,timeSteps);

    wstring chunk,chunky;
    int n=0;
    for (int i=0; i<N; i++) {
        wstring smp = txt.sample(timeSteps+1);
        chunk=smp.substr(0,timeSteps);
        chunky=smp.substr(1,timeSteps);
        for (int t=0; t<timeSteps; t++) {
            Xr(i,t)=txt.w2v[chunk[t]];
            yr(i,t)=txt.w2v[chunky[t]];
        }
        ++n;
    }

    int maxN = 100000;
    if (n>maxN) n=maxN;
    int n1=n*0.9;
    int dn=(n-n1)/2;
    if (n1+2*dn > n) {
        cerr << "Math doesn't work out." << endl;
    }

    cerr << n1 << " datasets, " << dn << " test-sets, " << dn << " validation-sets" << endl;

    MatrixN X=Xr.block(0,0,n1,timeSteps);
    MatrixN y=yr.block(0,0,n1,timeSteps);
    MatrixN xVal=Xr.block(n1,0,dn,timeSteps);
    MatrixN yVal=yr.block(n1,0,dn,timeSteps);
    MatrixN xTest=Xr.block(n1+dn,0,dn,timeSteps);
    MatrixN yTest=yr.block(n1+dn,0,dn,timeSteps);
    
    unsigned int ctm=(unsigned int)std::time(0);
    std::srand(ctm);

    cpInitCompute("Rnnreader");
    registerLayers();

    // LayerBlockOldStyle lb(R"({"name":"rnnreader","init":"orthonormal"})"_json);
    LayerBlockOldStyle lb(R"({"name":"rnnreader","init":"orthogonal"})"_json);
    int vocabularySize=txt.vocsize();
    int H=128; // 400;
    int batchSize=64;
    float clip=5.0;

    string rnntype="LSTM"; // or "RNN"
    cerr << "RNN-type: " << rnntype << endl;

    json j0;
    string oName{"OH0"};
    j0["inputShape"]=vector<int>{timeSteps};
    j0["V"]=vocabularySize;
    lb.addLayer("OneHot",oName,j0,{"input"});

    string nName;
    json j1;
    j1["inputShape"]=vector<int>{vocabularySize,timeSteps};
    j1["N"]=batchSize;
    j1["H"]=H;
    j1["forgetgateinitones"]=true;
    //j1["forgetbias"]=0.85;
    //j1["clip"]=clip;
    int layerDepth=2; // 6;
    j1["H"]=H;
    for (auto l=0; l<layerDepth; l++) {
        if (l>0) j1["inputShape"]=vector<int>{H,timeSteps};
        nName="lstm"+std::to_string(l);
        lb.addLayer(rnntype,nName,j1,{oName});
        oName=nName;
    }

    json j10;
    j10["inputShape"]=vector<int>{H,timeSteps};
    j10["M"]=vocabularySize;
    lb.addLayer("TemporalAffine","af1",j10,{oName});

    json j11;
    j11["inputShape"]=vector<int>{vocabularySize,timeSteps}; // currently inputShape of TempSoftmax MUST match inputShape of TemporalLoss
    lb.addLayer("TemporalSoftmax","sm1",j11,{"af1"});

    if (!lb.checkTopology(true)) {
        allOk=false;
        cerr << red << "Topology-check for LayerBlock: ERROR." << def << endl;
        exit(-1);
    } else {
        cerr << green << "Topology-check for LayerBlock: ok." << def << endl;
    }

    floatN episodes=5.0; // 70.0;
    floatN currentEpoch=0.0;

    json train_params(R"({"verbose":true,"shuffle":false,"notests":false,"nofragmentbatches":true})"_json);
    train_params["lossfactor"]=1.0/(floatN)timeSteps;  // Allows to normalize the loss with timeSteps.
    train_params["epochs"]=(floatN)episodes;
    train_params["batch_size"]=batchSize;
    // train_params["lr_decay"] = 1.0; // every 40 epochs, lr = lr/10 (0.945^40 = 0.104)

    json j_opt(R"({"learning_rate": 2.75e-3})"_json);
    Optimizer *pOpt=optimizerFactory("Adam",j_opt);
    t_cppl OptimizerState{};
    
    json j_loss(R"({"name":"temporalsoftmax"})"_json);
    j_loss["inputShape"]=vector<int>{vocabularySize,timeSteps};
    Loss *pLoss=lossFactory("TemporalCrossEntropy",j_loss);

    cerr << "Training parameters: " << train_params.dump(4) << endl;
    cerr << "Optimizer parameters: " << pOpt->getOptimizerParameters().dump(4) << endl;
    cerr << "Loss parameters: " << pLoss->getLossParameters().dump(4) << endl;

    for (int i=0; i<1000; i++) {
        train_params["startepoch"]=(floatN)currentEpoch;
        t_cppl states;
        t_cppl statesVal;
        states["y"] = new MatrixN(y);
        statesVal["y"] = new MatrixN(yVal);
        lb.train(X, &states, xVal, &statesVal, pOpt, &OptimizerState, pLoss, train_params);
        cppl_delete(&states);
        cppl_delete(&statesVal);

        currentEpoch+=episodes;

        int pos=rand() % 1000 + 5000;
        wstring instr=txt.text.substr(pos,timeSteps);

        MatrixN xg(1,timeSteps);
        for (int i=0; i<timeSteps; i++) {
            xg(0,i)=txt.w2v[instr[i]];
       }
        wstring sout{};
        t_cppl statesg{};
        Layer* plstm0=lb.layerMap["lstm0"];
        plstm0->genZeroStates(&statesg, 1);
        int g,t,v;
        TemporalSoftmax *pSm1=(TemporalSoftmax *)lb.layerMap["sm1"];

        for (floatN temp : {0.8, 1.0, 1.2, 1.4, 1.7}) {
            pSm1->setTemperature(temp); // XXX we should reset LSTM states here.
            cerr << "---- Temperature: " << temp << " ------------------" << endl;
            sout=wstring{};
            for (g=0; g<300; g++) {
                t_cppl cache{};

                MatrixN probst=lb.forward(xg,&cache, &statesg);
                MatrixN probsd=MatrixN(timeSteps,vocabularySize);
                for (t=0; t<timeSteps; t++) {
                    for (v=0; v<vocabularySize; v++) {
                        probsd(t,v)=probst(0,t*vocabularySize+v);
                    }
                }
                int li = -1;
                for (t=0; t<timeSteps; t++) {
                    vector<floatN> probs(vocabularySize);
                    vector<floatN> index(vocabularySize);
                    for (v=0; v<vocabularySize; v++) {
                        probs[v]=probsd(t,v);
                        index[v]=v;
                    }
                    li=(int)index[randomChoice(index, probs)];
                }
                cppl_delete(&cache);

                for (int t=0; t<timeSteps-1; t++) {
                    xg(0,t)=xg(0,t+1);
                }
                xg(0,timeSteps-1)=li;
                sout += txt.v2w[li];
            }
            wcout << sout << endl;
        }
        pSm1->setTemperature(1.0);
        wstring timestr;
        currentDateTime(timestr);
        std::wofstream fl("rnnreader.txt", std::ios_base::app);
        fl << "---- " << timestr << ", ep:" << currentEpoch << " ---" << endl;
        fl << sout << endl;
        fl.close();
        cppl_delete(&statesg);
    }
    delete pOpt;
    cppl_delete(&OptimizerState);
    delete pLoss;
}
