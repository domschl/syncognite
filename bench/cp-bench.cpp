#include "cp-neural.h"
#include <iomanip>
#include <curses.h>

// Manual build:
// g++ -g -ggdb -I ../cpneural -I /usr/local/include/eigen3 bench.cpp -L ../Build/cpneural/ -lcpneural -lpthread -o bench

using std::cerr; using std::endl;
using std::setprecision; using std::setw; using std::fixed;

map<string, string> benchRecipes = {
    {"Affine", "{benchN=100;inputShape=[1024];hidden=1024}"},
    {"Relu", "{benchN=100;inputShape=[1024]}"},
    {"Nonlinearity", "{benchN=100;inputShape=[1024]}"},
    {"AffineRelu", "{benchN=100;inputShape=[1024];hidden=1024}"},
    {"Convolution", "{benchN=100;inputShape=[3,32,32];kernel=[64,5,5];stride=1;pad=2}"},
    {"Pooling", "{benchN=100;inputShape=[3,64,64];stride=2}"},
    {"Dropout", "{benchN=100;inputShape=[1024];drop=0.5}"},
    {"LSTM", "{benchN=100;inputShape=[100,80];N=100;H=256}"},
    {"RNN", "{benchN=100;inputShape=[100,80];N=100;H=256}"},

// BatchNorm, OneHot, Softmax, SpatialBatchNorm, Svm, TemporalAffine, TemporalSoftmax, TwoLayerNet, WordEmbedding
//    {"Relu", "{inputShape=[1024]}"},
//    {"Relu", "{inputShape=[1024]}"},
//    {"Relu", "{inputShape=[1024]}"}

};


bool matComp(MatrixN& m0, MatrixN& m1, string msg="", floatN eps=1.e-6) {
    if (m0.cols() != m1.cols() || m0.rows() != m1.rows()) {
        cerr << msg << ": Incompatible shapes " << shape(m0) << "!=" << shape(m1) << endl;
        return false;
    }
    MatrixN d = m0 - m1;
    floatN dif = d.cwiseProduct(d).sum();
    if (dif < eps) {
        cerr << msg << " err=" << dif << endl;
        return true;
    } else {
        IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        cerr << msg << " m0:" << endl << m0.format(CleanFmt) << endl;
        cerr << msg << " m1:" << endl << m1.format(CleanFmt) << endl;
        cerr << "err=" << dif << endl;
        return false;
    }
}

bool benchLayer(string name, Layer* player, MatrixN &X, MatrixN &y, int row) {
    Timer tcpu;
    float tcus, tcusn, tfn, tf,tb, tfx, tbx;
    t_cppl cache;
    t_cppl grads;
    t_cppl states;
    MatrixN ya;

    int N=X.rows();
    states["y"] = new MatrixN(y);

    cerr.precision(3);
    cerr << fixed;

    tfn=1e8; tf=1e8; tb=1e8;

    tcpu.startCpu();
    ya=player->forward(X,&cache,&states,0);
    tcus=tcpu.stopCpuMicro()/1000.0;
    if (tcus<tf) tf=tcus;

    tcpu.startCpu();
    player->backward(ya,&cache, &states, &grads, 0);
    tcus=tcpu.stopCpuMicro()/1000.0;
    if (tcus<tb) tb=tcus;

    cppl_delete(&cache);
    cppl_delete(&grads);
    cppl_delete(&states);

    tfx= tf / (double)N;
    tbx= tb / (double)N;

    move(row,17); printw("%8.4f", tf);
    move(row,24); printw("%8.4f", tfx);
    move(row,33); printw("%8.4f", tb);
    move(row,40); printw("%8.4f", tbx);

    return true;
}

int doBench() {
    bool allOk=true;
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);
    //Eigen::setNbThreads(0);
    int mreps=10;
    clear();
    for (int mrep=0; mrep<mreps; mrep++) {
        int row=0;
        move(0,0); printw("Layer                 fw(ms) fw/N(ms)      bw(ms) bw/N(ms)");
        for (auto it : benchRecipes) {
            ++row;
            move(row,0); printw(it.first.c_str());
            t_layer_props_entry te=_syncogniteLayerFactory.mapprops[it.first];
            if (benchRecipes.find(it.first)==benchRecipes.end()) {
                move(row, 24); printw("No bench recipe for layer %s", it.first.c_str());
            } else {
                CpParams cp(benchRecipes[it.first]);
                int bs=cp.getPar("benchN",(int)0);
                if (bs==0) {
                    cerr << "No benchN batch size defined in recipe for layer " << it.first << endl;
                    continue;
                }
                string bd=cp.getPar("benchDataType",(string)"");
                vector<int> isv=cp.getPar("inputShape", vector<int>{});
                if (isv.size()==0) {
                    cerr << "No inputShape batch size defined in recipe for layer " << it.first << endl;
                    continue;
                }
                int is=1;
                for (auto k=0; k<isv.size(); k++) is *= isv[k];
                MatrixN X(bs,is);
                MatrixN y(bs,1);
                if (bd=="") {
                    X.setRandom();
                } else if (bd=="int100") {
                    for (auto k=0; k<X.size(); k++) {
                        X(k)=rand()%100;
                    }
                } else {
                    cerr << "No benchDataType: " << bd << " is not understood in recipe for layer " << it.first << endl;
                    continue;
                }
                cp.setPar("train", true);
                Layer *pl = CREATE_LAYER(it.first, cp)
                if (pl->layerType & LayerType::LT_LOSS) {
                    for (auto k=0; k<y.size(); k++) {
                        y(k)=rand()%is;
                    }
                }
                if (!benchLayer(it.first, pl, X, y, row)) {
                    cerr << "Error" << endl;
                    allOk=false;
                }
                delete pl;
            }
        }
        move(row+1,0);
        refresh();
    }
    printw("Press enter...");
    getch();
    endwin();
    return allOk;
}

#ifdef USE_CUDA
void cudaBench() {
    int N=801, K=20000, M=70;
    MatrixN a(N,K),b(K,M),c0(N,M),c1(N,M);
    a.setRandom();
    b.setRandom();
    Timer t1;
    t1.startWall();
    c0=a*b;
    cerr << "Eigen: " << t1.stopWallMicro() << endl << endl;
    t1.startWall();
    c1=matmul(&a,&b,1,true);
    cerr << "Cuda : " << t1.stopWallMicro() << endl << endl;
    matComp(c0,c1,"cuda-divergence",0.01);
}
#endif

//HO = 1 + (H + 2 * pad - HH) / stride;
//WO = 1 + (W + 2 * pad - WW) / stride;
void colmagic() {
    int N=7;
    Convolution cv{"{inputShape=[2,3,3];kernel=[4,3,3];stride=1;pad=1}"};
    vector<int> vo=cv.getOutputShape();
    int F=vo[0], HO=vo[1], WO=vo[2];
    int rws=4;
    MatrixN m(rws,HO * WO * F * N / rws);
    m.setZero();
    for (int i=0; i<m.size(); i++) m(i%rws,i/rws)=i;
    cerr << "OLD:" << endl;
    cv.col2imx(m,N);
    cerr << "NEW:" << endl;
    cv.col2im(m,N);
}

void icolmagic() {
    int N=19;
    Convolution cv{"{inputShape=[2,3,3];kernel=[4,3,3];stride=1;pad=1}"};
    vector<int> vo=cv.getOutputShape();
    int F=vo[0], HO=vo[1], WO=vo[2];
    int rws=4;
    //MatrixN m(rws,HO * WO * F * N / rws);
    MatrixN m(N,HO * WO * F);
    m.setZero();
    for (int i=0; i<m.size(); i++) m(i)=i;
    cerr << "OLD:" << endl;
    MatrixN x1=cv.icol2imx(m,N);
    cerr << "NEW:" << endl;
    MatrixN x2=cv.icol2im(m,N);
    if (matComp(x1,x2)) {
        cerr << "seems good" << endl;
    }
}


int main() {
    initscr();
    cpInitCompute("Bench",nullptr,0);
    registerLayers();
    int ret=0;
    ret=doBench();

    #ifdef USE_CUDA
    if (cpGetNumGpuThreads()>0){
        cerr << "Cuda tests:" << endl;
        cudaBench();
    }
    #endif

    //icolmagic();

    cpExitCompute();
    return ret;
}
