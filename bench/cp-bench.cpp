#include "cp-neural.h"
#include <iomanip>
#include <curses.h>
#include <fstream>
#include <iostream>

// Manual build:
// g++ -g -ggdb -I ../cpneural -I /usr/local/include/eigen3 bench.cpp -L ../Build/cpneural/ -lcpneural -lpthread -o bench

using std::cerr; using std::endl;
using std::setprecision; using std::setw; using std::fixed;

map<string, string> benchRecipes = {
    {"OneHot", "{benchIdx=0;benchName='OneHot';benchDataType='int100';benchN=100;V=100;inputShape=[100]}"},
    {"Affine", "{benchIdx=1;benchName='Affine';benchN=100;inputShape=[1024];hidden=1024}"},
    {"AffineRelu", "{benchIdx=2;benchName='AffineRelu';benchN=100;inputShape=[1024];hidden=1024}"},
    {"Relu", "{benchIdx=3;benchName='Relu';benchN=100;inputShape=[1024]}"},
    {"Nonlinearity0", "{benchIdx=4;benchName='Nonlin-Relu';benchN=100;type='relu';inputShape=[1024]}"},
    {"Nonlinearity1", "{benchIdx=5;benchName='Nonlin-Sgmd';benchN=100;type='sigmoid';inputShape=[1024]}"},
    {"Nonlinearity2", "{benchIdx=6;benchName='Nonlin-Tanh';benchN=100;type='tanh';inputShape=[1024]}"},
    {"Dropout", "{benchIdx=7;benchName='Dropout';benchN=100;inputShape=[1024];drop=0.5}"},
    {"BatchNorm", "{benchIdx=8;benchName='BatchNorm';benchN=100;inputShape=[1024]}"},
    {"SpatialBatchNorm", "{benchIdx=9;benchName='SpatialBtchNrm';benchN=100;N=100;inputShape=[3,32,32]}"},
    {"Convolution", "{benchIdx=10;benchName='Convolution';benchN=100;inputShape=[3,32,32];kernel=[64,5,5];stride=1;pad=2}"},
    {"Pooling", "{benchIdx=11;benchName='Pooling';benchN=100;inputShape=[3,64,64];stride=2}"},
    {"LSTM", "{benchIdx=12;benchName='LSTM';benchN=100;inputShape=[100,80];N=100;H=256}"},
    {"RNN", "{benchIdx=13;benchName='RNN';benchN=100;inputShape=[100,80];N=100;H=256}"},
    {"TemporalAffine", "{benchIdx=14;benchName='TemporalAff';benchN=100;inputShape=[100,80];N=100;M=256}"},
    {"WordEmbedding", "{benchIdx=15;benchName='WordEmbed';benchN=100;inputShape=[100];D=100;V=256}"},
    {"Svm", "{benchIdx=16;benchName='SVM';benchN=100;inputShape=[1024]}"},
    {"Softmax", "{benchIdx=17;benchName='Softmax';benchN=100;inputShape=[1024]}"},
    {"TemporalSoftmax", "{benchIdx=18;benchName='TemporalSM';benchN=100;inputShape=[100,80]}"},
    {"TwoLayerNet", "{benchIdx=19;benchName='TwoLayerNet';benchN=100;inputShape=[1024];hidden=[1024,1024]}"},
};

MatrixN benchMean;
MatrixN historyMean;

bool saveBench(const string fname) {
	std::ofstream file(fname);
	if (file.is_open()) {
		file << benchMean << '\n';
        file.close();
        return true;
	}
    return false;
}

void split(const string &s, const char delim, vector<string> &elems) {
    std::stringstream ss;
    ss.str(s);
    string item;
    while (getline(ss, item, delim)) {
        if (item!=" " && item!="") elems.push_back(item);
    }
}

bool loadBench(const string fname) {
    std::ifstream file(fname);
	if (file.is_open()) {
        for (int i=0; i<historyMean.rows(); i++) {
            string line;
            getline(file,line);
            vector<string>vs;
            split(line,' ',vs);
            if (vs.size()==8) {
                for (int j=0; j<8; j++) {
                    historyMean(i,j)=(floatN)stod(vs[j]);
                }
            } else {
                cerr << "BADsize!=8: " << vs.size() << endl;
                file.close();
                return false;
            }
        }
        file.close();
        return true;
	}
    return false;
}
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

float DM=1.0;
float updateMean(int y,int x,float f) {
    if (benchMean(y,x)==0) benchMean(y,x)=f;
    else benchMean(y,x)=(benchMean(y,x)*DM+f)/(DM+1);
    return benchMean(y,x);
}

void colprint(floatN f) {
    if (f<0.0) {
        attron(COLOR_PAIR(1));
        printw("%4.1f",f);
        attroff(COLOR_PAIR(1));
    } else if (f>0.0) {
        attron(COLOR_PAIR(2));
        printw("%4.1f",f);
        attroff(COLOR_PAIR(2));
    } else printw("%4.1f",f);
}

bool benchLayer(string name, Layer* player, MatrixN &X, MatrixN &y, int row0) {
    Timer tcpu;
    float tcus, tcusn, tfn, tf,tb, tfx, tbx;
    t_cppl cache;
    t_cppl grads;
    t_cppl states;
    MatrixN ya;
    floatN htf,htb;

    int row=row0+1;
    int N=X.rows();
    states["y"] = new MatrixN(y);

    cerr.precision(3);
    cerr << fixed;

    tfn=1e8; tf=1e8; tb=1e8;
    floatN ptf=0.0; floatN ptb=0.0;

    tcpu.startCpu();
    ya=player->forward(X,&cache,&states,0);
    tcus=tcpu.stopCpuMicro()/1000.0;
    if (tcus<tf) tf=tcus;
    tf=updateMean(row0,0,tf);
    htf=historyMean(row0,0);
    if (htf!=0.0) ptf=(htf-tf)/tf*100.0;

    tcpu.startCpu();
    player->backward(ya,&cache, &states, &grads, 0);
    tcus=tcpu.stopCpuMicro()/1000.0;
    if (tcus<tb) tb=tcus;
    tb=updateMean(row0,1,tb);
    htb=historyMean(row0,1);
    if (htb!=0.0) ptb=(htb-tb)/tb*100.0;

    cppl_delete(&cache);
    cppl_delete(&grads);
    cppl_delete(&states);

    tfx= tf / (double)N;
    tfx=updateMean(row0,2,tfx);
    tbx= tb / (double)N;
    tbx=updateMean(row0,3,tbx);

    move(row,17); printw("%8.4f", tf);
    move(row,24); printw("%8.4f", tfx);
    if (ptf!=0.0) { move(row,33); colprint(ptf); }
    move(row,38); printw("%8.4f", tb);
    move(row,45); printw("%8.4f", tbx);
    if (ptb!=0.0) { move(row,54); colprint(ptb); }
    return true;
}

int doBench() {
    bool allOk=true;
    bool doSave=false;
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);
    //Eigen::setNbThreads(0);
    int mreps=100;
    benchMean=MatrixN(benchRecipes.size(),8);
    benchMean.setZero();
    historyMean=MatrixN(benchRecipes.size(),8);
    historyMean.setZero();
    int maxrow=0;
    clear();
    for (int mrep=0; mrep<mreps; mrep++) {
        clear();
        move(0,0); printw("Layer             fw(ms) fw/N(ms) bw(ms) bw/N(ms)");
        for (auto it : benchRecipes) {
            string classname=it.first;
            while (isdigit(classname[classname.size()-1])) {
                classname=classname.substr(0,classname.size()-1);
            }
            CpParams cp(benchRecipes[it.first]);
            int row=cp.getPar("benchIdx",(int)-1);
            if (row>=benchRecipes.size()) {
                cerr << "benchIdx must not be equal or greater than size of benchRecipes, error in: " << it.first << endl;
                exit(-1);
            }
            if (row==-1) {
                cerr << "No benchIdx defined for: " << it.first << endl;
                exit(-1);
            }
            string name=cp.getPar("benchName",it.first);
            move(row+1,0); printw(name.c_str());
            if (row+1 > maxrow) maxrow=row+1;
            t_layer_props_entry te=_syncogniteLayerFactory.mapprops[classname];
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
            MatrixN y;
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
            Layer *pl = CREATE_LAYER(classname, cp)
            if (pl->layerType & LayerType::LT_LOSS) {
                y=MatrixN(bs,pl->outputShape[0]);
                for (auto k=0; k<y.size(); k++) {
                    y(k)=rand()%isv[0];
                }
            }
            if (!benchLayer(classname, pl, X, y, row)) {
                cerr << "Error" << endl;
                allOk=false;
            }
            delete pl;
        }
        move(maxrow+1,0);
        printw("q - quite   s - snapshot   c - compare       ");
        refresh();
        if (mrep<mreps/1.3) {
            DM += 0.8;
        }
        int c=getch();
        if (c=='q') break;
        else if (c=='s') {doSave=true; saveBench("bench.txt");}
        else if (c=='c') loadBench("bench.txt");
    }
    move(maxrow+1,0);
    endwin();
    if (doSave) saveBench("bench.txt");
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
    initscr(); // curses
    cbreak(); // no character buffering
    noecho(); // no echo
    nodelay(stdscr, TRUE); // async keyboard checks
    // scrollok(stdscr, TRUE); // if scrolling is needed.
    start_color();			/* Start color 			*/
    init_pair(1, COLOR_RED, COLOR_BLACK);
    init_pair(2, COLOR_GREEN, COLOR_BLACK);

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
