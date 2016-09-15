#include "cp-neural.h"
#include <iomanip>

// Manual build:
// g++ -g -ggdb -I ../cpneural -I /usr/local/include/eigen3 testneural.cpp -L ../Build/cpneural/ -lcpneural -lpthread -o test

using std::cout; using std::endl;
using std::setprecision; using std::setw; using std::fixed;

bool matComp(MatrixN& m0, MatrixN& m1, string msg="", floatN eps=1.e-6) {
    if (m0.cols() != m1.cols() || m0.rows() != m1.rows()) {
        cout << msg << ": Incompatible shapes " << shape(m0) << "!=" << shape(m1) << endl;
        return false;
    }
    MatrixN d = m0 - m1;
    floatN dif = d.cwiseProduct(d).sum();
    if (dif < eps) {
        cout << msg << " err=" << dif << endl;
        return true;
    } else {
        IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        cout << msg << " m0:" << endl << m0.format(CleanFmt) << endl;
        cout << msg << " m1:" << endl << m1.format(CleanFmt) << endl;
        cout << "err=" << dif << endl;
        return false;
    }
}

bool benchLayer(string name, Layer* player, int N, int M) {
    MatrixN x(N,M);
    MatrixN y(N,1);
    x.setRandom();
    y.setRandom();
    for (int i=0; i<y.size(); i++) {
        y(i)=(floatN)(int)((y(i)+1)*5);
    }
    Timer tcpu;
    double tcus, tcusn, tfn, tf,tb, tfx, tbx;
    t_cppl cache;
    t_cppl grads;
    string sname;

    sname=name;

    while (sname.size() < 12) sname += " ";

    cout.precision(3);
    cout << fixed;

    tfn=1e8; tf=1e8; tb=1e8;
    for (int rep=0; rep<5; rep++) {

        /*
        tcpu.startCpu();
        player->forward(x,nullptr);
        tcus=tcpu.stopCpuMicro()/1000.0;
        if (tcus<tfn) tfn=tcus;
        */
        tcpu.startCpu();
        if (player->layerType==LayerType::LT_NORMAL) {
            player->forward(x,&cache);
        } else {
            player->forward(x,y,&cache);
        }
        tcus=tcpu.stopCpuMicro()/1000.0;
        if (tcus<tf) tf=tcus;

        if (name=="BatchNorm") {
            return false;
        }

        tcpu.startCpu();
        if (player->layerType==LayerType::LT_NORMAL) {
            player->backward(x,&cache, &grads);
        } else {
            player->backward(y,&cache, &grads);
        }
        tcus=tcpu.stopCpuMicro()/1000.0;
        if (tcus<tb) tb=tcus;

        cppl_delete(&cache);
        cppl_delete(&grads);
    }

    tfx= tf / (double)N;
    tfx= tb / (double)N;
    //cout << sname << " forward (no cache):    " << fixed << setw(8) << tcus << "ms (cpu), " << endl;
    cout << sname << " forward (with cache):  " << fixed << setw(8) << tf << "ms (cpu), " << tfx << "ms (/sample)." << endl;
    cout << sname << " backward (with cache): " << fixed << setw(8) << tb << "ms (cpu), " << tbx << "ms (/sample)." << endl;

    return true;
}

int doBench() {
    bool allOk=true;
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);
    int nr=0;
    int N=200;
    int M=1000;
    for (auto it : _syncogniteLayerFactory.mapl) {
        ++nr;
        t_layer_props_entry te=_syncogniteLayerFactory.mapprops[it.first];
        CpParams cp;

        if (te>1) cout << nr << ".: " << it.first << " N=" << N << " dim=[" << M << "x" << M << "]"<< endl;
        else cout << nr << ".: " << it.first << " N=" << N << " dim=[" << M << "]" << endl;

        std::vector<int> tp(te);
        for (auto i=0; i< tp.size(); i++) tp[i]=M;
        cp.setPar("topo",tp);
        Layer *l = CREATE_LAYER(it.first, cp)
        if (!benchLayer(it.first, l, N, M)) {
            cout << "Error" << endl;
            allOk=false;
        }
        delete l;
    }
    cout << "bench end, nr=" << nr << endl;
    return allOk;
}

int main() {
    Eigen::initParallel();
    registerLayers();

    int ret=0;
    ret=doBench();
    return ret;
}
