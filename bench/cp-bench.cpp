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

bool benchLayer(string name, Layer* player) {
    MatrixN x(100,1000);
    MatrixN y(100,1);
    x.setRandom();
    y.setRandom();
    for (int i=0; i<y.size(); i++) {
        y(i)=(floatN)(int)((y(i)+1)*5);
    }
    Timer tcpu;
    double tcus;
    t_cppl cache;
    t_cppl grads;
    string sname;

    sname=name;

    while (sname.size() < 12) sname += " ";

    cout.precision(3);
    cout << fixed;
    /*
    tcpu.startCpu();
    player->forward(x,nullptr);
    tcus=tcpu.stopCpuMicro()/1000.0;
    cout << sname << " forward (no cache):    " << fixed << setw(8) << tcus << "ms (cpu)." << endl;
    */
    tcpu.startCpu();
    if (player->layerType==LayerType::LT_NORMAL) {
        player->forward(x,&cache);
    } else {
        player->forward(x,y,&cache);
    }
    tcus=tcpu.stopCpuMicro()/1000.0;
    cout << sname << " forward (with cache):  " << fixed << setw(8) << tcus << "ms (cpu)." << endl;

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
    cout << sname << " backward (with cache): " << fixed << setw(8) << tcus << "ms (cpu)." << endl;
    cppl_delete(&cache);
    cppl_delete(&grads);

    return true;
}

int doBench() {
    bool allOk=true;
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);
    cout << "bench start" << endl;
    int nr=0;
    for (auto it : _syncogniteLayerFactory.mapl) {
        ++nr;
        cout << nr << ".: " << it.first << endl;
        t_layer_props_entry te=_syncogniteLayerFactory.mapprops[it.first];
        CpParams cp;
        std::vector<int> tp(te);
        for (auto i=0; i< tp.size(); i++) tp[i]=1000;
        cp.setPar("topo",tp);
        Layer *l = CREATE_LAYER(it.first, cp)
        for (int rep=0; rep<4; rep++) {
            if (!benchLayer(it.first, l)) {
                cout << "Error" << endl;
                allOk=false;
            }
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
