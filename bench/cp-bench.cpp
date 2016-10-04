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

bool benchLayer(string name, Layer* player, int N, int M, int reps) {
    MatrixN x(N,M);
    MatrixN y(N,1);
    MatrixN ya;
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
    for (int rep=0; rep<reps; rep++) {
        t_cppl cache;
        t_cppl grads;
        /*
        tcpu.startCpu();
        player->forward(x,nullptr);
        tcus=tcpu.stopCpuMicro()/1000.0;
        if (tcus<tfn) tfn=tcus;
        */
        tcpu.startCpu();
        if (player->layerType==LayerType::LT_NORMAL) {
            ya=player->forward(x,&cache,0);
        } else {
            ya=player->forward(x,y,&cache,0);
        }
        tcus=tcpu.stopCpuMicro()/1000.0;
        // cout << "Transform (" << name << "): " << shape(x) << "->" << shape(ya) << endl;
        if (tcus<tf) tf=tcus;

/*        if (name=="BatchNorm") {
            return false;
        }
*/
        tcpu.startCpu();
        if (player->layerType==LayerType::LT_NORMAL) {
            player->backward(ya,&cache, &grads, 0);
        } else {
            player->backward(y,&cache, &grads, 0);
        }
        tcus=tcpu.stopCpuMicro()/1000.0;
        if (tcus<tb) tb=tcus;

        cppl_delete(&cache);
        cppl_delete(&grads);
    }


    tfx= tf / (double)N;
    tbx= tb / (double)N;
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
    int reps=0;

    //Eigen::setNbThreads(0);
    for (int mrep=0; mrep<1; mrep++) {
    /*    if (mrep > 0) {
            cout << "Eigen thread count: " << mrep << endl;
            Eigen::setNbThreads(mrep);
        }
*/        for (auto it : _syncogniteLayerFactory.mapl) {
            ++nr;
            //if (mrep > 0 && it.first!="Affine") continue;

            t_layer_props_entry te=_syncogniteLayerFactory.mapprops[it.first];
            CpParams cp;

            if (te==2) cout << nr << ".: " << it.first << " N=" << N << " dim=[" << M << "x" << M << "]"<< endl;
            else if (te==1) cout << nr << ".: " << it.first << " N=" << N << " dim=[" << M << "]" << endl;
            else cout  << nr << ".: " << it.first << " N=" << N << endl;

            std::vector<int> tp(te);
            for (auto i=0; i< tp.size(); i++) tp[i]=M;
            int MI=M;
            if (it.first=="Convolution") {
/*                tp[0]=3;
                tp[1]=100;
                tp[2]=100;
                MI=100*100*3;
                tp[3]=8;
                tp[4]=5;
                tp[5]=5;
                cp.setPar("pad",2);
                cp.setPar("stride",3);
*/                tp[0]=3;
                tp[1]=31;
                tp[2]=31;
                MI=31*31*3;
                N=100;
                cp.setPar("kernel",vector<int>{25,3,3});
                cp.setPar("pad",1);
                cp.setPar("stride",1);
                cp.setPar("verbose",true);
            } else if (it.first=="Pooling") {
                tp[0]=3;
                tp[1]=100;
                tp[2]=100;
                MI=100*100*3;
                cp.setPar("pad",2);
                cp.setPar("stride",2);
            } else if (it.first=="SpatialBatchNorm") {
                tp[0]=3;
                tp[1]=100;
                tp[2]=100;
                cp.setPar("batch_norm",N);
                MI=100*100*3;
            }
            cp.setPar("topo",tp);
            cp.setPar("train", true);
            Layer *l = CREATE_LAYER(it.first, cp)
            if (reps==0) {
                //reps=10; // warm-up for first measurement
                //cout << "Warmup..." << endl;
                reps=1;
            }
            else reps=1;
            if (!benchLayer(it.first, l, N, MI, reps)) {
                cout << "Error" << endl;
                allOk=false;
            }
            delete l;
        }
        cout << endl;
    }
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
    cout << "Eigen: " << t1.stopWallMicro() << endl << endl;
    t1.startWall();
    c1=matmul(&a,&b,1,true);
    cout << "Cuda : " << t1.stopWallMicro() << endl << endl;
    matComp(c0,c1,"cuda-divergence",0.01);
}
#endif

int main() {
    cpInitCompute("Bench");


    registerLayers();
    int ret=0;
    ret=doBench();

    #ifdef USE_CUDA
    if (cpGetNumGpuThreads()>0){
        cout << "Cuda tests:" << endl;
        cudaBench();
    }
    #endif

    cpExitCompute();
    return ret;
}
