#include <cp-neural.h>
#include <random>

// Manual build:
// g++ -g -ggdb -I ../cpneural -I /usr/local/include/eigen3 minimal.cpp -L ../build/cpneural/ -lcpneural -lpthread -o test

using std::cerr; using std::endl;

int tFunc(floatN x, std::mt19937 &gen, bool disturb=true) {
    std::uniform_real_distribution<> dis(-0.1, 0.1);
    int y=0;
    if (disturb) {
        y=(int)((x+dis(gen))*2+5);
    } else {
        y=(int)((x)*2+5);
    }
    if (y<0) y=0;
    if (y>10) y=10;
    //cerr << x << ":" << y << " " << endl;
    return y;
}

bool trainTest(string init) {
    bool allOk=true;
    json j;
    int N=5000,NV=500,NT=500,I=5,H=50,C=10;
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

    j["inputShape"]=vector<int>{I};
    j["hidden"]=vector<int>{H,C};
    j["init"]=init;
    TwoLayerNet tln(j);

    MatrixN X(N,I);
    X.setRandom();
    MatrixN y(N,1);
    for (unsigned i=0; i<y.rows(); i++) {
        floatN s = 0.0;
        for (unsigned j=0; j<1; j++) {
            s += X(i,j);
        }
        y(i,0)=tFunc(s,gen,true);
    }

    MatrixN Xv(NV,I);
    Xv.setRandom();
    MatrixN yv(NV,1);
    for (unsigned i=0; i<yv.rows(); i++) {
        floatN s = 0.0;
        for (unsigned j=0; j<1; j++) {
            s += Xv(i,j);
        }
        yv(i,0)=tFunc(s,gen,true);
    }

    MatrixN Xt(NT,I);
    Xt.setRandom();
    MatrixN yt(NT,1);
    for (unsigned i=0; i<yt.rows(); i++) {
        floatN s = 0.0;
        for (unsigned j=0; j<1; j++) {
            s += Xt(i,j);
        }
        yt(i,0)=tFunc(s,gen,false);
    }

    json jo(R"({"verbose":false,"epochs":10000.0,"batch_size":100,"learning_rate":1e-2,"threads":8})"_json);

    floatN train_err,test_err,val_err;

    tln.train(X, y, Xv, yv, "Adam", jo);
    //tln.train(X, y, Xv, yv, "Sdg", cpo);
    train_err=tln.test(X, y);
    val_err=tln.test(Xv, yv);
    test_err=tln.test(Xt, yt);

    cerr << "Train-test, train-err=" << train_err << endl;
    cerr << "       validation-err=" << val_err << endl;
    cerr << "       final test-err=" << test_err << endl;
    if (test_err>0.2 || val_err>0.2 || train_err>0.2) allOk=false;
    return allOk;
}

void initTest() {
    cerr << "standard init=============";// << endl;
    trainTest("standard");
    cerr << "normal init===============";// << endl;
    trainTest("normal");
    cerr << "orthonormal init==========";// << endl;
    trainTest("orthonormal");
    cerr << "orthogonal init==========";// << endl;
    trainTest("orthogonal");
}

int main(int argc, char *argv[]) {
    string name="test";
    cpInitCompute(name);
    int ret=0;

    initTest();
    return ret;

}
