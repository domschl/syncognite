#include <cp-neural.h>

// Manual build:
// g++ -g -ggdb -I ../cpneural -I /usr/local/include/eigen3 minitest.cpp -L ../Build/cpneural/ -lcpneural -lpthread -o test

using std::cerr; using std::endl;

int doTests() {
    MatrixN w(10,10);
    MatrixN wx;
    cerr << "-------------Standard-init" << endl;
    wx=xavierInit(w,XavierMode::XAV_STANDARD);
    cerr << wx << endl;
    cerr << "-------------Normal-init" << endl;
    wx=xavierInit(w,XavierMode::XAV_NORMAL);
    cerr << wx << endl;
    cerr << "-------------Orthonormal-init" << endl;
    wx=xavierInit(w,XavierMode::XAV_ORTHONORMAL);
    cerr << wx << endl;

    cerr << "-------------Orthonormal-init (check multi)" << endl;
    MatrixN o=wx * wx.transpose();
    cerr << o << endl;

    return 0;
}

int tFunc(floatN x, int c) {
    int y=(int)(((sin(x*2.0)+1.0)/2.0)*(floatN)c);
    //cerr << x << ":" << y << " " << endl;
    return y;
}

bool trainTest(string init) {
    bool allOk=true;
    json j;
    int N=3000,NV=300,NT=300,I=5,H=20,C=4;
    j["inputShape"]=vector<int>{I};
    j["hidden"]=vector<int>{H,C};
    j["init"]=init;
    TwoLayerNet tln(j);

    MatrixN X(N,I);
    X.setRandom();
    MatrixN y(N,1);
    for (unsigned i=0; i<y.rows(); i++) y(i,0)=tFunc(X(i,0),C);

    MatrixN Xv(NV,I);
    Xv.setRandom();
    MatrixN yv(NV,1);
    for (unsigned i=0; i<yv.rows(); i++) yv(i,0)=tFunc(Xv(i,0),C);

    MatrixN Xt(NT,I);
    Xt.setRandom();
    MatrixN yt(NT,1);
    for (unsigned i=0; i<yt.rows(); i++) yt(i,0)=tFunc(Xt(i,0),C);

    json jo(R"({"verbose":true, "epochs":100.0, "batch_size":20, "lr_decay":1.0, "threads":2})"_json);
	jo["regularization"]=(floatN)2e-8;

    json j_opt(R"({"name":"Adam","beta1":0.9,"beta2":0.999,"epsilon":1e-8})"_json);
	j_opt["learning_rate"]=(floatN)1e-2;
    json j_loss(R"({"name":"CrossEntropy"})"_json);
    Optimizer *pOptimizer=optimizerFactory("Adam", j_opt);
    t_cppl OptimizerState{};
    Loss *pLoss=lossFactory("SparseCategoricalCrossEntropy", j_loss);

	tln.train(X, y, Xv, yv, pOptimizer, &OptimizerState, pLoss, jo);

    delete pOptimizer;
    cppl_delete(&OptimizerState);
    delete pLoss;

    floatN train_err,test_err,val_err;
    //tln.train(X, y, Xv, yv, "SDG", cpo);
    train_err=tln.test(X, y);
    val_err=tln.test(Xv, yv);
    test_err=tln.test(Xt, yt);

    cerr << "Train-test, train-err=" << train_err << endl;
    cerr << "       validation-err=" << val_err << endl;
    cerr << "       final test-err=" << val_err << endl;
    if (test_err>0.2 || val_err>0.2 || train_err>0.2) allOk=false;
    return allOk;
}

void initTest() {
    cerr << "standard init=============" << endl;
    trainTest("standard");
    cerr << "normal init===============" << endl;
    trainTest("normal");
    cerr << "orthonormal init==========" << endl;
    trainTest("orthonormal");
    cerr << "orthogonal init==========" << endl;
    trainTest("orthogonal");
}

void jsonTest() {
    json j;
    j["test"]=vector<int>{3,4,5};
    j["turbo"]["traffic"]["tangente"]=3.13;
    cerr << j << endl;
    if (j["tandem"]["kat"]==nullptr)
        cerr << "not defined"<< endl;
    string a;
    a=j.value("murksel","tada");
    cerr << a << endl;
    j["murksel"]="ding";
    a=j.value("murksel","tada");
    cerr << a << endl;
    int b;
    b=j.value("hidden",13);
    cerr << b << endl;
    j["hidden"]=1024;
    b=j.value("hidden",13);
    cerr << b << endl;


    json j2 = R"({"test":32,"turbo":"loader","c2": [2]})"_json;
    cerr << j2 << endl;
    string l=j2["turbo"];
    cerr << j2["turbo"]<<endl;
    cerr << j.dump(4) << endl;
    cerr << j2.dump(4) << endl;
}

void hd5Test() {
    int inputShapeFlat=1024;
    int hidden=1024;
    string inittype{"standard"};
    floatN initfactor=1.0;

    json j1;
    j1["inputShape"]=vector<int>{inputShapeFlat};
    j1["hidden"]=hidden;
    j1["init"]=inittype;
    j1["initfactor"]=initfactor;

    string filepath{"minitest.h5"};
    H5::H5File file((H5std_string)filepath, H5F_ACC_TRUNC );
    // H5File file(FILE_NAME, H5F_ACC_RDWR);

    Affine *af1=new Affine(j1);
    af1->saveParameters(&file);
    file.close();

    Affine *af2=new Affine(j1);
    H5::H5File filer((H5std_string)filepath, H5F_ACC_RDONLY);
    af2->loadParameters(&filer);
    filer.close();

    if (matCompare(*af1->params["W"], *af2->params["W"], "W")) cerr << "W: ok." << endl;
    else cerr << "W: test failure." << endl;

    if (matCompare(*af1->params["b"], *af2->params["b"], "b")) cerr << "b: ok." << endl;
    else cerr << "b: test failure." << endl;
}

int main(int argc, char *argv[]) {
    string name="test";
    cpInitCompute(name);
    int ret=0;

    doTests();
    jsonTest();
    hd5Test();
    initTest();
    return ret;

}
