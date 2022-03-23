#ifndef _CPL_AFFINE_H
#define _CPL_AFFINE_H

#include "../cp-layers.h"

class Affine : public Layer {
private:
    int numGpuThreads;
    int numCpuThreads;
    int hidden;
    floatN initfactor;
    void setup(const json& jx) {
        j=jx;
        layerName=j.value("name",(string)"Affine");
        layerClassName="Affine";
        inputShapeRang=1;
        layerType=LayerType::LT_NORMAL;
        vector<int> inputShape=j.value("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        hidden=j.value("hidden",1024);
        string inittype=j.value("init",(string)"standard");
        XavierMode initmode=xavierInitType(inittype);
        initfactor=j.value("initfactor",(floatN)1.0);
        outputShape={hidden};

        MatrixN W = xavierInit(MatrixN(inputShapeFlat,hidden),initmode,initfactor);
        MatrixN b = xavierInit(MatrixN(1,hidden),initmode,initfactor);
        cppl_set(&params, "W", new MatrixN(W)); // W
        cppl_set(&params, "b", new MatrixN(b)); // b
        numCpuThreads=cpGetNumCpuThreads();

        layerInit=true;
    }
public:
    Affine(const json& jx) {
        setup(jx);
    }
    Affine(const string conf) {
        setup(json::parse(conf));
    }
    ~Affine() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        if (params["W"]->rows() != x.cols()) {
            cerr << layerName << ": " << "Forward: dimension mismatch in x*W: x:" << shape(x) << " W:" << shape(*params["W"]) << endl;
            MatrixN y(0,0);
            return y;
        }
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));

        MatrixN y=(x * (*(params["W"]))).rowwise() + RowVectorN(*params["b"]);
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) override {
        MatrixN x(*(*pcache)["x"]);
        MatrixN dx(x.rows(),x.cols());
        MatrixN W(*params["W"]);
        MatrixN dW(W.rows(),W.cols());
        dx = dchain * (*params["W"]).transpose(); // dx
        cppl_set(pgrads, "W", new MatrixN((*(*pcache)["x"]).transpose() * dchain)); //dW
        cppl_set(pgrads, "b", new MatrixN(dchain.colwise().sum())); //db
        return dx;
    }
};

#endif
