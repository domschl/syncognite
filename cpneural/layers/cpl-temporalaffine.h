#ifndef _CPL_TEMPORALAFFINE_H
#define _CPL_TEMPORALAFFINE_H

#include "../cp-layers.h"

class  TemporalAffine : public Layer {
private:
    int hidden;
    void setup(const CpParams& cx) {
        layerName=" TemporalAffine";
        inputShapeRang=1;
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        vector<int> inputShape=cp.getPar("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        hidden=cp.getPar("hidden",1024);
        outputShape={hidden};

        cppl_set(&params, "W", new MatrixN(inputShapeFlat,hidden)); // W
        cppl_set(&params, "b", new MatrixN(1,hidden)); // b

        params["W"]->setRandom();
        floatN xavier = 1.0/std::sqrt((floatN)(inputShapeFlat+hidden)); // (setRandom is [-1,1]-> fakt 0.5, xavier is 2/(ni+no))
        *params["W"] *= xavier;
        params["b"]->setRandom();
        *params["b"] *= xavier;
        layerInit=true;
    }
public:
     TemporalAffine(const CpParams& cx) {
        setup(cx);
    }
     TemporalAffine(const string conf) {
        setup(CpParams(conf));
    }
    ~ TemporalAffine() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id=0) override {
        if (params["W"]->rows() != x.cols()) {
            cerr << layerName << ": " << "Forward: dimension mismatch in x*W: x:" << shape(x) << " W:" << shape(*params["W"]) << endl;
            MatrixN y(0,0);
            return y;
        }
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));

        MatrixN y(x.rows(), (*params["W"]).cols());
        y=(x * (*params["W"])).rowwise() + RowVectorN(*params["b"]);
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
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
