#ifndef _CPL_RELU_H
#define _CPL_RELU_H

#include "../cp-layers.h"

class Relu : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="Relu";
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        inputShapeRang=1;
        vector<int> inputShape=cp.getPar("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        outputShape=inputShape;
        layerInit=true;
    }
public:
    Relu(const CpParams& cx) {
        setup(cx);
    }
    Relu(string conf) {
        setup(CpParams(conf));
    }
    ~Relu() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl *pcache, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        MatrixN ych=x;
        for (unsigned int i=0; i<ych.size(); i++) {
            if (x(i)<0.0) {
                ych(i)=0.0;
            }
        }
        return ych;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl *pcache, t_cppl *pgrads, int id=0) override {
        MatrixN y=*((*pcache)["x"]);
        for (unsigned int i=0; i<y.size(); i++) {
            if (y(i)>0.0) y(i)=1.0;
            else y(i)=0.0;
        }
        MatrixN dx = y.cwiseProduct(dchain); // dx
        return dx;
    }
};

#endif
