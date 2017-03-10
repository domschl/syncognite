#ifndef _CPL_RELU_H
#define _CPL_RELU_H

#include "../cp-layers.h"

class Relu : public Layer {
private:
    void setup(const json& jx) {
        layerName="Relu";
        layerClassName="Relu";
        layerType=LayerType::LT_NORMAL;
        j=jx;
        inputShapeRang=1;
        vector<int> inputShape=j.value("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        outputShape=inputShape;
        layerInit=true;
    }
public:
    Relu(const json& jx) {
        setup(jx);
    }
    Relu(const string conf) {
        setup(json::parse(conf));
    }
    ~Relu() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl *pcache, t_cppl* pstates, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        return (x.array().max(0)).matrix();
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl *pcache, t_cppl* pstates, t_cppl *pgrads, int id=0) override {
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
