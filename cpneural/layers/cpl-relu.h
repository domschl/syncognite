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
    virtual MatrixN forward(const MatrixN& x, t_cppl *pcache, t_cppl* pstates, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        //MatrixN ych(x);
        //cerr << "RL:" << x.size() << endl << x << endl;
        return (x.array().max(0)).matrix();
/*
        for (int n=0; n<x.rows(); n++) {
            for (int i=0; i<x.cols(); i++) {
                if (x(n,i)<0.0) ych(n,i)=0.0;
            }
        } */
/*        for (unsigned int i=0; i<ych.size(); i++) {
            if (ych(i)<0.0) {
                ych(i)=0.0;
            }
        } */
        // return ych;
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
