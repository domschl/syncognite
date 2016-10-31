#ifndef _CPL_NONLINEARITY_H
#define _CPL_NONLINEARITY_H

#include "../cp-layers.h"

namespace Nonlin {
    enum Nonlin {
        NL_INVALID = 0,
        NL_RELU = 1,
        NL_SIGMOID = 2,
        NL_TANH = 3
    };
}


class Nonlinearity : public Layer {
private:
    Nonlin::Nonlin nonlintype=Nonlin::NL_INVALID;
    void setup(const CpParams& cx) {
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        inputShapeRang=1;
        vector<int> inputShape=cp.getPar("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        string nonlintypestr=cp.getPar("nonlinearitytype", (string)"relu");
        layerName="Nonlinearity-"+nonlintypestr;
        if (nonlintypestr=="relu") nonlintype=Nonlin::NL_RELU;
        else if (nonlintypestr=="sigmoid") nonlintype=Nonlin::NL_SIGMOID;
        else if (nonlintypestr=="tanh") nonlintype=Nonlin::NL_TANH;
        outputShape=inputShape;
        if (nonlintype!=Nonlin::NL_INVALID) layerInit=true;
        else cerr << "Invalid type for nonlinearitytype: " << nonlintypestr << endl;
    }
public:
    MatrixN Sigmoid(const MatrixN& m) {
        MatrixN mn(m);
        //mn.colwise() -=  mn.rowwise().maxCoeff();
        return (1.0/(1.0+(mn.array() * -1.0).exp())).matrix();
    }
    MatrixN dSigmoid(const MatrixN& y) {
        return (y.array()*(y.array()-1.0) * -1.0).matrix();
    }
    MatrixN Tanh(const MatrixN& m) {
        return (m.array().tanh()).matrix();
    }
    MatrixN dTanh(const MatrixN& y) {
        return (1.0-y.array()*y.array()).matrix();
    }
    MatrixN Relu(const MatrixN& m) {
        return (m.array().max(0)).matrix();
    }
    MatrixN dRelu(const MatrixN& x) {
        MatrixN y=x;
        for (unsigned int i=0; i<y.size(); i++) {
            if (y(i)>0.0) y(i)=1.0;
            else y(i)=0.0;
        }
        return y;
    }
    Nonlinearity(const CpParams& cx) {
        setup(cx);
    }
    Nonlinearity(string conf) {
        setup(CpParams(conf));
    }
    ~Nonlinearity() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl *pcache, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        MatrixN y=x;
        switch (nonlintype) {
            case Nonlin::NL_RELU:
                y=Relu(x);
                break;
            case Nonlin::NL_SIGMOID:
                y=Sigmoid(x);
                if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
                break;
            case Nonlin::NL_TANH:
                y=Tanh(x);
                if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
                break;
            default:
                cerr << "Bad initialization for nonlinearity, no known type!" << endl;
                break;
        }
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl *pcache, t_cppl *pgrads, int id=0) override {
        MatrixN dxc;
        MatrixN x,y;
        switch (nonlintype) {
            case Nonlin::NL_RELU:
                x=*((*pcache)["x"]);
                dxc=dRelu(x);
                break;
            case Nonlin::NL_SIGMOID:
                y=*((*pcache)["y"]);
                dxc=dSigmoid(y);
                break;
            case Nonlin::NL_TANH:
                y=*((*pcache)["y"]);
                dxc=dTanh(y);
                break;
            default:
                cerr << "Bad initialization for nonlinearity, no known type!" << endl;
                break;
        }
        MatrixN dx = dxc.cwiseProduct(dchain); // dx
        return dx;
    }
};

#endif
