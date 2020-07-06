#ifndef _CPL_NONLINEARITY_H
#define _CPL_NONLINEARITY_H

#include "../cp-layers.h"

namespace Nonlin {
    enum Nonlin {
        NL_INVALID = 0,
        NL_RELU = 1,
        NL_SIGMOID = 2,
        NL_TANH = 3,
        NL_SELU = 4,
        NL_RESILU = 5
    };
}


class Nonlinearity : public Layer {
private:
    Nonlin::Nonlin nonlintype=Nonlin::NL_INVALID;
    void setup(const json& jx) {
        layerType=LayerType::LT_NORMAL;
        j=jx;
        inputShapeRang=1;
        vector<int> inputShape=j.value("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        string nonlintypestr=j.value("type", (string)"relu");
        string layerNameS=j.value("name",(string)"Nonlin");
        layerName=layerNameS+"-"+nonlintypestr;
        layerClassName="Nonlinearity";
        if (nonlintypestr=="relu") nonlintype=Nonlin::NL_RELU;
        else if (nonlintypestr=="sigmoid") nonlintype=Nonlin::NL_SIGMOID;
        else if (nonlintypestr=="tanh") nonlintype=Nonlin::NL_TANH;
        else if (nonlintypestr=="selu") nonlintype=Nonlin::NL_SELU;
        else if (nonlintypestr=="resilu") nonlintype=Nonlin::NL_RESILU;
        outputShape=inputShape;
        if (nonlintype!=Nonlin::NL_INVALID) layerInit=true;
        else cerr << "Invalid type for nonlinearity: " << nonlintypestr << endl;
    }
public:
    MatrixN Sigmoid(const MatrixN& m) {
        MatrixN mn(m);
        MatrixN y;
        // Standard sigmoid is not numerically stable (large -x instable)
        //y=(1.0/(1.0+(mn.array() * -1.0).exp()));
        // Alternative via tanh is stable and doesn't need case distinctions for large + - inf values of x.
        // alt see: http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        y=((mn.array()/2.0).tanh()-1.0)/2.0+1.0;
        return y;
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
    MatrixN Selu(const MatrixN& x) {
        // "scaled exponential linear units" (SELUs), https://arxiv.org/abs/1706.02515
        floatN alpha = 1.6732632423543772848170429916717;
        floatN scale = 1.0507009873554804934193349852946;
        MatrixN y=x;
        // return scale*np.where(x>=0.0, x, alpha*np.exp(x)-alpha)
        for (unsigned int i=0; i<(unsigned int)y.size(); i++) {
            if (y(i)<0.0) y(i)=alpha*exp(y(i))-alpha;
        }
        y=y*scale;
        return y;
    }
    MatrixN dSelu(const MatrixN& x) {
        // "scaled exponential linear units" (SELUs), https://arxiv.org/abs/1706.02515
        MatrixN y=x;
        floatN alpha = 1.6732632423543772848170429916717;
        floatN scale = 1.0507009873554804934193349852946;
        for (unsigned int i=0; i<(unsigned int)y.size(); i++) {
            if (y(i)>=0) y(i)=scale;
            else y(i)=alpha*exp(y(i))*scale;
        }
        return y;
    }
    MatrixN Resilu(const MatrixN& x) {
        // Resilu should combine a non-linearity (similar to relu) and a residual, linear connection
        // r(x) = 1/(1-exp(-x/a)). Variing a limits in relu. r(x) can be rewritten as r(x)=x/(exp(x)-1) + x  (the linear, additive residual part)
        // Problem: phase-transition at 0, hence:
        // approximate with taylor-expansion up to O(6) for -h<x<h
        floatN h=0.001;
        floatN x2;
        MatrixN y=x.array()/(1.0-(x.array() * -1.0).exp());
        for (unsigned int i=0; i<(unsigned int)x.size(); i++) {
            if (x(i) < h && x(i) > -1.0 * h ) {
                x2=x(i)*x(i);
                y(i)=1.0+x(i)/2.0+x2/12.0-x2*x2/720.0+x2*x2*x2/30240.0;
            }
        }
        return y; 
    }
    MatrixN dResilu(const MatrixN& x) {
        // approximate with taylor-expansion up to O(6) for -h<x<h
        floatN h=0.001;
        floatN x2;
        MatrixN e=x.array().exp();
        MatrixN e1 = e.array()-1.0;
        MatrixN y = e.array()*(e.array()-x.array()-1.0)/e1.array()/e1.array();
        for (unsigned int i=0; i<(unsigned int)x.size(); i++) {
            if (x(i) < h && x(i) > -1.0 * h ) {
                x2 = x(i)*x(i);
                y(i)=0.5+x(i)/6.0-x2*x(i)/180.0+x2*x2*x(i)/5040.0-x2*x2*x2*x(i)/151200; //  //x(i)/6.0+0.5819767068693265;  // 0.58.. == 1/(e-1)
            }
        }
        return y; 

    }

    Nonlinearity(const json& jx) {
        setup(jx);
    }
    Nonlinearity(const string conf) {
        setup(json::parse(conf));
    }
    ~Nonlinearity() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl *pcache, t_cppl* pstates, int id=0) override {
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
        case Nonlin::NL_SELU:
            y=Selu(x);
            if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
            break;
        case Nonlin::NL_RESILU:
            y=Resilu(x);
            if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
            break;
        default:
            cerr << "Bad initialization for nonlinearity, no known type!" << endl;
            break;
        }
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl *pcache, t_cppl* pstates, t_cppl *pgrads, int id=0) override {
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
            case Nonlin::NL_SELU:
                y=*((*pcache)["x"]);
                dxc=dSelu(y);
                break;
            case Nonlin::NL_RESILU:
                y=*((*pcache)["x"]);
                dxc=dResilu(y);
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
