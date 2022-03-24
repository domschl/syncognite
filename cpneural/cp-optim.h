#pragma once

#include "cp-neural.h"

/** @brief Stochastic gradient descent optimizer.
 *
 */
class SDG : public Optimizer {
    floatN lr;
public:
    SDG(const json& jx) {
        /** Most simple optimizer: stochastic gradient descent.
         *
         * @param jx JSON object that should contain "learning_rate".
         */
        j=jx;
    }
    virtual MatrixN update(MatrixN& w, MatrixN& dw, string wName, t_cppl* pCache) override {
        /** Update the parameters of the neural network.
         *
         * @param w Matrix of parameters.
         * @param dw Gradient of the loss function.
         * @param wName Name of the parameter w. (Unused for SDG, since no state)
         * @param pCache Pointer to the state-cache for the optimizer. (Unused for SDG, since no state)
         * @return Updated parameters.
         */
        lr=j.value("learning_rate", (floatN)1e-2);
        w=w-lr*dw;
        return w;
    }
};

/*
  Performs stochastic gradient descent with momentum. [CS231]
  Params:
  - learning_rate: Scalar learning rate.
  - momentum: Scalar between 0 and 1 giving the momentum value.
    Setting momentum = 0 reduces to sgd.
*/
class SDGmomentum : public Optimizer {
    floatN lr;
    floatN mm;
public:
    SDGmomentum(const json& jx) {
        j=jx;
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl* pocache) override {
        lr=j.value("learning_rate", (floatN)1e-2);
        mm=j.value("momentum",(floatN)0.9);
        string cname=var+"-velocity";
        if (pocache->find("cname_v")==pocache->end()) {
            MatrixN z=MatrixN(x);
            z.setZero();
            cppl_update(pocache, cname, &z);
        }
        MatrixN dxw;
        MatrixN v;
        v=*(*pocache)[cname];
        dxw = lr*dx - mm*v;
        *(*pocache)[cname]= (-1.0) * dxw;
        x=x-dxw;
        return x;
    }
};


//  Uses the RMSProp update rule, which uses a moving average of squared gradient
//  values to set adaptive per-parameter learning rates. [CS231]
class RMSprop : public Optimizer {
    floatN lr;
    floatN dc,ep;
public:
    RMSprop(const json& jx) {
        j=jx;
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl* pocache) override {
        lr=j.value("learning_rate", (floatN)1e-2);
        dc=j.value("decay_rate", (floatN)0.99);
        ep=j.value("epsilon", (floatN)1e-8);
        string cname=var+"-movavr";
        if (pocache->find("cname")==pocache->end()) {
            MatrixN z=MatrixN(x);
            z.setZero();
            cppl_update(pocache, cname, &z);
        }
        *(*pocache)[cname]=dc * (*(*pocache)[cname]) + ((1.0 - dc) * (dx.array() * dx.array())).matrix();
        MatrixN dv=((*(*pocache)[cname]).array().sqrt() + ep).matrix();
        for (int i=0; i<dv.size(); i++) {
            if (dv(i)>0.0) dv(i)=1.0/dv(i);
            else cerr<<"BAD ALGO!" << endl;
        }
        x = x - (lr * dx.array() * dv.array()).matrix();
        return x;
    }
};


// Uses the Adam update rule, which incorporates moving averages of both the
// gradient and its square and a bias correction term. [CS231]
// - learning_rate: Scalar learning rate.
// - beta1: Decay rate for moving average of first moment of gradient.
// - beta2: Decay rate for moving average of second moment of gradient.
// - epsilon: Small scalar used for smoothing to avoid dividing by zero.
class Adam : public Optimizer {
    floatN lr;
    floatN b1,b2,ep;
public:
    Adam(const json& jx) {
        j=jx;
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl* pocache) override {
        lr=j.value("learning_rate", (floatN)1e-2);
        b1=j.value("beta1", (floatN)0.9);
        b2=j.value("beta2", (floatN)0.999);
        ep=j.value("epsilon", (floatN)1e-8);
        string cname_m=var+"-m";
        if (pocache->find("cname_m")==pocache->end()) {
            MatrixN z=MatrixN(x);
            z.setZero();
            cppl_update(pocache, cname_m, &z);
        }
        string cname_v=var+"-v";
        if (pocache->find("cname_v")==pocache->end()) {
            MatrixN z=MatrixN(x);
            z.setZero();
            cppl_update(pocache, cname_v, &z);
        }
        string cname_t=var+"-t";
        if (pocache->find("cname_t")==pocache->end()) {
            MatrixN z1(1,1);
            z1.setZero();
            cppl_update(pocache, cname_t, &z1);
        }
        floatN t=(*(*pocache)[cname_t])(0,0) + 1.0;
        (*(*pocache)[cname_t])(0,0)=t;
        *(*pocache)[cname_m]=b1 * (*(*pocache)[cname_m]) + (1.0 -b1) * dx;
        *(*pocache)[cname_v]=b2 * (*(*pocache)[cname_v]).array() + (1.0 -b2) * (dx.array() * dx.array());
        MatrixN mc = 1.0/(1.0-pow(b1,t)) * (*(*pocache)[cname_m]);
        MatrixN vc = 1.0/(1.0-pow(b2,t)) * (*(*pocache)[cname_v]);
        x = x.array() - lr * mc.array() / (vc.array().sqrt() + ep);
        return x;
    }
};

Optimizer *optimizerFactory(string name, const json& j) {
    if (name=="SDG") return (Optimizer *)new SDG(j);
    if (name=="SDGmomentum") return (Optimizer *)new SDGmomentum(j);
    if (name=="RMSprop") return (Optimizer *)new RMSprop(j);
    if (name=="Adam") return (Optimizer *)new Adam(j);
    cerr << "optimizerFactory called for unknown optimizer " << name << "." << endl;
    return nullptr;
}
