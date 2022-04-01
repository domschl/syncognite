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
    virtual MatrixN update(MatrixN& w, MatrixN& dw, string wName, t_cppl* pOptimizerState) override {
        /** Update the parameters of the neural network.
         *
         * @param w Matrix of parameters.
         * @param dw Gradient of the loss function.
         * @param wName Name of the parameter w. (Unused for SDG, since no state)
         * @param pOptimizerState Pointer to the state-cache for the optimizer. (Unused for SDG, since no state)
         * @return Updated parameters.
         */
        lr=j.value("learning_rate", (floatN)1e-2);
        w=w-lr*dw;
        return w;
    }
};

/** @brief Stochastic gradient descent optimizer with momentum.
 * Algorithm based on CS231 course notes.
 */
class SDGmomentum : public Optimizer {
    floatN lr;
    floatN mm;
public:
    SDGmomentum(const json& jx) {
        /** Stochastic gradient descent with momentum.
         *
         * Momentum should be between 0 and 1, a value of 0 gives SDG.
         *
         * @param jx JSON object that should contain "learning_rate" and "momentum".
         */
        j=jx;
    }
    virtual MatrixN update(MatrixN& w, MatrixN& dw, string wName, t_cppl* pOptimizerState) override {
        /** Update the parameters of the neural network.
         *
         * @param w Matrix of parameters.
         * @param dw Gradient of the loss function.
         * @param wName Name of the parameter w. (used for naming the cache-state of momentum)
         * @param pOptimizerState Pointer to the state-cache for the optimizer. (will hold momentum state)
         * @return Updated parameters.
         */
        lr=j.value("learning_rate", (floatN)1e-2);
        mm=j.value("momentum",(floatN)0.9);
        string vName=wName+"-velocity";
        if (pOptimizerState->find(vName)==pOptimizerState->end()) {
            MatrixN z=MatrixN(w);
            z.setZero();
            cppl_update(pOptimizerState, vName, &z);
        }
        MatrixN dwm;
        MatrixN v;
        v=*(*pOptimizerState)[vName];
        dwm = lr*dw - mm*v;
        *(*pOptimizerState)[vName]= (-1.0) * dwm;
        w=w-dwm;
        return w;
    }
};


/** @brief RMSProp optimizer.
 *  Algorithm based on CS231 course notes. 
 */
class RMSprop : public Optimizer {
    floatN lr; /** learning rate */
    floatN dc; /** decay rate */
    floatN ep; /** epsilon */
public:
    RMSprop(const json& jx) {
        /** RMSProp optimizer.
         *
         * @param jx JSON object that should contain "learning_rate", "decay_rate" and "epsilon".
         */
        j=jx;
    }
    virtual MatrixN update(MatrixN& w, MatrixN& dw, string wName, t_cppl* pOptimizerState) override {
        /** Update the parameters of the neural network.
         *
         * @param w Matrix of parameters.
         * @param dw Gradient of the loss function.
         * @param wName Name of the parameter w. (used for naming the cache-state of moving average)
         * @param pOptimizerState Pointer to the state-cache for the optimizer. (will hold moving average state)
         * @return Updated parameters.
         */
        lr=j.value("learning_rate", (floatN)1e-3);
        dc=j.value("decay_rate", (floatN)0.9);
        ep=j.value("epsilon", (floatN)1e-7);
        string cName=wName+"-movavr";
        if (pOptimizerState->find(cName)==pOptimizerState->end()) {
            MatrixN z=MatrixN(w);
            z.setZero();
            cppl_update(pOptimizerState, cName, &z);
        }
        *(*pOptimizerState)[cName]=dc * (*(*pOptimizerState)[cName]) + ((1.0 - dc) * (dw.array() * dw.array())).matrix();
        MatrixN dv=((*(*pOptimizerState)[cName]).array().sqrt() + ep).matrix();
        for (int i=0; i<dv.size(); i++) {
            if (dv(i)>0.0) dv(i)=1.0/dv(i);
            else cerr<<"BAD ALGO!" << endl;
        }
        w = w - (lr * dw.array() * dv.array()).matrix();
        return w;
    }
};


// Uses the Adam update rule, which incorporates moving averages of both the
// gradient and its square and a bias correction term. [CS231]
// - learning_rate: Scalar learning rate.
// - beta1: Decay rate for moving average of first moment of gradient.
// - beta2: Decay rate for moving average of second moment of gradient.
// - epsilon: Small scalar used for smoothing to avoid dividing by zero.
/** @brief Adam optimizer.
 *  Algorithm based on CS231 course notes. 
 */
class Adam : public Optimizer {
    floatN b1; /** beta1: decay rate for first moment */
    floatN b2; /** beta2: decay rate for second moment */
    floatN ep; /** epsilon */
public:
    floatN lr; /** learning rate */
    Adam(const json& jx) {
        /** Adam optimizer.
         *
         * @param jx JSON object that should contain "learning_rate", "beta1" and "beta2".
         */
        j=jx;
    }
    virtual MatrixN update(MatrixN& w, MatrixN& dw, string wName, t_cppl* pOptimizerState) override {
        /** Update the parameters of the neural network.
         *
         * @param w Matrix of parameters.
         * @param dw Gradient of the loss function.
         * @param wName Name of the parameter w. (used for naming the state-variables of optimizer state parameters)
         * @param pOptimizerState Pointer to the state for the optimizer. (will hold optimizer state parameters)
         * @return Updated parameters.
         */
        lr=j.value("learning_rate", (floatN)1e-3);
        b1=j.value("beta1", (floatN)0.9);
        b2=j.value("beta2", (floatN)0.999);
        ep=j.value("epsilon", (floatN)1e-7);
        string cName_m=wName+"-m";
        if (pOptimizerState->find(cName_m)==pOptimizerState->end()) {
            MatrixN z=MatrixN(w);
            z.setZero();
            cppl_update(pOptimizerState, cName_m, &z);
        }
        string cName_v=wName+"-v";
        if (pOptimizerState->find(cName_v)==pOptimizerState->end()) {
            MatrixN z=MatrixN(w);
            z.setZero();
            cppl_update(pOptimizerState, cName_v, &z);
        }
        string cName_t=wName+"-t";
        if (pOptimizerState->find(cName_t)==pOptimizerState->end()) {
            MatrixN z1(1,1);
            z1.setZero();
            cppl_update(pOptimizerState, cName_t, &z1);
        }
        floatN t=(*(*pOptimizerState)[cName_t])(0,0) + 1.0;
        (*(*pOptimizerState)[cName_t])(0,0)=t;
        *(*pOptimizerState)[cName_m]=b1 * (*(*pOptimizerState)[cName_m]) + (1.0 -b1) * dw;
        *(*pOptimizerState)[cName_v]=b2 * (*(*pOptimizerState)[cName_v]).array() + (1.0 -b2) * (dw.array() * dw.array());
        MatrixN mc = 1.0/(1.0-pow(b1,t)) * (*(*pOptimizerState)[cName_m]);
        MatrixN vc = 1.0/(1.0-pow(b2,t)) * (*(*pOptimizerState)[cName_v]);
        w = w.array() - lr * mc.array() / (vc.array().sqrt() + ep);
        return w;
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
