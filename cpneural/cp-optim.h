#ifndef _CP_OPTIM_H
#define _CP_OPTIM_H

#include "cp-layer.h"

class Optimizer {
public:
    cp_t_params<int> iparams;
    cp_t_params<floatN> fparams;
    floatN getPar(string par, floatN def) {
        auto it=fparams.find(par);
        if (it==fparams.end()) {
            fparams[par]=def;
        }
        return fparams[par];
    }
    int getPar(string par, int def) {
        auto it=iparams.find(par);
        if (it==iparams.end()) {
            iparams[par]=def;
        }
        return iparams[par];
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx) {return x;};
};

class sdg : public Optimizer {
    floatN lr;
public:
    sdg(cp_t_params<floatN>& ps) {
        fparams=ps;
        lr=getPar("learning_rate", 1e-2);
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx) override {
        x=x-lr*dx;
        return x;
    }
};

Optimizer *optimizerFactory(string name, cp_t_params<floatN> params) {
    if (name=="sdg") return (Optimizer *)new sdg(params);
    cout << "optimizerFactory called for unknown optimizer " << name << "." << endl;
    return nullptr;
}


floatN Layer::train(MatrixN& x, MatrixN& y, string optimizer, cp_t_params<int> ipars, cp_t_params<floatN> fpars) {
    Optimizer* popti=optimizerFactory("sdg", fpars);
    popti->fparams=fpars;
    popti->iparams=ipars;
    int ep=popti->getPar("epochs", 1);
    int bs=popti->getPar("batch_size", 100);
    floatN lr = popti->getPar("learning_rate", 1.0e-3);
    cout << ep << " " << bs << " " << lr << endl;

    for (unsigned int i=0; i<ep; i++) {

        //forward(x);
        //loss(y);
        //backward(y);
        // optimize(poti);
    }
    return 0.0;
}
#endif
