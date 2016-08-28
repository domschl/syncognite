#ifndef _CP_LAYERS_H
#define _CP_LAYERS_H

#include "cp-layer.h"

class optimizer {
public:
    cp_t_params<int> iparams;
    cp_t_params<floatN> fparams;
    floatN getPar(string par, floatN def) {
        auto it=fparams.find[par];
        if (it=ps.end()) {
            fparams[par]=def;
        }
        return fparams[par];
    }
    int getPar(string par, int def) {
        auto it=iparams.find[par];
        if (it=ps.end()) {
            iparams[par]=def;
        }
        return iparams[par];
    }
    virtual update(MatrixN& x, MatrixN& dx) {return x;};
}

/*
def sgd(w, dw, config=None):
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  w -= config['learning_rate'] * dw
  return w, config
*/
class sdg : optimizer {
    floatN lr;
    sdg(cp_t_params<floatN>& ps) {
        fparam=ps;
        lr=getPar("learning_rate", 1e-2);
    }
    virtual update(MatrixN& x, MatrixN& dx) override {
        x=x-lr*dx;
        return x;
    };
}

Optimizer *optimizerFactory(string name, t_params params) {
    if (name=="sdg") return new sdg(params);
    cout << "optimizerFactory called for unknown optimizer " << name << "." << endl;
    return nullptr;
}


floatN Layer::floatN train(MatrixN& x, MatrixN& y, string optimizer, cp_t_params<int> ipars, cp_t_params<floatN> fpars) {
    fparams=fpars;
    iparams=ipars;
    Optimizer* popti=optimizerFactory("sdg", fpars);
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
