#ifndef _CP_LAYERS_H
#define _CP_LAYERS_H

#include "cp-layer.h"

class optimizer {
public:
    t_params params;
        /*
        optimizer(map<string, floatN>& ps) {
        for (parmap iterator = ps.begin; iterator != ps.end(); iterator++) {
            params[iterator->first]=iterator.second;
            params=ps;
            }
        }
        */
    floatN getPar(string par, floatN def) {
        auto it=param.find[par];
        if (it=ps.end()) {
            param[par]=def;
        }
        return param[par];
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
    sdg(parmap& ps) {
        param=ps;
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

floatN train(MatrixN& x, MatrixN& y, string optimizer, t_params pars) {
    Optimizer* popti=optimizerFactory("sdg", pars);
    unsigned int ep=(unsigned int)poti->getPar("epochs", 1.0);
    unsigned int bs=(unsigned int)poti->getPar("batch_size", 100.0);

    for (unsigned int i=0; i<ep; i++) {
        forward(x);
        loss(y);
        backward(y);
        // optimize(poti);
    }
    return 0.0;
}
#endif
