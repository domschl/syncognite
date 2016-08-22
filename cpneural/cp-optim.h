#ifndef _CP_LAYERS_H
#define _CP_LAYERS_H

#include "cp-layer.h"

typedef map<string, floatN> parmap;

class optimizer {
public:
    parmap params;
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

#endif
