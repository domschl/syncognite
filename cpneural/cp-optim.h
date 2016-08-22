#ifndef _CP_LAYERS_H
#define _CP_LAYERS_H

#include "cp-layer.h"

typedef map<string, floatN> parmap;

class optimizer {
public:
    parmap params;
    optimizer(map<string, floatN>& ps) {
        for (parmap iterator = ps.begin; iterator != ps.end(); iterator++) {
            params[iterator->first]=iterator.second;
        }
    }
    update(MatrixN& x, MatrixN& dx);
}

class sdg : optimizer {

}

#endif
