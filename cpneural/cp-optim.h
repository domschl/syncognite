#ifndef _CP_OPTIM_H
#define _CP_OPTIM_H

#include "cp-layer.h"


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
    bool verbose;
    if (popti->getPar("verbose", 0) == 0) verbose=false;
    else verbose=true;
    floatN lr = popti->getPar("learning_rate", 1.0e-2);
    cout << ep << " " << bs << " " << lr << endl;

    floatN l=0.0;
    int chunks=(x.rows()+bs-1) / bs;
    for (int e=0; e<ep; e++) {
        cout << "epoch: " << e << endl;
        for (int b=0; b<chunks; b++) {
            int y0,dy;
            y0=b*bs;
            if (y0+bs > x.rows()) dy=x.rows()-y0;
            else dy=bs;
            MatrixN xb=x.block(y0,0,dy,x.cols());
            MatrixN yb=y.block(y0,0,dy,y.cols());
            //cout << "chunk: " << b << " x:" << shape(xb) << " y:" << shape(yb) << endl;
            forward(xb);
            l=loss(yb);
            backward(yb);
            update(popti);
            cout << l << " ";
            if ((b+1)%8==0) cout << endl;
        }
        cout << endl << "Ep." << e << " Loss:" << l << endl;
    }
    return 0.0;
}
#endif
