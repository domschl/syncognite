#ifndef _CP_OPTIM_H
#define _CP_OPTIM_H

#include "cp-layer.h"
#include "cp-timer.h"


class sdg : public Optimizer {
    floatN lr;
public:
    sdg(cp_t_params<floatN>& ps) {
        fparams=ps;
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx) override {
        lr=getPar("learning_rate", 1e-2);
        x=x-lr*dx;
        return x;
    }
};

Optimizer *optimizerFactory(string name, cp_t_params<floatN> params) {
    if (name=="sdg") return (Optimizer *)new sdg(params);
    cout << "optimizerFactory called for unknown optimizer " << name << "." << endl;
    return nullptr;
}


floatN Layer::train(MatrixN& x, MatrixN& y, MatrixN &xv, MatrixN &yv, string optimizer, cp_t_params<int> ipars, cp_t_params<floatN> fpars) {
    Optimizer* popti=optimizerFactory("sdg", fpars);
    popti->fparams=fpars;
    popti->iparams=ipars;
    int ep=popti->getPar("epochs", 1);
    int bs=popti->getPar("batch_size", 100);
    floatN lr_decay=popti->getPar("lr_decay", 1.0);
    bool verbose;
    if (popti->getPar("verbose", 0) == 0) verbose=false;
    else verbose=true;
    floatN lr = popti->getPar("learning_rate", 1.0e-2);
    cout << ep << " " << bs << " " << lr << endl;
    Timer t1;
    double dfus, dbus;

    floatN l=0.0;
    int chunks=(x.rows()+bs-1) / bs;
    for (int e=0; e<ep; e++) {
        cout << "Epoch: " << e+1 << " learning-rate:" << lr << endl;
        for (int b=0; b<chunks; b++) {
            int y0,dy;
            t_cppl cache;
            t_cppl grads;
            y0=b*bs;
            if (y0+bs > x.rows()) dy=x.rows()-y0;
            else dy=bs;
            MatrixN xb=x.block(y0,0,dy,x.cols());
            MatrixN yb=y.block(y0,0,dy,y.cols());
            //cout << "chunk: " << b << " x:" << shape(xb) << " y:" << shape(yb) << endl;
            //t1.startCpu();
            forward(xb, &cache);
            //dfus=t1.stopCpuMicro()/(double)dy;
            //t1.startCpu();
            l=loss(yb, &cache);
            backward(yb, &cache, &grads);
            //dbus=t1.stopCpuMicro()/(double)dy;
            update(popti, &grads);
            //if ((b+1)%20==0) cout << dfus << " " << dbus << endl;
            cppl_delete(&cache);
            cppl_delete(&grads);
        }
        cout << "Loss:" << l << " err(validation):" << test(xv,yv) << endl;
        if (lr_decay!=1.0) {
            lr *= lr_decay;
            popti->setPar("learning_rate", lr);
        }
    }
    return 0.0;
}
#endif
