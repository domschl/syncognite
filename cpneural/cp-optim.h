#ifndef _CP_OPTIM_H
#define _CP_OPTIM_H

#include <future>
#include <list>
#include "cp-layer.h"
#include "cp-timer.h"
#include "cp-layers.h"


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
/*
Timer t1;
double dfus, dbus;
bool timeit=true;
// cout << "chunk: " << b << " x:" << shape(xb) << " y:" << shape(yb) << endl;
if (timeit) {
    t1.startCpu();
}
if (timeit) {
    dfus=t1.stopCpuMicro()/(double)dy;
    t1.startCpu();
}
if (timeit) {
    dbus=t1.stopCpuMicro()/(double)dy;
}
if (timeit) {
    cout << "forward pass: " << dfus << "µs, backward pass: " << dbus << "µs." << endl;
    timeit=false;
}

*/

t_cppl Layer::workerThread(const MatrixN& xb, const MatrixN& yb, floatN *ploss) {
    t_cppl cache;
    t_cppl grads;
    forward(xb, yb, &cache);
    *ploss=loss(yb, &cache);
    backward(yb, &cache, &grads);
    cppl_delete(&cache);
    return grads;
}

floatN Layer::train(const MatrixN& x, const MatrixN& y, const MatrixN &xv, const MatrixN &yv,
                string optimizer, cp_t_params<int> ipars, cp_t_params<floatN> fpars) {
    Optimizer* popti=optimizerFactory("sdg", fpars);
    popti->fparams=fpars;
    popti->iparams=ipars;
    int ep=popti->getPar("epochs", 1); //Default only!
    int bs=popti->getPar("batch_size", 100); // Defaults only! are overwritten!
    int nt=popti->getPar("threads",1); // Default only!
    floatN lr_decay=popti->getPar("lr_decay", 1.0); //Default only!
    bool verbose;
    if (popti->getPar("verbose", 0) == 0) verbose=false;
    else verbose=true;
    floatN lr = popti->getPar("learning_rate", 1.0e-2); // Default only!
    cout << ep << " " << bs << " " << lr << endl;

    floatN l=0.0;
    bs=bs/nt;
    int ebs=bs*nt;
    int chunks=(x.rows()+ebs-1) / ebs;
    cout << "Data-size" << x.rows() << ", chunks: " << chunks << ", batch_size: " << bs << ", threads: " << nt << endl;
    for (int e=0; e<ep; e++) {
        cout << "Epoch: " << e+1 << endl; // << " learning-rate:" << lr << endl;
        for (int b=0; b<chunks; b += nt) {
            std::list<std::future<t_cppl>> grads;
            for (int bi=0; bi<nt; bi++) {
                int y0,dy;
                y0=(b*nt+bi)*bs;
                if (y0+bs > x.rows()) dy=x.rows()-y0-1;
                else dy=bs;
                if (y0+dy>=x.rows() || dy<=0) {
                    cout << "Muuuh" << y0+dy << " " << y0 << " " << dy << endl;
                    continue;
                }
                //cout << "Processing: [" << y0 << "," << y0+dy-1 << "] ";
                MatrixN xb=x.block(y0,0,dy,x.cols());
                MatrixN yb=y.block(y0,0,dy,y.cols());
                grads.push_back(std::async(std::launch::async, [this, xb, yb, &l]{ return this->workerThread(xb, yb, &l); }));
            }

            t_cppl sgrad;
            bool first=true;
            for (std::list<std::future<t_cppl>>::iterator it=grads.begin(); it != grads.end(); ++it) {
                t_cppl grd = (*it).get();
                if (first) {
                    for (auto g : grd) {
                        MatrixN gx=*(g.second);
                        sgrad[g.first] = new MatrixN(gx);
                        first=false;
                    }
                } else {
                    for (auto g : grd) {
                        *(sgrad[g.first]) += *(g.second);
                    }
                }
                cppl_delete(&grd);
            }
            update(popti, &sgrad);
            cppl_delete(&sgrad);
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
