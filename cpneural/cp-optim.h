#ifndef _CP_OPTIM_H
#define _CP_OPTIM_H

#include <future>
#include <list>
#include <set>
#include "cp-layer.h"
#include "cp-timer.h"
#include "cp-layers.h"


class Sdg : public Optimizer {
    floatN lr;
    std::set<string> debugnames;
public:
    Sdg(cp_t_params<floatN>& ps) {
        for (auto pi : ps) {
            setPar(pi.first, pi.second);
        }
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl* pcache) override {
        lr=getPar("learning_rate", 1e-2);
        if (debugnames.find(var)==debugnames.end()) {
            debugnames.insert(var);
            cout << "Optimizer knows states of var: " << var << endl;
        }
        x=x-lr*dx;
        return x;
    }
};

/*
  Performs stochastic gradient descent with momentum. [CS231]
  Params:
  - learning_rate: Scalar learning rate.
  - momentum: Scalar between 0 and 1 giving the momentum value.
    Setting momentum = 0 reduces to sgd.
*/
class SdgMomentum : public Optimizer {
    floatN lr;
    floatN mm;
public:
    SdgMomentum(cp_t_params<floatN>& ps) {
        for (auto pi : ps) {
            setPar(pi.first, pi.second);
        }
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl* pocache) override {
        lr=getPar("learning_rate", 1e-2);
        mm=getPar("momentum",0.9);
        string cname=var+"-velocity";
        if (pocache->find(cname)==pocache->end()) {
            MatrixN *z=new MatrixN(x);
            z->setZero();
            (*pocache)[cname]=z;
            cout << cname << " initialized." << endl;
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
// Params:
// learning_rate 1e-2
// decay_rate 0.99
// epsilon: 1e-8
class RmsProp : public Optimizer {
    floatN lr;
    floatN dc,ep;
public:
    RmsProp(cp_t_params<floatN>& ps) {
        for (auto pi : ps) {
            setPar(pi.first, pi.second);
        }
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl* pocache) override {
        lr=getPar("learning_rate", 1e-2);
        dc=getPar("decay_rate", 0.99);
        ep=getPar("epsilon", 1e-8);
        string cname=var+"-movavr";
        if (pocache->find(cname)==pocache->end()) {
            MatrixN *z=new MatrixN(x);
            z->setZero();
            (*pocache)[cname]=z;
            cout << cname << " initialized." << endl;
        }
        *(*pocache)[cname]=dc * (*(*pocache)[cname]) + ((1.0 - dc) * (dx.array() * dx.array())).matrix();
        MatrixN dv=((*(*pocache)[cname]).array().sqrt() + ep).matrix();
        for (int i=0; i<dv.size(); i++) {
            if (dv(i)>0.0) dv(i)=1.0/dv(i);
            else cout<<"BAD ALGO!" << endl;
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
    Adam(cp_t_params<floatN>& ps) {
        for (auto pi : ps) {
            setPar(pi.first, pi.second);
        }
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl* pocache) override {
        lr=getPar("learning_rate", 1e-2);
        b1=getPar("beta1", 0.9);
        b2=getPar("beta2", 0.999);
        ep=getPar("epsilon", 1e-8);
        string cname_m=var+"-m";
        if (pocache->find(cname_m)==pocache->end()) {
            MatrixN *z=new MatrixN(x);
            z->setZero();
            (*pocache)[cname_m]=z;
            cout << cname_m << " initialized." << endl;
        }
        string cname_v=var+"-v";
        if (pocache->find(cname_v)==pocache->end()) {
            MatrixN *z=new MatrixN(x);
            z->setZero();
            (*pocache)[cname_v]=z;
            cout << cname_v << " initialized." << endl;
        }
        string cname_t=var+"-t";
        if (pocache->find(cname_t)==pocache->end()) {
            MatrixN *z=new MatrixN(1,1);
            z->setZero();
            (*pocache)[cname_t]=z;
            cout << cname_t << " initialized." << endl;
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

Optimizer *optimizerFactory(string name, cp_t_params<floatN> params) {
    if (name=="Sdg") return (Optimizer *)new Sdg(params);
    if (name=="SdgMomentum") return (Optimizer *)new SdgMomentum(params);
    if (name=="RmsProp") return (Optimizer *)new RmsProp(params);
    if (name=="Adam") return (Optimizer *)new Adam(params);
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
    Optimizer* popti=optimizerFactory(optimizer, fpars);
    for (auto pi : fpars) {
        popti->setPar(pi.first, pi.second);
    }
    for (auto pi : ipars) {
        popti->setPar(pi.first, pi.second);
    }
    t_cppl optiCache;
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
    lr=lr/nt;
    Timer tw;
    popti->setPar("learning_rate", lr); // adpated to thread-count XXX here?
    int ebs=bs*nt;
    int chunks=(x.rows()+ebs-1) / ebs;
    cout << "Data-size: " << x.rows() << ", chunks: " << chunks << ", batch_size: " << bs;
    cout << ", threads: " << nt << " (batch_size*chunks*threads): " << chunks*bs*nt << endl;
    for (int e=0; e<ep; e++) {
        cout << "Epoch: " << e+1 << endl; // << " learning-rate:" << lr << endl;
        tw.startWall();
        for (int b=0; b<chunks*nt; b += nt) {
            std::list<std::future<t_cppl>> grads;
            for (int bi=0; bi<nt; bi++) {
                int y0,dy;
                y0=b*bs+bi*bs;
                if (y0>=x.rows()) continue;
                if (y0+bs > x.rows()) dy=x.rows()-y0-1;
                else dy=bs;
                if (y0+dy>x.rows() || dy<=0) {
                    cout << "Muuuh" << y0+dy << " " << y0 << " " << dy << endl;
                }
                //cout << "[" << y0 << "," << y0+dy-1 << "] ";
                MatrixN xb=x.block(y0,0,dy,x.cols());
                MatrixN yb=y.block(y0,0,dy,y.cols());
                grads.push_back(std::async(std::launch::async, [this, xb, yb, &l]{ return this->workerThread(xb, yb, &l); }));
            }
            //cout << endl;

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
            // cout << "db2:" << *(sgrad["af2-b"]) << endl;
            update(popti, &sgrad, "", &optiCache);
            cppl_delete(&sgrad);
            grads.clear();
        }
        cout << "Time: "<< tw.stopWallMicro()/1000000.0 << "s, loss:" << l << " err(validation):" << test(xv,yv) << endl;
        if (lr_decay!=1.0) {
            lr *= lr_decay;
            popti->setPar("learning_rate", lr);
        }
    }
    cppl_delete(&optiCache);
    return 0.0;
}
#endif
