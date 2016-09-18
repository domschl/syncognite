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
public:
    Sdg(const CpParams& cx) {
        cp=cx;
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl* pcache) override {
        lr=cp.getPar("learning_rate", (floatN)1e-2);
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
    SdgMomentum(const CpParams& cx) {
        cp=cx;
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl* pocache) override {
        lr=cp.getPar("learning_rate", (floatN)1e-2);
        mm=cp.getPar("momentum",(floatN)0.9);
        string cname=var+"-velocity";
        if (pocache->find("cname_v")==pocache->end()) {
            MatrixN z=MatrixN(x);
            z.setZero();
            cppl_update(pocache, cname, &z);
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
    RmsProp(const CpParams& cx) {
        cp=cx;
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl* pocache) override {
        lr=cp.getPar("learning_rate", (floatN)1e-2);
        dc=cp.getPar("decay_rate", (floatN)0.99);
        ep=cp.getPar("epsilon", (floatN)1e-8);
        string cname=var+"-movavr";
        if (pocache->find("cname")==pocache->end()) {
            MatrixN z=MatrixN(x);
            z.setZero();
            cppl_update(pocache, cname, &z);
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
    Adam(const CpParams& cx) {
        cp=cx;
    }
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl* pocache) override {
        lr=cp.getPar("learning_rate", (floatN)1e-2);
        b1=cp.getPar("beta1", (floatN)0.9);
        b2=cp.getPar("beta2", (floatN)0.999);
        ep=cp.getPar("epsilon", (floatN)1e-8);
        string cname_m=var+"-m";
        if (pocache->find("cname_m")==pocache->end()) {
            MatrixN z=MatrixN(x);
            z.setZero();
            cppl_update(pocache, cname_m, &z);
        }
        string cname_v=var+"-v";
        if (pocache->find("cname_v")==pocache->end()) {
            MatrixN z=MatrixN(x);
            z.setZero();
            cppl_update(pocache, cname_v, &z);
        }
        string cname_t=var+"-t";
        if (pocache->find("cname_t")==pocache->end()) {
            MatrixN z1(1,1);
            z1.setZero();
            cppl_update(pocache, cname_t, &z1);
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

Optimizer *optimizerFactory(string name, const CpParams& cp) {
    if (name=="Sdg") return (Optimizer *)new Sdg(cp);
    if (name=="SdgMomentum") return (Optimizer *)new SdgMomentum(cp);
    if (name=="RmsProp") return (Optimizer *)new RmsProp(cp);
    if (name=="Adam") return (Optimizer *)new Adam(cp);
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

t_cppl Layer::workerThread(const MatrixN& xb, const MatrixN& yb, floatN *ploss, int id) {
    t_cppl cache;
    t_cppl grads;

    //cout << "Context start: " << id << endl;
    forward(xb, yb, &cache, id);
    //cout << "fw" << id << endl;
    *ploss=loss(yb, &cache);
    backward(yb, &cache, &grads, id);
    //cout << "bw" << id << endl;
    cppl_delete(&cache);
    //cout << "Context end: " << id << endl;
    return grads;
}

floatN Layer::train(const MatrixN& x, const MatrixN& y, const MatrixN &xv, const MatrixN &yv,
                string optimizer, const CpParams& cp) {
    Optimizer* popti=optimizerFactory(optimizer, cp);
    t_cppl optiCache;
    setFlag("train",true);
    bool bShuffle=true;

    int ep=popti->cp.getPar("epochs", (int)1); //Default only!
    int bs=popti->cp.getPar("batch_size", (int)100); // Defaults only! are overwritten!
    int nt=cpGetNumPoolThreads(); // popti->cp.getPar("threads",(int)1); // Default only!
    floatN lr_decay=popti->cp.getPar("lr_decay", (floatN)1.0); //Default only!
    bool verbose=popti->cp.getPar("verbose", (bool)false);
    floatN lr = popti->cp.getPar("learning_rate", (floatN)1.0e-2); // Default only!
    floatN regularization = popti->cp.getPar("regularization", (floatN)0.0); // Default only!
    //cout << ep << " " << bs << " " << lr << endl;

    vector<unsigned int> ack(x.rows());
    vector<unsigned int> shfl(x.rows());
    for (unsigned int i=0; i<shfl.size(); i++) shfl[i]=i;

    floatN l=0.0;
    bs=bs/nt;
    lr=lr/nt;
    Timer tw;
    popti->cp.setPar("learning_rate", lr); // adpated to thread-count XXX here?
    int ebs=bs*nt;
    int chunks=(x.rows()+ebs-1) / ebs;
    cout << "Training net: data-size: " << x.rows() << ", chunks: " << chunks << ", batch_size/threads: " << bs;
    cout << ", threads: " << nt << " (bz*ch*thr): " << chunks*bs*nt << endl;
    for (int e=0; e<ep; e++) {
        std::random_shuffle(shfl.begin(), shfl.end());
        for (unsigned int i=0; i<ack.size(); i++) ack[i]=0;
        if (verbose) cout << "Epoch: " << e+1 << endl; // << " learning-rate:" << lr << endl;
        tw.startWall();
        for (int b=0; b<chunks*nt; b += nt) {
            std::list<std::future<t_cppl>> gradsFut;
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

                MatrixN xb(dy,x.cols());
                MatrixN yb(dy,y.cols());
                if (bShuffle) {
                    for (int i=y0; i<y0+dy; i++) {
                        int in=i-y0;
                        for (int j=0; j<x.cols(); j++) {
                            xb(in,j) = x(shfl[i],j);
                        }
                        for (int j=0; j<y.cols(); j++) {
                            yb(in,j) = y(shfl[i],j);
                        }
                        if (shfl[i]>=ack.size()) cout << "UUHH trying to learn non-existant shuffle data-point no: " << shfl[i] << endl;
                        else {
                            ack[shfl[i]]++;
                        }
                    }

                } else {
                    xb=x.block(y0,0,dy,x.cols());
                    yb=y.block(y0,0,dy,y.cols());
                    for (unsigned int i=y0; i<(unsigned int)(y0+dy); i++) {
                        if (i>=ack.size()) cout << "UUHH trying to learn non-existant data-point no: " << i << endl;
                        else {
                            ack[i]++;
                        }
                    }
                }
                gradsFut.push_back(std::async(std::launch::async, [this, xb, yb, &l, bi]{ return this->workerThread(xb, yb, &l, bi); }));
            }
            //cout << endl;

            t_cppl sgrad;
            bool first=true;
            for (std::list<std::future<t_cppl>>::iterator it=gradsFut.begin(); it != gradsFut.end(); ++it) {
                t_cppl grd = it->get();
                if (first) {
                    for (auto g : grd) {
                        sgrad[g.first] = new MatrixN(*(g.second));
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
            if (regularization!=0.0) { // Loss is not adapted for regularization, since that's anyway only cosmetics.
                for (auto gi : sgrad) {
                    *(sgrad[gi.first]) += *(params[gi.first]) * regularization;
                }
            }
            update(popti, &sgrad, "", &optiCache);
            cppl_delete(&sgrad);
            gradsFut.clear();
            if (verbose && b%50==0) cout << "At: " << (floatN)b/(floatN)(chunks*nt)*100.0 << "\% of epoch, loss: " << l << endl;
        }


        //floatN errtra=test(x,y);
        floatN errval=test(xv,yv);
        floatN accval=1.0-errval;
        if (verbose) cout << "Time: "<< tw.stopWallMicro()/1000000.0 << "s, loss:" << l << " err(val):" << errval << " acc(val):" << accval << endl;
        setFlag("train",true);
        if (lr_decay!=1.0) {
            lr *= lr_decay;
            popti->cp.setPar("learning_rate", lr);
        }
        for (unsigned int i=0; i<ack.size(); i++) {
            if (ack[i]!=1) {
                cout << "Datapoint: " << i << " should be active once, was active: " << ack[i] << endl;
            }
        }
    }
    cppl_delete(&optiCache);
    delete popti;
    return 0.0;
}
#endif
