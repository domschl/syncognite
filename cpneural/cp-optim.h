#ifndef _CP_OPTIM_H
#define _CP_OPTIM_H

#include "cp-neural.h"

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
            else cerr<<"BAD ALGO!" << endl;
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
    cerr << "optimizerFactory called for unknown optimizer " << name << "." << endl;
    return nullptr;
}

floatN Layer::test(const MatrixN& x, t_cppl* pstates, int batchsize=100)  {
    setFlag("train",false);
    int bs=batchsize;
    int N=shape(x)[0];
    MatrixN xb,yb;
    int co=0;

    if (pstates->find("y") == pstates->end()) {
        cerr << "Layer::test: pstates does not contain y -> fatal!" << endl;
    }
    MatrixN y = *((*pstates)["y"]);
    MatrixN *py = (*pstates)["y"];

    for (int ck=0; ck<(N+bs-1)/bs; ck++) {
        int x0=ck*bs;
        int dl=bs;
        if (x0+dl>N) dl=N-x0;
        xb=x.block(x0,0,dl,x.cols());
        yb=y.block(x0,0,dl,y.cols());
        (*pstates)["y"]=&yb;
        MatrixN yt=forward(xb, nullptr, pstates, 0);
        if (yt.rows() != yb.rows()) {
            cerr << "test: incompatible row count!" << endl;
            (*pstates)["y"] = py;
            return -1000.0;
        }
        for (int i=0; i<yt.rows(); i++) {
            int ji=-1;
            floatN pr=-10000;
            for (int j=0; j<yt.cols(); j++) {
                if (yt(i,j)>pr) {
                    pr=yt(i,j);
                    ji=j;
                }
            }
            if (ji==(-1)) {
                cerr << "Internal: at " << layerName << "could not identify max-index for y-row-" << i << ": " << yt.row(i) << endl;
                (*pstates)["y"] = py;
                return -1000.0;
            }
            if (ji==yb(i,0)) ++co;
        }
    }
    floatN err=1.0-(floatN)co/(floatN)y.rows();
    (*pstates)["y"] = py;
    return err;
}

floatN Layer::test(const MatrixN& x, const MatrixN& y, int batchsize=100) {
    t_cppl states;
    MatrixN yvol = y;
    states["y"]=&yvol;
    return test(x, &states, batchsize);
}

retdict Layer::workerThread(MatrixN *pxb, t_cppl* pstates, int id) {
    t_cppl cache;
    t_cppl grads;
    retdict rd;
    if (pstates->find("y") == pstates->end()) {
        cerr << "workerThread: pstates does not contain y -> fatal!" << endl;
    }
    MatrixN yb=*((*pstates)["y"]);
    forward(*pxb, &cache, pstates, id);
    floatN thisloss=loss(&cache, pstates);
    lossQueueMutex.lock();
    lossQueue.push(thisloss);
    lossQueueMutex.unlock();
    backward(yb, &cache, pstates, &grads, id);
    cppl_delete(&cache);
    // cppl_delete(pstates);
    delete pxb;
    rd["grads"] = grads;
    rd["states"] = *pstates;
    return rd;
}

floatN Layer::train(const MatrixN& x, t_cppl* pstates, const MatrixN &xv, t_cppl* pstatesv,
                string optimizer, const CpParams& cp) {
    Optimizer* popti=optimizerFactory(optimizer, cp);
    t_cppl optiCache;
    t_cppl states[MAX_NUMTHREADS];
    MatrixN* pxbi[MAX_NUMTHREADS];

    setFlag("train",true);
    floatN lastAcc;
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier gray(Color::FG_LIGHT_BLUE);
    Color::Modifier def(Color::FG_DEFAULT);

    if (pstates->find("y") == pstates->end()) {
        cerr << "Layer::train: pstates does not contain y -> fatal!" << endl;
    }
    MatrixN y = *((*pstates)["y"]);
    if (pstatesv->find("y") == pstatesv->end()) {
        cerr << "pstates does not contain y -> fatal!" << endl;
    }
    MatrixN yv = *((*pstatesv)["y"]);

    float epf=popti->cp.getPar("epochs", (float)1.0); //Default only!
    int ep=(int)ceil(epf);
    float sepf=popti->cp.getPar("startepoch", (float)0.0);
    int bs=popti->cp.getPar("batch_size", (int)100); // Defaults only! are overwritten!
    floatN lr_decay=popti->cp.getPar("lr_decay", (floatN)1.0); //Default only!
    bool verbose=popti->cp.getPar("verbose", (bool)false);
    bool bShuffle=popti->cp.getPar("shuffle", (bool)false);
    bool bPreserveStates=popti->cp.getPar("preservestates", (bool)false);
    floatN lr = popti->cp.getPar("learning_rate", (floatN)1.0e-2); // Default only!
    floatN regularization = popti->cp.getPar("regularization", (floatN)0.0); // Default only!
    //cerr << ep << " " << bs << " " << lr << endl;

    int nt=cpGetNumCpuThreads() + cpGetNumGpuThreads(); // popti->cp.getPar("threads",(int)1); // Default only!
    int maxThreads=popti->cp.getPar("maxthreads",(int)0);
    if (maxThreads>1 && bPreserveStates) {
        cerr << "ERROR: cannnot preserve states, if thread-count > 1, reducint to 1." << endl;
        maxThreads=1;
    }
    if (maxThreads!=0) {
        if (nt>maxThreads) nt=maxThreads;
    }

    std::ofstream logfile;
    if (sepf>0.0) {
        logfile.open("log.txt", std::ios_base::app);
    } else {
        logfile.open("log.txt");
        logfile << "# Epoche\tLoss\tMeanloss\tmeanloss2\tAccuracy\tmeanacc" << endl;
    }
    std::flush(logfile);
    vector<unsigned int> ack(x.rows());
    vector<unsigned int> shfl(x.rows());
    for (unsigned int i=0; i<shfl.size(); i++) shfl[i]=i;

    floatN meanloss=0.0;
    floatN m2loss=0.0;
    floatN meanacc=0.0;
    floatN lastloss=0.0;
    floatN accval=0.0;
    //bs=bs/nt;
    lr=lr/nt;
    Timer tw;
    popti->cp.setPar("learning_rate", lr); // adpated to thread-count XXX here?
    //int ebs=bs*nt;
    int chunks=(x.rows()+bs-1) / bs;
    cerr << "Training net: data-size: " << x.rows() << ", chunks: " << chunks << ", batch_size: " << bs;
    cerr << ", threads: " << nt << " (bz*ch): " << chunks*bs << endl;
    bool fracend=false;
    for (int e=sepf; e<sepf+ep && !fracend; e++) {
        std::random_shuffle(shfl.begin(), shfl.end());
        for (unsigned int i=0; i<ack.size(); i++) ack[i]=0;
        if (verbose) {
            cerr << "Epoch: " << green << e+1 << def << "\r"; // << " learning-rate:" << lr << "\r";
            std::flush(cerr);
        }
        tw.startWall();
        int th=0;
        //std::list<std::future<t_cppl>> gradsFut;
        std::vector<std::future<retdict>> retFut;
        int bold=0;
        for (int b=0; b<chunks && !fracend; b++) {
            int bi=b%nt;
            int y0,dy;
            y0=b*bs;
            if (y0>=x.rows()) continue;
            if (y0+bs > x.rows()) dy=x.rows()-y0;
            else dy=bs;
            if (y0+dy>x.rows() || dy<=0) {
                cerr << "Muuuh" << y0+dy << " " << y0 << " " << dy << endl;
            }
            //cerr << "[" << y0 << "," << y0+dy-1 << "] ";

            MatrixN xb(dy,x.cols());
            MatrixN yb(dy,y.cols());
            xb.setZero();
            yb.setZero();
            if (bShuffle) {
                for (int i=y0; i<y0+dy; i++) {
                    int in=i-y0;
                    for (int j=0; j<x.cols(); j++) {
                        xb(in,j) = x(shfl[i],j);
                    }
                    for (int j=0; j<y.cols(); j++) {
                        yb(in,j) = y(shfl[i],j);
                    }
/*                    if (shfl[i]>=ack.size()) cerr << "UUHH trying to learn non-existant shuffle data-point no: " << shfl[i] << endl;
                    else {
                        ack[shfl[i]]++;
                    }
    */            }

            } else {
                xb=x.block(y0,0,dy,x.cols());
                yb=y.block(y0,0,dy,y.cols());
            /*    for (unsigned int i=y0; i<(unsigned int)(y0+dy); i++) {
                    if (i>=ack.size()) cerr << "UUHH trying to learn non-existant data-point no: " << i << endl;
                    else {
                        ack[i]++;
                    }
                }*/
            }
            ++th;
            states[bi].clear();
            for (auto st : *pstates) {
                if (st.first != "y") {
                    cppl_set(&(states[bi]), st.first, new MatrixN(*st.second));
                }
            }
            cppl_set(&(states[bi]), "y", new MatrixN(yb));
            pxbi[bi] = new MatrixN(xb);
            MatrixN *pxb = pxbi[bi];
            t_cppl *pst = &(states[bi]);
            retFut.push_back(std::async(std::launch::async, [this, pxb, pst, bi]{ return this->workerThread(pxb, pst, bi); }));
            if (bi==nt-1 || b==chunks-1) {
                t_cppl sgrad;
                bool first=true;
                for (auto &it : retFut) {
                    --th;
                    retdict rvg=it.get();
                    t_cppl retGrads = rvg["grads"];
                    if (first) {
                        for (auto g : retGrads) {
                            sgrad[g.first] = new MatrixN(*(g.second));
                            first=false;
                        }
                    } else {
                        for (auto g : retGrads) {
                            *(sgrad[g.first]) += *(g.second);
                        }
                    }
                    cppl_delete(&retGrads);
                    t_cppl retStates = rvg["states"];
                    if (bPreserveStates) {
                        // XXX here we need to copy the states back!
                        // There is something missing for thread-state coupling!

                        // Hacky, yet acceptable, solution, only one thread on bPreserveStates...
                        *pstates=retStates;

                    }
                    cppl_delete(&retStates);
                }
                if (regularization!=0.0) { // Loss is not adapted for regularization, since that's anyway only cosmetics.
                    for (auto gi : sgrad) {
                        *(sgrad[gi.first]) += *(params[gi.first]) * regularization;
                    }
                }
                update(popti, &sgrad, "", &optiCache);
                cppl_delete(&sgrad);
                retFut.clear();

                float dv1=10.0;
                float dv2=50.0;

                lossQueueMutex.lock();
                while (!lossQueue.empty()) {
                    lastloss=lossQueue.front();
                    lossQueue.pop();
                    //cerr << "lossQueue pop: " << lastloss << endl;
                    if (meanloss==0) meanloss=lastloss;
                    else meanloss=((dv1-1.0)*meanloss+lastloss)/dv1;
                    if (m2loss==0) m2loss=lastloss;
                    else m2loss=((dv2-1.0)*m2loss+lastloss)/dv2;
                }
                lossQueueMutex.unlock();

                if (verbose && b-bold>=5) {
                    floatN twt=tw.stopWallMicro();
                    floatN ett=twt/1000000.0 / (floatN)b * (floatN)chunks;
                    floatN eta=twt/1000000.0-ett;
                    floatN chtr=twt/1000.0/(floatN)(b*bs);
                    if (e+5.0<dv1) dv1=e+5.0;
                    if (e+20.0<dv2) dv2=e+20.0;
                    cerr << gray << "At: " << std::fixed << std::setw(4) << green << (int)((floatN)b/(floatN)chunks*100.0) << "\%" << gray << " of epoch " << green << e+1 << gray <<", " << chtr << " ms/data, ett: " << (int)ett << "s, eta: " << (int)eta << "s, loss: " << meanloss << ", " << m2loss << def << "\r";
                    std::flush(cerr);
                    logfile << e+(floatN)b/(floatN)chunks << "\t" << lastloss << "\t" meanloss << "\t" << m2loss << "\t" << accval << "\t" << meanacc << endl;
                    std::flush(logfile);
                    bold=b;
                }
                //cerr << "UD" << endl;
                if ((float)e+(floatN)b/(floatN)chunks > epf+sepf) fracend=true;
            }
        }
        //cerr << "ED" << endl;

        //floatN errtra=test(x,y);
        Timer tt;
        tt.startWall();
        floatN errval=test(xv,pstatesv,bs);
        floatN ttst=tt.stopWallMicro();
        accval=1.0-errval;
        lastAcc=accval;
        if (verbose) cerr << "Ep: " << e+1 << ", Time: "<< (int)(tw.stopWallMicro()/1000000.0) << "s, (" << (int)(ttst/1000000.0) << "s test) loss:" << m2loss << " err(val):" << errval << green << " acc(val):" << accval << def << endl;
        if (meanacc==0.0) meanacc=accval;
        else meanacc=(meanacc+2.0*accval)/3.0;
        logfile << e+1.0 << "\t" << lastloss << "\t" << meanloss<< "\t" << m2loss << "\t" << accval << "\t" << meanacc << endl;
        std::flush(logfile);
        setFlag("train",true);
        if (lr_decay!=1.0) {
            lr *= lr_decay;
            popti->cp.setPar("learning_rate", lr);
        }
/*        for (unsigned int i=0; i<ack.size(); i++) {
            if (ack[i]!=1) {
                cerr << "Datapoint: " << i << " should be active once, was active: " << ack[i] << endl;
            }
        }
*/    }
    cppl_delete(&optiCache);
    delete popti;
    return lastAcc;
}

floatN Layer::train(const MatrixN& x, const MatrixN& y, const MatrixN &xv, const MatrixN& yv,
                string optimizer, const CpParams& cp) {
    t_cppl states, statesv;
    states["y"]=new MatrixN(y);
    statesv["y"]=new MatrixN(yv);
    auto loss = train(x, &states, xv, &statesv, optimizer, cp);
    cppl_delete(&states);
    cppl_delete(&statesv);
    return loss;
}

/*floatN Layer::train(const MatrixN& x, const MatrixN& y, const MatrixN &xv, const MatrixN &yv,
                string optimizer, const CpParams& cp) {
    return train(const MatrixN& x, const MatrixN& y, const MatrixN &xv, const MatrixN &yv,
                    string optimizer, const CpParams& cp, nullptr);
    }*/
#endif
