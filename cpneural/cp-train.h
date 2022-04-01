#pragma once

#include "cp-neural.h"

floatN Layer::test(const MatrixN& x, t_cppl* pstates, int batchsize=100)  {
    setFlag("train",false);
    int bs=batchsize;
    int N=shape(x)[0];
    MatrixN xb,yb;
    int co=0;
    int nt=0;

    //if ((int)x.rows() < batchsize) batchsize=(int)x.rows();
    
    if (pstates->find("y") == pstates->end()) {
        cerr << "Layer::test: pstates does not contain y -> fatal!" << endl;
    }
    MatrixN y = *((*pstates)["y"]);
    MatrixN *py = (*pstates)["y"];
    //int nrr=(int)y.rows();

    for (int ck=0; ck<(N+bs-1)/bs; ck++) {
        int x0=ck*bs;
        int dl=bs;
        if (x0+dl>N) continue; // no fractional batches, problem with RNNs: dl=N-x0;
        xb=x.block(x0,0,dl,x.cols());
        //yb=y.block(x0,0,dl,y.cols());
        MatrixN ybte=y.block(x0,0,dl,y.cols());
        MatrixN ybtet=ybte.transpose();
        Eigen::Map<MatrixN> ybt(ybtet.data(),dl*y.cols(),1);
        MatrixN yb(ybt);
        (*pstates)["y"]=&yb;
        t_cppl cache;
        MatrixN yt=forward(xb, &cache, pstates, 0);
        if (yt.rows() != yb.rows()) {
            int BS=(int)xb.rows();
            int T=(int)yb.rows()/BS;
            int D=(int)yt.cols()/T;

            MatrixN ytn=MatrixN(BS*T,D);
            for (int n=0; n<BS; n++) {
                for (int t=0; t<T; t++) {
                    for (int d=0; d<D; d++) {
                        ytn(n*T+t,d) = yt(n,t*D+d);
                    }
                }
            }
            yt=ytn;
            if (yt.rows() !=yb.rows()) {
                cerr << "test: incompatible row count! Can't remap! testdata:" << yb.rows() << ", result:" << yt.rows() << endl;
                cerr << "X:" << shape(xb) << " y:" << shape(yb) << " f(X):" << shape(yt) << endl;
                (*pstates)["y"] = py;
                return -1000.0;
            }
        }
        //nrr=(int)yt.rows();
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
            ++nt;
        }
        cppl_delete(&cache);
    }
    // floatN err=1.0-(floatN)co/(floatN)(nrr*bs);
    floatN err=1.0-(floatN)co/(floatN)(nt);
    (*pstates)["y"] = py;
    return err;
}

floatN Layer::test(const MatrixN& x, const MatrixN& y, int batchsize=100) {
    t_cppl states;
    MatrixN yvol = y;
    states["y"]=&yvol;
    return test(x, &states, batchsize);
}

retdict Layer::workerThread(MatrixN *pxb, t_cppl* pstates, int id, Loss *pLoss) {
    t_cppl cache;
    t_cppl grads;
    retdict rd;
    if (pstates->find("y") == pstates->end()) {
        cerr << "workerThread: pstates does not contain y -> fatal!" << endl;
    }
    MatrixN yb=*((*pstates)["y"]);
    MatrixN yhat = forward(*pxb, &cache, pstates, id);
    //MatrixN yhat=*((*pstates)["probs"]); // XXX: just use return of forward()? old: ["y"]);
    floatN thisloss=pLoss->loss(yhat, yb, pstates);
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
                Optimizer *popti, t_cppl* pOptimizerState, Loss *pLoss, const json& job_params) {
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

    float epf=job_params.value("epochs", (float)1.0); //Default only!
    int ep=(int)ceil(epf);
    float sepf=job_params.value("startepoch", (float)0.0);
    int bs=job_params.value("batch_size", (int)100); // Defaults only! are overwritten!
    floatN lr_decay=job_params.value("lr_decay", (floatN)1.0); //Default only!
    bool verbose=job_params.value("verbose", (bool)false);
    bool verbosetitle=job_params.value("verbosetitle", (bool)true);
    bool bShuffle=job_params.value("shuffle", (bool)false);
    bool bPreserveStates=job_params.value("preservestates", (bool)false);
    bool noTests=job_params.value("notests", (bool)false);
    bool noFragmentBatches=job_params.value("nofragmentbatches",false);
    
    floatN regularization = job_params.value("regularization", (floatN)0.0); // Default only!
    floatN regularization_decay = job_params.value("regularization_decay", (floatN)1.0); // Default only!
    float lossfactor=job_params.value("lossfactor",(float)1.0);
    
    floatN lr = popti->j.value("learning_rate", (floatN)1.0e-2); // Default only!
    
    int nt=cpGetNumCpuThreads();
    int maxThreads=popti->j.value("maxthreads",(int)0);
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
    popti->j["learning_rate"]=lr; // adpated to thread-count XXX here?
    //int ebs=bs*nt;
    int chunks=((int)x.rows()+bs-1) / bs;
    if (verbosetitle) {
        cerr << endl << "Training net: data-size: " << x.rows() << ", chunks: " << chunks << ", batch_size: " << bs;
        cerr << ", threads: " << nt << " (bz*ch): " << chunks*bs << endl;
    }
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
            if (y0>=(int)x.rows()) continue;
            int xrws=(int)x.rows();
            if (y0+bs > xrws) dy=xrws-y0;
            else dy=bs;
            if (y0+dy>(int)x.rows() || dy<=0) {
                cerr << "Muuuh" << y0+dy << " " << y0 << " " << dy << endl;
            }
            //cerr << "[" << y0 << "," << y0+dy-1 << "] ";
            if (dy!=bs && noFragmentBatches) {
                break;  // RNN and LSTMs can't handle different batch-sizes.
            }
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
            
            
            retFut.push_back(std::async(std::launch::async, [this, pxb, pst, bi, pLoss]{ return workerThread(pxb, pst, bi, pLoss); }));
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
                update(popti, &sgrad, "", pOptimizerState);
                cppl_delete(&sgrad);
                retFut.clear();

                floatN dv1=10.0;
                floatN dv2=50.0;

                lossQueueMutex.lock();
                while (!lossQueue.empty()) {
                    lastloss=lossQueue.front()*lossfactor;
                    lossQueue.pop();
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
                    //if (e+5.0<dv1) dv1=e+5.0;
                    //if (e+20.0<dv2) dv2=e+20.0;
                    cerr << gray << "At: " << std::fixed << std::setw(4) << green << (int)((floatN)b/(floatN)chunks*100.0) << "\%" << gray << " of epoch " << green << e+1 << gray <<", " << std::setprecision(2) << chtr << " ms/data, ett: " << (int)ett << "s, eta: " << (int)eta << "s, loss: " << std::setprecision(4) << meanloss << def << "\r";
                    std::flush(cerr);
                    //logfile << e+(floatN)b/(floatN)chunks << "\t" << lastloss << "\t" << meanloss << "\t" << m2loss << "\t" << accval << "\t" << meanacc << endl;
                    logfile << e+(floatN)b/(floatN)chunks << "\t" << lastloss << "\t" << meanloss << "\t" << m2loss << "\t" /*<< accval*/ << "\t" << /* meanacc <<*/ endl;
                    std::flush(logfile);
                    bold=b;
                }
                //cerr << "UD" << endl;
                if ((float)e+(floatN)b/(floatN)chunks > epf+sepf) fracend=true;
            }
        }
        //cerr << "ED" << endl;

        //floatN errtra=test(x,y);
        if (!noTests) {
            Timer tt;
            tt.startWall();
            floatN errval=test(xv,pstatesv,bs);
            floatN ttst=tt.stopWallMicro();
            accval=1.0-errval;
            lastAcc=accval;
            if (verbose) cerr << "Ep: " << e+1 << ", Time: "<< std::setprecision(2) << (int)(tw.stopWallMicro()/1000000.0) << "s, (" << std::setprecision(2) << (int)(ttst/1000000.0) << "s test) loss:" << std::setprecision(4) << m2loss << " err(val):" << errval << green << " acc(val):" << accval << def << endl;
            if (meanacc==0.0) meanacc=accval;
            else meanacc=(meanacc+2.0*accval)/3.0;
            logfile << e+1.0 << "\t" << lastloss << "\t" << meanloss<< "\t" << m2loss << "\t" << accval << "\t" << meanacc << endl;
            std::flush(logfile);
        }
        setFlag("train",true);
        if (lr_decay!=1.0) {
            lr *= lr_decay;
            popti->j["learning_rate"]=lr;
        }
        if (regularization_decay!=0.0) {
            regularization *= regularization_decay;
        }
/*        for (unsigned int i=0; i<ack.size(); i++) {
            if (ack[i]!=1) {
                cerr << "Datapoint: " << i << " should be active once, was active: " << ack[i] << endl;
            }
        }
*/    }
    return lastAcc;
}

// Legacy interface
/*
floatN Layer::train(const MatrixN& x, const MatrixN& y, const MatrixN &xv, const MatrixN& yv,
                string optimizer, const json& j) {
    t_cppl states, statesv;
    cerr << "train: legacy interface (1) called, using default optimizer and losses" << endl;
    Optimizer* pOptimizer=optimizerFactory(optimizer, j);
    Loss *pLoss=lossFactory("SparseCategoricalCrossEntropy", j);
    states["y"]=new MatrixN(y);
    statesv["y"]=new MatrixN(yv);
    auto loss = train(x, &states, xv, &statesv, pOptimizer, pLoss, j);
    cppl_delete(&states);
    cppl_delete(&statesv);
    delete pLoss;
    delete pOptimizer;
    return loss;
}
*/
floatN Layer::train(const MatrixN& x, const MatrixN& y, const MatrixN &x_val, const MatrixN& y_val,
        Optimizer *pOptimizer, t_cppl *pOptimizerState, Loss *pLoss, const json& job_parameters) {
    t_cppl states, states_val;
    states["y"]=new MatrixN(y);
    states_val["y"]=new MatrixN(y_val);


    auto loss = train(x, &states, x_val, &states_val, pOptimizer, pOptimizerState, pLoss, job_parameters);
    cppl_delete(&states);
    cppl_delete(&states_val);
    return loss;
}

// legacy interface
/*
floatN Layer::train(const MatrixN& x, t_cppl *pStates, const MatrixN &xv, t_cppl *pStatesVal,
        string optimizer, const json& j) {
    cerr << "train: legacy interface (2) called, using default optimizer and losses" << endl;
    Optimizer* pOptimizer=optimizerFactory(optimizer, j);
    Loss *pLoss=lossFactory("SparseCategoricalCrossEntropy", j);
    auto loss = train(x, pStates, xv, pStatesVal, pOptimizer, pLoss, j);
    delete pLoss;
    delete pOptimizer;
    return loss;
}
*/
