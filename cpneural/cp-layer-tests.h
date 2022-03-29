#ifndef _CP_LAYER_TESTS_H
#define _CP_LAYER_TESTS_H

#include "cp-neural.h"

bool Layer::checkForward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, 
                         floatN eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    bool allOk=true;
    MatrixN yt;
    MatrixN y;

    yt=forward(x, pcache, pstates, 0);
    if (layerType & LayerType::LT_LOSS) {
        if (pstates->find("y") == pstates->end()) {
            cerr << "  checkForward: pstates does not contain y -> fatal!" << endl;
        }
        y = *((*pstates)["y"]);
    }
    MatrixN yic(0, yt.cols());
    for (unsigned int i=0; i<x.rows(); i++) {
        MatrixN xi=x.row(i);
        MatrixN yi;
        if (layerType & LayerType::LT_LOSS) {
            MatrixN yii=y.row(i);
            t_cppl sti{};
            cppl_set(&sti, "y", new MatrixN(yii));
            yi = forward(xi, nullptr, &sti, 0);
            cppl_delete(&sti);
        } else {
            yi = forward(xi, nullptr, pstates, 0);
        }
        MatrixN yn(yi.rows() + yic.rows(),yi.cols());
        yn << yic, yi;
        yic = yn;
    }

    //MatrixN yc = forward(x, nullptr);
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    MatrixN d = yic - yt;
    floatN dif = d.cwiseProduct(d).sum();

    if (dif < eps) {
        if (verbose>1) cerr << "  Forward vectorizer OK, err=" << dif << endl;
    } else {
        if (verbose>0) {
            cerr << "  Forward vectorizer Error, err=" << dif << endl;
            cerr << "  yic:" << yic.format(CleanFmt) << endl;
            cerr << "  yt:" << yt.format(CleanFmt) << endl;
        }
        allOk=false;
    }
    return allOk;
}

bool Layer::checkBackward(const MatrixN& x, t_cppl *pcache, t_cppl* pstates, 
                          floatN eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    bool allOk =true;
    IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
    t_cppl cache;
    t_cppl grads;
    t_cppl rgrads;

    MatrixN dyc;
    MatrixN y;
    if ((layerType & LayerType::LT_NORMAL) && !(layerType & LayerType::LT_LOSS)) {
        dyc = forward(x, &cache, pstates, 0);
        dyc.setRandom();
    } else if (layerType & LayerType::LT_LOSS) {
        if (pstates->find("y") == pstates->end()) {
            if (verbose>0) cerr << "  checkBackward: pstates does not contain y -> fatal!" << endl;
        }
        y = *((*pstates)["y"]);
        forward(x, &cache, pstates, 0);
        dyc=y; // XXX: no!
    } else {
        cerr << "  BAD LAYER TYPE!" << layerType << endl;
        return false;
    }
    if (dyc.rows() != x.rows()) {
        cerr << "  internal error: y:" << shape(dyc) << " x:" << shape(x) << " - unequal number of rows!" << endl;
        return false;
    }
    MatrixN dx = x;
    dx.setZero();

    // if (cache.find("x")==cache.end()) cerr << "  WARNING: x is not in cache!" << endl;

    MatrixN dxc = backward(dyc, &cache, pstates, &grads, 0);

    for (auto it : grads) {
        cppl_set(&rgrads, it.first, new MatrixN(*grads[it.first])); // grads[it.first].rows(),grads[it.first].cols());
        rgrads[it.first]->setZero();
    }

    for (unsigned int i=0; i<x.rows(); i++) {
        t_cppl chi;
        t_cppl gdi;
        MatrixN xi=x.row(i);
        MatrixN dyi2;

        MatrixN dyi = dyc.row(i);
        t_cppl sti{}; // XXX: init with pstates? // XXX: RNN h!
        cppl_set(&sti, "y", new MatrixN(dyi));
        dyi2 = forward(xi, &chi, &sti, 0);
        MatrixN dyt = backward(dyi, &chi, &sti, &gdi, 0);
        dx.row(i) = dyt.row(0);
        for (auto it : grads) {
            *rgrads[it.first] += *gdi[it.first];
        }
        cppl_delete(&gdi);
        cppl_delete(&chi);
        cppl_delete(&sti);
    }
    if (layerType & LayerType::LT_LOSS) {
        dx /= x.rows(); // XXX is this sound for all types of loss-layers?
        for (auto it : grads) { // XXX this is beyound soundness...
             *(grads[it.first]) *= x.rows();
        }
    }
    MatrixN d = dx - dxc;
    floatN dif = d.cwiseProduct(d).sum();
    if (dif < eps) {
        if (verbose>1) cerr << "    Backward vectorizer dx OK, err=" << dif << endl;
    } else {
        if (verbose>0) {
            cerr << "    Backward vectorizer dx:" << endl << dx.format(CleanFmt) << endl;
            cerr << "    dxc:" << endl << dxc.format(CleanFmt) << endl;
            cerr << "    Backward vectorizer dx Error, err=" << dif << endl;
        }
        allOk=false;
    }

    for (auto it : grads) {
        MatrixN d = *grads[it.first] - *rgrads[it.first];
        floatN dif = d.cwiseProduct(d).sum();
        if (dif < eps) {
            if (verbose>1) cerr << "    Backward vectorizer " << "d" << it.first << " OK, err=" << dif << endl;
        } else {
            if (verbose>0) {
                cerr << "    d" << it.first << ":" << endl << grads[it.first]->format(CleanFmt) << endl;
                cerr << "    d" << it.first << "c:" << endl << rgrads[it.first]->format(CleanFmt) << endl;
                cerr << "    Backward vectorizer " << "d" << it.first << " error, err=" << dif << endl;
            }
            allOk=false;
        }
    }
    cppl_delete(&cache);
    cppl_delete(&grads);
    cppl_delete(&rgrads);
    return allOk;
}

MatrixN *getVarPointerHack(string var, MatrixN *px, t_cppl& params, t_cppl* pstates, int verbose=1) {
    MatrixN *pm;
    if (var=="x") pm=px;
    else {
        if (params.find(var)==params.end()) {
            if (pstates->find(var)!=pstates->end()) {
                if (verbose>1) cerr << "  Warning: trying to differentiate numerically a state" << endl;
                pm = (*pstates)[var];
            } else {
                string var0 = var.substr(0, var.size()-1);
                if (var0.size()>0 && pstates->find(var0)!=pstates->end()) {
                    if (verbose>2) cerr << "  Numerical check, mapping gradient " << var << " to state -> " << var0 << endl;
                    pm = (*pstates)[var0];
                    if (verbose>2) cerr << "  pm set" << endl;
                } else {
                    pm = nullptr;
                    if (verbose>0) cerr << "  Cannot find corresponding variable to gradient: " << var << " -> FATAL" << endl;
                }
            }
        }  else {
            pm = params[var];
        }
    }
    return pm;
}

MatrixN Layer::calcNumGrad(const MatrixN& xorg, const MatrixN& dchain, t_cppl* pcache, t_cppl* pstates, string var, floatN h=CP_DEFAULT_NUM_H, int verbose=1) {
    if (verbose>2) cerr << "  checking numerical gradient for " << var << "..." << endl;
    MatrixN *pm;
    MatrixN x;
    if (pcache->find("x")==pcache->end()) { // XXX that is quite a mess!
        x=xorg;
    } else {
        x=*((*pcache)["x"]);
    }
    pm=getVarPointerHack(var, &x, params, pstates);

    MatrixN grad((*pm).rows(), (*pm).cols());

    floatN pxold;
    for (unsigned int i=0; i<grad.size(); i++) {
        MatrixN y0,y1;
        t_cppl st;

        pxold = (*pm)(i);
        (*pm)(i) = (*pm)(i) - h;
        cppl_copy(pstates, &st);
        y0 = forward(x, nullptr, &st, 0);
        cppl_delete(&st);
        (*pm)(i) = pxold + h;
        cppl_copy(pstates, &st);
        y1 = forward(x, nullptr, &st, 0);
        cppl_delete(&st);
        (*pm)(i) = pxold;

        MatrixN dy=y1-y0;
        MatrixN dd;
        dd = dy.cwiseProduct(dchain);

        floatN drs = (dd / floatN(2.0 * h)).sum();
        grad(i)=drs;
    }
    return grad;
}

MatrixN Layer::calcNumGradLoss(const MatrixN& xorg, t_cppl *pcache, t_cppl* pstates, 
                               string var, floatN h=CP_DEFAULT_NUM_H, int verbose=1,
                               Loss *pLoss=nullptr) {
    MatrixN *pm;
    MatrixN x=xorg;
    MatrixN y=*((*pstates)["y"]);

    pm=getVarPointerHack(var, &x, params, pstates);
    MatrixN grad(pm->rows(), pm->cols());

    floatN pxold;
    for (unsigned int i=0; i<grad.size(); i++) {
        t_cppl cache;
        MatrixN y0,y1;
        float sy0, sy1;

        pxold = (*pm)(i);
        (*pm)(i) = (*pm)(i) - h;
        y0 = forward(x, &cache, pstates, 0);
        MatrixN yhat = y0; // name cleanup
        cerr << "This implementation is incomplete and wrong" << endl;
        // XXX this is a hack to get the loss working
        sy0 = pLoss->loss(yhat, y, pstates);
        cppl_delete(&cache);
        (*pm)(i) = pxold + h;
        y1 = forward(x, &cache, pstates, 0);
        yhat = y1; // name cleanup
        sy1 = pLoss->loss(yhat, y, pstates);
        cppl_delete(&cache);
        (*pm)(i) = pxold;

        floatN dy=sy1-sy0;
        floatN drs = dy / (2.0 * h);
        grad(i)=drs;
    }
    return grad;
}

bool Layer::calcNumGrads(const MatrixN& x, const MatrixN& dchain, t_cppl *pcache, 
                         t_cppl* pstates, t_cppl *pgrads, t_cppl *pnumGrads, 
                         floatN h=CP_DEFAULT_NUM_H, bool lossFkt=false, int verbose=1,
                         Loss *pLoss=nullptr) {
    for (auto it : *pgrads) {
        MatrixN g;
        if (!lossFkt) {
            g = calcNumGrad(x, dchain, pcache, pstates, it.first, h, verbose);
        } else {
            g = calcNumGradLoss(x, pcache, pstates, it.first, h, verbose);
        }
        cppl_set(pnumGrads, it.first, new MatrixN(g));
    }
    return true;
}

bool Layer::checkGradients(const MatrixN& x, const MatrixN& y, const MatrixN& dchain, 
                           t_cppl *pcache, t_cppl *pstates, floatN h=CP_DEFAULT_NUM_H, 
                           floatN eps=CP_DEFAULT_NUM_EPS, bool lossFkt=false, int verbose=1,
                           Loss *pLoss=nullptr) {
    bool allOk=true;
    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    Color::Modifier lred(Color::FG_LIGHT_RED);
    Color::Modifier lgreen(Color::FG_LIGHT_GREEN);
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    t_cppl grads;
    // MatrixN yt=forward(x, pcache);
    MatrixN dx;
    MatrixN yt;
    t_cppl st;
    cppl_copy(pstates, &st);
    if (lossFkt) {
        yt=forward(x, pcache, &st, 0);
        pLoss->loss(pcache,&st);
        dx=backward(y, pcache, &st, &grads, 0);
    } else {
        yt=forward(x, pcache, &st, 0);
        dx=backward(dchain, pcache, &st, &grads, 0);
    }
    cppl_delete(&st);
    if (dx.rows()>0 && dx.cols()>0)  // Some layers (e.g. WordEmbeddings have no dx!)
        grads["x"]=new MatrixN(dx);

    t_cppl numGrads;
    cppl_copy(pstates, &st);
    calcNumGrads(x, dchain, pcache, &st, &grads, &numGrads, h, lossFkt, verbose, pLoss);
    cppl_delete(&st);

    for (auto it : grads) {
        IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        MatrixN d = *(grads[it.first]) - *(numGrads[it.first]);
        floatN df = (d.cwiseProduct(d)).sum();
        if (df < eps) {
            if (verbose>1) cerr << "    " << layerName << ": " << "∂/∂" << it.first << green << " OK, err=" << df << def << endl;
        } else {
            if (verbose>0) {
                cerr << "    eps:" << eps << " h:" << h << endl;
                cerr << "    ∂/∂" << it.first << "[num]: " << endl << (*(numGrads[it.first])).format(CleanFmt) << endl;
                cerr << "    ∂/∂" << it.first << "[the]: " << endl << (*(grads[it.first])).format(CleanFmt) << endl;
                cerr << "      ð" << it.first << "    : " << endl << ((*(grads[it.first])) - (*(numGrads[it.first]))).format(CleanFmt) << endl;
                cerr << layerName << ": " << "∂/∂" << it.first << red << " ERROR, err=" << df << def << endl;
            }
            allOk=false;
        }
    }
    cppl_delete(&grads);
    cppl_delete(&numGrads);
    return allOk;
}

bool Layer::checkLayer(const MatrixN& x, const MatrixN& y, const MatrixN& dchain, 
                       t_cppl *pcache, t_cppl* pstates, floatN h=CP_DEFAULT_NUM_H, 
                       floatN eps=CP_DEFAULT_NUM_EPS, bool lossFkt=false, int verbose=1,
                       Loss *pLoss=nullptr) {
    // XXX: y parameter?
    bool allOk=true, ret;
    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    Color::Modifier lred(Color::FG_LIGHT_RED);
    Color::Modifier lgreen(Color::FG_LIGHT_GREEN);
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    //if (verbose>2) cerr << "  CheckLayer start" << endl;
    std::flush(cerr);

    if (!j.value("noVectorizationTests", false)) {
        if (verbose>2) cerr << "  check forward vectorizer " << layerName << "..." << endl;
        ret=checkForward(x, nullptr, pstates, eps, verbose);
        if (!ret) {
            if (verbose>0) cerr << "  " << layerName << ": " << red << "Forward vectorizing test failed!" << def << endl;
            return ret;
        } else {
            if (verbose>1) cerr << "  " << layerName << ": "<< green << "Forward vectorizing test OK!" << def << endl;
        }

        if (verbose>2) cerr << "  check backward vectorizer " << layerName << "..." << endl;
        t_cppl cache;
        ret=checkBackward(x, &cache, pstates, eps, verbose);
        cppl_delete(&cache);
        if (!ret) {
            if (verbose>0) cerr << "  " << layerName << ": " << red << "Backward vectorizing test failed!" << def << endl;
            allOk=false; //return ret;
        } else {
            if (verbose>1) cerr << "  " << layerName << ": " << green << "Backward vectorizing test OK!" << def << endl;
        }
    }

    if (verbose>2) cerr << "  check numerical gradients " << layerName << "..." << endl;
    t_cppl cache2;
    ret=checkGradients(x, y, dchain, &cache2, pstates, h, eps, lossFkt, verbose, pLoss);
    cppl_delete(&cache2);
    if (!ret) {
        if (verbose>0) cerr << "  " << layerName << ": " << red << "Gradient numerical test failed!" << def << endl;
        return ret;
    } else {
        if (verbose>1) cerr << "  " << layerName << ": " << green << "Gradient numerical test OK!" << def << endl;
    }

    if (allOk) {
        if (verbose>1) cerr << "  " << layerName << ": " << green << "checkLayer: Numerical gradient check tests ok!" << def << endl;
    } else {
        if (verbose>0) cerr << "  " << layerName << ": " << red << "checkLayer: Numerical gradient check ended with error!" << def << endl;
    }
    return allOk;
}

bool Layer::selfTest(const MatrixN& x, t_cppl* pstates, floatN h=CP_DEFAULT_NUM_H, 
                     floatN eps=CP_DEFAULT_NUM_EPS, int verbose=1, Loss *pLoss=nullptr) {
    bool lossFkt=false, ret;
    MatrixN dchain;
    t_cppl cache;
    MatrixN y;
    if (verbose>1) cerr << "  SelfTest for: " << layerName << endl;
    t_cppl st;
    cppl_copy(pstates, &st);
    MatrixN yf = forward(x, nullptr, &st, 0);
    cppl_delete(&st);

    if ((layerType & LayerType::LT_NORMAL) && !(layerType & LayerType::LT_LOSS)) {
        dchain = yf;
        dchain.setRandom();
    } else if (layerType & LayerType::LT_LOSS) {
        //cppl_set(&cache, "probs", new MatrixN(yf));
        //cppl_set(&cache, "y", new MatrixN(y));
        if (pstates->find("y") == pstates->end()) {
            cerr << "  selfTest: pstates does not contain y -> fatal!" << endl;
        }
        y = *((*pstates)["y"]);
        dchain = y;
        lossFkt=true;
    }
    ret=checkLayer(x, y, dchain, &cache, pstates, h , eps, lossFkt, verbose, pLoss);
    cppl_delete(&cache);
    return ret;
}
#endif
