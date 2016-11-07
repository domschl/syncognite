#ifndef _CP_LAYER_TESTS_H
#define _CP_LAYER_TESTS_H

#include "cp-neural.h"

bool Layer::checkForward(const MatrixN& x, const MatrixN& y, floatN eps=CP_DEFAULT_NUM_EPS) {
    bool allOk=true;
    MatrixN yt;
    if (layerType==LayerType::LT_LOSS) {
        yt=forward(x, y, nullptr, 0);
    } else {
        yt=forward(x, nullptr, 0);
    }
    MatrixN yic(0, yt.cols());
    for (unsigned int i=0; i<x.rows(); i++) {
        MatrixN xi=x.row(i);
        MatrixN yi;
        if (layerType==LayerType::LT_LOSS) {
            MatrixN yii=y.row(i);
            yi = forward(xi, yii, nullptr, 0);
        } else {
            yi = forward(xi, nullptr, 0);
        }
        MatrixN yn(yi.rows() + yic.rows(),yi.cols());
        yn << yic, yi;
        yic = yn;
    }

    //MatrixN yc = forward(x, nullptr);
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    MatrixN d = yic - yt;
    floatN dif = d.cwiseProduct(d).sum();

    if (dif < eps) {cerr << "Forward vectorizer OK, err=" << dif << endl;}
    else {
        cerr << "Forward vectorizer Error, err=" << dif << endl;
        cerr << "yic:" << yic.format(CleanFmt) << endl;
        cerr << "yt:" << yt.format(CleanFmt) << endl;
        allOk=false;
    }
    return allOk;
}

bool Layer::checkBackward(const MatrixN& x, const MatrixN& y, t_cppl *pcache, floatN eps=CP_DEFAULT_NUM_EPS) {
    bool allOk =true;
    IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
    t_cppl cache;
    t_cppl grads;
    t_cppl rgrads;

    MatrixN dyc;
    if (layerType==LayerType::LT_NORMAL) {
        dyc = forward(x, &cache, 0);
        dyc.setRandom();
    } else if (layerType==LayerType::LT_LOSS) {
        forward(x, y, &cache, 0);
        dyc=y;
    } else {
        cerr << "BAD LAYER TYPE!" << layerType << endl;
        return false;
    }
    if (dyc.rows() != x.rows()) {
        cerr << "internal error: y:" << shape(dyc) << " x:" << shape(x) << " - unequal number of rows!" << endl;
        return false;
    }
    MatrixN dx = x;
    dx.setZero();

    if (cache.find("x")==cache.end()) cerr << "WARNING: x is not in cache!" << endl;

    MatrixN dxc = backward(dyc, &cache, &grads, 0);

    for (auto it : grads) {
        cppl_set(&rgrads, it.first, new MatrixN(*grads[it.first])); // grads[it.first].rows(),grads[it.first].cols());
        rgrads[it.first]->setZero();
    }

    for (unsigned int i=0; i<x.rows(); i++) {
        t_cppl chi;
        t_cppl gdi;
        MatrixN xi=x.row(i);
        MatrixN dyi2;
        if (layerType==LayerType::LT_LOSS) {
            dyi2 = forward(xi, dyc.row(i), &chi, 0);
        } else {
            dyi2 = forward(xi, &chi, 0);
        }
        MatrixN dyi = dyc.row(i);
        MatrixN dyt = backward(dyi, &chi, &gdi, 0);
        dx.row(i) = dyt.row(0);
        for (auto it : grads) {
            *rgrads[it.first] += *gdi[it.first];
        }
        cppl_delete(&gdi);
        cppl_delete(&chi);
    }
    if (layerType==LayerType::LT_LOSS) {
        dx /= x.rows(); // XXX is this sound for all types of loss-layers?
        for (auto it : grads) { // XXX this is beyound soundness...
             *(grads[it.first]) *= x.rows();
        }
    }
    MatrixN d = dx - dxc;
    floatN dif = d.cwiseProduct(d).sum();
    if (dif < eps) {cerr << "Backward vectorizer dx OK, err=" << dif << endl;}
    else {
        cerr << "Backward vectorizer dx:" << endl << dx.format(CleanFmt) << endl;
        cerr << "dxc:" << endl << dxc.format(CleanFmt) << endl;
        cerr << "Backward vectorizer dx Error, err=" << dif << endl;
        allOk=false;
    }

    for (auto it : grads) {
        MatrixN d = *grads[it.first] - *rgrads[it.first];
        floatN dif = d.cwiseProduct(d).sum();
        if (dif < eps) {
            cerr << "Backward vectorizer " << "d" << it.first << " OK, err=" << dif << endl;
        } else {
            cerr << "d" << it.first << ":" << endl << grads[it.first]->format(CleanFmt) << endl;
            cerr << "d" << it.first << "c:" << endl << rgrads[it.first]->format(CleanFmt) << endl;
            cerr << "Backward vectorizer " << "d" << it.first << " error, err=" << dif << endl;
            allOk=false;
        }
    }
    cppl_delete(&cache);
    cppl_delete(&grads);
    cppl_delete(&rgrads);
    return allOk;
}

MatrixN Layer::calcNumGrad(const MatrixN& xorg, const MatrixN& dchain, t_cppl* pcache, string var, floatN h=CP_DEFAULT_NUM_H) {
    cerr << "  checking numerical gradient for " << var << "..." << endl;
    MatrixN *pm;
    MatrixN x;
    if (pcache->find("x")==pcache->end()) { // XXX that is quite a mess!
        x=xorg;
    } else {
        x=*((*pcache)["x"]);
    }
    if (var=="x") pm=&x;
    else pm = params[var];
    MatrixN grad((*pm).rows(), (*pm).cols());

    floatN pxold;
    for (unsigned int i=0; i<grad.size(); i++) {
        MatrixN y0,y1;
        if (var=="x") {
            pxold = x(i);
            x(i) = x(i) - h;
            y0 = forward(x, nullptr, 0);
            x(i) = pxold + h;
            y1 = forward(x, nullptr,0 );
            x(i) = pxold;
        } else {
            pxold = (*(params[var]))(i);
            (*(params[var]))(i) = (*(params[var]))(i) - h;
            y0 = forward(x, nullptr, 0);
            (*(params[var]))(i) = pxold + h;
            y1 = forward(x, nullptr, 0);
            (*(params[var]))(i) = pxold;
        }
        MatrixN dy=y1-y0;
        MatrixN dd;
        dd = dy.cwiseProduct(dchain);

        floatN drs = (dd / floatN(2.0 * h)).sum();
        grad(i)=drs;
    }
    return grad;
}

MatrixN Layer::calcNumGradLoss(const MatrixN& xorg, t_cppl *pcache, string var, floatN h=CP_DEFAULT_NUM_H) {
    MatrixN *pm;

    MatrixN x=*((*pcache)["x"]);
    //MatrixN x=xorg;
    MatrixN y=*((*pcache)["y"]);
    if (var=="x") pm=&x;
    else pm = params[var];
    MatrixN grad(pm->rows(), pm->cols());
    //cerr << var << "/dx-shape:" << shape(grad) << endl;

    floatN pxold;
    for (unsigned int i=0; i<grad.size(); i++) {
        t_cppl cache;
        MatrixN y0,y1;
        float sy0, sy1;
        if (var=="x") {
            pxold = x(i);
            x(i) = x(i) - h;
            y0 = forward(x, y, &cache, 0);
            sy0 = loss(y, &cache);
            cppl_delete(&cache);
            x(i) = pxold + h;
            y1 = forward(x, y, &cache, 0);
            sy1 = loss(y, &cache);
            cppl_delete(&cache);
            x(i) = pxold;
        } else {
            pxold = (*pm)(i);
            (*pm)(i) = (*pm)(i) - h;
            y0 = forward(x, y, &cache, 0);
            sy0 = loss(y, &cache);
            cppl_delete(&cache);
            (*pm)(i) = pxold + h;
            y1 = forward(x, y, &cache, 0);
            sy1 = loss(y, &cache);
            cppl_delete(&cache);
            (*pm)(i) = pxold;
        }
        floatN dy=sy1-sy0;
        floatN drs = dy / (2.0 * h);
        grad(i)=drs;
    }
    return grad;
}

bool Layer::calcNumGrads(const MatrixN& x, const MatrixN& dchain, t_cppl *pcache, t_cppl *pgrads, t_cppl *pnumGrads, floatN h=CP_DEFAULT_NUM_H, bool lossFkt=false) {
    for (auto it : *pgrads) {
        MatrixN g;
        if (!lossFkt) {
            g = calcNumGrad(x, dchain, pcache, it.first, h);
        } else {
            g = calcNumGradLoss(x, pcache, it.first, h);
        }
        cppl_set(pnumGrads, it.first, new MatrixN(g));
    }
    return true;
}

bool Layer::checkGradients(const MatrixN& x, const MatrixN& y, const MatrixN& dchain, t_cppl *pcache, floatN h=CP_DEFAULT_NUM_H, floatN eps=CP_DEFAULT_NUM_EPS, bool lossFkt=false){
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
    if (lossFkt) {
        yt=forward(x, y, pcache, 0);
        loss(y,pcache);
        dx=backward(y, pcache, &grads, 0);
    } else {
        yt=forward(x, pcache, 0);
        dx=backward(dchain, pcache, &grads, 0);
    }
    if (dx.rows()>0 && dx.cols()>0)  // Some layers (e.g. WordEmbeddings have no dx!)
        grads["x"]=new MatrixN(dx);

    t_cppl numGrads;
    calcNumGrads(x, dchain, pcache, &grads, &numGrads, h, lossFkt);

    for (auto it : grads) {
        IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        MatrixN d = *(grads[it.first]) - *(numGrads[it.first]);
        floatN df = (d.cwiseProduct(d)).sum();
        if (df < eps) {
            cerr << layerName << ": " << "∂/∂" << it.first << green << " OK, err=" << df << def << endl;
        } else {
            cerr << "eps:" << eps << " h:" << h << endl;
            cerr << "∂/∂" << it.first << "[num]: " << endl << (*(numGrads[it.first])).format(CleanFmt) << endl;
            cerr << "∂/∂" << it.first << "[the]: " << endl << (*(grads[it.first])).format(CleanFmt) << endl;
            cerr << "  ð" << it.first << "    : " << endl << ((*(grads[it.first])) - (*(numGrads[it.first]))).format(CleanFmt) << endl;
            cerr << layerName << ": " << "∂/∂" << it.first << red << " ERROR, err=" << df << def << endl;
            allOk=false;
        }
    }
    cppl_delete(&grads);
    cppl_delete(&numGrads);
    return allOk;
}

bool Layer::checkLayer(const MatrixN& x, const MatrixN& y, const MatrixN& dchain, t_cppl *pcache, floatN h=CP_DEFAULT_NUM_H, floatN eps=CP_DEFAULT_NUM_EPS, bool lossFkt=false) {
    bool allOk=true, ret;
    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    Color::Modifier lred(Color::FG_LIGHT_RED);
    Color::Modifier lgreen(Color::FG_LIGHT_GREEN);
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    if (!cp.getPar("noVectorizationTests", false)) {
        cerr << "  check forward vectorizer " << layerName << "..." << endl;
        ret=checkForward(x, y, eps);
        if (!ret) {
            cerr << layerName << ": " << red << "Forward vectorizing test failed!" << def << endl;
            return ret;
        } else {
            cerr << layerName << ": "<< green << "Forward vectorizing test OK!" << def << endl;
        }

        cerr << "  check backward vectorizer " << layerName << "..." << endl;
        t_cppl cache;
        ret=checkBackward(x, y, &cache, eps);
        cppl_delete(&cache);
        if (!ret) {
            cerr << layerName << ": " << red << "Backward vectorizing test failed!" << def << endl;
            allOk=false; //return ret;
        } else {
            cerr << layerName << ": " << green << "Backward vectorizing test OK!" << def << endl;
        }
    }

    cerr << "  check numerical gradients " << layerName << "..." << endl;
    t_cppl cache2;
    ret=checkGradients(x, y, dchain, &cache2, h, eps, lossFkt);
    cppl_delete(&cache2);
    if (!ret) {
        cerr << layerName << ": " << red << "Gradient numerical test failed!" << def << endl;
        return ret;
    } else {
        cerr << layerName << ": " << green << "Gradient numerical test OK!" << def << endl;
    }

    if (allOk) {
        cerr << layerName << ": " << green << "checkLayer: Numerical gradient check tests ok!" << def << endl;
    } else {
        cerr << layerName << ": " << red << "checkLayer: tests ended with error!" << def << endl;
    }
    return allOk;
}

bool Layer::selfTest(const MatrixN& x, const MatrixN& y, floatN h=CP_DEFAULT_NUM_H, floatN eps=CP_DEFAULT_NUM_EPS) {
    bool lossFkt=false, ret;
    MatrixN dchain;
    t_cppl cache;
    cerr << "SelfTest for: " << layerName << " -----------------" << endl;
    MatrixN yf = forward(x, &cache, 0);
    if (layerType == LayerType::LT_NORMAL) {
        dchain = yf;
        dchain.setRandom();
    } else if (layerType == LayerType::LT_LOSS) {
        //cppl_set(&cache, "probs", new MatrixN(yf));
        //cppl_set(&cache, "y", new MatrixN(y));
        dchain = y;
        lossFkt=true;
    }
    ret=checkLayer(x, y, dchain, &cache, h , eps, lossFkt);
    cppl_delete(&cache);
    return ret;
}
#endif
