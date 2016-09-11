#ifndef _CP_LAYER_TESTS_H
#define _CP_LAYER_TESTS_H

#include <iostream>
#include <string>
#include <vector>

#include "cp-layer.h"
#include "cp-math.h"
#include "cp-util.h"

bool Layer::checkForward(const MatrixN& x, const MatrixN& y, floatN eps=CP_DEFAULT_NUM_EPS) {
    bool allOk=true;
    MatrixN yt;
    if (layerType==LayerType::LT_LOSS) {
        yt=forward(x, y, nullptr);
    } else {
        yt=forward(x, nullptr);
    }
    MatrixN yic(0, yt.cols());
    for (unsigned int i=0; i<x.rows(); i++) {
        MatrixN xi=x.row(i);
        MatrixN yi;
        if (layerType==LayerType::LT_LOSS) {
            MatrixN yii=y.row(i);
            yi = forward(xi, yii, nullptr);
        } else {
            yi = forward(xi, nullptr);
        }
        MatrixN yn(yi.rows() + yic.rows(),yi.cols());
        yn << yic, yi;
        yic = yn;
    }

    //MatrixN yc = forward(x, nullptr);
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    MatrixN d = yic - yt;
    floatN dif = d.cwiseProduct(d).sum();

    if (dif < eps) {cout << "Forward vectorizer OK, err=" << dif << endl;}
    else {
        cout << "Forward vectorizer Error, err=" << dif << endl;
        cout << "yic:" << yic.format(CleanFmt) << endl;
        cout << "yt:" << yt.format(CleanFmt) << endl;
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
        dyc = forward(x, &cache);
        dyc.setRandom();
    } else if (layerType==LayerType::LT_LOSS) {
        forward(x, y, &cache);
        dyc=y;
    } else {
        cout << "BAD LAYER TYPE!" << layerType << endl;
        return false;
    }
    if (dyc.rows() != x.rows()) {
        cout << "internal error: y:" << shape(dyc) << " x:" << shape(x) << " - unequal number of rows!" << endl;
        return false;
    }
    MatrixN dx = x;
    dx.setZero();

    if (cache.find("x")==cache.end()) cout << "WARNING: x is not in cache!" << endl;

    MatrixN dxc = backward(dyc, &cache, &grads);

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
            dyi2 = forward(xi, dyc.row(i), &chi);
        } else {
            dyi2 = forward(xi, &chi);
        }
        MatrixN dyi = dyc.row(i);
        MatrixN dyt = backward(dyi, &chi, &gdi);
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
    if (dif < eps) {cout << "Backward vectorizer dx OK, err=" << dif << endl;}
    else {
        cout << "Backward vectorizer dx:" << endl << dx.format(CleanFmt) << endl;
        cout << "dxc:" << endl << dxc.format(CleanFmt) << endl;
        cout << "Backward vectorizer dx Error, err=" << dif << endl;
        allOk=false;
    }

    for (auto it : grads) {
        MatrixN d = *grads[it.first] - *rgrads[it.first];
        floatN dif = d.cwiseProduct(d).sum();
        if (dif < eps) {
            cout << "Backward vectorizer " << "d" << it.first << " OK, err=" << dif << endl;
        } else {
            cout << "d" << it.first << ":" << endl << grads[it.first]->format(CleanFmt) << endl;
            cout << "d" << it.first << "c:" << endl << rgrads[it.first]->format(CleanFmt) << endl;
            cout << "Backward vectorizer " << "d" << it.first << " error, err=" << dif << endl;
            allOk=false;
        }
    }
    cppl_delete(&cache);
    cppl_delete(&grads);
    cppl_delete(&rgrads);
    return allOk;
}

MatrixN Layer::calcNumGrad(const MatrixN& xorg, const MatrixN& dchain, t_cppl* pcache, string var, floatN h=CP_DEFAULT_NUM_H) {
    cout << "  checking numerical gradiend for " << var << "..." << endl;
    MatrixN *pm;
    MatrixN x=xorg;
    // MatrixN x=*((*pcache)["x"]);
    if (var=="x") pm=&x;
    else pm = params[var];
    MatrixN grad((*pm).rows(), (*pm).cols());

    floatN pxold;
    for (unsigned int i=0; i<grad.size(); i++) {
        MatrixN y0,y1;
        if (var=="x") {
            pxold = x(i);
            x(i) = x(i) - h;
            y0 = forward(x, nullptr);
            x(i) = pxold + h;
            y1 = forward(x, nullptr);
            x(i) = pxold;
        } else {
            pxold = (*(params[var]))(i);
            (*(params[var]))(i) = (*(params[var]))(i) - h;
            y0 = forward(x, nullptr);
            (*(params[var]))(i) = pxold + h;
            y1 = forward(x, nullptr);
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

    // MatrixN x=*((*pcache)["x"]);
    MatrixN x=xorg;
    MatrixN y=*((*pcache)["y"]);
    if (var=="x") pm=&x;
    else pm = params[var];
    MatrixN grad(pm->rows(), pm->cols());
    //cout << var << "/dx-shape:" << shape(grad) << endl;

    floatN pxold;
    for (unsigned int i=0; i<grad.size(); i++) {
        t_cppl cache;
        MatrixN y0,y1;
        float sy0, sy1;
        if (var=="x") {
            pxold = x(i);
            x(i) = x(i) - h;
            y0 = forward(x, y, &cache);
            sy0 = loss(y, &cache);
            cppl_delete(&cache);
            x(i) = pxold + h;
            y1 = forward(x, y, &cache);
            sy1 = loss(y, &cache);
            cppl_delete(&cache);
            x(i) = pxold;
        } else {
            pxold = (*pm)(i);
            (*pm)(i) = (*pm)(i) - h;
            y0 = forward(x, y, &cache);
            sy0 = loss(y, &cache);
            cppl_delete(&cache);
            (*pm)(i) = pxold + h;
            y1 = forward(x, y, &cache);
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
        yt=forward(x, y, pcache);
        loss(y,pcache);
        dx=backward(y, pcache, &grads);
    } else {
        yt=forward(x, pcache);
        dx=backward(dchain, pcache, &grads);
    }
    grads["x"]=new MatrixN(dx);

    t_cppl numGrads;
    calcNumGrads(x, dchain, pcache, &grads, &numGrads, h, lossFkt);

    for (auto it : grads) {
        IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        MatrixN d = *(grads[it.first]) - *(numGrads[it.first]);
        floatN df = (d.cwiseProduct(d)).sum();
        if (df < eps) {
            cout << layerName << ": " << "∂/∂" << it.first << green << " OK, err=" << df << def << endl;
        } else {
            cout << "∂/∂" << it.first << "[num]: " << endl << (*(numGrads[it.first])).format(CleanFmt) << endl;
            cout << "∂/∂" << it.first << "[the]: " << endl << (*(grads[it.first])).format(CleanFmt) << endl;
            cout << "  ð" << it.first << "    : " << endl << ((*(grads[it.first])) - (*(numGrads[it.first]))).format(CleanFmt) << endl;
            cout << layerName << ": " << "∂/∂" << it.first << red << " ERROR, err=" << df << def << endl;
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
        cout << "  check forward vectorizer " << layerName << "..." << endl;
        ret=checkForward(x, y, eps);
        if (!ret) {
            cout << layerName << ": " << red << "Forward vectorizing test failed!" << def << endl;
            return ret;
        } else {
            cout << layerName << ": "<< green << "Forward vectorizing test OK!" << def << endl;
        }

        cout << "  check backward vectorizer " << layerName << "..." << endl;
        t_cppl cache;
        ret=checkBackward(x, y, &cache, eps);
        cppl_delete(&cache);
        if (!ret) {
            cout << layerName << ": " << red << "Backward vectorizing test failed!" << def << endl;
            allOk=false; //return ret;
        } else {
            cout << layerName << ": " << green << "Backward vectorizing test OK!" << def << endl;
        }
    }

    cout << "  check numerical gradients " << layerName << "..." << endl;
    t_cppl cache2;
    ret=checkGradients(x, y, dchain, &cache2, h, eps, lossFkt);
    cppl_delete(&cache2);
    if (!ret) {
        cout << layerName << ": " << red << "Gradient numerical test failed!" << def << endl;
        return ret;
    } else {
        cout << layerName << ": " << green << "Gradient numerical test OK!" << def << endl;
    }

    if (allOk) {
        cout << layerName << ": " << green << "checkLayer: Numerical gradient check tests ok!" << def << endl;
    } else {
        cout << layerName << ": " << red << "checkLayer: tests ended with error!" << def << endl;
    }
    return allOk;
}

bool Layer::selfTest(const MatrixN& x, const MatrixN& y, floatN h=CP_DEFAULT_NUM_H, floatN eps=CP_DEFAULT_NUM_EPS) {
    bool lossFkt=false, ret;
    MatrixN dchain;
    t_cppl cache;
    cout << "SelfTest for: " << layerName << " -----------------" << endl;
    MatrixN yf = forward(x, &cache);
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
