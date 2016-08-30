#ifndef _CP_LAYER_TESTS_H
#define _CP_LAYER_TESTS_H

#include <iostream>
#include <string>
#include <vector>

#include "cp-layer.h"
#include "cp-math.h"
#include "cp-util.h"

bool Layer::checkForward(MatrixN& x, floatN eps=CP_DEFAULT_NUM_EPS) {
    bool allOk=true;
    MatrixN yt=forward(x, nullptr);
    MatrixN y(0, yt.cols());
    for (unsigned int i=0; i<x.rows(); i++) {
        MatrixN xi=x.row(i);
        MatrixN yi = forward(xi, nullptr);
        MatrixN yn(yi.rows() + y.rows(),yi.cols());
        yn << y, yi;
        y = yn;
    }

    //MatrixN yc = forward(x, nullptr);
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    MatrixN d = y - yt;
    floatN dif = d.cwiseProduct(d).sum();

    if (dif < eps) {cout << "Forward vectorizer OK, err=" << dif << endl;}
    else {
        cout << "Forward vectorizer Error, err=" << dif << endl;
        cout << "y:" << y.format(CleanFmt) << endl;
        cout << "yt:" << yt.format(CleanFmt) << endl;
        allOk=false;
    }
    return allOk;
}

bool Layer::checkBackward(MatrixN& x, floatN eps=CP_DEFAULT_NUM_EPS) {
    bool allOk =true;
    IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
    t_cppl cache;
    t_cppl grads;
    t_cppl rgrads;

    MatrixN dyc = forward(x, &cache);
    dyc.setRandom();
    MatrixN dx = x;
    dx.setZero();

    MatrixN dxc = backward(dyc, &cache, &grads);
    for (auto it : grads) {
        cppl_set(&rgrads, it.first, new MatrixN(*grads[it.first])); // grads[it.first].rows(),grads[it.first].cols());
        rgrads[it.first]->setZero();
        //cout << shape(x) << " " << it.first << " "<< shape(dx) << shape(*rgrads[it.first]) << shape(*grads[it.first]) << endl;
    }
    for (unsigned int i=0; i<x.rows(); i++) {
        t_cppl chi;
        t_cppl gdi;
        MatrixN xi=x.row(i);
        MatrixN dyi2 = forward(xi, &chi);
        MatrixN dyi = dyc.row(i);
        MatrixN dyt = backward(dyi, &chi, &gdi);
        //cout << "B-SHAPE_INS" << shape(dx) << shape(dyt) << endl;
        dx.row(i) = dyt.row(0);
        for (auto it : grads) {
            *rgrads[it.first] += *gdi[it.first];
        }
        cppl_delete(&gdi);
        cppl_delete(&chi);
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
            cout << "Backward vectorizer " << "d" << it.first << "Error, err=" << dif << endl;
            allOk=false;
        }
    }
    cppl_delete(&cache);
    cppl_delete(&grads);
    cppl_delete(&rgrads);
    return allOk;
}

MatrixN Layer::calcNumGrad(MatrixN& dchain, t_cppl* pcache, string var, floatN h=CP_DEFAULT_NUM_H) {
    MatrixN *pm = params[var];
    MatrixN grad((*pm).rows(), (*pm).cols());

    floatN pxold;
    for (unsigned int i=0; i<grad.size(); i++) {
        pxold = (*(params[var]))(i);
        (*(params[var]))(i) = (*(params[var]))(i) - h;
        MatrixN y0 = forward(*((*pcache)["x"]), nullptr);
        (*(params[var]))(i) = pxold + h;
        MatrixN y1 = forward(*((*pcache)["x"]), nullptr);
        (*(params[var]))(i) = pxold;
        MatrixN dy=y1-y0;
        MatrixN dd;
        dd = dy.cwiseProduct(dchain);

        floatN drs = (dd / floatN(2.0 * h)).sum();
        grad(i)=drs;
    }
    return grad;
}

MatrixN Layer::calcNumGradLoss(t_cppl *pcache, string var, floatN h=CP_DEFAULT_NUM_H) {
    MatrixN *pm = params[var];
    MatrixN grad((*pm).rows(), (*pm).cols());

    floatN pxold;
    for (unsigned int i=0; i<grad.size(); i++) {
        t_cppl cache;
        pxold = (*(params[var]))(i);
        (*(params[var]))(i) = (*(params[var]))(i) - h;
        MatrixN y0 = forward(*((*pcache)["x"]), &cache);
        floatN sy0 = loss(*((*pcache)["y"]), &cache);
        cppl_delete(&cache);
        (*(params[var]))(i) = pxold + h;
        MatrixN y1 = forward(*(params[var]), &cache);
        floatN sy1 = loss(*((*pcache)["y"]), &cache);
        cppl_delete(&cache);
        (*(params[var]))(i) = pxold;
        floatN dy=sy1-sy0;
        floatN drs = dy / (2.0 * h);
        grad(i)=drs;
    }
    return grad;
}

bool Layer::calcNumGrads(MatrixN& dchain, t_cppl *pcache, t_cppl *pgrads, t_cppl *pnumGrads, floatN h=CP_DEFAULT_NUM_H, bool lossFkt=false) {
    for (auto it : *pgrads) {
        MatrixN g;
        if (!lossFkt) {
            g = calcNumGrad(dchain, pcache, it.first, h);
        } else {
            g = calcNumGradLoss(pcache, it.first, h);
        }
        cppl_set(pnumGrads, it.first, new MatrixN(g));
    }
    return true;
}

bool Layer::checkGradients(MatrixN& dchain, t_cppl *pcache, floatN h=CP_DEFAULT_NUM_H, floatN eps=CP_DEFAULT_NUM_EPS, bool lossFkt=false){
    bool allOk=true;
    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    Color::Modifier lred(Color::FG_LIGHT_RED);
    Color::Modifier lgreen(Color::FG_LIGHT_GREEN);
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    t_cppl grads;
    MatrixN x=*(*pcache)["x"];
    MatrixN yt=forward(x, pcache);
    if (lossFkt) { // XXX probably not needed!
        loss(dchain, pcache);
    }
    backward(dchain, pcache, &grads);

    t_cppl numGrads;
    calcNumGrads(dchain, pcache, &grads, &numGrads, h, lossFkt);

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

bool Layer::checkLayer(MatrixN& dchain, t_cppl *pcache, floatN h=CP_DEFAULT_NUM_H, floatN eps=CP_DEFAULT_NUM_EPS, bool lossFkt=false) {
    bool allOk=true, ret;
    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    Color::Modifier lred(Color::FG_LIGHT_RED);
    Color::Modifier lgreen(Color::FG_LIGHT_GREEN);
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    ret=checkForward(*(*pcache)["x"], eps);
    if (!ret) {
        cout << layerName << ": " << red << "Forward vectorizing test failed!" << def << endl;
        return ret;
    } else {
        cout << layerName << ": "<< green << "Forward vectorizing test OK!" << def << endl;
    }

    ret=checkBackward(*(*pcache)["x"], eps);
    if (!ret) {
        cout << layerName << ": " << red << "Backward vectorizing test failed!" << def << endl;
        return ret;
    } else {
        cout << layerName << ": " << green << "Backward vectorizing test OK!" << def << endl;
    }

    ret=checkGradients(dchain, pcache, h, eps, lossFkt);
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

bool Layer::selfTest(MatrixN& x, MatrixN& y, floatN h=CP_DEFAULT_NUM_H, floatN eps=CP_DEFAULT_NUM_EPS) {
    bool lossFkt=false, ret;
    MatrixN dchain;
    t_cppl cache;
    MatrixN yf = forward(x, &cache);
    if (layerType == LayerType::LT_NORMAL) {
        dchain = yf;
        dchain.setRandom();
    } else if (layerType == LayerType::LT_LOSS) {
        cppl_set(&cache, "probs", new MatrixN(yf));
        cppl_set(&cache, "y", new MatrixN(y));
        dchain = y;
        lossFkt=true;
    }
    ret=checkLayer(dchain, &cache, h, eps, lossFkt);
    cppl_delete(&cache);
    return ret;
}
#endif
