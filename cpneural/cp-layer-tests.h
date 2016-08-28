#ifndef _CP_LAYER_TESTS_H
#define _CP_LAYER_TESTS_H

#include <iostream>
#include <string>
#include <vector>

#include "cp-layer.h"
#include "cp-math.h"

bool Layer::checkForward(MatrixN& x, floatN eps=1.0e-6) {
    bool allOk=true;
    MatrixN yt=forward(x);
    MatrixN y(0, yt.cols());
    for (unsigned int i=0; i<x.rows(); i++) {
        MatrixN xi=x.row(i);
        MatrixN yi = forward(xi);
        MatrixN yn(yi.rows() + y.rows(),yi.cols());
        yn << y, yi;
        y = yn;
    }

    MatrixN yc = forward(x);
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    MatrixN d = y - yc;
    floatN dif = d.cwiseProduct(d).sum();

    if (dif < eps) {cout << "Forward vectorizer OK, err=" << dif << endl;}
    else {
        cout << "Forward vectorizer Error, err=" << dif << endl;
        cout << "y:" << y.format(CleanFmt) << endl;
        cout << "yc:" << yc.format(CleanFmt) << endl;
        allOk=false;
    }
    return allOk;
}

bool Layer::checkBackward(MatrixN& dchain, floatN eps=CP_DEFAULT_NUM_EPS) {
    bool allOk =true;
    IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
    MatrixN x = *(params[0]);
    MatrixN dyc = forward(x);
    dyc.setRandom();
    MatrixN dx = x;
    dx.setZero();

    MatrixN dxc = backward(dyc);
    // vector<MatrixN *> dparso = getG(dyc);
    vector<MatrixN *> dpars(grads.size());
    for (unsigned int i=1; i<dpars.size(); i++) {
        MatrixN *pm = grads[i];
        dpars[i]= new MatrixN((*pm).rows(), (*pm).cols());
        dpars[i]->setZero();
    }
    //cout << shape(x) << shape(dx) << shape(dxc) << shape(yc) << endl;
    for (unsigned int i=0; i<x.rows(); i++) {
        MatrixN xi=x.row(i);
        MatrixN dyi2 = forward(xi);
        MatrixN dyi = dyc.row(i);
        MatrixN dyt = backward(dyi);
        //cout << "B-SHAPE_INS" << shape(dx) << shape(dyt) << endl;
        dx.row(i) = dyt.row(0);
        //vector<MatrixN *> dparsi = getG(dyi);
        for (unsigned int j=1; j<grads.size(); j++) {
            MatrixN *pmi = grads[j];
            //cout << i << "-" << j <<"-0:" << (*(dpars[j])).format(CleanFmt) << endl;
            *(dpars[j]) = *(dpars[j]) + *pmi;
            //cout << i << "-" << j << "-1:" << (*(dpars[j])).format(CleanFmt) << endl;
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

    MatrixN dyc2 = forward(x);
    dxc = backward(dyc);
    //vector<MatrixN *> dparsc = getG(dyc);
    for (unsigned int i=1; i<grads.size(); i++) {
        MatrixN *pmc = grads[i];
        MatrixN *pm = dpars[i];
        MatrixN d = *pm - *pmc;
        floatN dif = d.cwiseProduct(d).sum();
        if (dif < eps) {
            cout << "Backward vectorizer " << "d" << names[i] << " OK, err=" << dif << endl;
        } else {
            cout << "d" << names[i] << ":" << endl << (*pm).format(CleanFmt) << endl;
            cout << "d" << names[i] << "c:" << endl << (*pmc).format(CleanFmt) << endl;
            cout << "Backward vectorizer " << "d" << names[i] << "Error, err=" << dif << endl;
            allOk=false;
        }
    }
    for (unsigned int i=1; i<dpars.size(); i++) {
        delete dpars[i];
        dpars[i]=nullptr;
    }
    return allOk;
}

MatrixN Layer::calcNumGrad(MatrixN& dchain, unsigned int ind, floatN h=CP_DEFAULT_NUM_H) {
    if (ind >= params.size()) {
        cout << "bad index! " << ind << endl;
        return MatrixN(0,0);
    }
    MatrixN *pm = params[ind];
    MatrixN grad((*pm).rows(), (*pm).cols());

    floatN pxold;
    for (unsigned int i=0; i<grad.size(); i++) {
        pxold = (*(params[ind]))(i);
        (*(params[ind]))(i) = (*(params[ind]))(i) - h;
        MatrixN y0 = forward(*(params[0]));
        (*(params[ind]))(i) = pxold + h;
        MatrixN y1 = forward(*(params[0]));
        (*(params[ind]))(i) = pxold;
        MatrixN dy=y1-y0;
        MatrixN dd;
        dd = dy.cwiseProduct(dchain);

        floatN drs = (dd / floatN(2.0 * h)).sum();
        grad(i)=drs;
    }
    return grad;
}

MatrixN Layer::calcNumGradLoss(MatrixN& dchain, unsigned int ind, floatN h=CP_DEFAULT_NUM_H) {
    if (ind >= params.size()) {
        cout << "bad index! " << ind << endl;
        return MatrixN(0,0);
    }
    MatrixN *pm = params[ind];
    MatrixN grad((*pm).rows(), (*pm).cols());

    floatN pxold;
    for (unsigned int i=0; i<grad.size(); i++) {
        pxold = (*(params[ind]))(i);
        (*(params[ind]))(i) = (*(params[ind]))(i) - h;
        MatrixN y0 = forward(*(params[0]));
        floatN sy0 = loss(*(cache[1]));
        (*(params[ind]))(i) = pxold + h;
        MatrixN y1 = forward(*(params[0]));
        floatN sy1 = loss(*(cache[1]));
        (*(params[ind]))(i) = pxold;
        floatN dy=sy1-sy0;
        floatN drs = dy / (2.0 * h);
        grad(i)=drs;
    }
    return grad;
}

bool Layer::calcNumGrads(MatrixN& dchain, vector<MatrixN *>numGrads, floatN h=CP_DEFAULT_NUM_H, bool lossFkt=false) {
    MatrixN y=forward(*(params[0]));
    if (params.size() != grads.size()) {
        cout << "Internal error: params has " << params.size() << " parameters, grads has " << grads.size() << " parameters. Count must be equal!" << endl;
        return false;
    }
    if (params.size() != numGrads.size()) {
        cout << "Internal error: params has " << params.size() << " parameters, numGrads has " << numGrads.size() << " parameters. Count must be equal!" << endl;
        return false;
    }
    if (params.size() != names.size()) {
        cout << "Internal error: params has " << params.size() << " parameters, names-parameter nms has " << names.size() << " parameters. Count must be equal!" << endl;
        return false;
    }
    for (unsigned int i=0; i<params.size(); i++) {
        if (((*(params[i])).cols() != (*(grads[i])).cols()) || (*(params[i])).rows() != (*(grads[i])).rows()) {
            vector<unsigned int> s0, s1;
            s0 = shape(*(params[i]));
            s1 = shape(*(grads[i]));
            cout << "Dimension mismatch grads/params ind[" << i << "]: " << names[i] << s0 << " != d" << names[i] << s1 << endl;
            return false;
        }
    }
    for (unsigned int m=0; m<grads.size(); m++) {
        MatrixN g;
        if (!lossFkt) {
            g = calcNumGrad(dchain, m, h);
        } else {
            g = calcNumGradLoss(dchain, m, h);
        }
        *(numGrads[m]) = g;
    }
    return true;
}

bool Layer::checkGradients(MatrixN& x, MatrixN& dchain, floatN h=CP_DEFAULT_NUM_H, floatN eps=CP_DEFAULT_NUM_EPS, bool lossFkt=false){
    bool allOk=true;
    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    Color::Modifier lred(Color::FG_LIGHT_RED);
    Color::Modifier lgreen(Color::FG_LIGHT_GREEN);
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    MatrixN yt=forward(x);
    if (lossFkt) { // XXX probably not needed!
        loss(dchain);
    }
    backward(dchain);

    if (params.size() != grads.size()) {
        cout << layerName << ": " << "Internal error: params has " << params.size() << " parameters, ngrads has " << grads.size() << " parameters. Count must be equal!" << endl;
        return false;
    }
    if (params.size() != names.size()) {
        cout << layerName << ": " << "Internal error: params has " << params.size() << " parameters, names-parameter nms has " << names.size() << " parameters. Count must be equal!" << endl;
        return false;
    }
    vector<MatrixN *> numGrads(grads.size());
    for (unsigned int i=0; i<numGrads.size(); i++) {
        MatrixN *pm = grads[i];
        numGrads[i]= new MatrixN(pm->rows(), pm->cols());
        numGrads[i]->setZero();
    }
    for (unsigned int i=0; i<numGrads.size(); i++) {
        if ((params[i]->cols() != numGrads[i]->cols()) || (params[i]->rows() != numGrads[i]->rows())) {
            vector<unsigned int> s0, s1;
            s0 = shape(*(params[i]));
            s1 = shape(*(numGrads[i]));
            cout << layerName << ": " << "Dimension mismatch params/numGrads ind[" << names[i] << "]: " << i << ", " << names[i] << s0 << " != d" << names[i] << s1 << endl;
            return false;
        }
    }

    if (!calcNumGrads(dchain, numGrads, h, lossFkt)) {
        cout << "Bad init!";
        return false;
    }
    for (unsigned int m=0; m<numGrads.size(); m++) {
        IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        MatrixN d = *(grads[m]) - *(numGrads[m]);
        floatN df = (d.cwiseProduct(d)).sum();
        if (df < eps) {
            cout << layerName << ": " << "∂/∂" << names[m] << green << " OK, err=" << df << def << endl;
        } else {
            cout << "∂/∂" << names[m] << "[num]: " << endl << (*(numGrads[m])).format(CleanFmt) << endl;
            cout << "∂/∂" << names[m] << "[the]: " << endl << (*(grads[m])).format(CleanFmt) << endl;
            cout << "  ð" << names[m] << "    : " << endl << ((*(grads[m])) - (*(numGrads[m]))).format(CleanFmt) << endl;
            cout << layerName << ": " << "∂/∂" << names[m] << red << " ERROR, err=" << df << def << endl;
            allOk=false;
        }
    }
    for (unsigned int i=0; i<numGrads.size(); i++) {
        delete numGrads[i];
        numGrads[i]=nullptr;
    }
    return allOk;
}

bool Layer::checkLayer(MatrixN& x, MatrixN& dchain, floatN h=CP_DEFAULT_NUM_H, floatN eps=CP_DEFAULT_NUM_EPS, bool lossFkt=false) {
    bool allOk=true, ret;
    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    Color::Modifier lred(Color::FG_LIGHT_RED);
    Color::Modifier lgreen(Color::FG_LIGHT_GREEN);
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    ret=checkForward(*(params[0]), eps);
    if (!ret) {
        cout << layerName << ": " << red << "Forward vectorizing test failed!" << def << endl;
        return ret;
    } else {
        cout << layerName << ": "<< green << "Forward vectorizing test OK!" << def << endl;
    }

    ret=checkBackward(dchain, eps);
    if (!ret) {
        cout << layerName << ": " << red << "Backward vectorizing test failed!" << def << endl;
        return ret;
    } else {
        cout << layerName << ": " << green << "Backward vectorizing test OK!" << def << endl;
    }

    ret=checkGradients(x, dchain, h, eps, lossFkt);
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
    bool lossFkt=false;
    MatrixN dchain;
    MatrixN yf = forward(x);
    if (layerType == LayerType::LT_NORMAL) {
        dchain = yf;
        dchain.setRandom();
    } else if (layerType == LayerType::LT_LOSS) {
        if (cache.size() >= 2) {
            MatrixN probs = yf;
            *cache[0]=probs;
            *cache[1]=y;
            dchain = y;
        } else {
            cout << "Internal error, cache not set for loss-layer" << endl;
            return false;
        }
        lossFkt=true;
    }
    return checkLayer(x, dchain, h, eps, lossFkt);
}
#endif
