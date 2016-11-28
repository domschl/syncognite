#ifndef _CPLT_DROPOUT_H
#define _CPLT_DROPOUT_H

#include "../testneural.h"

bool checkDropout(float eps=3.0e-2) {
    bool allOk=true;
    MatrixN x(500,500);
    x.setRandom();
    floatN dl=10.0;
    floatN dop=0.8;
    x.array() += dl;

    Dropout dp("{inputShape=[500];train=true}");
    dp.cp.setPar("drop",dop);
    MatrixN y=dp.forward(x, nullptr);
    dp.cp.setPar("train",false);
    MatrixN yt=dp.forward(x, nullptr);

    floatN xm=x.mean();
    floatN ym=y.mean();
    floatN ytm=yt.mean();

    cerr << "Dropout: x-mean:" << xm << endl;
    cerr << "  y-mean:" << ym << endl;
    cerr << "  yt-mean:" << ytm << endl;
    cerr << "  drop:" << dop << endl;
    cerr << "  offs:" << dl << endl;

    floatN err1=std::abs(ytm-ym);
    if (err1 > eps) {
        allOk=false;
        cerr << "Dropout: difference between test-mean:" << ytm << " and train-mean:" << ym << " too high:" << err1 << endl;
    }
    floatN err2=std::abs(xm-dl);
    if (err2 > eps) {
        allOk=false;
        cerr << "Dropout: difference between x-mean and random-offset too high:"  << err2 << endl;
    }
    floatN err3=std::abs(dl*dop-ym);
    if (err3 > eps) {
        allOk=false;
        cerr << "Dropout: difference between y-mean*offset and droprate too high"  << err3 << endl;
    }
    if (allOk) cerr << "Dropout: statistics tests ok, err1:" << err1 << " err2:" << err2 << " err3:" << err3 << endl;
    return allOk;
}

#endif
