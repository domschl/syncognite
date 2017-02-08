#ifndef _CPLT_DROPOUT_H
#define _CPLT_DROPOUT_H

#include "../testneural.h"

bool checkDropout(float eps=3.0e-2, int verbose=1) {
    bool allOk=true;
    t_cppl states;
    MatrixN x(500,500);
    x.setRandom();
    floatN dl=10.0;
    floatN dop=0.8;
    x.array() += dl;

    Dropout dp("{inputShape=[500];train=true}");
    dp.cp.setPar("drop",dop);
    MatrixN y=dp.forward(x, nullptr, &states);
    dp.cp.setPar("train",false);
    MatrixN yt=dp.forward(x, nullptr, &states);

    floatN xm=x.mean();
    floatN ym=y.mean();
    floatN ytm=yt.mean();

    if (verbose>1) {
        cerr << "  Dropout: x-mean:" << xm << endl;
        cerr << "    y-mean:" << ym << endl;
        cerr << "    yt-mean:" << ytm << endl;
        cerr << "    drop:" << dop << endl;
        cerr << "    offs:" << dl << endl;
    }

    floatN err1=std::abs(ytm-ym);
    if (err1 > eps) {
        allOk=false;
        if (verbose>0) cerr << "  Dropout: difference between test-mean:" << ytm << " and train-mean: " << ym << " too high: " << err1 << endl;
    }
    floatN err2=std::abs(xm-dl);
    if (err2 > eps) {
        allOk=false;
        if (verbose>0) cerr << "  Dropout: difference between x-mean and random-offset too high: "  << err2 << endl;
    }
    floatN err3=std::abs(dl*dop-ym);
    if (err3 > eps) {
        allOk=false;
        if (verbose>0) cerr << "  Dropout: difference between y-mean*offset and droprate too high: "  << err3 << endl;
    }
    if (allOk) {
        if (verbose>1) cerr << "  Dropout: statistics tests ok, err1: " << err1 << " err2: " << err2 << " err3: " << err3 << endl;
    }
    return allOk;
}

bool testDropout(int verbose) {
    Color::Modifier lblue(Color::FG_LIGHT_BLUE);
    Color::Modifier def(Color::FG_DEFAULT);
	bool bOk=true;
	t_cppl s1;
	cerr << lblue << "Dropout Layer: " << def << endl;
	// Numerical gradient
    // Dropout
	Dropout dp("{inputShape=[5];train=true;noVectorizationTests=true;freeze=true;drop=0.8}");
	MatrixN xdp(3, 5);
	xdp.setRandom();
	floatN h = 1e-6;
	if (h < CP_DEFAULT_NUM_H)
		h = CP_DEFAULT_NUM_H;
	floatN eps = 1e-8;
	if (eps < CP_DEFAULT_NUM_EPS)
		eps = CP_DEFAULT_NUM_EPS;
	bool res=dp.selfTest(xdp, &s1, h, eps, verbose);
	registerTestResult("Dropout", "Numerical gradient", res, "");
	if (!res) bOk = false;

    eps=3e-2;
	res=checkDropout(eps, verbose);
	registerTestResult("Dropout", "Forward (with test-data)", res, "");
	if (!res) bOk = false;

	return bOk;
}

#endif
