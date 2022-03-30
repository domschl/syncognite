#ifndef _CPLT_SVM_H
#define _CPLT_SVM_H

#include "../testneural.h"

bool checkSvm(float eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    bool allOk=true;
    MatrixN x(10,5);
    x << 2.48040968e-04,   5.60446668e-04,  -3.52994957e-04,
         1.01572982e-03,   4.14494264e-04,
        -6.31693635e-04,  -8.15563788e-04,  -1.20636602e-03,
         -2.10174557e-03,   5.53294928e-04,
          1.14679595e-03,   1.24827753e-03,  -6.61989763e-04,
          9.55559461e-04,  -4.28180029e-04,
          4.46347111e-04,   6.23103141e-04,  -8.31752231e-04,
         -8.16901550e-04,  -3.51481858e-04,
          5.99420847e-04,   7.99136992e-04,   7.48694922e-04,
         -1.31792142e-03,  -1.41278790e-03,
          7.83720049e-04,  -1.87400705e-03,   6.83413931e-04,
         -3.33278182e-05,  -8.23791353e-04,
          4.48433013e-04,  -1.90826829e-04,  -1.18725164e-03,
          8.57369270e-04,  -2.03127259e-04,
         -8.12742999e-04,  -8.77664600e-04,   9.59702869e-04,
         -4.21470554e-05,  -1.26450252e-04,
          7.75822790e-04,  -9.17338786e-04,   6.60689034e-04,
          2.50740181e-04,   1.58892909e-03,
         -1.07719599e-03,  -1.12323192e-04,   7.62566128e-06,
         -2.26193130e-04,   9.21699517e-04;
    MatrixN y(10,1);
    y << 2, 3, 0, 2, 4, 3, 1, 0, 1, 4;
    MatrixN margins(10,5);
    margins << 1.00060104,  1.00091344,  0.        ,  1.00136872,  1.00076749,
               1.00147005,  1.00128618,  1.00089538,  0.        ,  1.00265504,
               0.        ,  1.00010148,  0.99819121,  0.99980876,  0.99842502,
               1.0012781 ,  1.00145486,  0.        ,  1.00001485,  1.00048027,
               1.00201221,  1.00221192,  1.00216148,  1.00009487,  0.        ,
               1.00081705,  0.99815932,  1.00071674,  0.        ,  0.99920954,
               1.00063926,  0.        ,  0.99900358,  1.0010482 ,  0.9999877 ,
               0.        ,  0.99993508,  1.00177245,  1.0007706 ,  1.00068629,
               1.00169316,  0.        ,  1.00157803,  1.00116808,  1.00250627,
               0.9980011 ,  0.99896598,  0.99908593,  0.99885211,  0.;
    floatN loss=4.00207888295;
    MatrixN dx(10,5);
    dx << 0.1,  0.1, -0.4,  0.1,  0.1,
          0.1,  0.1,  0.1, -0.4,  0.1,
         -0.4,  0.1,  0.1,  0.1,  0.1,
          0.1,  0.1, -0.4,  0.1,  0.1,
          0.1,  0.1,  0.1,  0.1, -0.4,
          0.1,  0.1,  0.1, -0.4,  0.1,
          0.1, -0.4,  0.1,  0.1,  0.1,
         -0.4,  0.1,  0.1,  0.1,  0.1,
          0.1, -0.4,  0.1,  0.1,  0.1,
          0.1,  0.1,  0.1,  0.1, -0.4;

    Svm sv(R"({"inputShape":[5]})"_json);
    t_cppl cache;
    t_cppl grads;
    t_cppl states;
    states["y"] = &y;
    MatrixN margins0=sv.forward(x, &cache, &states);
    bool ret=matCompT(margins,margins0,"Svm probabilities",eps,verbose);
    if (!ret) allOk=false;
    json j(R"({})"_json);
    Loss *pLoss=lossFactory("SVMMargin", j);
    floatN loss0=pLoss->loss(margins0, y, &states);
    // floatN loss0=sv.loss(&cache, &states);
    floatN d=loss-loss0;
    floatN err=std::abs(d);
    if (err > eps) {
        if (verbose>0) cerr << "  Loss error: correct:" << loss << " got: " << loss0 << ", err=" << err << endl;
        allOk=false;
    } else {
        if (verbose>1) cerr << "  Loss ok, loss=" << loss0 << " (ref: " << loss << "), err=" << err << endl;
    }
    MatrixN dx0=sv.backward(y, &cache, &states, &grads);
    ret=matCompT(dx,dx0,"Softmax dx",eps,verbose);
    if (!ret) allOk=false;
    delete pLoss;
    cppl_delete(&grads);
    cppl_delete(&cache);
    return allOk;
}

bool testSvm(int verbose) {
    Color::Modifier lblue(Color::FG_LIGHT_BLUE);
    Color::Modifier def(Color::FG_DEFAULT);
	bool bOk=true;
	t_cppl s1;
	cerr << lblue << "SVM Layer: " << def << endl;
	// Numerical gradient
    // SVM
	int svN = 10, svC = 5;
	json j2;
	j2["inputShape"]=vector<int>{svC};
	Svm sv(j2);
	t_cppl svmstates;
	MatrixN xsv(svN, svC);
	xsv.setRandom();
	MatrixN yv(svN, 1);
	for (unsigned i = 0; i < yv.rows(); i++)
		yv(i, 0) = (rand() % svC);
	svmstates["y"] = &yv;
	floatN h = 1e-3;
	if (h < CP_DEFAULT_NUM_H)
		h = CP_DEFAULT_NUM_H;
	floatN eps = 1e-6;
	if (eps < CP_DEFAULT_NUM_EPS)
		eps = CP_DEFAULT_NUM_EPS;
    Loss *pLoss=lossFactory("SVMMargin", R"({})");
	bool res=sv.selfTest(xsv, &svmstates, h, eps, verbose, pLoss);
    delete pLoss;
	registerTestResult("SVM", "Numerical gradient", res, "");
	if (!res) bOk = false;

	res=checkSvm(CP_DEFAULT_NUM_EPS, verbose);
	registerTestResult("SVM", "Check (with test-data)", res, "");
	if (!res) bOk = false;

	return bOk;
}

#endif
