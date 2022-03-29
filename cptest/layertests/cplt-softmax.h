#ifndef _CPLT_SOFTMAX_H
#define _CPLT_SOFTMAX_H

#include "../testneural.h"

bool checkSoftmax(float eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    bool allOk=true;
    MatrixN x(10,5);
    x << -5.53887846e-04,  -1.66357895e-05,   9.78587865e-04, -1.32038284e-03,   4.77159634e-04,
          9.20053947e-04,  -7.19330590e-05,  -5.98042560e-04, -2.20287950e-03,  -1.90102146e-03,
         -2.50299417e-04,   4.90923653e-04,  -1.20710791e-04, -1.59583803e-03,   7.76216493e-04,
          5.69312741e-04,   1.04067712e-03,   7.80831584e-04, 1.59436445e-04,  -4.02010213e-04,
         -1.92711171e-04,   4.38969012e-04,   3.51890037e-04, 8.72617659e-04,  -2.67204717e-04,
         -1.93907739e-04,   5.80622659e-05,  -1.35256160e-03, 6.45579573e-04,   3.07149694e-04,
          7.88018217e-05,  -5.08851258e-04,  -5.68082221e-04, -9.08716816e-04,  -4.28502983e-04,
         -7.81674859e-05,   2.58156281e-04,   9.68529476e-04, 7.02486610e-04,   1.02575914e-03,
          7.97130342e-04,  -7.56924427e-04,  -5.05724689e-05, -4.83491308e-04,   5.32065794e-04,
         -4.82766795e-04,   5.50968630e-05,   5.90486482e-04, 4.08029314e-04,   2.16114208e-04;
    MatrixN y(10,1);
    y << 2, 2, 1, 4, 2, 3, 2, 3, 4, 1;
    MatrixN probs(10,5);
    probs << 0.19990659,  0.20001402,  0.20021317,  0.19975342,  0.20011281,
             0.20033832,  0.20013968,  0.20003441,  0.19971365,  0.19977394,
             0.19997786,  0.20012615,  0.20000378,  0.19970897,  0.20018325,
             0.20002791,  0.20012222,  0.20007022,  0.19994594,  0.19983371,
             0.19991332,  0.20003964,  0.20002222,  0.2001264 ,  0.19989842,
             0.1999826 ,  0.200033  ,  0.19975102,  0.20015055,  0.20008283,
             0.20010919,  0.19999163,  0.19997979,  0.19991168,  0.2000077 ,
             0.19986932,  0.19993655,  0.20007863,  0.20002541,  0.20009008,
             0.20015793,  0.19984711,  0.19998832,  0.19990176,  0.20010488,
             0.199872  ,  0.19997953,  0.20008662,  0.20005012,  0.20001173;
    floatN loss=1.60920315915;
    MatrixN dx(10,5);
    dx << 0.01999066,  0.0200014 , -0.07997868,  0.01997534,  0.02001128,
          0.02003383,  0.02001397, -0.07999656,  0.01997136,  0.01997739,
          0.01999779, -0.07998739,  0.02000038,  0.0199709 ,  0.02001832,
          0.02000279,  0.02001222,  0.02000702,  0.01999459, -0.08001663,
          0.01999133,  0.02000396, -0.07999778,  0.02001264,  0.01998984,
          0.01999826,  0.0200033 ,  0.0199751 , -0.07998494,  0.02000828,
          0.02001092,  0.01999916, -0.08000202,  0.01999117,  0.02000077,
          0.01998693,  0.01999366,  0.02000786, -0.07999746,  0.02000901,
          0.02001579,  0.01998471,  0.01999883,  0.01999018, -0.07998951,
          0.0199872 , -0.08000205,  0.02000866,  0.02000501,  0.02000117;

    Softmax sm(R"({"inputShape":[5]})"_json);
    t_cppl cache;
    t_cppl grads;
    t_cppl states;
    states["y"] = &y;
    MatrixN probs0=sm.forward(x, &cache, &states);
    bool ret=matCompT(probs,probs0,"Softmax probabilities",eps,verbose);
    if (!ret) allOk=false;
    json j(R"({})"_json);
    t_cppl lossStates;
    Loss *pLoss = lossFactory("SparseCategoricalCrossEntropy", j);
    floatN loss0=pLoss->loss(probs0, y, &lossStates);
    // floatN loss0=sm.loss(&cache, &states);
    floatN d=loss-loss0;
    floatN err=std::abs(d);
    if (err > eps) {
        if (verbose>0) cerr << "  Loss error: correct:" << loss << " got: " << loss0 << ", err=" << err << endl;
        allOk=false;
    } else {
        if (verbose>1) cerr << "  Loss ok, loss=" << loss0 << " (ref: " << loss << "), err=" << err << endl;
    }
    //MatrixN dchain=x;
    //dchain.setOnes();
    MatrixN dx0=sm.backward(y, &cache, &states, &grads);
    ret=matCompT(dx,dx0,"Softmax dx",eps,verbose);
    if (!ret) allOk=false;
    cppl_delete(&grads);
    cppl_delete(&cache);
    cppl_delete(&lossStates);
    delete pLoss;
    return allOk;
}

bool testSoftmax(int verbose) {
    Color::Modifier lblue(Color::FG_LIGHT_BLUE);
    Color::Modifier def(Color::FG_DEFAULT);
	bool bOk=true;
	t_cppl s1;
	cerr << lblue << "Softmax Layer: " << def << endl;
	// Numerical gradient
    // Softmax
	int smN = 10, smC = 4;
	json j1;
	j1["inputShape"]=vector<int>{smC};
	Softmax mx(j1);
	t_cppl smstates;
	MatrixN xmx(smN, smC);
	xmx.setRandom();
	MatrixN y(smN, 1);
	for (unsigned i = 0; i < y.rows(); i++)
		y(i, 0) = (rand() % smC);
	smstates["y"] = &y;
	floatN h = 1e-3;
	if (h < CP_DEFAULT_NUM_H)
		h = CP_DEFAULT_NUM_H;
	floatN eps = 1e-6;
	if (eps < CP_DEFAULT_NUM_EPS)
		eps = CP_DEFAULT_NUM_EPS;
    Loss *pLoss=lossFactory("SparseCategoricalCrossEntropy", R"({"inputShape":[5]})"_json);
	bool res=mx.selfTest(xmx, &smstates, h, eps, verbose, pLoss);
	registerTestResult("Softmax", "Numerical gradient", res, "");
	if (!res) bOk = false;

	res=checkSoftmax(CP_DEFAULT_NUM_EPS, verbose);
	registerTestResult("Softmax", "Check (with test-data)", res, "");
	if (!res) bOk = false;

	return bOk;
}

#endif
