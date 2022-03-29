#ifndef _CPLT_TWOLAYERNET_H
#define _CPLT_TWOLAYERNET_H

#include "../testneural.h"

bool checkTwoLayerNet(float eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    bool allOk=true;   // N=3, D=5, H=4, C=2
    int N=3, D=5, H=4, C=2;
    t_cppl states;
    MatrixN x(N,D);
    x << -5.5       , -3.35714286, -1.21428571,  0.92857143,  3.07142857,
         -4.78571429, -2.64285714, -0.5       ,  1.64285714,  3.78571429,
         -4.07142857, -1.92857143,  0.21428571,  2.35714286,  4.5;
    MatrixN yc(N,1);
    yc << 0, 1, 1;
    MatrixN W1(D,H);
    W1 << -0.7       , -0.64736842, -0.59473684, -0.54210526,
          -0.48947368, -0.43684211, -0.38421053, -0.33157895,
          -0.27894737, -0.22631579, -0.17368421, -0.12105263,
          -0.06842105, -0.01578947,  0.03684211,  0.08947368,
           0.14210526,  0.19473684,  0.24736842,  0.3;
    MatrixN b1(1,H);
    b1 << -0.1       ,  0.23333333,  0.56666667,  0.9;
    MatrixN W2(H,C);
    W2 << -0.3, -0.2,
          -0.1,  0.,
           0.1,  0.2,
           0.3,  0.4;
    MatrixN b2(1,C);
    b2 << -0.9,  0.1;

    MatrixN sc(N,C); // Scores
    sc << -0.88621554,  2.56401003,
         -0.69824561,  2.46626566,
         -0.51027569,  2.3685213;

    json j;
    j["inputShape"]=vector<int>{D};
    j["hidden"]=vector<int>{H,C};
    TwoLayerNet tln(j);

    *(tln.params["af1-W"])=W1;
    *(tln.params["af1-b"])=b1;
    *(tln.params["af2-W"])=W2;
    *(tln.params["af2-b"])=b2;

    t_cppl cache;
    t_cppl grads;
    states["y"] = &yc;
    MatrixN sc0=tln.forward(x,&cache,&states);
    bool ret=matCompT(sc,sc0,"TwoLayerNetScores",eps,verbose);
    if (!ret) allOk=false;

    MatrixN dW1(D,H);
    dW1 << -0.16400759, -0.16400759, -0.16400759, -0.16400759,
           -0.10147167, -0.10147167, -0.10147167, -0.10147167,
           -0.03893575, -0.03893575, -0.03893575, -0.03893575,
            0.02360017,  0.02360017,  0.02360017,  0.02360017,
            0.08613609,  0.08613609,  0.08613609,  0.08613609;
    MatrixN db1(1,H);
    db1 << 0.02918343,  0.02918343,  0.02918343,  0.02918343;
    MatrixN dW2(H,C);
    dW2 << -1.83041352,  1.83041352,
           -1.82522911,  1.82522911,
           -1.8200447 ,  1.8200447 ,
           -1.81486029,  1.81486029;
    MatrixN db2(1,C);
    db2 << -0.29183429,  0.29183429;

    // XXX reg parameter
    floatN reg=0.0;
    json jl(R"({})"_json);
    t_cppl lossStates;
    Loss *pLoss=lossFactory("SparseCategoricalCrossEntropy",jl);
    floatN ls = pLoss->loss(sc0, yc, &lossStates);
    floatN lsc = 1.1925059294331903;
    floatN lse=std::abs(ls-lsc);
    if (lse < eps) {
        if (verbose>1) cerr << "  TwoLayerNet: loss-err: " << lse << " for reg=" << reg << " OK." << endl;
    } else {
        if (verbose>0) cerr << "  TwoLayerNet: loss-err: " << lse << " for reg=" << reg << " incorrect: " << ls << ", expected: " << lsc << endl;
        allOk=false;
    }
    MatrixN dx0=tln.backward(yc,&cache,&states, &grads);

    if (verbose>2) {
        cerr << "  Got grads: ";
        for (auto gi : grads) {
            cerr <<  gi.first << " ";
        }
        cerr << endl;
    }
    ret=matCompT(dW1,*(grads["af1-W"]),"TwoLayerNet dW1",eps,verbose);
    if (!ret) allOk=false;
    ret=matCompT(db1,*(grads["af1-b"]),"TwoLayerNet db1",eps,verbose);
    if (!ret) allOk=false;
    ret=matCompT(dW2,*(grads["af2-W"]),"TwoLayerNet dW2",eps,verbose);
    if (!ret) allOk=false;
    ret=matCompT(db2,*(grads["af2-b"]),"TwoLayerNet db2",eps,verbose);
    if (!ret) allOk=false;

    cppl_delete(&cache);
    cppl_delete(&grads);
    cppl_delete(&lossStates);
    delete pLoss;
    return allOk;
}

bool testTwoLayerNet(int verbose) {
    Color::Modifier lblue(Color::FG_LIGHT_BLUE);
    Color::Modifier def(Color::FG_DEFAULT);
	bool bOk=true;
	t_cppl s1;
	cerr << lblue << "TwoLayerNet Layer: " << def << endl;
	// Numerical gradient
    // TwoLayerNet
	int ntl1 = 4, ntl2 = 5, ntl3 = 6, ntlN = 30;
	json j;
	j["inputShape"]=vector<int>{ntl1};
	j["hidden"]=vector<int>{ntl2, ntl3};
	j["init"]=(string) "standard";
	TwoLayerNet tl(j);
	MatrixN xtl(ntlN, ntl1);
	xtl.setRandom();
	MatrixN y2(ntlN, 1);
	for (unsigned i = 0; i < y2.rows(); i++)
		y2(i, 0) = (rand() % ntl3);
	floatN h = 1e-3;
	if (h < CP_DEFAULT_NUM_H)
		h = CP_DEFAULT_NUM_H;
	floatN eps = 1e-5;
	if (eps < CP_DEFAULT_NUM_EPS)
		eps = CP_DEFAULT_NUM_EPS;
	t_cppl tlstates;
	tlstates["y"] = &y2;
    Loss *pLoss=lossFactory("SparseCategoricalCrossEntropy",j);
	bool res=tl.selfTest(xtl, &tlstates, h, eps, verbose, pLoss);
	registerTestResult("TwoLayerNet", "Numerical gradient", res, "");
	if (!res) bOk = false;

	res=checkTwoLayerNet(CP_DEFAULT_NUM_EPS, verbose);
	registerTestResult("TwoLayerNet", "Check (with test-data)", res, "");
	if (!res) bOk = false;

	return bOk;
}
#endif
