#ifndef _CPLT_AFFINERELU_H
#define _CPLT_AFFINERELU_H

#include "../testneural.h"

bool checkAffineRelu(float eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    bool allOk=true;
    t_cppl states;
    MatrixN x(2,4);
    x << -2.44826954,  0.81707546,  1.31506197, -0.0965869,
         -1.58810595,  0.61785734, -0.44616526, -0.82397868;
    MatrixN W(4,5);
    W << 0.86699529, -1.01282323, -0.38693827, -0.74054919, -1.1270489,
        -2.27456327,  0.190157  , -1.26097006, -0.33208802,  0.16781256,
         0.08560445, -0.24551482,  0.30694568, -1.61658197, -3.02608437,
         0.18890925,  1.7598865 , -0.14769698, -0.59141176, -0.85895842;
    MatrixN b(1,5);
    b << 1.81489966,  1.27103839,  1.58359929, -0.8527733 ,  1.24037006;
    MatrixN y(2,5);
    y << 0.        ,  3.41322609,  1.91853897,  0.        ,  0.24028072,
         0.        ,  1.65643012,  1.40374932,  1.32668765,  5.19182449;

    AffineRelu arl(R"({"inputShape":[4],"hidden":5})"_json);
    t_cppl cache;
    t_cppl grads;
    *(arl.params["af-W"])=W;
    *(arl.params["af-b"])=b;
    MatrixN y0=arl.forward(x, &cache, &states);
    bool ret=matCompT(y,y0,"AffineRelu",eps,verbose);
    if (!ret) allOk=false;

    MatrixN dx(2,4);
    dx << 2.29906266, -1.48504925,  6.47154972,  1.00439731,
          1.8610674 , -0.74000018, -0.6322688 , -4.68601953;
    MatrixN dW(4,5);
    dW << 0.        ,  4.7081366 , -2.63137168,  0.27607488,  4.10764656,
          0.        , -1.78497082,  0.90776884, -0.10740775, -1.32401681,
          0.        ,  0.6314218 ,  0.97587313,  0.07756096, -2.89927552,
          0.        ,  2.03769315, -0.36020745,  0.1432397 , -0.24398387;
    MatrixN db(1,5);
    db << 0.        , -2.77768192,  1.19310997, -0.17383908, -1.49039921;
    MatrixN dchain(2,5);
    dchain << -1.08201385, -0.34514762,  0.8563332 ,  0.7021515 , -2.02372516,
              -0.26158065, -2.43253431,  0.33677677, -0.17383908,  0.53332595;

    MatrixN dx0=arl.backward(dchain, &cache, &states, &grads);

    ret=matCompT(dx,dx0,"AffineRelu dx",eps,verbose);
    if (!ret) allOk=false;
    ret=matCompT(dW,*(grads["af-W"]),"AffineRelu dW",eps,verbose);
    if (!ret) allOk=false;
    ret=matCompT(db,*(grads["af-b"]),"AffineRelu db",eps,verbose);
    if (!ret) allOk=false;

    cppl_delete(&cache);
    cppl_delete(&grads);

    return allOk;
}

bool testAffineRelu(int verbose) {
    Color::Modifier lblue(Color::FG_LIGHT_BLUE);
    Color::Modifier def(Color::FG_DEFAULT);
	bool bOk=true;
	t_cppl s1;
	cerr << lblue << "AffineRelu Layer: " << def << endl;
	// Numerical gradient
    AffineRelu rx(R"({"inputShape":[2],"hidden":3})"_json);
	MatrixN xarl(30, 2);
	xarl.setRandom();
	floatN h = 1e-6;
	if (h < CP_DEFAULT_NUM_H)
		h = CP_DEFAULT_NUM_H;
	floatN eps = 1e-6;
	if (eps < CP_DEFAULT_NUM_EPS)
		eps = CP_DEFAULT_NUM_EPS;
	bool res=rx.selfTest(xarl, &s1, h, eps, verbose);
	registerTestResult("AffineRelu", "Numerical gradient", res, "");
	if (!res) bOk = false;

	res=checkAffineRelu(CP_DEFAULT_NUM_EPS, verbose);
	registerTestResult("AffineRelu", "Forward/Backward (with test-data)", res, "");
	if (!res) bOk = false;

	return bOk;
}
#endif
