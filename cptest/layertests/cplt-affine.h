#ifndef _CPLT_AFFINE_H
#define _CPLT_AFFINE_H

#include "../testneural.h"

bool checkAffineForward(floatN eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    t_cppl states;
    MatrixN x(2,4);
    x << -0.1       , -0.01428571,  0.07142857,  0.15714286,
          0.24285714,  0.32857143,  0.41428571,  0.5;
    MatrixN W(4,3);
    W << -0.2       , -0.15454545, -0.10909091,
         -0.06363636, -0.01818182,  0.02727273,
          0.07272727,  0.11818182,  0.16363636,
          0.20909091,  0.25454545,  0.3;
    MatrixN b(1,3);
    b << -0.3, -0.1,  0.1;
    MatrixN y(2,3);
    y << -0.24103896, -0.03584416,  0.16935065,
         -0.23480519,  0.03272727,  0.30025974;

    Affine pe( R"({"inputShape":[4],"hidden":3})"_json );
    *(pe.params["W"])= W;
    *(pe.params["b"])=b;
    MatrixN y0=pe.forward(x, nullptr, &states);
    return matCompT(y,y0,"AffineForward",eps,verbose);
}

bool checkAffineBackward(float eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    t_cppl states;
    MatrixN x(2,4);
    x << 1.31745392, -0.61371249,  0.45447287, -0.27054087,
         0.10106874,  1.00650622,  0.47243961, -0.42940807;
    MatrixN W(4,5);
    W << -0.33297518, -0.34410449, -0.84123035, -0.04845468, -2.35649863,
          0.50012296,  0.11834242, -0.95766758,  1.03839053,  0.88182165,
         -0.08384473,  0.74101315, -0.6128059 , -0.10586676, -0.70638727,
         -0.69378517,  0.23008973, -0.16988779, -1.66077535,  0.10843451;
    MatrixN b(1,5);
    b <<  1.42152833, -0.50754731,  0.09331398,  0.83707801,  1.39097462;
    MatrixN dx(2,4);
    dx << -2.59324406, -2.10880392, -3.29279846, -2.19694245,
          -3.81171235,  4.25370933, -1.5117824 , -3.21306015;
    MatrixN dW(4,5);
    dW << 1.1393111 , -2.22498409,  3.93887327,  0.77438161,  0.35517399,
         -0.14120118,  0.51706313, -2.45941632,  1.63133349,  1.74156205,
          0.55479794, -0.98325637,  1.09937315,  1.09447527,  0.91454051,
         -0.38504418,  0.65836473, -0.56660463, -0.93167997, -0.8126062;
    MatrixN db(1,5);
    db << 1.20613441, -2.1440201 ,  2.44244227,  2.33348303,  1.9407547;
    MatrixN dchain(2,5);
    dchain << 0.83641977, -1.65103186,  3.03523817,  0.44273757,  0.13073521,
              0.36971463, -0.49298824, -0.5927959 ,  1.89074546,  1.81001949;
    Affine pe(R"({"inputShape":[4],"hidden":5})"_json);
    *(pe.params["W"])=W;
    *(pe.params["b"])=b;
    t_cppl cache;
    t_cppl grads;
    MatrixN y=pe.forward(x, &cache, &states);
    MatrixN dx0=pe.backward(dchain, &cache, &states, &grads);
    bool allOk=true;
    bool ret=matCompT(dx,dx0,"AffineBackward dx",eps, verbose);
    if (!ret) allOk=false;
    ret=matCompT(dW,*(grads["W"]),"AffineBackward dW",eps, verbose);
    if (!ret) allOk=false;
    ret=matCompT(db,*(grads["b"]),"AffineBackward bx",eps, verbose);
    if (!ret) allOk=false;
    cppl_delete(&cache);
    cppl_delete(&grads);
    return allOk;
}

bool testAffine(int verbose) {
    Color::Modifier lblue(Color::FG_LIGHT_BLUE);
    Color::Modifier def(Color::FG_DEFAULT);
	bool bOk=true;
	t_cppl s1;
	cerr << lblue << "Affine Layer: " << def << endl;
	// Numerical gradient
	Affine pc(R"({"inputShape":[30],"hidden":20})"_json);
	MatrixN x(10, 30);
	x.setRandom();
	bool res = pc.selfTest(x, &s1, CP_DEFAULT_NUM_H, CP_DEFAULT_NUM_EPS, verbose);
	registerTestResult("Affine", "Numerical gradient", res, "");
	if (!res) bOk = false;

	res=checkAffineForward(CP_DEFAULT_NUM_EPS, verbose);
	registerTestResult("Affine", "Forward (with test-data)", res, "");
	if (!res) bOk = false;

	res=checkAffineBackward(CP_DEFAULT_NUM_EPS, verbose);
	registerTestResult("Affine", "Backward (with test-data)", res, "");
	if (!res) bOk = false;
	return bOk;
}

#endif
