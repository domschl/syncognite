#ifndef _CPLT_NONLINEARITY_H
#define _CPLT_NONLINEARITY_H

#include "../testneural.h"

bool checkNonlinearityForward(floatN eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    t_cppl states;
    bool allOk=true;
    MatrixN x(3,4);
    x << -0.5       , -0.40909091, -0.31818182, -0.22727273,
         -0.13636364, -0.04545455,  0.04545455,  0.13636364,
          0.22727273,  0.31818182,  0.40909091,  0.5;

    MatrixN y(3,4);
    y << 0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.04545455,  0.13636364,
         0.22727273,  0.31818182,  0.40909091,  0.5;

    Nonlinearity  nlr(R"({"inputShape":[4],"type":"relu"})"_json);
    MatrixN y0=nlr.forward(x, nullptr, &states);
    if (!matCompT(y,y0,"NonlinearityForwardRelu",eps,verbose)) allOk=false;

    x << -0.5,        -0.40909091, -0.31818182, -0.22727273,
         -0.13636364, -0.04545455,  0.04545455,  0.13636364,
          0.22727273,  0.31818182,  0.40909091,  0.5;

    y << 0.37754067,  0.39913012,  0.42111892,  0.44342513,
         0.46596182,  0.48863832,  0.51136168,  0.53403818,
         0.55657487,  0.57888108,  0.60086988,  0.62245933;

    Nonlinearity  nlr2(R"({"inputShape":[4],"type":"sigmoid"})"_json);
    y0=nlr2.forward(x, nullptr, &states);
    if (!matCompT(y,y0,"NonlinearityForwardSigmoid",eps,verbose)) allOk=false;

    x << -0.5       , -0.40909091, -0.31818182, -0.22727273,
         -0.13636364, -0.04545455,  0.04545455,  0.13636364,
          0.22727273,  0.31818182,  0.40909091,  0.5;

    y << -0.46211716, -0.38770051, -0.30786199, -0.22343882,
         -0.13552465, -0.04542327,  0.04542327,  0.13552465,
          0.22343882,  0.30786199,  0.38770051,  0.46211716;

    Nonlinearity  nlr3(R"({"inputShape":[4],"type":"tanh"})"_json);
    y0=nlr3.forward(x, nullptr, &states);
    if (!matCompT(y,y0,"NonlinearityForwardTanh",eps,verbose)) allOk=false;

    return allOk;
}

bool checkNonlinearityBackward(float eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    t_cppl states;
    MatrixN x(4,4);
    x << -0.56204781, -0.30204112,  0.7685022 , -0.74405281,
         -1.46482614, -0.3824993 ,  0.23478267,  0.81716411,
         -0.7702258 ,  0.25433918,  0.33381382,  0.22000994,
          0.43154112, -0.2128072 ,  0.90312084,  1.32935976;
    MatrixN dx(4,4);
    dx << -0.        , -0.        ,  1.23506749,  0.,
          -0.        ,  0.        , -0.68054398,  2.23784401,
          -0.        ,  0.36303492, -0.08854093,  0.63582723,
          -0.07389104, -0.        , -1.18782779, -0.8492151;
    MatrixN dchain(4,4);
    dchain << -0.53996216, -1.18478937,  1.23506749,  0.0695497,
              -1.10965119,  0.24569561, -0.68054398,  2.23784401,
              -0.39696365,  0.36303492, -0.08854093,  0.63582723,
              -0.07389104, -0.38178744, -1.18782779, -0.8492151;
    Nonlinearity nl(R"({"inputShape":[4],"type":"relu"})"_json);
    t_cppl cache;
    t_cppl grads;
    MatrixN y=nl.forward(x, &cache, &states);
    MatrixN dx0=nl.backward(dchain, &cache, &states, &grads);
    bool allOk=true;
    bool ret=matCompT(dx,dx0,"NonlinearityBackward (relu) dx",eps,verbose);
    if (!ret) allOk=false;
    cppl_delete(&cache);
    cppl_delete(&grads);

    x << 1.43767491,  1.80314254,  0.6648782 ,  0.07930361,
         -0.04041539,  2.0385067 ,  0.95919656,  0.78136866,
         -0.60190026, -0.59705516,  0.46964069,  2.49260531,
          1.81210733, -0.75791144, -0.05050857,  2.18823795;

    dx << -0.14967124,  0.10171973, -0.04627804,  0.21278199,
          0.18218001, -0.02431661,  0.09598144,  0.33520963,
          0.06603975, -0.09736881, -0.06869742,  0.02529933,
         -0.24602253,  0.18381851,  0.39258962, -0.14610137;

    dchain << -0.96513598,  0.83750624, -0.20633479,  0.85246687,
         0.72901767, -0.23853026,  0.47921611,  1.55612321,
         0.28881523, -0.42522819, -0.29022231,  0.35862906,
        -2.03870191,  0.84601717,  1.57136025, -1.61173135;

    Nonlinearity nl2(R"({"inputShape":[4],"type":"sigmoid"})"_json);
    y=nl2.forward(x, &cache, &states);
    dx0=nl2.backward(dchain, &cache, &states, &grads);
    ret=matCompT(dx,dx0,"NonlinearityBackward (sigmoid) dx",eps,verbose);
    if (!ret) allOk=false;
    cppl_delete(&cache);
    cppl_delete(&grads);


    x << 0.62237522, -0.67630521, -0.73180118,  0.09480968,
         -0.38519315,  0.54151353,  0.20554376, -0.83341587,
          0.50075282, -0.298898  , -0.61444481, -1.43827731,
         -0.51080252,  1.01150184,  0.33966994, -0.04912771;

    dx << 1.86839943,  0.98946334, -1.34211481, -1.81021704,
         -0.36344808,  0.362766  , -0.13229278,  0.04144855,
          0.93192044,  1.57711458,  0.37771666, -0.45210137,
         -0.87189663, -0.2880611 , -0.33361629, -0.33249507;

    dchain <<  2.69053302,  1.51538108, -2.19868614, -1.82653767,
         -0.42009465,  0.47995538, -0.13796107,  0.0775524,
          1.18579972,  1.72226031,  0.53919526, -2.23895634,
         -1.11987843, -0.69806274, -0.37361077, -0.3332982;

    Nonlinearity nl3(R"({"inputShape":[4],"type":"tanh"})"_json);
    y=nl3.forward(x, &cache, &states);
    dx0=nl3.backward(dchain, &cache, &states, &grads);
    ret=matCompT(dx,dx0,"NonlinearityBackward (tanh) dx",eps,verbose);
    if (!ret) allOk=false;
    cppl_delete(&cache);
    cppl_delete(&grads);

    return allOk;
}

bool testNonlinearity(int verbose) {
     Color::Modifier lblue(Color::FG_LIGHT_BLUE);
     Color::Modifier def(Color::FG_DEFAULT);
	bool bOk=true;
     bool res=false;
	t_cppl s1 {};
     cerr << lblue << "Nonlinearity Layers: " << def << endl;
     vector<string> nls = {"relu", "sigmoid", "tanh", "selu", "resilu"};
     for (string nlsi : nls) {
          json j(R"({"inputShape":[20]})"_json);
          j["type"]=(string)nlsi;
          Nonlinearity nlr(j);
          MatrixN xnl(10, 20);
          xnl.setRandom();
          if (nlsi == "resilu") {
               res=nlr.selfTest(xnl, &s1, 0.001, 0.01, verbose);
          } else {
               res=nlr.selfTest(xnl, &s1, CP_DEFAULT_NUM_H, CP_DEFAULT_NUM_EPS, verbose);
          }
          registerTestResult("Nonlinearity", nlsi+", numerical gradient", res, "");
         	if (!res) bOk = false;
     }

	res=checkNonlinearityForward(CP_DEFAULT_NUM_EPS, verbose);
	registerTestResult("Nonlinearity", "Forward (with test-data)", res, "");
	if (!res) bOk = false;

	res=checkNonlinearityBackward(CP_DEFAULT_NUM_EPS, verbose);
	registerTestResult("Nonlinearity", "Backward (with test-data)", res, "");
	if (!res) bOk = false;
	return bOk;
}

#endif
