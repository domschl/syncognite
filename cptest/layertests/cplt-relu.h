#ifndef _CPLT_RELU_H
#define _CPLT_RELU_H

#include "../testneural.h"

bool checkReluForward(floatN eps=CP_DEFAULT_NUM_EPS) {
    t_cppl states;
    MatrixN x(3,4);
    x << -0.5       , -0.40909091, -0.31818182, -0.22727273,
         -0.13636364, -0.04545455,  0.04545455,  0.13636364,
          0.22727273,  0.31818182,  0.40909091,  0.5;

    MatrixN y(3,4);
    y << 0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.04545455,  0.13636364,
         0.22727273,  0.31818182,  0.40909091,  0.5;

    Relu rl(CpParams("{inputShape=[4]}"));
    MatrixN y0=rl.forward(x, nullptr, &states);
    return matComp(y,y0,"ReluForward",eps);
}

bool checkReluBackward(float eps=CP_DEFAULT_NUM_EPS) {
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
    Relu rl("{inputShape=[4]}");
    t_cppl cache;
    t_cppl grads;
    MatrixN y=rl.forward(x, &cache, &states);
    MatrixN dx0=rl.backward(dchain, &cache, &states, &grads);
    bool allOk=true;
    bool ret=matComp(dx,dx0,"ReluBackward dx",eps);
    if (!ret) allOk=false;
    cppl_delete(&cache);
    cppl_delete(&grads);
    return allOk;
}

#endif
