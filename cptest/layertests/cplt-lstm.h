#ifndef _CPLT_LSTM_H
#define _CPLT_LSTM_H

#include "../testneural.h"

bool checkLSTMStepForward(floatN eps=CP_DEFAULT_NUM_EPS) {
    MatrixN x(3,4);    // N, D  // N, D, H = 3, 4, 5
    x << -0.4       , -0.25454545, -0.10909091,  0.03636364,
         0.18181818,  0.32727273,  0.47272727,  0.61818182,
         0.76363636,  0.90909091,  1.05454545,  1.2;
    MatrixN Wxh(4,4*5); // D, 4*H
    MatrixN Whh(5,4*5); // H, 4*H
    Wxh << -2.1       , -2.05696203, -2.01392405, -1.97088608, -1.9278481 ,
        -1.88481013, -1.84177215, -1.79873418, -1.7556962 , -1.71265823,
        -1.66962025, -1.62658228, -1.5835443 , -1.54050633, -1.49746835,
        -1.45443038, -1.41139241, -1.36835443, -1.32531646, -1.28227848,
        -1.23924051, -1.19620253, -1.15316456, -1.11012658, -1.06708861,
        -1.02405063, -0.98101266, -0.93797468, -0.89493671, -0.85189873,
        -0.80886076, -0.76582278, -0.72278481, -0.67974684, -0.63670886,
        -0.59367089, -0.55063291, -0.50759494, -0.46455696, -0.42151899,
        -0.37848101, -0.33544304, -0.29240506, -0.24936709, -0.20632911,
        -0.16329114, -0.12025316, -0.07721519, -0.03417722,  0.00886076,
         0.05189873,  0.09493671,  0.13797468,  0.18101266,  0.22405063,
         0.26708861,  0.31012658,  0.35316456,  0.39620253,  0.43924051,
         0.48227848,  0.52531646,  0.56835443,  0.61139241,  0.65443038,
         0.69746835,  0.74050633,  0.7835443 ,  0.82658228,  0.86962025,
         0.91265823,  0.9556962 ,  0.99873418,  1.04177215,  1.08481013,
         1.1278481 ,  1.17088608,  1.21392405,  1.25696203,  1.3;
    Whh << -0.7       , -0.67070707, -0.64141414, -0.61212121, -0.58282828,
        -0.55353535, -0.52424242, -0.49494949, -0.46565657, -0.43636364,
        -0.40707071, -0.37777778, -0.34848485, -0.31919192, -0.28989899,
        -0.26060606, -0.23131313, -0.2020202 , -0.17272727, -0.14343434,
        -0.11414141, -0.08484848, -0.05555556, -0.02626263,  0.0030303 ,
         0.03232323,  0.06161616,  0.09090909,  0.12020202,  0.14949495,
         0.17878788,  0.20808081,  0.23737374,  0.26666667,  0.2959596 ,
         0.32525253,  0.35454545,  0.38383838,  0.41313131,  0.44242424,
         0.47171717,  0.5010101 ,  0.53030303,  0.55959596,  0.58888889,
         0.61818182,  0.64747475,  0.67676768,  0.70606061,  0.73535354,
         0.76464646,  0.79393939,  0.82323232,  0.85252525,  0.88181818,
         0.91111111,  0.94040404,  0.96969697,  0.9989899 ,  1.02828283,
         1.05757576,  1.08686869,  1.11616162,  1.14545455,  1.17474747,
         1.2040404 ,  1.23333333,  1.26262626,  1.29191919,  1.32121212,
         1.35050505,  1.37979798,  1.40909091,  1.43838384,  1.46767677,
         1.4969697 ,  1.52626263,  1.55555556,  1.58484848,  1.61414141,
         1.64343434,  1.67272727,  1.7020202 ,  1.73131313,  1.76060606,
         1.78989899,  1.81919192,  1.84848485,  1.87777778,  1.90707071,
         1.93636364,  1.96565657,  1.99494949,  2.02424242,  2.05353535,
         2.08282828,  2.11212121,  2.14141414,  2.17070707,  2.2;
    MatrixN bh(1,4*5); // 4*H
    bh << 0.3       ,  0.32105263,  0.34210526,  0.36315789,  0.38421053,
        0.40526316,  0.42631579,  0.44736842,  0.46842105,  0.48947368,
        0.51052632,  0.53157895,  0.55263158,  0.57368421,  0.59473684,
        0.61578947,  0.63684211,  0.65789474,  0.67894737,  0.7;
    MatrixN h(3,5);  // N, H (prev_h)
    h << -0.3       , -0.22857143, -0.15714286, -0.08571429, -0.01428571,
         0.05714286,  0.12857143,  0.2       ,  0.27142857,  0.34285714,
         0.41428571,  0.48571429,  0.55714286,  0.62857143,  0.7;
    MatrixN hn(3,5); // N, H (next_h)
    hn << 0.24635157,  0.28610883,  0.32240467,  0.35525807,  0.38474904,
         0.49223563,  0.55611431,  0.61507696,  0.66844003,  0.7159181 ,
         0.56735664,  0.66310127,  0.74419266,  0.80889665,  0.858299;
    MatrixN c(3,5);  // N, H (prev_c)
    c << -0.4       , -0.30714286, -0.21428571, -0.12142857, -0.02857143,
         0.06428571,  0.15714286,  0.25      ,  0.34285714,  0.43571429,
         0.52857143,  0.62142857,  0.71428571,  0.80714286,  0.9;
    MatrixN cn(3,5); // N, H (next_c)
    cn << 0.32986176,  0.39145139,  0.451556  ,  0.51014116,  0.56717407,
         0.66382255,  0.76674007,  0.87195994,  0.97902709,  1.08751345,
         0.74192008,  0.90592151,  1.07717006,  1.25120233,  1.42395676;

    LSTM lstm("{name='testlstm';inputShape=[4,1];H=5;N=3}");
    *(lstm.params["Wxh"])= Wxh;
    *(lstm.params["Whh"])= Whh;
    *(lstm.params["bh"])=bh;
    t_cppl cache;
    t_cppl states;
    cppl_set(&states,"testlstm-h",new MatrixN(h));
    cppl_set(&states,"testlstm-c",new MatrixN(c));
    t_cppl cp=lstm.forward_step(x, &cache, &states, 0);
    cppl_delete(&cache);
    cppl_delete(&states);
    bool allOk=true;
    if (!matComp(hn,*(cp["testlstm-h0"]),"LSTMForwardStep",eps)) {
        allOk = false;
    }
    if (!matComp(cn,*(cp["testlstm-c0"]),"LSTMForwardStep",eps)) {
        allOk = false;
    }
    cppl_delete(&cp);
    return allOk;
}
/*
bool checkLSTMStepBackward(float eps=CP_DEFAULT_NUM_EPS) {
    MatrixN x(4,5);     // N, D, H = 4, 5, 6
    x << ;
    MatrixN Wxh(5,6);
    Wxh << ;
    MatrixN Whh(6,6);
    Whh << ;
    MatrixN bh(1,6);
    bh <<  ;
    MatrixN h0(4,6);
    h0 << ;

    MatrixN dx(4,5);
    dx << ;
    MatrixN dWxh(5,6);
    dWxh << ;
    MatrixN dWhh(6,6);
    dWhh << ;
    MatrixN dbh(1,6);
    dbh << ;
    MatrixN dh0(4,6);
    dh0 << ;

    MatrixN dchain(4,6);
    dchain << ;
    LSTM lstm("{name='test2lstm';inputShape=[5,1;H=6;N=4}");
    *(lstm.params["Wxh")=Wxh;
    *(lstm.params["Whh")=Whh;
    *(lstm.params["bh")=bh;
    t_cppl cache;
    t_cppl grads;
    t_cppl states;
    cppl_set(&states,"test2lstm-h",new MatrixN(h0));
    MatrixN y=lstm.forward_step(x, &cache, &states);
    MatrixN dx0=lstm.backward_step(dchain, &cache, &states, &grads);
    bool allOk=true;
    bool ret=matComp(dx,dx0,"LSTMStepBackward dx",eps);
    if (!ret) allOk=false;
    ret=matComp(dWxh,*(grads["Wxh"),"LSTMStepBackward dWxh",eps);
    if (!ret) allOk=false;
    ret=matComp(dWhh,*(grads["Whh"),"LSTMStepBackward dWhh",eps);
    if (!ret) allOk=false;
    ret=matComp(dbh,*(grads["bh"),"LSTMStepBackward bh",eps);
    if (!ret) allOk=false;
    ret=matComp(dh0,*(grads["test2lstm-h0"),"LSTMStepBackward h0",eps);
    if (!ret) allOk=false;
    cppl_delete(&cache);
    cppl_delete(&grads);
    cppl_delete(&states);
    return allOk;
}

bool checkLSTMForward(floatN eps=CP_DEFAULT_NUM_EPS) {
    MatrixN x(2,12);   // N, T, D, H = 2, 3, 4, 5
    x << ;
    MatrixN Wxh(4,5);
    MatrixN Whh(5,5);
    Wxh << ;
    Whh << ;
    MatrixN bh(1,5);
    bh << -0.7, -0.5, -0.3, -0.1,  0.1;
    MatrixN h0(2,5);
    h0 << ;
    MatrixN hn(2,15);
    hn << ;

//                        D,T
    LSTM lstm("{name='lstm3';inputShape=[4,3;H=5;N=2}");
    *(lstm.params["Wxh")= Wxh;
    *(lstm.params["Whh")= Whh;
    *(lstm.params["bh")=bh;
    //*(lstm.params)["ho"=h0;
    t_cppl cache;
    t_cppl states;
    states["lstm3-h" = new MatrixN(h0);
    MatrixN hn0=lstm.forward(x, &cache, &states, 0);
    cppl_delete(&cache);
    cppl_delete(&states);
    return matComp(hn,hn0,"LSTMForward",eps);
}

bool checkLSTMBackward(float eps=CP_DEFAULT_NUM_EPS) {
    MatrixN x(2,30);   // N, D, T, H = 2, 3, 10, 5
    x << ;
    MatrixN Wxh(3,5);
    Wxh << ;
    MatrixN Whh(5,5);
    Whh << ;
    MatrixN bh(1,5);
    bh <<  ;
    MatrixN h0(2,5);
    h0 << ;

    MatrixN dx(2,30);
    dx << ;
    MatrixN dWxh(3,5);
    dWxh << ;
    MatrixN dWhh(5,5);
    dWhh << ;
    MatrixN dbh(1,5);
    dbh << ;
    MatrixN dh0(2,5);
    dh0 << ;

    MatrixN dchain(2,50);
    dchain << ;
//                        D,T
    LSTM lstm("{name='lstm4';inputShape=[3,10;H=5;N=2}");   //inputShape=D, hidden=H
    *(lstm.params["Wxh")=Wxh;
    *(lstm.params["Whh")=Whh;
    *(lstm.params["bh")=bh;
    //*(lstm.params["ho")=h0;

    t_cppl cache;
    t_cppl states;
    t_cppl grads;
    states["lstm4-h" = new MatrixN(h0);
    MatrixN y=lstm.forward(x, &cache, &states);
    cppl_update(&states, "lstm4-h", &h0);
    MatrixN dx0=lstm.backward(dchain, &cache, &states, &grads);
    bool allOk=true;
    bool ret=matComp(dx,dx0,"LSTMBackward dx",eps);
    if (!ret) allOk=false;
    ret=matComp(dWxh,*(grads["Wxh"),"LSTMBackward dWxh",eps);
    if (!ret) allOk=false;
    ret=matComp(dWhh,*(grads["Whh"),"LSTMBackward dWhh",eps);
    if (!ret) allOk=false;
    ret=matComp(dbh,*(grads["bh"),"LSTMBackward bh",eps);
    if (!ret) allOk=false;
    ret=matComp(dh0,*(grads["lstm4-h0"),"LSTMBackward h0",eps); // XXX: Uhhh!
    if (!ret) allOk=false;

    cppl_delete(&cache);
    cppl_delete(&grads);
    cppl_delete(&states);
    return allOk;
}
*/
#endif
