#include "cp-neural.h"

using std::cout; using std::endl;

bool matComp(MatrixN& m0, MatrixN& m1, string msg="", floatN eps=1.e-6) {
    if (m0.cols() != m1.cols() || m0.rows() != m1.rows()) {
        cout << msg << ": Incompatible shapes " << shape(m0) << "!=" << shape(m1) << endl;
        return false;
    }
    MatrixN d = m0 - m1;
    floatN dif = d.cwiseProduct(d).sum();
    if (dif < eps) {
        cout << msg << " err=" << dif << endl;
        return true;
    } else {
        IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        cout << msg << " m0:" << endl << m0.format(CleanFmt) << endl;
        cout << msg << " m1:" << endl << m1.format(CleanFmt) << endl;
        cout << "err=" << dif << endl;
        return false;
    }
}

bool checkAffineForward(floatN eps=1.e-6) {
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

    Affine pe({4,3});
    *(pe.params["W"])= W;
    *(pe.params["b"])=b;
    MatrixN y0=pe.forward(x, nullptr);
    return matComp(y,y0,"AffineForward",eps);
}

bool checkAffineBackward(float eps=1.0e-6) {
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
    Affine pe({4,5});
    *(pe.params["W"])=W;
    *(pe.params["b"])=b;
    t_cppl cache;
    t_cppl grads;
    MatrixN y=pe.forward(x, &cache);
    MatrixN dx0=pe.backward(dchain, &cache, &grads);
    bool allOk=true;
    bool ret=matComp(dx,dx0,"AffineBackward dx",eps);
    if (!ret) allOk=false;
    ret=matComp(dW,*(grads["W"]),"AffineBackward dW",eps);
    if (!ret) allOk=false;
    ret=matComp(db,*(grads["b"]),"AffineBackward bx",eps);
    if (!ret) allOk=false;
    cppl_delete(&cache);
    cppl_delete(&grads);
    return allOk;
}

bool checkReluForward(floatN eps=1.e-6) {
    MatrixN x(3,4);
    x << -0.5       , -0.40909091, -0.31818182, -0.22727273,
         -0.13636364, -0.04545455,  0.04545455,  0.13636364,
          0.22727273,  0.31818182,  0.40909091,  0.5;

    MatrixN y(3,4);
    y << 0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.04545455,  0.13636364,
         0.22727273,  0.31818182,  0.40909091,  0.5;

    Relu rl({4});
    MatrixN y0=rl.forward(x, nullptr);
    return matComp(y,y0,"ReluForward",eps);
}

bool checkReluBackward(float eps=1.0e-6) {
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
    Relu rl({4});
    t_cppl cache;
    t_cppl grads;
    MatrixN y=rl.forward(x, &cache);
    MatrixN dx0=rl.backward(dchain, &cache, &grads);
    bool allOk=true;
    bool ret=matComp(dx,dx0,"ReluBackward dx",eps);
    if (!ret) allOk=false;
    cppl_delete(&cache);
    cppl_delete(&grads);
    return allOk;
}

bool checkAffineRelu(float eps=1.0e-6) {
    bool allOk=true;
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

    AffineRelu arl({4,5});
    t_cppl cache;
    t_cppl grads;
    *(arl.params["af-W"])=W;
    *(arl.params["af-b"])=b;
    MatrixN y0=arl.forward(x, &cache);
    bool ret=matComp(y,y0,"AffineRelu",eps);
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

    MatrixN dx0=arl.backward(dchain, &cache, &grads);

    ret=matComp(dx,dx0,"AffineRelu dx",eps);
    if (!ret) allOk=false;
    ret=matComp(dW,*(grads["af-W"]),"AffineRelu dW",eps);
    if (!ret) allOk=false;
    ret=matComp(db,*(grads["af-b"]),"AffineRelu db",eps);
    if (!ret) allOk=false;

    cppl_delete(&cache);
    cppl_delete(&grads);
    return allOk;
}

bool checkSoftmax(float eps=1.0e-6) {
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

    Softmax sm({5});
    t_cppl cache;
    t_cppl grads;
    MatrixN probs0=sm.forward(x, y, &cache);
    bool ret=matComp(probs,probs0,"Softmax probabilities",eps);
    if (!ret) allOk=false;
    floatN loss0=sm.loss(y, &cache);
    floatN d=loss-loss0;
    floatN err=abs(d);
    if (err > eps) {
        cout << "Loss error: correct:" << loss << " got: " << loss0 << ", err=" << err << endl;
        allOk=false;
    } else {
        cout << "Loss ok, loss=" << loss0 << " (ref: " << loss << "), err=" << err << endl;
    }
    //MatrixN dchain=x;
    //dchain.setOnes();
    MatrixN dx0=sm.backward(y, &cache, &grads);
    ret=matComp(dx,dx0,"Softmax dx",eps);
    if (!ret) allOk=false;
    cppl_delete(&grads);
    cppl_delete(&cache);
    return allOk;
}


bool checkSvm(float eps=1.0e-6) {
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

    Svm sv({5});
    t_cppl cache;
    t_cppl grads;
    MatrixN margins0=sv.forward(x, y, &cache);
    bool ret=matComp(margins,margins0,"Svm probabilities",eps);
    if (!ret) allOk=false;
    floatN loss0=sv.loss(y, &cache);
    floatN d=loss-loss0;
    floatN err=abs(d);
    if (err > eps) {
        cout << "Loss error: correct:" << loss << " got: " << loss0 << ", err=" << err << endl;
        allOk=false;
    } else {
        cout << "Loss ok, loss=" << loss0 << " (ref: " << loss << "), err=" << err << endl;
    }
    MatrixN dx0=sv.backward(y, &cache, &grads);
    ret=matComp(dx,dx0,"Softmax dx",eps);
    if (!ret) allOk=false;
    cppl_delete(&grads);
    cppl_delete(&cache);
    return allOk;
}

bool checkTwoLayer(float eps=1.0e-6) {
    bool allOk=true;   // N=3, D=5, H=4, C=2
    MatrixN x(3,5);
    x << -5.5       , -3.35714286, -1.21428571,  0.92857143,  3.07142857,
         -4.78571429, -2.64285714, -0.5       ,  1.64285714,  3.78571429,
         -4.07142857, -1.92857143,  0.21428571,  2.35714286,  4.5;
    MatrixN yc(3,1);
    yc << 0, 1, 1;
    MatrixN W1(5,4);
    W1 << -0.7       , -0.64736842, -0.59473684, -0.54210526,
          -0.48947368, -0.43684211, -0.38421053, -0.33157895,
          -0.27894737, -0.22631579, -0.17368421, -0.12105263,
          -0.06842105, -0.01578947,  0.03684211,  0.08947368,
           0.14210526,  0.19473684,  0.24736842,  0.3;
    MatrixN b1(1,4);
    b1 << -0.1       ,  0.23333333,  0.56666667,  0.9;
    MatrixN W2(4,2);
    W2 << -0.3, -0.2,
          -0.1,  0.,
           0.1,  0.2,
           0.3,  0.4;
    MatrixN b2(1,2);
    b2 << -0.9,  0.1;

    MatrixN sc(3,2); // Scores
    sc << -0.88621554,  2.56401003,
         -0.69824561,  2.46626566,
         -0.51027569,  2.3685213;

    TwoLayerNet tln({5,4,2});
    cout << "tln-w1" << *(tln.params["af1-W"]) << endl;
    *(tln.params["af1-W"])=W1;
    cout << "tln-w2" << *(tln.params["af1-W"]) << endl;
    *(tln.params["af1-b"])=b1;
    *(tln.params["af2-W"])=W2;
    *(tln.params["af2-b"])=b2;

    t_cppl cache;
    t_cppl grads;
    MatrixN sc0=tln.forward(x,yc,&cache);
    bool ret=matComp(sc,sc0,"TwoLayerNetScores",eps);
    if (!ret) allOk=false;

    MatrixN dW1(5,4);
    dW1 << -0.16400759, -0.16400759, -0.16400759, -0.16400759,
           -0.10147167, -0.10147167, -0.10147167, -0.10147167,
           -0.03893575, -0.03893575, -0.03893575, -0.03893575,
            0.02360017,  0.02360017,  0.02360017,  0.02360017,
            0.08613609,  0.08613609,  0.08613609,  0.08613609;
    MatrixN db1(1,4);
    db1 << 0.02918343,  0.02918343,  0.02918343,  0.02918343;
    MatrixN dW2(4,2);
    dW2 << -1.83041352,  1.83041352,
           -1.82522911,  1.82522911,
           -1.8200447 ,  1.8200447 ,
           -1.81486029,  1.81486029;
    MatrixN db2(1,2);
    db2 << -0.29183429,  0.29183429;

    // XXX reg parameter
    floatN reg=0.0;
    floatN ls = tln.loss(yc,&cache);
    floatN lsc = 1.1925059294331903;
    floatN lse=abs(ls-lsc);
    if (lse < eps) {
        cout << "TwoLayerNet: loss-err: " << lse << " for reg=" << reg << " OK." << endl;
    } else {
        cout << "TwoLayerNet: loss-err: " << lse << " for reg=" << reg << " incorrect: " << ls << ", expected: " << lsc << endl;
        allOk=false;
    }
    MatrixN dx0=tln.backward(yc,&cache,&grads);

    ret=matComp(dW1,*(grads["af1-W"]),"TwoLayerNet dW1",eps);
    if (!ret) allOk=false;
    ret=matComp(db1,*(grads["b1-W"]),"TwoLayerNet db1",eps);
    if (!ret) allOk=false;
    ret=matComp(dW2,*(grads["af2-W"]),"TwoLayerNet dW2",eps);
    if (!ret) allOk=false;
    ret=matComp(db2,*(grads["b2-W"]),"TwoLayerNet db2",eps);
    if (!ret) allOk=false;

    cppl_delete(&cache);
    cppl_delete(&grads);
    return allOk;
}

bool registerTest() {
    bool allOk=true;
    cout << "Registered Layers:" << endl;
    int nr=1;
    for (auto it : _syncogniteLayerFactory.mapl) {
        cout << nr << ".: " << it.first << " ";
        t_layer_props_entry te=_syncogniteLayerFactory.mapprops[it.first];
        t_layer_topo tp=std::vector<int>(te);
        Layer *l = CREATE_LAYER(it.first, tp)
        if (l->layerType == LT_NORMAL) {
            cout << "normal layer" << endl;
        } else if (l->layerType==LT_LOSS) {
            cout << "loss-layer (final)" << endl;
        } else {
            cout << "unspecified layer -- ERROR!" << endl;
            allOk=false;
        }
        delete l;
        ++nr;
    }
    return allOk;
}


bool trainTest() {
    bool allOk=true;
    TwoLayerNet tln({5,4,2});
    cout << "NOT IMPLEMENTED!" << endl;
    return allOk;
}

int main() {
    MatrixN yz=MatrixN(0,0);
    cout << "=== 0.: Init: registering layers" << endl;
    registerLayers();
    cout << "=== 1.: Numerical gradient tests" << endl;
    bool allOk=true;
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    Affine pc({30,20});
    MatrixN x(10,30);
    x.setRandom();
    if (!pc.selfTest(x,yz)) {
        allOk=false;
    }

    Relu rl({30});
    MatrixN xr(20,30);
    xr.setRandom();
    if (!rl.selfTest(xr,yz)) {
        allOk=false;
    }

    AffineRelu rx({2,3});
    MatrixN xarl(30,2);
    xarl.setRandom();
    if (!rx.selfTest(xarl,yz)) {
        allOk=false;
    }

    int ntl1=4, ntl2=5, ntl3=6, ntlN=30;
    TwoLayerNet tl({ntl1,ntl2,ntl3});
    MatrixN xtl(ntlN,ntl1);
    xtl.setRandom();
    MatrixN y2(ntlN,1);
    for (unsigned i=0; i<y2.rows(); i++) y2(i,0)=(rand()%ntl3);
    if (!tl.selfTest(xtl,y2)) {
        allOk=false;
        cout << red << "Numerical gradient for TwoLayerNet: ERROR." << def << endl;
    }

    int smN=10, smC=4;
    Softmax mx({smC});
    MatrixN xmx(smN,smC);
    xmx.setRandom();
    MatrixN y(smN,1);
    for (unsigned i=0; i<y.rows(); i++) y(i,0)=(rand()%smC);
    if (!mx.selfTest(xmx, y, 1e-3, 1e-6)) {
        allOk=false;
    }

    int svN=10, svC=5;
    Svm sv({svC});
    MatrixN xsv(svN,svC);
    xsv.setRandom();
    MatrixN yv(svN,1);
    for (unsigned i=0; i<yv.rows(); i++) yv(i,0)=(rand()%svC);
    if (!sv.selfTest(xsv, yv, 1e-3, 1e-6)) {
        allOk=false;
    }

    cout << "=== 2.: Test-data tests" << endl;

    if (checkAffineForward()) {
        cout << green << "AffineForward (Affine) with test data: OK." << def << endl;
    } else {
        cout << red << "AffineForward (Affine) with test data: ERROR." << def << endl;
        allOk=false;
    }

    if (checkAffineBackward()) {
        cout << green << "AffineBackward (Affine) with test data: OK." << def << endl;
    } else {
        cout << red << "AffineBackward (Affine) with test data: ERROR." << def << endl;
        allOk=false;
    }

    if (checkReluForward()) {
        cout << green << "ReluForward with test data: OK." << def << endl;
    } else {
        cout << red << "ReluForward with test data: ERROR." << def << endl;
        allOk=false;
    }

    if (checkReluBackward()) {
        cout << green << "ReluBackward (Affine) with test data: OK." << def << endl;
    } else {
        cout << red << "ReluBackward (Affine) with test data: ERROR." << def << endl;
        allOk=false;
    }

    if (checkAffineRelu()) {
        cout << green << "AffineRelu with test data: OK." << def << endl;
    } else {
        cout << red << "AffineRelu with test data: ERROR." << def << endl;
        allOk=false;
    }

    if (checkSoftmax()) {
        cout << green << "Softmax with test data: OK." << def << endl;
    } else {
        cout << red << "Softmax with test data: ERROR." << def << endl;
        allOk=false;
    }
    if (checkSvm()) {
        cout << green << "Svm with test data: OK." << def << endl;
    } else {
        cout << red << "Svm with test data: ERROR." << def << endl;
        allOk=false;
    }

    if (checkTwoLayer()) {
        cout << green << "TwoLayerNet with test data: OK." << def << endl;
    } else {
        cout << red << "TwoLayerNet with test data: ERROR." << def << endl;
        allOk=false;
    }


    if (registerTest()) {
        cout << green << "RegisterTest: OK." << def << endl;
    } else {
        cout << red << "RegisterTest: ERROR." << def << endl;
        allOk=false;
    }


/*    if (trainTest()) {
        cout << green << "TrainTest: OK." << def << endl;
    } else {
        cout << red << "TrainTest: ERROR." << def << endl;
        allOk=false;
    }
*/
    if (allOk) {
        cout << green << "All tests ok." << def << endl;
    } else {
        cout << red << "Tests failed." << def << endl;
    }

    return 0;
}
