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
    *(pe.params[1])=W;
    *(pe.params[2])=b;
    MatrixN y0=pe.forward(x);
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
    *(pe.params[1])=W;
    *(pe.params[2])=b;
    MatrixN y=pe.forward(x);
    MatrixN dx0=pe.backward(dchain);
    bool allOk=true;
    bool ret=matComp(dx,dx0,"AffineBackward dx",eps);
    if (!ret) allOk=false;
    ret=matComp(dW,*(pe.grads[1]),"AffineBackward dW",eps);
    if (!ret) allOk=false;
    ret=matComp(db,*(pe.grads[2]),"AffineBackward bx",eps);
    if (!ret) allOk=false;
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
    MatrixN y0=rl.forward(x);
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
    MatrixN y=rl.forward(x);
    MatrixN dx0=rl.backward(dchain);
    bool allOk=true;
    bool ret=matComp(dx,dx0,"ReluBackward dx",eps);
    if (!ret) allOk=false;
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
    *(arl.af->params[1])=W;
    *(arl.af->params[2])=b;
    MatrixN y0=arl.forward(x);
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

    MatrixN dx0=arl.backward(dchain);

    ret=matComp(dx,dx0,"AffineRelu dx",eps);
    if (!ret) allOk=false;
    ret=matComp(dW,*(arl.af->grads[1]),"AffineRelu dW",eps);
    if (!ret) allOk=false;
    ret=matComp(db,*(arl.af->grads[2]),"AffineRelu db",eps);
    if (!ret) allOk=false;

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
    MatrixN probs0=sm.forward(x);
    bool ret=matComp(probs,probs0,"Softmax probabilities",eps);
    if (!ret) allOk=false;
    floatN loss0=sm.loss(y);
    floatN d=loss-loss0;
    floatN err=d*d;
    if (err > eps) {
        cout << "Loss error: correct:" << loss << " got: " << loss0 << ", err=" << err << endl;
        allOk=false;
    } else {
        cout << "Loss ok, loss=" << loss0 << " (ref: " << loss << "), err=" << err << endl;
    }
    MatrixN dchain=x;
    dchain.setOnes();
    MatrixN dx0=sm.backward(y);
    ret=matComp(dx,dx0,"Softmax dx",eps);
    if (!ret) allOk=false;

    return allOk;
}

int main() {
    bool allOk=true;
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    Affine pc({30,20});
    MatrixN x(10,30);
    x.setRandom();
    if (!pc.checkAll(x)) {
        allOk=false;
    }

    Relu rl({30});
    MatrixN xr(20,30);
    xr.setRandom();
    if (!rl.checkAll(xr)) {
        allOk=false;
    }

    AffineRelu rx({2,3});
    MatrixN xarl(30,2);
    xarl.setRandom();
    if (!rx.checkAll(xarl)) {
        allOk=false;
    }

    int smN=10, smC=4;
    Softmax mx({smC});
    MatrixN xmx(smN,smC);
    xmx.setRandom();
    MatrixN y(smN,1);
    for (unsigned i=0; i<y.rows(); i++) y(i,0)=(rand()%smC);
    if (!mx.checkLoss(xmx, y)) {
        allOk=false;
    }

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

    if (allOk) {
        cout << green << "All tests ok." << def << endl;
    } else {
        cout << red << "Tests failed." << def << endl;
    }

    return 0;
}
