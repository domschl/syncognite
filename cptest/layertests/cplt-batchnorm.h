#ifndef _CPLT_BATCHNORM_H
#define _CPLT_BATCHNORM_H

#include "../testneural.h"

bool checkBatchNormForward(floatN eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    t_cppl cache;
    t_cppl states;
    bool allOk=true;
    MatrixN x(10,3);
    x << -1.76682167,  0.76232051, -1.31336665,
         -1.5949946 ,  0.96014   ,  0.71301837,
          1.64497495,  2.10628039, -0.79136981,
         -0.06795316, -0.47106827,  1.18868314,
          0.92155587, -0.08303236,  0.52682692,
         -3.12821021,  2.16827829, -0.93670303,
         -0.6552201 ,  0.35542666,  0.52397302,
         -3.60689284,  2.35628056,  1.01625476,
          0.38807454, -0.35816073,  0.69088183,
          1.06266819,  1.04606173, -0.53434315;

    MatrixN xn(10,3);
    xn << -0.6369156 , -0.12222813, -1.65495685,
         -0.53619278,  0.07607152,  0.70380848,
          1.36303558,  1.22499387, -1.04733883,
          0.35893921, -1.35861064,  1.25749474,
          0.93897664, -0.96963287,  0.48707675,
         -1.434944  ,  1.28714225, -1.21651051,
          0.01469091, -0.53010961,  0.48375473,
         -1.71554159,  1.47560085,  1.05678359,
          0.62625677, -1.24542905,  0.67804096,
          1.02169486,  0.1622018 , -0.74815305;

    BatchNorm bn(R"({"inputShape":[3],"train":true})"_json);
    MatrixN xn0=bn.forward(x, &cache, &states);
    MatrixN mean=xn0.colwise().mean();
    if (verbose>1) cerr << "    Mean:" << mean << endl;
    MatrixN xme = xn0.rowwise() - RowVectorN(mean.row(0));
    MatrixN xmsq = ((xme.array() * xme.array()).colwise().sum()/xn0.rows()).array().sqrt();
    if (verbose>1) cerr << "    StdDev:" << xmsq << endl;

    if (!matCompT(xn,xn0,"BatchNormForward",eps,verbose)) {
        if (verbose>0) cerr << "  BatchNorm forward failed" << endl;
        allOk=false;
    }  else {
        if (verbose>1) cerr << "  BatchNorm forward ok." << endl;
    }

    MatrixN runmean(1,3);
    runmean << -0.06802819,  0.08842527,  0.01083855;
    MatrixN runvar(1,3);
    runvar << 0.170594  ,  0.09975786,  0.08590872;
    if (!matCompT(*(cache["running_mean"]),runmean,"BatchNorm run-mean", eps, verbose)) {
        if (verbose>0) cerr << "  BatchNorm running-mean failed" << endl;
        allOk=false;
    } else {
        if (verbose>1) cerr << "  BatchNorm running mean ok." << endl;
    }
    if (!matCompT(*(cache["running_var"]),runvar,"BatchNorm run-var", eps, verbose)) {
        if (verbose>0) cerr << "  BatchNorm running-var failed" << endl;
        allOk=false;
    } else {
        if (verbose>1) cerr << "  BatchNorm running var ok." << endl;
    }
    cppl_delete(&cache);

    t_cppl cache2;
    MatrixN xn2(10,3);
    xn2 << 10.3630844 ,  11.75554375,   8.03512946,
           10.46380722,  12.15214304,  15.11142543,
           12.36303558,  14.44998773,   9.8579835 ,
           11.35893921,   9.28277872,  16.77248423,
           11.93897664,  10.06073426,  14.46123025,
            9.565056  ,  14.5742845 ,   9.35046847,
           11.01469091,  10.93978079,  14.45126418,
            9.28445841,  14.95120171,  16.17035077,
           11.62625677,   9.5091419 ,  15.03412288,
           12.02169486,  12.3244036 ,  10.75554084;


    BatchNorm bn2(R"({"inputShape":[3],"train":true})"_json);
    *(bn2.params["gamma"]) << 1.0, 2.0, 3.0;
    *(bn2.params["beta"]) << 11.0, 12.0, 13.0;
    MatrixN xn20=bn2.forward(x, &cache2, &states);
    MatrixN mean2=xn20.colwise().mean();
    if (verbose>1) cerr << "    Mean:" << mean2 << endl;
    MatrixN xme2 = xn20.rowwise() - RowVectorN(mean2.row(0));
    MatrixN xmsq2 = ((xme2.array() * xme2.array()).colwise().sum()/xn20.rows()).array().sqrt();
    if (verbose>1) cerr << "    StdDev:" << xmsq2 << endl;

    if (!matCompT(xn2,xn20,"BatchNormForward",eps,verbose)) {
        if (verbose>0) cerr << "  BatchNorm with beta/gamma forward failed" << endl;
        allOk=false;
    }  else {
        if (verbose>1) cerr << "  BatchNorm beta/gamma forward ok." << endl;
    }

    MatrixN runmean2(1,3);
    runmean2 << -0.06802819,  0.08842527,  0.01083855;
    MatrixN runvar2(1,3);
    runvar2 << 0.170594  ,  0.09975786,  0.08590872;
    if (!matCompT(*(cache2["running_mean"]),runmean2, "BatchNormForward", eps, verbose)) {
        if (verbose>0) cerr << "  BatchNorm running-mean2 failed" << endl;
        allOk=false;
    } else {
        if (verbose>1) cerr << "  BatchNorm running mean2 ok." << endl;
    }
    if (!matCompT(*(cache2["running_var"]),runvar2, "BatchNormForward", eps, verbose)) {
        if (verbose>0) cerr << "  BatchNorm running-var2 failed" << endl;
        allOk=false;
    } else {
        if (verbose>1) cerr << "  BatchNorm running var2 ok." << endl;
    }
    cppl_delete(&cache2);

    t_cppl cache3;
    int nnr=200;
    MatrixN xt(nnr,3);
    BatchNorm bn3(R"({"inputShape":[3],"train":true})"_json);
    *(bn3.params["gamma"]) << 1.0, 2.0, 3.0;
    *(bn3.params["beta"]) << 0.0, -1.0, 4.0;
    for (int i=0; i<nnr; i++) {
        xt.setRandom();
        bn3.forward(xt,&cache3, &states);
    }
    if (verbose>1) {
        cerr << "    Running mean after " << nnr << " cycl: " << *(cache3["running_mean"]) << endl;
        cerr << "    Running stdvar after " << nnr << " cycl: " << *(cache3["running_var"]) << endl;
    }
    if (verbose>1) cerr << "  switching test" << endl;
    bn3.j["train"]=false;
    if (bn3.j.value("train", true)) cerr << "INTERNAL ERROR: parSet boolean failed!" << endl;
    xt.setRandom();
    MatrixN xn30=bn3.forward(xt, &cache3, &states);
    MatrixN mean3=xn30.colwise().mean();
    if (verbose>1) cerr << "    Mean:" << mean3 << endl;
    if (!matCompT(*(bn3.params["beta"]), mean3, "Batchnorm train/test sequence: mean", 0.1, verbose)) {
        allOk=0;
    }
    MatrixN xme3 = xn30.rowwise() - RowVectorN(mean3.row(0));
    MatrixN xmsq3 = ((xme3.array() * xme3.array()).colwise().sum()/xn30.rows()).array().sqrt();
    if (verbose>1) cerr << "    StdDev:" << xmsq3 << endl;
    if (!matCompT(*(bn3.params["gamma"]), xmsq3, "Batchnorm train/test sequence: stdderi", 0.1, verbose)) {
        allOk=0;
    }
    cppl_delete(&cache3);

    return allOk;
}

bool checkBatchNormBackward(float eps=CP_DEFAULT_NUM_EPS, int verbose=1) {
    bool allOk=true;
    t_cppl states;
    MatrixN x(4,5);
    x << 15.70035538,  10.9836183 ,  12.60007796,   8.40461897, 6.73940903,
         18.85269464,  17.58441018,  14.44920968,   7.33474882, 11.35816205,
         17.20864889,   5.43166315,  14.90657023,  13.22667748, 15.03122597,
          4.33556564,   2.44751717,  12.06870335,   5.74243521, 12.16941296;
    MatrixN gamma(1,5);
    gamma << 0.31896744,  1.67004399,  1.57384876, -0.79775794,  0.15197293;
    MatrixN beta(1,5);
    beta << 1.6099422 ,  1.55804396,  0.88364562,  1.21053159,  0.50543461;

    MatrixN dx(4,5);
    dx << -0.04546749,  0.126882  ,  0.4864624 ,  0.18757926,  0.02899233,
          -0.01486729, -0.09133611, -0.26401447, -0.24233534, -0.0726569 ,
           0.05690624,  0.10035256,  0.13037742, -0.01516446,  0.03441372,
           0.00342854, -0.13589844, -0.35282534,  0.06992054,  0.00925085;
    MatrixN dgamma(1,5);
    dgamma << -1.20718803, -1.45268518, -0.05539113, -2.07077292,  0.89780683;
    MatrixN dbeta(1,5);
    dbeta << -4.45908107, -1.23582122,  0.04775023,  3.14325285,  1.0989124;

    MatrixN dchain(4,5);
    dchain << -2.01650781,  0.01176043,  0.39260716,  0.17978038,  0.49731936,
              -1.63610541, -1.15773852, -0.19987901,  1.88290138, -1.14657413,
              -0.26565248,  0.26942709,  0.09496168, -0.00460701,  1.22847938,
              -0.54081537, -0.35927023, -0.23993959,  1.0851781 ,  0.51968779;

    BatchNorm bn(R"({"inputShape":[5],"train":true})"_json);
    *(bn.params["gamma"])=gamma;
    *(bn.params["beta"])=beta;

    t_cppl cache;
    t_cppl grads;
    MatrixN y=bn.forward(x, &cache, &states);
    MatrixN dx0=bn.backward(dchain, &cache, &states, &grads);

    bool ret=matCompT(dx,dx0,"BatchNormBackward dx",eps,verbose);
    if (!ret) allOk=false;
    ret=matCompT(dgamma,*grads["gamma"],"BatchNormBackward dgamma",eps,verbose);
    if (!ret) allOk=false;
    ret=matCompT(dbeta,*grads["beta"],"BatchNormBackward dbeta",eps,verbose);
    if (!ret) allOk=false;

    cppl_delete(&cache);
    cppl_delete(&grads);
    return allOk;
}

bool testBatchNorm(int verbose) {
    Color::Modifier lblue(Color::FG_LIGHT_BLUE);
    Color::Modifier def(Color::FG_DEFAULT);
	bool bOk=true;
	t_cppl s1;
	cerr << lblue << "BatchNorm Layer: " << def << endl;
	// Numerical gradient
    // Batchnorm - still some strangities:
    BatchNorm bn(R"({"inputShape":[10],"train":true,"noVectorizationTests":true})"_json);
    MatrixN xbr(20, 10);
    xbr.setRandom();
    bool res=bn.selfTest(xbr, &s1, 1e-4, 1e-3, verbose);
	registerTestResult("BatchNorm", "Numerical gradient", res, "");
	if (!res) bOk = false;

	res=checkBatchNormForward(CP_DEFAULT_NUM_EPS, verbose);
	registerTestResult("BatchNorm", "Forward (with test-data)", res, "");
	if (!res) bOk = false;

	res=checkBatchNormBackward(CP_DEFAULT_NUM_EPS, verbose);
	registerTestResult("BatchNorm", "Backward (with test-data)", res, "");
	if (!res) bOk = false;
	return bOk;
}
#endif
