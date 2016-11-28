#ifndef _CPLT_SPATIALBATCHNORM_H
#define _CPLT_SPATIALBATCHNORM_H

#include "../testneural.h"

/*
bool checkSpatialBatchNormForward(floatN eps=CP_DEFAULT_NUM_EPS) {
    t_cppl cache;
    bool allOk=true;
    int N=2,C=3,H=4,W=5;
    MatrixN x(N,C*H*W);
    x << 8.78962706,   8.0291637 ,  11.60778889,   4.99186032,
            6.13744244],
         [ 14.59732005,  14.10668797,  13.37234881,   3.00249161,
            9.62136543],
         [  4.22327468,  10.00268404,  14.81851005,   9.75125777,
           11.43856752],
         [  3.51450781,  16.89735717,  12.65431168,   6.38401118,
           12.2657403 ]],

        [[ 10.87153462,   8.68198036,   6.73262304,   0.5534183 ,
            4.27954268],
         [  8.69555438,   6.40090306,   6.73451431,  11.92334719,
           17.68919521],
         [ 13.45727092,   7.89658869,   1.4195175 ,  11.61283967,
           10.72618318],
         [ 15.11274493,   7.51741157,  12.75017856,  10.5682995 ,
           -0.03013656]],

        [[  4.88465059,   6.40105871,  10.05622138,  12.837527  ,
           11.16735146],
         [ 13.57428873,   8.56767183,   9.9347655 ,  16.26914829,
            6.9370306 ],
         [ 14.17225379,   6.19287826,  17.23022609,   7.06608598,
           13.95560841],
         [ 16.40296331,   3.98507805,  15.32086583,   2.56310822,
            7.72864566]]],


       [[[  5.85471718,   2.72990482,   4.54645332,   9.44162877,
           13.86911841],
         [ 14.07873185,   8.54849704,  11.180672  ,   5.63914828,
           10.28946611],
         [ 16.42387985,   6.37100762,  11.14480784,  19.79021647,
           13.78566717],
         [ 15.67697588,   5.64131222,  10.28372126,   5.28879624,
           10.75321074]],

        [[  8.98089762,   7.86301729,   5.23153917,   9.30918146,
            7.16110546],
         [  0.51168663,   5.85359042,   9.99794186,   7.05953747,
            5.54801046],
         [ 17.26655238,  12.44949015,  12.79318267,  14.58963923,
            9.6533302 ],
         [ 13.68057759,   2.97747567,   6.33454499,   6.39428531,
           13.38036783]],

        [[  3.22744799,  15.85551598,   7.15545571,   9.47726025,
            7.500317  ],
         [ 12.32933878,   2.98779115,  17.81234035,  12.13771252,
           12.78516613],
         [  8.60556051,  11.55277572,   9.64684081,   7.15129868,
           10.71853786],
         [  7.70506503,   2.78056639,   5.40666492,   5.97151951,
            8.66736087;

    MatrixN xn(10,3);
    xn << ;

    SpatialBatchNorm sbn("{inputShape=[3];train=true}");
    //bn.setPar("trainMode", true);
    MatrixN xn0=sbn.forward(x, &cache);
    MatrixN mean=xn0.colwise().mean();
    cerr << "Mean:" << mean << endl;
    MatrixN xme = xn0.rowwise() - RowVectorN(mean.row(0));
    MatrixN xmsq = ((xme.array() * xme.array()).colwise().sum()/xn0.rows()).array().sqrt();
    cerr << "StdDev:" << xmsq << endl;

    if (!matComp(xn,xn0,"SpatialBatchNormForward",eps)) {
        cerr << "SpatialBatchNorm forward failed" << endl;
        allOk=false;
    }  else {
        cerr << "  SpatialBatchNorm forward ok." << endl;
    }

    MatrixN runmean(1,3);
    runmean << ;
    MatrixN runvar(1,3);
    runvar << ;
    if (!matComp(*(cache["running_mean"]),runmean)) {
        cerr << "SpatialBatchNorm running-mean failed" << endl;
        allOk=false;
    } else {
        cerr << "  SpatialBatchNorm running mean ok." << endl;
    }
    if (!matComp(*(cache["running_var"]),runvar)) {
        cerr << "SpatialBatchNorm running-var failed" << endl;
        allOk=false;
    } else {
        cerr << "  SpatialBatchNorm running var ok." << endl;
    }
    cppl_delete(&cache);

    t_cppl cache2;
    MatrixN xn2(10,3);
    xn2 << ;


    SpatialBatchNorm sbn2("{inputShape=[3];train=true}");
    *(sbn2.params["gamma"]) << 1.0, 2.0, 3.0;
    *(sbn2.params["beta"]) << 11.0, 12.0, 13.0;
    //bn.setPar("trainMode", true);
    MatrixN xn20=sbn2.forward(x, &cache2);
    MatrixN mean2=xn20.colwise().mean();
    cerr << "Mean:" << mean2 << endl;
    MatrixN xme2 = xn20.rowwise() - RowVectorN(mean2.row(0));
    MatrixN xmsq2 = ((xme2.array() * xme2.array()).colwise().sum()/xn20.rows()).array().sqrt();
    cerr << "StdDev:" << xmsq2 << endl;

    if (!matComp(xn2,xn20,"SpatialBatchNormForward",eps)) {
        cerr << "SpatialBatchNorm with beta/gamma forward failed" << endl;
        allOk=false;
    }  else {
        cerr << "  SpatialBatchNorm beta/gamma forward ok." << endl;
    }

    MatrixN runmean2(1,3);
    runmean2 << ;
    MatrixN runvar2(1,3);
    runvar2 << ;
    if (!matComp(*(cache2["running_mean"]),runmean2)) {
        cerr << "SpatialBatchNorm running-mean2 failed" << endl;
        allOk=false;
    } else {
        cerr << "  SpatialBatchNorm running mean2 ok." << endl;
    }
    if (!matComp(*(cache2["running_var"]),runvar2)) {
        cerr << "SpatialBatchNorm running-var2 failed" << endl;
        allOk=false;
    } else {
        cerr << "  SpatialBatchNorm running var2 ok." << endl;
    }
    cppl_delete(&cache2);

    t_cppl cache3;
    int nnr=200;
    MatrixN xt(nnr,3);
    BatchNorm sbn3("{inputShape=[3];train=true}");
    *(sbn3.params["gamma"]) << 1.0, 2.0, 3.0;
    *(sbn3.params["beta"]) << 0.0, -1.0, 4.0;
    for (int i=0; i<nnr; i++) {
        xt.setRandom();
        sbn3.forward(xt,&cache3);
    }
    cerr << "  Running mean after " << nnr << " cycl: " << *(cache3["running_mean"]) << endl;
    cerr << "  Running stdvar after " << nnr << " cycl: " << *(cache3["running_var"]) << endl;
    cerr << "switching test" << endl;
    sbn3.cp.setPar("train", false);
    if (sbn3.cp.getPar("train", true)) cerr << "INTERNAL ERROR: parSet boolean failed!" << endl;
    xt.setRandom();
    MatrixN xn30=sbn3.forward(xt, &cache3);
    MatrixN mean3=xn30.colwise().mean();
    cerr << "  Mean:" << mean3 << endl;
    if (!matComp(*(bn3.params["beta"]), mean3, "SpatialBatchnorm train/test sequence: mean", 0.1)) {
        allOk=0;
    }
    MatrixN xme3 = xn30.rowwise() - RowVectorN(mean3.row(0));
    MatrixN xmsq3 = ((xme3.array() * xme3.array()).colwise().sum()/xn30.rows()).array().sqrt();
    cerr << "  StdDev:" << xmsq3 << endl;
    if (!matComp(*(bn3.params["gamma"]), xmsq3, "SpatialBatchnorm train/test sequence: stdderi", 0.1)) {
        allOk=0;
    }
    cppl_delete(&cache3);

    return allOk;
}

bool checkSpatialBatchNormBackward(float eps=CP_DEFAULT_NUM_EPS) {
    bool allOk=true;
    MatrixN x(4,5);
    x << ;
    MatrixN gamma(1,5);
    gamma << ;
    MatrixN beta(1,5);
    beta << ;

    MatrixN dx(4,5);
    dx << ;
    MatrixN dgamma(1,5);
    dgamma << ;
    MatrixN dbeta(1,5);
    dbeta << ;

    MatrixN dchain(4,5);
    dchain << ;

    SpatialBatchNorm sbn("{inputShape=[5];train=true}");
    *(sbn.params["gamma"])=gamma;
    *(sbn.params["beta"])=beta;

    t_cppl cache;
    t_cppl grads;
    MatrixN y=sbn.forward(x, &cache);
    MatrixN dx0=sbn.backward(dchain, &cache, &grads);

    bool ret=matComp(dx,dx0,"SpatialBatchNormBackward dx",eps);
    if (!ret) allOk=false;
    ret=matComp(dgamma,*grads["gamma"],"SpatialBatchNormBackward dgamma",eps);
    if (!ret) allOk=false;
    ret=matComp(dbeta,*grads["beta"],"SpatialBatchNormBackward dbeta",eps);
    if (!ret) allOk=false;

    cppl_delete(&cache);
    cppl_delete(&grads);
    return allOk;
}

*/
#endif
