#ifndef _CPLT_TEMPORALSOFTMAX_H
#define _CPLT_TEMPORALSOFTMAX_H

#include "../testneural.h"

float getTemporalSMLoss(int N, int T, int V, float p) {
    MatrixN x(N,V*T);
    x.setRandom();
    x = (x.array()+1.0) * 0.005;
    MatrixN y(N,T);
    for (int i=0; i<y.size(); i++) y(i)=rand() % V;
    MatrixN mask(N,T);
    for (int i=0; i<mask.size(); i++) {
        if (rand()%1000 < p*1000.0) mask(i)=1.0;
        else mask(i)=0.0;
    }
    CpParams cp;
    cp.setPar("inputShape",vector<int>{V,T});
    TemporalSoftmax tsm(cp);
    t_cppl cache;
    cppl_set(&cache,"mask",new MatrixN(mask));
    tsm.forward(x,y,&cache);
    float loss;
    loss=tsm.loss(y,&cache);
    cppl_delete(&cache);
    return loss;
}

bool checkTemporalSoftmaxLoss(float eps=CP_DEFAULT_NUM_EPS) {
    bool allOk=true;
    float loss;
    cerr << "Checking TemporalSoftmaxLoss:" << endl;
    loss=getTemporalSMLoss(1000, 1, 10, 1.0);   // Should be about 2.3
    if (std::abs(loss-2.3)>eps) {
        cerr << "  TemporalSMLoss check failed for ex (1): " << loss << ", should be 2.3" << endl;
        allOk=false;
    } else {
        cerr << "  TemporalSMLoss check OK for ex (1): " << loss << ", theoretical: 2.3" << endl;
    }
    loss=getTemporalSMLoss(1000, 10, 10, 1.0);  // Should be about 23
    if (std::abs(loss-23.0)>eps) {
        cerr << "TemporalSMLoss check failed for ex (2): " << loss << ", should be 23" << endl;
        allOk=false;
    } else {
        cerr << "  TemporalSMLoss check OK for ex (2): " << loss << ", theoretical: 23" << endl;
    }
    loss=getTemporalSMLoss(50000, 10, 10, 0.1); // Should be about 2.3
    if (std::abs(loss-2.3)>eps) {
        cerr << "  TemporalSMLoss check failed for ex (3): " << loss << ", should be 2.3" << endl;
        allOk=false;
    } else {
        cerr << "  TemporalSMLoss check OK for ex (3): " << loss << ", theoretical: 2.3" << endl;
    }
    return allOk;
}

#endif
