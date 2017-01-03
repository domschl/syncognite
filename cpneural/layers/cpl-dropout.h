#ifndef _CPL_DROPOUT_H
#define _CPL_DROPOUT_H

#include "../cp-layers.h"

// Dropout: with probability drop, a neuron's value is dropped. (1-p)?
class Dropout : public Layer {
private:
    bool freeze;
    void setup(const CpParams& cx) {
        layerName="Dropout";
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        inputShapeRang=1;
        vector<int> inputShape=cp.getPar("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        outputShape={inputShape};
        drop = cp.getPar("drop", (floatN)0.5);
        trainMode = cp.getPar("train", (bool)false);
        freeze = cp.getPar("freeze", (bool)false);
        if (freeze) srand(123);
        else srand(time(nullptr));
        layerInit=true;
    }
public:
    floatN drop;
    bool trainMode;

    Dropout(const CpParams& cx) {
        setup(cx);
    }
    Dropout(string conf) {
        setup(CpParams(conf));
    }
    ~Dropout() {
        cppl_delete(&params);
    }

    /* not thread safe
    unsigned long fastrand(void) {          //period 2^96-1
    static unsigned long x=123456789, y=362436069, z=521288629;
    unsigned long t;
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

       t = x;
       x = y;
       y = z;
       z = t ^ x ^ y;

      return z;
    } */
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        drop = cp.getPar("drop", (floatN)0.5);
        if (drop==1.0) return x;
        trainMode = cp.getPar("train", false);
        freeze = cp.getPar("freeze", false);
        if (freeze) srand(123);
        MatrixN xout;
        if (trainMode) {
            MatrixN* pmask;
            if (freeze && pcache!=nullptr && pcache->find("dropmask")!=pcache->end()) {
                pmask=(*pcache)["dropmask"];
                xout = x.array() * (*pmask).array();
            } else {
                pmask=new MatrixN(x);
                // pmask->setRandom();
                //pmask->setZero();
                int dr=(int)(drop*1000.0);
                for (int i=0; i<x.size(); i++) {
                    //if (((*pmask)(i)+1.0)/2.0 < drop) (*pmask)(i)=1.0;
                    //else (*pmask)(i)=0.0;
                    //if (fastrand()%1000 < dr) (*pmask)(i)=1.0;
                    if (rand()%1000 < dr) (*pmask)(i)=1.0;
                    else (*pmask)(i)=0.0;
                }
                xout = x.array() * (*pmask).array();
                if (pcache!=nullptr) cppl_set(pcache, "dropmask", pmask);
                else delete pmask;
            }
            //cerr << "drop:" << drop << endl << xout << endl;
        } else {
            xout = x * drop;
        }
        return xout;
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) override {
        MatrixN dx;
        trainMode = cp.getPar("train", false);
        if (trainMode && drop!=1.0) {
            MatrixN* pmask=(*pcache)["dropmask"];
            dx=y.array() * (*pmask).array();
        } else {
            dx=y;
        }
        return dx;
    }
};

#endif
