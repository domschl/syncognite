#ifndef _CPL_DROPOUT_H
#define _CPL_DROPOUT_H

#include "../cp-layers.h"

// Dropout: with probability drop, a neuron's value is dropped. (1-p)?
class Dropout : public Layer {
private:
    bool freeze;
    void setup(const json& jx) {
        j=jx;
        layerName=j.value("name",(string)"Dropout");
        layerClassName="Dropout";
        layerType=LayerType::LT_NORMAL;
        inputShapeRang=1;
        vector<int> inputShape=j.value("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        outputShape={inputShape};
        drop = j.value("drop", (floatN)0.5);
        trainMode = j.value("train", (bool)false);
        freeze = j.value("freeze", (bool)false);
        if (freeze) srand(123);
        else srand(time(nullptr));
        layerInit=true;
    }
public:
    floatN drop;
    bool trainMode;

    Dropout(const json& jx) {
        setup(jx);
    }
    Dropout(const string conf) {
        setup(json::parse(conf));
    }
    ~Dropout() {
        cppl_delete(&params);
    }


    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        drop = j.value("drop", (floatN)0.5);
        if (drop==1.0) return x;
        trainMode = j.value("train", false);
        freeze = j.value("freeze", false);
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
        trainMode = j.value("train", false);
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
