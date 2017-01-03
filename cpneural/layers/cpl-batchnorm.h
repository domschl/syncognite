#ifndef _CPL_BATCHNORM_H
#define _CPL_BATCHNORM_H

#include "../cp-layers.h"

// Batch normalization
class BatchNorm : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="BatchNorm";
        layerType=LayerType::LT_NORMAL;
        inputShapeRang=1;
        cp=cx;
        eps = cp.getPar("eps", (floatN)1e-5);
        momentum = cp.getPar("momentum", (floatN)0.9);
        trainMode = cp.getPar("train", (bool)false);
        vector<int> inputShape=cp.getPar("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        outputShape={inputShape};

        MatrixN *pgamma=new MatrixN(1,inputShapeFlat);
        pgamma->setOnes();
        cppl_set(&params,"gamma",pgamma);
        MatrixN *pbeta=new MatrixN(1,inputShapeFlat);
        pbeta->setZero();
        cppl_set(&params,"beta",pbeta);
        layerInit=true;
    }
public:
    floatN eps;
    floatN momentum;
    bool trainMode;

    BatchNorm(const CpParams& cx) {
        setup(cx);
    }
    BatchNorm(string conf) {
        setup(CpParams(conf));
    }
    ~BatchNorm() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        MatrixN *prm, *prv;
        MatrixN *pbeta, *pgamma;
        MatrixN xout;
        trainMode = cp.getPar("train", false);
        if (pcache==nullptr || pcache->find("running_mean")==pcache->end()) {
            prm=new MatrixN(1,shape(x)[1]);
            prm->setZero();
            if (pcache!=nullptr) cppl_set(pcache,"running_mean",prm);
        } else {
            prm=(*pcache)["running_mean"];
        }
        if (pcache==nullptr || pcache->find("running_var")==pcache->end()) {
            prv=new MatrixN(1,shape(x)[1]);
            prv->setZero(); //setOnes();
            if (pcache != nullptr) cppl_set(pcache,"running_var",prv);
        } else {
            prv=(*pcache)["running_var"];
        }
        pgamma=params["gamma"];
        pbeta=params["beta"];

        floatN N=shape(x)[0];
        MatrixN mean=(x.colwise().sum()/N); //.row(0);
        RowVectorN meanv=mean.row(0);
        MatrixN xme=x.rowwise()-meanv;
        MatrixN sqse=(xme.array() * xme.array()).colwise().sum()/N+eps;
        MatrixN stdv=sqse.array().sqrt(); //.row(0);
        RowVectorN stdvv=stdv.row(0);

        if (pcache!=nullptr) {
            *(*pcache)["running_mean"] = *prm * momentum + mean * (1.0-momentum);
            *(*pcache)["running_var"]  = *prv * momentum + stdv * (1.0-momentum);
        } else {
            delete prm;
            prm=nullptr;
            delete prv;
            prv=nullptr;
        }

        if (trainMode) {
            MatrixN x2 = xme.array().rowwise() / RowVectorN(stdv.row(0)).array();
            xout = x2.array().rowwise() * RowVectorN((*pgamma).row(0)).array();
            xout.rowwise() += RowVectorN((*pbeta).row(0));

            if (pcache != nullptr) {
                cppl_update(pcache,"sqse",&sqse);
                cppl_update(pcache,"xme",&xme);
                cppl_update(pcache, "x2", &x2);

                if (momentum==1.0) {
                    cerr << "ERROR: momentum should never be 1" << endl;
                }
            }
        } else { // testmode
            MatrixN xot = x.rowwise() - meanv;
            MatrixN xot2 = xot.array().rowwise() / stdvv.array();
            MatrixN xot3 = xot2.array().rowwise() * RowVectorN((*pgamma).row(0)).array();
            xout = xot3.rowwise() + RowVectorN((*pbeta).row(0));
        }
        return xout;
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) override {
        if (pcache->find("sqse")==pcache->end()) cerr << "Bad: no cache entry for sqse!" << endl;
        MatrixN sqse=*((*pcache)["sqse"]);
        if (pcache->find("xme")==pcache->end()) cerr << "Bad: no cache entry for xme!" << endl;
        MatrixN xme=*((*pcache)["xme"]);
        if (pcache->find("x2")==pcache->end()) cerr << "Bad: no cache entry for x2!" << endl;
        MatrixN x2=*((*pcache)["x2"]);
        if (params.find("gamma")==params.end()) cerr << "Bad: no params entry for gamma!" << endl;
        MatrixN gamma=*(params["gamma"]);
        MatrixN dbeta=y.colwise().sum();

        MatrixN dgamma=(x2.array() * y.array()).colwise().sum();
        if (shape(gamma) != shape(dgamma)) cerr << "bad: dgamma has wrong shape: " << shape(gamma) << shape(dgamma) << endl;

        floatN N=y.rows();
        MatrixN d1=MatrixN(y).setOnes();
        MatrixN dx0 = gamma.array() * (sqse.array().pow(-0.5)) / N;
        MatrixN dx1 = (y*N).rowwise() - RowVectorN(y.colwise().sum());
        MatrixN iv = sqse.array().pow(-1.0);
        MatrixN dx21 = (xme.array().rowwise() * RowVectorN(iv).array());
        MatrixN dx22 = (y.array() * xme.array()).colwise().sum();
        MatrixN dx2 = dx21.array().rowwise() * RowVectorN(dx22).array();
        // cerr << "d1:" << shape(d1) << ", dx0:" << shape(dx0) << endl;
        //MatrixN dx=(dx1 - dx2).array().rowwise() * RowVectorN(d1*dx0).array() ;
        MatrixN dx=(dx1 - dx2).array().rowwise() * RowVectorN(dx0).array() ;
        cppl_set(pgrads,"gamma",new MatrixN(dgamma));
        cppl_set(pgrads,"beta",new MatrixN(dbeta));

        return dx;
    }

};

#endif
