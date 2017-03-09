#ifndef _CPL_SOFTMAX_H
#define _CPL_SOFTMAX_H

#include "../cp-layers.h"


class Softmax : public Layer {
private:
    void setup(const json& jx) {
        layerName="Softmax";
        layerType=LayerType::LT_LOSS;
        j=jx;
        inputShapeRang=1;
        vector<int> inputShape=j.value("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        outputShape={1};
        layerInit=true;
    }
public:
    Softmax(const json& jx) {
        setup(jx);
    }
    Softmax(const string conf) {
        setup(json::parse(conf));
    }
    ~Softmax() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        //if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        if (pstates->find("y") == pstates->end()) {
            cerr << "SM-fw: pstates does not contain y -> fatal!" << endl;
            exit(-1);
        }
        MatrixN y = *((*pstates)["y"]);
        VectorN mxc = x.rowwise().maxCoeff();
        MatrixN xn = x;
        xn.colwise() -=  mxc;
        MatrixN xne = xn.array().exp().matrix();
        VectorN xnes = xne.rowwise().sum();
        for (unsigned int i=0; i<xne.rows(); i++) { // XXX broadcasting?
            if (xnes(i)==0.0) {
                cerr << "zero-divider in softmax: " << i;
                exit(-1);
            }
            xne.row(i) = xne.row(i) / xnes(i);
        }
        MatrixN probs = xne;
        if (pcache!=nullptr) cppl_set(pcache, "probs", new MatrixN(probs));
        return probs;
    }
    virtual floatN loss(t_cppl* pcache, t_cppl* pstates) override {
        if (pstates->find("y") == pstates->end()) {
            cerr << "SM-loss: pstates does not contain y -> fatal!" << endl;
        }
        MatrixN y = *((*pstates)["y"]);
        if (pcache->find("probs")==pcache->end()) {
            cerr << "SM-loss: probs unknown, not in cache -> fatal!" << endl;
        }
        MatrixN probs=*((*pcache)["probs"]);
        if (y.rows() != probs.rows() || y.cols() != 1) {
            cerr << layerName << ": "  << "Loss, dimension mismatch in Softmax(x), Probs: ";
            cerr << shape(probs) << " y:" << shape(y) << " y.cols=" << y.cols() << "(should be 1)" << endl;
            return 1000.0;
        }
        //if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        floatN loss=0.0;
        for (unsigned int i=0; i<probs.rows(); i++) {
            if (y(i,0)>=probs.cols()) {
                cerr << "internal error: y(" << i << ",0) >= " << probs.cols() << endl;
                return -10000.0;
            }
            floatN pi = probs(i,y(i,0));
            if (pi==0.0) cerr << "Invalid zero log-probability at " << i << endl;
            else loss -= log(pi);
        }
        loss /= probs.rows();
        return loss;
    }
    virtual MatrixN backward(const MatrixN& dy, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) override {
        if (pstates->find("y") == pstates->end()) {
            cerr << "SM-bw: pstates does not contain y -> fatal!" << endl;
        }
        MatrixN y = *((*pstates)["y"]);
        MatrixN probs=*((*pcache)["probs"]);

        MatrixN dx=probs;
        for (unsigned int i=0; i<probs.rows(); i++) {
            dx(i,y(i,0)) -= 1.0;
        }
        dx /= dx.rows();
        return dx;
    }
};

#endif
