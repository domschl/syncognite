#ifndef _CPL_SVM_H
#define _CPL_SVM_H

#include "../cp-layers.h"


// Multiclass support vector machine Svm
class Svm : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="Svm";
        layerType=LayerType::LT_LOSS;
        cp=cx;
        inputShapeRang=1;
        vector<int> inputShape=cp.getPar("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        outputShape={inputShapeFlat};
        layerInit=true;
    }
public:
    Svm(const CpParams& cx) {
        setup(cx);
    }
    Svm(string conf) {
        setup(CpParams(conf));
    }
    ~Svm() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        VectorN correctClassScores(x.rows());
        for (unsigned int i=0; i<x.rows(); i++) {
            correctClassScores(i)=x(i,(int)y(i));
        }
        MatrixN margins=x;
        margins.colwise() -= correctClassScores;
        margins = (margins.array() + 1.0).matrix();
        for (unsigned int i=0; i < margins.size(); i++) {
            if (margins(i)<0.0) margins(i)=0.0;
        }
        for (unsigned int i=0; i<margins.rows(); i++) {
            margins(i,y(i,0)) = 0.0;
        }

        if (pcache!=nullptr) cppl_set(pcache, "margins", new MatrixN(margins));
        return margins;
    }
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) override {
        MatrixN margins=*((*pcache)["margins"]);
        floatN loss = margins.sum() / margins.rows();
        return loss;
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        MatrixN margins=*((*pcache)["margins"]);
        MatrixN x=*((*pcache)["x"]);
        VectorN numPos(x.rows());
        MatrixN dx=x;
        dx.setZero();
        for (unsigned int i=0; i<margins.rows(); i++) {
            int num=0;
            for (unsigned int j=0; j<margins.cols(); j++) {
                if (margins(i,j)>0.0) ++num;
            }
            numPos(i) = (floatN)num;
        }
        for (unsigned int i=0; i<dx.size(); i++) {
            if (margins(i) > 0.0) dx(i)=1.0;
        }
        for (unsigned int i=0; i<dx.rows(); i++) {
            dx(i,y(i,0)) -= numPos(i);
        }
        dx /= dx.rows();
        return dx;
    }
};

#endif
