#ifndef _CP_LAYERS_H
#define _CP_LAYERS_H

#include "cp-math.h"
#include "cp-layer.h"

class Nonlinearities {
public:
    MatrixN sigmoid(MatrixN& m) {
        ArrayN ar = m.array();
        return (1/(1+(ar.maxCoeff() - ar).exp())).matrix();
    }
    MatrixN tanh(MatrixN& m) {
        ArrayN ar = m.array();
        return (ar.tanh()).matrix();
    }
    MatrixN Relu(MatrixN& m) {
        ArrayN ar = m.array();
        return (ar.max(0)).matrix();
    }
};

class Affine : public Layer {
public:
    Affine(t_layer_topo topo) {
        assert (topo.size()==2);
        layerName="Affine";
        layerType=LayerType::LT_NORMAL;
        cppl_set(&params, "W", new MatrixN(topo[0],topo[1])); // W
        cppl_set(&params, "b", new MatrixN(1,topo[1])); // b

        params["W"]->setRandom();
        floatN xavier = 1.0/(floatN)(topo[0]+topo[1]); // (setRandom is [-1,1]-> fakt 0.5, xavier is 2/(ni+no))
        *params["W"] *= xavier;
        params["b"]->setRandom();
        *params["b"] *= xavier;
    }
    ~Affine() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache) override {
        if (params["W"]->rows() != x.cols()) {
            cout << layerName << ": " << "Forward: dimension mismatch in x*W: x:" << shape(x) << " W:" << shape(*params["W"]) << endl;
            MatrixN y(0,0);
            return y;
        }
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        MatrixN y = x * (*params["W"]);
        RowVectorN b = *params["b"];
        y.rowwise() += b;
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pgrads) override {
        MatrixN dx = dchain * (*params["W"]).transpose(); // dx
        cppl_set(pgrads, "W", new MatrixN((*(*pcache)["x"]).transpose() * dchain)); //dW
        cppl_set(pgrads, "b", new MatrixN(dchain.colwise().sum())); //db
        return dx;
    }
};

class Relu : public Layer {
public:
    Relu(t_layer_topo topo) {
        assert (topo.size()==1);
        layerName="Relu";
        layerType=LayerType::LT_NORMAL;
    }
    ~Relu() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl *pcache) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        MatrixN ych=x;
        for (unsigned int i=0; i<ych.size(); i++) {
            if (x(i)<0.0) {
                ych(i)=0.0;
            }
        }
        return ych;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl *pcache, t_cppl *pgrads) override {
        MatrixN y=*((*pcache)["x"]);
        for (unsigned int i=0; i<y.size(); i++) {
            if (y(i)>0.0) y(i)=1.0;
            else y(i)=0.0;
        }
        MatrixN dx = y.cwiseProduct(dchain); // dx
        return dx;
    }
};

void mlPush(string prefix, t_cppl *src, t_cppl *dst) {
    if (dst!=nullptr) {
        for (auto pi : *src) {
            cppl_set(dst, prefix+"-"+pi.first, pi.second);
        }
    } else {
        cppl_delete(src);
    }
}

void mlPop(string prefix, t_cppl *src, t_cppl *dst) {
    for (auto ci : *src) {
        if (ci.first.substr(0,prefix.size()+1)==prefix+"-") cppl_set(dst, ci.first.substr(prefix.size()+1), ci.second);
    }
}

class AffineRelu : public Layer {
public:
    Affine *af;
    Relu *rl;
    AffineRelu(t_layer_topo topo) {
        assert (topo.size()==2);
        layerName="AffineRelu";
        layerType=LayerType::LT_NORMAL;
        af=new Affine({topo[0],topo[1]});
        mlPush("af", &(af->params), &params);
        rl=new Relu({topo[1]});
        mlPush("re", &(rl->params), &params);
    }
    ~AffineRelu() {
        delete af;
        af=nullptr;
        delete rl;
        rl=nullptr;
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        t_cppl tcacheaf;
        MatrixN y0=af->forward(x, &tcacheaf);
        mlPush("af", &tcacheaf, pcache);
        t_cppl tcachere;
        MatrixN y=rl->forward(y0, &tcachere);
        mlPush("re", &tcachere, pcache);
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl *pcache, t_cppl *pgrads) override {
        t_cppl tcachere;
        t_cppl tgradsre;
        mlPop("re",pcache,&tcachere);
        MatrixN dx0=rl->backward(dchain, &tcachere, &tgradsre);
        mlPush("re",&tgradsre,pgrads);
        t_cppl tcacheaf;
        t_cppl tgradsaf;
        mlPop("af",pcache,&tcacheaf);
        MatrixN dx=af->backward(dx0, &tcacheaf, &tgradsaf);
        mlPush("af",&tgradsaf,pgrads);
        return dx;
    }
    virtual bool update(Optimizer *popti, t_cppl *pgrads, string var, t_cppl *pocache) override {
        t_cppl tgradsaf;
        mlPop("af",pgrads,&tgradsaf);
        af->update(popti, &tgradsaf, var+"afre1-", pocache); // XXX push/pop for pocache?
        t_cppl tgradsre;
        mlPop("re",pgrads,&tgradsre);
        rl->update(popti, &tgradsre, var+"afre2-", pocache);
        return true;
    }
};


// Multiclass support vector machine Svm
class Svm : public Layer {
public:
    Svm(t_layer_topo topo) {
        assert (topo.size()==1);
        layerName="Svm";
        layerType=LayerType::LT_LOSS;
    }
    ~Svm() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache) override {
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
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads) override {
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

class Softmax : public Layer {
public:
    Softmax(t_layer_topo topo) {
        assert (topo.size()==1);
        layerName="Softmax";
        layerType=LayerType::LT_LOSS;
    }
    ~Softmax() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        VectorN mxc = x.rowwise().maxCoeff();
        MatrixN xn = x;
        xn.colwise() -=  mxc;
        MatrixN xne = xn.array().exp().matrix();
        VectorN xnes = xne.rowwise().sum();
        for (unsigned int i=0; i<xne.rows(); i++) { // XXX broadcasting?
            xne.row(i) = xne.row(i) / xnes(i);
        }
        MatrixN probs = xne;
        if (pcache!=nullptr) cppl_set(pcache, "probs", new MatrixN(probs));
        return probs;
    }
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) override {
        MatrixN probs=*((*pcache)["probs"]);
        if (y.rows() != probs.rows() || y.cols() != 1) {
            cout << layerName << ": "  << "Loss, dimension mismatch in Softmax(x), Probs: ";
            cout << shape(probs) << " y:" << shape(y) << " y.cols=" << y.cols() << "(should be 1)" << endl;
            return 1000.0;
        }
        //if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        floatN loss=0.0;
        for (unsigned int i=0; i<probs.rows(); i++) {
            if (y(i,0)>=probs.cols()) {
                cout << "internal error: y(" << i << ",0) >= " << probs.cols() << endl;
                return -10000.0;
            }
            floatN pi = probs(i,y(i,0));
            if (pi==0.0) cout << "Invalid zero log-probability at " << i << endl;
            else loss -= log(pi);
        }
        loss /= probs.rows();
        return loss;
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads) override {
        MatrixN probs=*((*pcache)["probs"]);

        MatrixN dx=probs;
        for (unsigned int i=0; i<probs.rows(); i++) {
            dx(i,y(i,0)) -= 1.0;
        }
        dx /= dx.rows();
        return dx;
    }
};


class TwoLayerNet : public Layer {
public:
    Affine *af1;
    Relu *rl;
    Affine *af2;
    Softmax *sm;
    TwoLayerNet(t_layer_topo topo) {
        assert (topo.size()==3);
        layerName="TwoLayerNet";
        layerType=LayerType::LT_LOSS;
        af1=new Affine({topo[0],topo[1]});
        mlPush("af1", &(af1->params), &params);
        rl=new Relu({topo[1]});
        mlPush("rl", &(rl->params), &params);
        af2=new Affine({topo[1],topo[2]});
        mlPush("af2", &(af2->params), &params);
        sm=new Softmax({topo[2]});
        mlPush("sm", &(sm->params), &params);
    }
    ~TwoLayerNet() {
        delete af1;
        af1=nullptr;
        delete rl;
        rl=nullptr;
        delete af2;
        af2=nullptr;
        delete sm;
        sm=nullptr;
    }
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        t_cppl c1;
        MatrixN y0=af1->forward(x,&c1);
        mlPush("af1",&c1,pcache);
        t_cppl c2;
        MatrixN y1=rl->forward(y0,&c2);
        mlPush("rl",&c2,pcache);
        t_cppl c3;
        MatrixN yo=af2->forward(y1,&c3);
        mlPush("af2",&c3,pcache);
        t_cppl c4;
        MatrixN yu=sm->forward(yo,y,&c4);
        mlPush("sm",&c4,pcache);
        return yo;
    }
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) override {
        t_cppl c4;
        mlPop("sm",pcache,&c4);
        return sm->loss(y, &c4);
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads) override {
        t_cppl c4;
        t_cppl g4;
        mlPop("sm",pcache,&c4);
        MatrixN dx3=sm->backward(y, &c4, &g4);
        mlPush("sm",&g4,pgrads);

        t_cppl c3;
        t_cppl g3;
        mlPop("af2",pcache,&c3);
        MatrixN dx2=af2->backward(dx3,&c3,&g3);
        mlPush("af2", &g3, pgrads);

        t_cppl c2;
        t_cppl g2;
        mlPop("rl",pcache,&c2);
        MatrixN dx1=rl->backward(dx2, &c2, &g2);
        mlPush("rl", &g2, pgrads);

        t_cppl c1;
        t_cppl g1;
        mlPop("af1",pcache,&c1);
        MatrixN dx=af1->backward(dx1, &c1, &g1);
        mlPush("af1", &g1, pgrads);

        return dx;
    }
    virtual bool update(Optimizer *popti, t_cppl *pgrads, string var, t_cppl *pocache) override {
        t_cppl g1;
        mlPop("af1",pgrads,&g1);
        af1->update(popti,&g1, var+"2l1", pocache); // XXX push/pop pocache?
        t_cppl g2;
        mlPop("rl",pgrads,&g2);
        rl->update(popti,&g2, var+"2l2", pocache);
        t_cppl g3;
        mlPop("af2",pgrads,&g3);
        af2->update(popti,&g3, var+"2l3", pocache);
        t_cppl g4;
        mlPop("sm",pgrads,&g4);
        sm->update(popti,&g4, var+"2l4", pocache);
        return true;
    }

};


void registerLayers() {
    REGISTER_LAYER("Affine", Affine, 2)
    REGISTER_LAYER("Relu", Relu, 1)
    REGISTER_LAYER("AffineRelu", AffineRelu, 2)
    REGISTER_LAYER("Softmax", Softmax, 1)
    REGISTER_LAYER("Svm", Svm, 1)
    REGISTER_LAYER("TwoLayerNet", TwoLayerNet, 3)
}
#endif
