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
private:
    void setup(const CpParams& cx) {
        layerName="Affine";
        topoParams=2;
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        vector<int> topo=cp.getPar("topo",vector<int>{0});
        assert (topo.size()==2);
        //cout << "CrAff:" << topo[0] <<"/" << topo[1] << endl;
        cppl_set(&params, "W", new MatrixN(topo[0],topo[1])); // W
        cppl_set(&params, "b", new MatrixN(1,topo[1])); // b

        params["W"]->setRandom();
        floatN xavier = 1.0/(floatN)(topo[0]+topo[1]); // (setRandom is [-1,1]-> fakt 0.5, xavier is 2/(ni+no))
        *params["W"] *= xavier;
        params["b"]->setRandom();
        *params["b"] *= xavier;
    }
public:
    Affine(const CpParams& cx) {
        setup(cx);
    }
    Affine(const string conf) {
        setup(CpParams(conf));
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
private:
    void setup(const CpParams& cx) {
        layerName="Relu";
        layerType=LayerType::LT_NORMAL;
        topoParams=1;
        cp=cx;
    }
public:
    Relu(const CpParams& cx) {
        setup(cx);
    }
    Relu(string conf) {
        setup(CpParams(conf));
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
private:
    void setup(const CpParams& cx) {
        layerName="AffineRelu";
        layerType=LayerType::LT_NORMAL;
        topoParams=2;
        cp=cx;
        vector<int> topo=cp.getPar("topo", vector<int>{0});
        assert (topo.size()==3);
        //cout << "CrAffRelu:" << topo[0] <<"/" << topo[1] << endl;
        CpParams ca;
        ca.setPar("topo", vector<int>{topo[0],topo[1]});
        af=new Affine(ca);
        mlPush("af", &(af->params), &params);
        CpParams cl;
        cl.setPar("topo", vector<int>{topo[1]});
        rl=new Relu(cl);
        mlPush("re", &(rl->params), &params);
    }
public:
    Affine *af;
    Relu *rl;
    AffineRelu(const CpParams& cx) {
        setup(cx);
    }
    AffineRelu(string conf) {
        setup(CpParams(conf));
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


// Batch normalization
class BatchNorm : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="BatchNorm";
        layerType=LayerType::LT_NORMAL;
        topoParams=1;
        cp=cx;
        eps = cp.getPar("eps", 1e-5);
        momentum = cp.getPar("momentum", 0.9);
        trainMode = cp.getPar("train", false);
        vector<int> topo=cp.getPar("topo", vector<int>{0});

        MatrixN *pgamma=new MatrixN(1,topo[0]);
        pgamma->setOnes();
        cppl_set(&params,"gamma",pgamma);
        MatrixN *pbeta=new MatrixN(1,topo[0]);
        pbeta->setZero();
        cppl_set(&params,"beta",pbeta);
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
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache) override {
        MatrixN *prm, *prv;
        MatrixN *pbeta, *pgamma;
        MatrixN xout;
        trainMode = cp.getPar("train", false);
        if (pcache==nullptr || pcache->find("running_mean")==pcache->end()) {
            prm=new MatrixN(1,shape(x)[1]);
            prm->setZero();
            if (pcache!=nullptr) cppl_set(pcache,"running_mean",prm);
            else delete prm;
        } else {
            prm=(*pcache)["running_mean"];
        }
        if (pcache==nullptr || pcache->find("running_var")==pcache->end()) {
            prv=new MatrixN(1,shape(x)[1]);
            prv->setZero();
            if (pcache != nullptr) cppl_set(pcache,"running_var",prv);
            else delete prv;
        } else {
            prv=(*pcache)["running_var"];
        }
        pgamma=params["gamma"];
        pbeta=params["beta"];
        if (trainMode) {
            int N=shape(x)[0];
            RowVectorN xm=x.colwise().sum()/(floatN)N;
            MatrixN xme=x.rowwise() - xm;
            MatrixN sqse=(xme.array()*xme.array()).colwise().sum()/(floatN)N + eps;
            RowVectorN v=sqse.array().sqrt();
            RowVectorN v2=v.array().pow(-1);
            MatrixN x2 = xme.array().rowwise() * v2.array();
            xout = x2.array().rowwise() * (RowVectorN(*pgamma).row(0)).array();
            xout.rowwise() += (*pbeta).row(0);

            if (pcache != nullptr) {
                cppl_update(pcache,"sqse",&sqse);
                cppl_update(pcache,"xme",&xme);
                cppl_update(pcache, "x2", &x2);

                *(*pcache)["running_mean"] = *((*pcache)["running_mean"]) * momentum + xm * (1.0-momentum);
                *(*pcache)["running_var"]  = (*((*pcache)["running_var"]) * momentum).rowwise() + v * (1.0-momentum);
            }
        } else {
            MatrixN xot = x.rowwise() - (*(*pcache)["running_mean"]).row(0);
            MatrixN xot2 = xot.array().rowwise() / (RowVectorN(*(*pcache)["running_var"])).array();
            MatrixN xot3 = xot2.array().rowwise() * (RowVectorN((*pgamma).row(0)).array());
            xout = xot3.rowwise()+RowVectorN((*pbeta).row(0));
            //(x - running_mean)/running_var*gamma+beta;
        }
        return xout;
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads) override {
        if (pcache->find("sqse")==pcache->end()) cout << "Bad: no cache entry for sqse!" << endl;
        MatrixN sqse=*((*pcache)["sqse"]);
        if (pcache->find("xme")==pcache->end()) cout << "Bad: no cache entry for xme!" << endl;
        MatrixN xme=*((*pcache)["xme"]);
        if (pcache->find("x2")==pcache->end()) cout << "Bad: no cache entry for x2!" << endl;
        MatrixN x2=*((*pcache)["x2"]);
        if (params.find("gamma")==params.end()) cout << "Bad: no params entry for gamma!" << endl;
        MatrixN gamma=*(params["gamma"]);
        MatrixN dbeta=y.colwise().sum();

        MatrixN dgamma=(x2.array() * y.array()).colwise().sum();
        if (shape(gamma) != shape(dgamma)) cout << "bad: dgamma has wrong shape: " << shape(gamma) << shape(dgamma) << endl;

        floatN N=y.rows();
        MatrixN d1=MatrixN(y).setOnes();
        MatrixN dx0 = gamma.array() * (sqse.array().pow(-0.5)) / N;
        MatrixN dx1 = (y*N).rowwise() - RowVectorN(y.colwise().sum());
        MatrixN iv = sqse.array().pow(-1.0);
        MatrixN dx21 = (xme.array().rowwise() * RowVectorN(iv).array());
        MatrixN dx22 = (y.array() * xme.array()).colwise().sum();
        MatrixN dx2 = dx21.array().rowwise() * RowVectorN(dx22).array();
        MatrixN dx=(dx1 - dx2).array().rowwise() * RowVectorN(d1*dx0).array() ;
        cppl_set(pgrads,"gamma",new MatrixN(dgamma));
        cppl_set(pgrads,"beta",new MatrixN(dbeta));

        return dx;
    }

};


// Dropout: with probability neuronDropProb, a neuron's value is dropped. (1-p)?
class Dropout : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="Dropout";
        layerType=LayerType::LT_NORMAL;
        topoParams=1;
        cp=cx;
        neuronDropProb = cp.getPar("neuronDropProb", 0.5);
        trainMode = cp.getPar("train", false);
        freeze = cp.getPar("freeze", false);
        vector<int> topo=cp.getPar("topo", vector<int>{0});
    }
public:
    floatN neuronDropProb;
    bool trainMode;
    bool freeze;

    Dropout(const CpParams& cx) {
        setup(cx);
    }
    Dropout(string conf) {
        setup(CpParams(conf));
    }
    ~Dropout() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache) override {
        trainMode = cp.getPar("train", false);
        neuronDropProb = cp.getPar("neuronDropProb", 0.5);
        freeze = cp.getPar("freeze", false);
        MatrixN xout;
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        if (trainMode) {
            MatrixN* pmask;
            freeze = cp.getPar("freeze", false);
            if (freeze && pcache!=nullptr && pcache->find("dropmask")!=pcache->end()) {
                pmask=(*pcache)["dropmask"];
                xout = x.array() * (*pmask).array();
            } else {
                pmask=new MatrixN(x);
                if (freeze) srand(123);
                pmask->setRandom();
                for (int i=0; i<x.size(); i++) {
                    if (((*pmask)(i)+1.0)/2.0 < neuronDropProb) (*pmask)(i)=1.0;
                    else (*pmask)(i)=0.0;
                }
                xout = x.array() * (*pmask).array();
                if (pcache!=nullptr) cppl_set(pcache, "dropmask", pmask);
                else delete pmask;
            }
            //cout << "drop:" << neuronDropProb << endl << xout << endl;
        } else {
            xout = x * neuronDropProb;
        }
        return xout;
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads) override {
        MatrixN dx;
        trainMode = cp.getPar("train", false);
        if (trainMode) {
            MatrixN* pmask=(*pcache)["dropmask"];
            dx=y.array() * (*pmask).array();
        } else {
            dx=y;
        }
        return dx;
    }
};


// Multiclass support vector machine Svm
class Svm : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="Svm";
        layerType=LayerType::LT_LOSS;
        topoParams=1;
        cp=cx;
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
private:
    void setup(const CpParams& cx) {
        layerName="Softmax";
        layerType=LayerType::LT_LOSS;
        topoParams=1;
        cp=cx;
    }
public:
    Softmax(const CpParams& cx) {
        setup(cx);
    }
    Softmax(string conf) {
        setup(CpParams(conf));
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
private:
    void setup(const CpParams& cx) {
        layerName="TwoLayerNet";
        layerType=LayerType::LT_LOSS;
        topoParams=3;
        cp=cx;
        vector<int> topo=cp.getPar("topo",vector<int>{0});
        CpParams c1,c2,c3,c4;
        c1.setPar("topo",vector<int>{topo[0],topo[1]});
        c2.setPar("topo",vector<int>{topo[1]});
        c3.setPar("topo",vector<int>{topo[1],topo[2]});
        c4.setPar("topo",vector<int>{topo[2]});
        af1=new Affine(c1);
        mlPush("af1", &(af1->params), &params);
        rl=new Relu(c2);
        mlPush("rl", &(rl->params), &params);
        af2=new Affine(c3);
        mlPush("af2", &(af2->params), &params);
        sm=new Softmax(c4);
        mlPush("sm", &(sm->params), &params);
    }
public:
    Affine *af1;
    Relu *rl;
    Affine *af2;
    Softmax *sm;
    TwoLayerNet(CpParams cx) {
        setup(cx);
    }
    TwoLayerNet(string conf) {
        setup(CpParams(conf));
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
    REGISTER_LAYER("BatchNorm", BatchNorm, 1)
    REGISTER_LAYER("Dropout", Dropout, 1)
    REGISTER_LAYER("Softmax", Softmax, 1)
    REGISTER_LAYER("Svm", Svm, 1)
    REGISTER_LAYER("TwoLayerNet", TwoLayerNet, 3)
}

class MultiLayer : public Layer {
public:
    map<string, Layer*> layerMap;
    map<string, vector<string>> layerInputs;

    MultiLayer(const CpParams& cp) {
        layerName="MultiLayer";


/*
        vector<int> topo=cp.getPar("topo",vector<int>{0});
        CpParams c1,c2,c3,c4;
        c1.setPar("topo",vector<int>{topo[0],topo[1]});
        c2.setPar("topo",vector<int>{topo[1]});
        c3.setPar("topo",vector<int>{topo[1],topo[2]});
        c4.setPar("topo",vector<int>{topo[2]});
        af1=new Affine(c1);
        mlPush("af1", &(af1->params), &params);
        rl=new Relu(c2);
        mlPush("rl", &(rl->params), &params);
        af2=new Affine(c3);
        mlPush("af2", &(af2->params), &params);
        sm=new Softmax(c4);
        mlPush("sm", &(sm->params), &params);*/
    }
    MultiLayer(string conf) {
        MultiLayer(CpParams(conf));
    }
    ~MultiLayer() {
/*        delete af1;
        af1=nullptr;
        delete rl;
        rl=nullptr;
        delete af2;
        af2=nullptr;
        delete sm;
        sm=nullptr;*/
    }
    void addLayer(string name, Layer* layer, vector<string> inputLayers) {
        layerMap[name]=layer;
        layerInputs[name]=inputLayers;
    }
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache) override {
/*        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
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
        return yo;*/
    }
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) override {
/*        t_cppl c4;
        mlPop("sm",pcache,&c4);
        return sm->loss(y, &c4);*/
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads) override {
/*        t_cppl c4;
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

        return dx;*/
    }
    virtual bool update(Optimizer *popti, t_cppl *pgrads, string var, t_cppl *pocache) override {
/*        t_cppl g1;
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
        return true;*/
    }
};

#endif
