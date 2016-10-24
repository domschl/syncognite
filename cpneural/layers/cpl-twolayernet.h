#ifndef _CPL_TWOLAYERNET_H
#define _CPL_TWOLAYERNET_H

#include "../cp-layers.h"


class TwoLayerNet : public Layer {
private:
    vector<int> hidden;
    void setup(const CpParams& cx) {
        bool retval=true;
        layerName="TwoLayerNet";
        layerType=LayerType::LT_LOSS;
        inputShapeRang=1;
        cp=cx;
        vector<int> inputShape=cp.getPar("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        hidden=cp.getPar("hidden",vector<int>{1024,1024});

        if (hidden.size()!=2) retval=false;

        outputShape={hidden[1]};

        CpParams c1,c2,c3,c4;
        c1.setPar("inputShape",vector<int>{inputShapeFlat});
        c1.setPar("hidden",hidden[0]);
        c2.setPar("inputShape",vector<int>{hidden[0]});
        c3.setPar("inputShape",vector<int>{hidden[0]});
        c3.setPar("hidden",hidden[1]);
        c4.setPar("inputShape",vector<int>{hidden[1]});
        af1=new Affine(c1);
        mlPush("af1", &(af1->params), &params);
        rl=new Relu(c2);
        mlPush("rl", &(rl->params), &params);
        af2=new Affine(c3);
        mlPush("af2", &(af2->params), &params);
        sm=new Softmax(c4);
        mlPush("sm", &(sm->params), &params);
        layerInit=retval;
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
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        t_cppl c1;
        MatrixN y0=af1->forward(x,&c1, id);
        mlPush("af1",&c1,pcache);
        t_cppl c2;
        MatrixN y1=rl->forward(y0,&c2, id);
        mlPush("rl",&c2,pcache);
        t_cppl c3;
        MatrixN yo=af2->forward(y1,&c3, id);
        mlPush("af2",&c3,pcache);
        t_cppl c4;
        MatrixN yu=sm->forward(yo,y,&c4, id);
        mlPush("sm",&c4,pcache);
        return yo;
    }
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) override {
        t_cppl c4;
        mlPop("sm",pcache,&c4);
        return sm->loss(y, &c4);
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        t_cppl c4;
        t_cppl g4;
        mlPop("sm",pcache,&c4);
        MatrixN dx3=sm->backward(y, &c4, &g4, id);
        mlPush("sm",&g4,pgrads);

        t_cppl c3;
        t_cppl g3;
        mlPop("af2",pcache,&c3);
        MatrixN dx2=af2->backward(dx3,&c3,&g3, id);
        mlPush("af2", &g3, pgrads);

        t_cppl c2;
        t_cppl g2;
        mlPop("rl",pcache,&c2);
        MatrixN dx1=rl->backward(dx2, &c2, &g2, id);
        mlPush("rl", &g2, pgrads);

        t_cppl c1;
        t_cppl g1;
        mlPop("af1",pcache,&c1);
        MatrixN dx=af1->backward(dx1, &c1, &g1, id);
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

#endif
