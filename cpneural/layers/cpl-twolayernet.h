#ifndef _CPL_TWOLAYERNET_H
#define _CPL_TWOLAYERNET_H

#include "../cp-layers.h"


class TwoLayerNet : public Layer {
private:
    vector<int> hidden;
    floatN initfactor=0.1;
    void setup(const json& jx) {
        bool retval=true;
        layerName="TwoLayerNet";
        layerClassName="TwoLayerNet";
        layerType=LayerType::LT_LOSS;
        inputShapeRang=1;
        j=jx;
        vector<int> inputShape=j.value("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        hidden=j.value("hidden",vector<int>{1024,1024});
        string inittype=j.value("init",(string)"standard");
        initfactor=j.value("initfactor", initfactor);

        if (hidden.size()!=2) {
            retval=false;
            cerr << "TwoLayer: bad hidden-vector init! size=" << hidden.size() << "!=2" << endl;
        }
        outputShape={1};

        json j1,j2,j3,j4;
        j1["inputShape"]=vector<int>{inputShapeFlat};
        j1["hidden"]=hidden[0];
        j1["init"]=inittype;
        j1["initfactor"]=initfactor;
        j1["name"]="af1";
        j2["inputShape"]=vector<int>{hidden[0]};
        j3["inputShape"]=vector<int>{hidden[0]};
        j3["hidden"]=hidden[1];
        j3["init"]=inittype;
        j3["initfactor"]=initfactor;
        j3["name"]="af2";
        j4["inputShape"]=vector<int>{hidden[1]};
        af1=new Affine(j1);
        mlPush("af1", &(af1->params), &params);
        rl=new Relu(j2);
        mlPush("rl", &(rl->params), &params);
        af2=new Affine(j3);
        mlPush("af2", &(af2->params), &params);
        sm=new Softmax(j4);
        mlPush("sm", &(sm->params), &params);
        layerInit=retval;
    }
public:
    Affine *af1;
    Relu *rl;
    Affine *af2;
    Softmax *sm;
    TwoLayerNet(const json& jx) {
        setup(jx);
    }
    TwoLayerNet(const string conf) {
        setup(json::parse(conf));
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
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        if (pstates->find("y") == pstates->end()) {
            cerr << "TLN-fw: pstates does not contain y -> fatal!" << endl;
            exit(-1);
        }
        MatrixN y(*((*pstates)["y"]));
        //if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        //if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        t_cppl c1;
        MatrixN y0=af1->forward(x,&c1, pstates, id);
        // cerr << "x:" << x << endl << "W:" << *((af1->params)["W"]) << endl;
        // cerr << "b:" << *((af1->params)["b"]) << endl << "y0:" << y0 << endl;
        mlPush("af1",&c1,pcache);
        t_cppl c2;
        MatrixN y1=rl->forward(y0,&c2, pstates, id);
        mlPush("rl",&c2,pcache);
        t_cppl c3;
        MatrixN yo=af2->forward(y1,&c3, pstates, id);
        mlPush("af2",&c3,pcache);
        t_cppl c4;
        MatrixN yu=sm->forward(yo,&c4, pstates, id);
        mlPush("sm",&c4,pcache);
        return yo;
    }
    virtual floatN loss(t_cppl* pcache, t_cppl* pstates) override {
        if (pstates->find("y") == pstates->end()) {
            cerr << endl << endl << "TLN-loss: pstates does not contain y -> fatal!" << endl << endl;
            exit(-1);
        }
        t_cppl c4;
        mlPop("sm",pcache,&c4);
        return sm->loss(&c4, pstates);
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) override {
        t_cppl c4;
        t_cppl g4;
        mlPop("sm",pcache,&c4);
        MatrixN dx3=sm->backward(y, &c4, pstates, &g4, id);
        mlPush("sm",&g4,pgrads);

        t_cppl c3;
        t_cppl g3;
        mlPop("af2",pcache,&c3);
        MatrixN dx2=af2->backward(dx3,&c3,pstates, &g3, id);
        mlPush("af2", &g3, pgrads);

        t_cppl c2;
        t_cppl g2;
        mlPop("rl",pcache,&c2);
        MatrixN dx1=rl->backward(dx2, &c2, pstates, &g2, id);
        mlPush("rl", &g2, pgrads);

        t_cppl c1;
        t_cppl g1;
        mlPop("af1",pcache,&c1);
        MatrixN dx=af1->backward(dx1, &c1, pstates, &g1, id);
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
    virtual bool saveParameters(H5::H5File* pfile) override {
        if (!af1->saveParameters(pfile)) return false;
        return af2->saveParameters(pfile);
    }
    virtual bool loadParameters(H5::H5File* pfile) override {
        if (!af1->loadParameters(pfile)) return false;
        return af2->loadParameters(pfile);
    }

};

#endif
