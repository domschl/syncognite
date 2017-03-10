#ifndef _CPL_AFFINERELU_H
#define _CPL_AFFINERELU_H

#include "../cp-layers.h"

class AffineRelu : public Layer {
private:
    int hidden;
    floatN initfactor;
    void setup(const json& jx) {
        layerName="AffineRelu";
        layerClassName="AffineRelu";
        layerType=LayerType::LT_NORMAL;
        inputShapeRang=2;
        j=jx;
        vector<int> inputShape=j.value("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        hidden=j.value("hidden",1024);
        string init=j.value("init",(string)"standard");
        //XavierMode inittype=xavierInitType(init);
        initfactor=j.value("initfactor",(floatN)1.0);

        outputShape={hidden};
        json ja;
        ja["inputShape"]=vector<int>{inputShapeFlat};
        ja["init"]=init;
        ja["initfactor"]=initfactor;
        ja["hidden"]=hidden;
        af=new Affine(ja);
        mlPush("af", &(af->params), &params);
        json jl;
        jl["inputShape"]=vector<int>{hidden};
        rl=new Relu(jl);
        mlPush("re", &(rl->params), &params);
        layerInit=true;
    }
public:
    Affine *af;
    Relu *rl;
    AffineRelu(const json& jx) {
        setup(jx);
    }
    AffineRelu(const string conf) {
        setup(json::parse(conf));
    }
    ~AffineRelu() {
        delete af;
        af=nullptr;
        delete rl;
        rl=nullptr;
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        t_cppl tcacheaf;
        MatrixN y0=af->forward(x, &tcacheaf, nullptr, id);
        mlPush("af", &tcacheaf, pcache);
        t_cppl tcachere;
        MatrixN y=rl->forward(y0, &tcachere, nullptr, id);
        mlPush("re", &tcachere, pcache);
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl *pcache, t_cppl* pstates, t_cppl *pgrads, int id=0) override {
        t_cppl tcachere;
        t_cppl tgradsre;
        mlPop("re",pcache,&tcachere);
        MatrixN dx0=rl->backward(dchain, &tcachere, nullptr, &tgradsre, id);
        mlPush("re",&tgradsre,pgrads);
        t_cppl tcacheaf;
        t_cppl tgradsaf;
        mlPop("af",pcache,&tcacheaf);
        MatrixN dx=af->backward(dx0, &tcacheaf, nullptr, &tgradsaf, id);
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

#endif
