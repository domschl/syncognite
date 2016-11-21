#ifndef _CPL_AFFINERELU_H
#define _CPL_AFFINERELU_H

#include "../cp-layers.h"

class AffineRelu : public Layer {
private:
    int hidden;
    void setup(const CpParams& cx) {
        layerName="AffineRelu";
        layerType=LayerType::LT_NORMAL;
        inputShapeRang=2;
        cp=cx;
        vector<int> inputShape=cp.getPar("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        hidden=cp.getPar("hidden",1024);
        XavierMode inittype=xavierInitType(cp.getPar("init",(string)"standard"));

        outputShape={hidden};
        CpParams ca;
        ca.setPar("inputShape", vector<int>{inputShapeFlat});
        ca.setPar("init",inittype);
        ca.setPar("hidden", hidden);
        af=new Affine(ca);
        mlPush("af", &(af->params), &params);
        CpParams cl;
        cl.setPar("inputShape", vector<int>{hidden});
        rl=new Relu(cl);
        mlPush("re", &(rl->params), &params);
        layerInit=true;
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
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        t_cppl tcacheaf;
        MatrixN y0=af->forward(x, &tcacheaf, id);
        mlPush("af", &tcacheaf, pcache);
        t_cppl tcachere;
        MatrixN y=rl->forward(y0, &tcachere, id);
        mlPush("re", &tcachere, pcache);
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl *pcache, t_cppl *pgrads, int id=0) override {
        t_cppl tcachere;
        t_cppl tgradsre;
        mlPop("re",pcache,&tcachere);
        MatrixN dx0=rl->backward(dchain, &tcachere, &tgradsre, id);
        mlPush("re",&tgradsre,pgrads);
        t_cppl tcacheaf;
        t_cppl tgradsaf;
        mlPop("af",pcache,&tcacheaf);
        MatrixN dx=af->backward(dx0, &tcacheaf, &tgradsaf, id);
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
