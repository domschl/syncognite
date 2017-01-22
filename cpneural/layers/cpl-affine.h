#ifndef _CPL_AFFINE_H
#define _CPL_AFFINE_H

#include "../cp-layers.h"

class Affine : public Layer {
private:
    int numGpuThreads;
    int numCpuThreads;
    int hidden;
    floatN initfactor;
    void setup(const CpParams& cx) {
        layerName="Affine";
        inputShapeRang=1;
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        vector<int> inputShape=cp.getPar("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        hidden=cp.getPar("hidden",1024);
        XavierMode inittype=xavierInitType(cp.getPar("init",(string)"standard"));
        initfactor=cp.getPar("initfactor",(floatN)1.0);
        outputShape={hidden};

        MatrixN W = xavierInit(MatrixN(inputShapeFlat,hidden),inittype,initfactor);
        MatrixN b = xavierInit(MatrixN(1,hidden),inittype,initfactor);
        cppl_set(&params, "W", new MatrixN(W)); // W
        cppl_set(&params, "b", new MatrixN(b)); // b
        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();

        layerInit=true;
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
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        if (params["W"]->rows() != x.cols()) {
            cerr << layerName << ": " << "Forward: dimension mismatch in x*W: x:" << shape(x) << " W:" << shape(*params["W"]) << endl;
            MatrixN y(0,0);
            return y;
        }
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));

        #ifdef USE_GPU
        int algo=1;
        #else
        int algo=0;
        #endif
        MatrixN y;
        if (algo==0 || id>=numGpuThreads) {
            y=(x * (*(params["W"]))).rowwise() + RowVectorN(*params["b"]);
        } else {
            #ifdef USE_GPU
            MatrixN x1(x.rows(),x.cols()+1);
            MatrixN xp1(x.rows(),1);
            xp1.setOnes();
            x1 << x, xp1;
            MatrixN Wb((*params["W"]).rows()+1,(*params["W"]).cols());
            Wb<<*params["W"], *params["b"];
            MatrixN y2;
            y=matmul(&x1,&Wb,id);
            #endif
        }
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) override {
        #ifdef USE_GPU
        int algo=1;
        #else
        int algo=0;
        #endif
        MatrixN x(*(*pcache)["x"]);
        MatrixN dx(x.rows(),x.cols());
        MatrixN W(*params["W"]);
        MatrixN dW(W.rows(),W.cols());
        if (algo==0 || id>=numGpuThreads) {
            dx = dchain * (*params["W"]).transpose(); // dx
            cppl_set(pgrads, "W", new MatrixN((*(*pcache)["x"]).transpose() * dchain)); //dW
            cppl_set(pgrads, "b", new MatrixN(dchain.colwise().sum())); //db
        } else {
            #ifdef USE_GPU
            MatrixN Wt;
            Wt=W.transpose();
            MatrixN xt;
            xt=x.transpose();
            MatrixN dc=dchain;
            dx=matmul(&dc,&Wt,id);
            cppl_set(pgrads, "W", new MatrixN(matmul(&xt,&dc,id)));
            cppl_set(pgrads, "b", new MatrixN(dchain.colwise().sum())); //db
            #endif
        }
        return dx;
    }
};

#endif
