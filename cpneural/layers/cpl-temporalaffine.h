#ifndef _CPL_TEMPORALAFFINE_H
#define _CPL_TEMPORALAFFINE_H

#include "../cp-layers.h"

class  TemporalAffine : public Layer {
private:
    int T,D,M;
    floatN initfactor;
    void setup(const CpParams& cx) {
        int allOk=true;
        layerName="TemporalAffine";
        inputShapeRang=1;
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        vector<int> inputShape=cp.getPar("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        D=inputShape[0]; // cp.getPar("D",128);
        T=inputShape[1]; // cp.getPar("T",128);
        M=cp.getPar("M",128);
        XavierMode inittype=xavierInitType(cp.getPar("init",(string)"standard"));
        initfactor=cp.getPar("initfactor",(floatN)1.0);

        outputShape={M,T};

        cppl_set(&params, "W", new MatrixN(xavierInit(MatrixN(D,M),inittype,initfactor))); // W
        cppl_set(&params, "b", new MatrixN(xavierInit(MatrixN(1,M),inittype,initfactor))); // b

/*
        params["W"]->setRandom();
        floatN xavier = 1.0/std::sqrt((floatN)(inputShapeFlat+M)); // (setRandom is [-1,1]-> fakt 0.5, xavier is 2/(ni+no))
        *params["W"] *= xavier;
        params["b"]->setRandom();
        *params["b"] *= xavier;
        */
        layerInit=allOk;
    }
public:
     TemporalAffine(const CpParams& cx) {
        setup(cx);
    }
     TemporalAffine(const string conf) {
        setup(CpParams(conf));
    }
    ~ TemporalAffine() {
        cppl_delete(&params);
    }
    /*
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, {T * D})
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, {T * M})
    - cache: Values needed for the backward pass
    """
    */
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        if (x.cols() != T*D) {
            cerr << layerName << ": " << "Forward: dimension mismatch TemporalAFfine in x*W: x(cols):" << x.cols() << " T*D:" << T*D << endl;
            MatrixN y(0,0);
            return y;
        }
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        int N=x.rows();
        // x: [N, (T * D)] -> [(N * T), D]
        MatrixN xt(N*T, D);
        for (int n=0; n<N; n++) {
            for (int t=0; t<T; t++) {
                for (int d=0; d<D; d++) {
                    xt(n*T+t,d)=x(n,t*D+d);
                }
            }
        }
        if (pcache!=nullptr) cppl_set(pcache, "xt", new MatrixN(xt));
        MatrixN yt=(xt * (*params["W"])).rowwise() + RowVectorN(*params["b"]);

        MatrixN y(N,T*M);
        for (int n=0; n<N; n++) {
            for (int t=0; t<T; t++) {
                for (int m=0; m<M; m++) {
                    y(n,t*M+m)=yt(n*T+t,m);
                }
            }
        }

        return y;
    }
    /*
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    */
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) override {
        int N=dchain.rows();
        MatrixN dchaint(N*T,M);
        for (int n=0; n<N; n++) {
            for (int t=0; t<T; t++) {
                for (int m=0; m<M; m++) {
                    dchaint(n*T+t,m)=dchain(n,t*M+m);
                }
            }
        }

        MatrixN dxt = dchaint * (*params["W"]).transpose(); // dx
        cppl_set(pgrads, "W", new MatrixN((*(*pcache)["xt"]).transpose() * dchaint)); //dW
        cppl_set(pgrads, "b", new MatrixN(dchaint.colwise().sum())); //db
        MatrixN dx(N, T*D);
        for (int n=0; n<N; n++) {
            for (int t=0; t<T; t++) {
                for (int d=0; d<D; d++) {
                    dx(n,t*D+d)=dxt(n*T+t,d);
                }
            }
        }
        return dx;
    }
};

#endif
