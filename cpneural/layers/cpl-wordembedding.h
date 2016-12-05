#ifndef _CPL_WORDEMBEDDING_H
#define _CPL_WORDEMBEDDING_H

#include "../cp-layers.h"

/* Word embeddings. We operate on minibatches of size N where
each sequence has length T. We assume a vocabulary of V words, assigning
each to a vector of dimension D.

Inputs:
- x: Integer array of shape (N, T) giving indices of words. Each element
  idx of x muxt be in the range 0 <= idx < V.
- W: Weight matrix of shape (V, D) giving word vectors for all words.

Returns a tuple of:
- out: Array of shape (N, T, D) giving word vectors for all input words.
- cache: Values needed for the backward pass
*/
class WordEmbedding : public Layer {
private:
    int numGpuThreads;
    int numCpuThreads;
    int T,D,V;
    MatrixN wVect;
    floatN initfactor;
    void setup(const CpParams& cx) {
        layerName="WordEmbedding";
        inputShapeRang=1;
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        vector<int> inputShape=cp.getPar("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        T=inputShape[0];
        V=cp.getPar("V",1024);
        D=cp.getPar("D",128);
        outputShape={D,T};
        XavierMode inittype=xavierInitType(cp.getPar("init",(string)"standard"));
        initfactor=cp.getPar("initfactor",(floatN)1.0);

        cppl_set(&params, "W", new MatrixN(xavierInit(MatrixN(V,D),inittype,initfactor)));
        wVect=MatrixN(V,V);
        wVect.setIdentity(); // Simple one-hot word vectors.
        // cppl_set(&params, "V", new MatrixN(wVect));   // XXX: we are implying to learn a better word vector repr. That was not done in CS231

        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();

        /*
        params["W"]->setRandom();
        floatN xavier = 1.0/std::sqrt((floatN)(inputShapeFlat+D)); // (setRandom is [-1,1]-> fakt 0.5, xavier is 2/(ni+no))
        *params["W"] *= xavier;
        */
        layerInit=true;
    }
public:
    WordEmbedding(const CpParams& cx) {
        setup(cx);
    }
    WordEmbedding(const string conf) {
        setup(CpParams(conf));
    }
    ~WordEmbedding() {
        cppl_delete(&params);
    }
    /*
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning
    each to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element
      idx of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    N, T = x.shape
    V, D = W.shape
    xoh = np.zeros((N, T, V))
    for ni in range(N):  # XXX vectors?
        for ti in range(T):
            xoh[ni, ti, x[ni, ti]] = 1.0
    xv = np.dot(xoh, W)
    out = xv
    cache = xoh
    return out, cache
    */
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id=0) override {
        int TT=cp.getPar("T-Steps",0);
        if (TT==0) TT=T;

        if (x.cols() != TT) {
            cerr << layerName << ": " << "Forward: dimension mismatch in x:" << shape(x) << " TT:" << TT << endl;
            MatrixN y(0,0);
            return y;
        }
        int N=shape(x)[0];
        MatrixN W(*params["W"]);
        // MatrixN wVect(*params["V"]);
        MatrixN xv(N*TT,V);
        for (int n=0; n<N; n++) {
            for (int t=0; t<TT; t++) {
                for (int vi=0; vi<V; vi++) {
                    xv(n*TT+t,vi)=wVect(x(n,t),vi);
                }
            }
        }
        if (pcache != nullptr) cppl_update(pcache, "xv", new MatrixN(xv));
        MatrixN y0 = xv * W;
        MatrixN y(N,D*TT);
        for (int n=0; n<N; n++) {
            for (int d=0; d<D; d++) {
                for (int t=0; t<TT; t++) {
                    y(n,d+t*D) = y0(n*TT+t,d);
                }
            }
        }
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        MatrixN dW(*params["W"]);
        dW.setZero();
        int N=shape(dchain)[0];
        MatrixN dx;
        MatrixN xv=*(*pcache)["xv"];
        // p2 = dout.reshape(-1, dout.shape[2])
        // p1 = xoh.reshape(-1, xoh.shape[2])
        //dW = np.dot(p1.T, p2)
        //cerr << "N:" << N << ", T:" << T << ", V:" << V << ", D:" << D << endl;
        //cerr << "dW:" << shape(dW) << ", xv:" << shape(xv) << ", dchain:" << shape(dchain) << endl;
        MatrixN dct(D,N*T);
        for (int n=0; n<N; n++) {
            for (int t=0; t<T; t++) {
                for (int d=0; d<D; d++) {
                    dct(d,n*T+t) = (dchain(n,t*D+d));
                }
            }
        }
        dW = (dct * xv).transpose();
        cppl_set(pgrads, "W", new MatrixN(dW));
        return dx; //Matrix of size(0,0): there is no dx, since x is discrete integers (word-indices)
    }
};

#endif
