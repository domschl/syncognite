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
        outputShape={T*D};

        cppl_set(&params, "W", new MatrixN(V,D));
        wVect=MatrixN(V,V);
        wVect.setIdentity(); // Simple one-hot word vectors.
        // cppl_set(&params, "V", new MatrixN(wVect));   // XXX: we are implying to learn a better word vector repr. That was not done in CS231

        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();

        params["W"]->setRandom();
        floatN xavier = 1.0/std::sqrt((floatN)(inputShapeFlat+D)); // (setRandom is [-1,1]-> fakt 0.5, xavier is 2/(ni+no))
        *params["W"] *= xavier;
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
        if (x.cols() != T) {
            cerr << layerName << ": " << "Forward: dimension mismatch in x:" << shape(x) << " T:" << T << endl;
            MatrixN y(0,0);
            return y;
        }
        int N=shape(x)[0];
        MatrixN W(*params["W"]);
        // MatrixN wVect(*params["V"]);
        MatrixN xv(N*T,V);
        for (int n=0; n<N; n++) {
            for (int t=0; t<T; t++) {
                for (int vi=0; vi<V; vi++)
                xv(n*T+t,vi)=wVect(x(n,t),vi);
            }
        }
        MatrixN y;

        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        MatrixN dWxh,dWhh,dbh;
        string name;
        int N=shape(dchain)[0];
        MatrixN dx(N,T*D);
        dx.setZero();
        return dx;
    }
};

#endif
