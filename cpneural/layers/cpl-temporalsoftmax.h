#ifndef _CPL_TEMPORALSOFTMAX_H
#define _CPL_TEMPORALSOFTMAX_H

#include "../cp-layers.h"

class  TemporalSoftmax : public Layer {
private:
    int T,D; //,V;
    void setup(const json& jx) {
        int allOk=true;
        layerName="TemporalSoftmax";
        inputShapeRang=2;
        layerType=LayerType::LT_LOSS;
        j=jx;
        vector<int> inputShape=j.value("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        D=inputShape[0]; // cp.getPar("D",128);
        T=inputShape[1]; // cp.getPar("T",128);
        //V=cp.getPar("V",10);
        outputShape={T};

        layerInit=allOk;
    }
public:
     TemporalSoftmax(const json& jx) {
        setup(jx);
    }
     TemporalSoftmax(const string conf) {
        setup(json::parse(conf));
    }
    ~ TemporalSoftmax() {
        cppl_delete(&params);
    }
    /*
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape ((N, T), V)  // XXX ? (NT - V) vs (N - TV)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    */
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
/*        if (pstates->find("y") == pstates->end()) {
            cerr << "TSM-fw: pstates does not contain y -> fatal!" << endl;
        }
        MatrixN y = *((*pstates)["y"]);
*/        if (x.cols() != D*T) {
            cerr << layerName << ": " << "Forward: dimension mismatch TemporalSoftmax in x(cols):" << x.cols() << " D*T:" << D*T << endl;
            MatrixN probs(0,0);
            return probs;
        }
        int N=x.rows();
        // x: [N, (T * D)] -> [(N * T), D]
        MatrixN xt=MatrixN(N*T, D);
        for (int n=0; n<N; n++) {
            for (int t=0; t<T; t++) {
                for (int d=0; d<D; d++) {
                    xt(n*T+t,d)=x(n,t*D+d);
                }
            }
        }
        if (xt.cols() != D) {
            cerr << layerName << ": " << "Forward: dimension mismatch TemporalSoftmax in xt(cols):" << xt.cols() << " D:" << D << endl;
            MatrixN probs(0,0);
            return probs;
        }
        if (N*T != xt.rows()) {
            cerr << layerName << ": " << "Forward: dimension mismatch TemporalSoftmax in xt(rows):" << xt.rows() << " N*T:" << N*T << endl;
            MatrixN probs(0,0);
            return probs;
        }
/*
        if (y.cols()!=T || y.rows()!=N) {
            cerr << layerName << ": " << "Forward: dimension mismatch TemporalSoftmax in y: " << shape(y) << " N,T:" << N << "," << T << endl;
            MatrixN probs(0,0);
            return probs;
        }

*/
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        if (pcache!=nullptr) cppl_set(pcache, "xt", new MatrixN(xt));
        //if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));

        VectorN mxc = xt.rowwise().maxCoeff();
        MatrixN xn = xt;
        xn.colwise() -=  mxc;
        MatrixN xne = xn.array().exp().matrix();
        VectorN xnes = xne.rowwise().sum();

        // Consistency checking, can be removed later
        if (xnes.size() != xne.rows()) {
            cerr << "Internal error when creating temporal softmax normalization" << endl;
        }
        // End checking

        for (unsigned int i=0; i<xne.rows(); i++) { // XXX broadcasting?
            xne.row(i) = xne.row(i) / xnes(i);
        }
        MatrixN probs = xne;

        // Consistency checking, can be removed later
        if (xnes.size() != xne.rows()) {
            cerr << "Internal error when creating temporal softmax normalization" << endl;
        }
        for (int n=0; n< probs.rows(); n++) {
            floatN sum=0.0;
            for (int j=0; j<probs.cols(); j++) {
                sum+=probs(n,j);
            }
            if (std::abs(sum-1.0)>1e-4) {
                cerr << "Un-normalized probablity detected in probs(" << n << ", " << ",:), sum=" << sum << endl;
            }
        }
        // End checking



        if (pcache!=nullptr) cppl_set(pcache, "probs", new MatrixN(probs));

        // probst: [(N * T), D] -> [N, (T * D)]
        MatrixN probst=MatrixN(N,T*D);
        for (int n=0; n<N; n++) {
            for (int t=0; t<T; t++) {
                for (int d=0; d<D; d++) {
                    probst(n,t*D+d)=probs(n*T+t,d);
                }
            }
        }
        return probst;
    }
    /*
    N, T, V = x.shape
    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)
    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]
    if verbose:
        print('dx_flat: ', dx_flat.shape)
    dx = dx_flat.reshape(N, T, V)
    return loss, dx
    */
    virtual floatN loss(t_cppl* pcache, t_cppl* pstates) override {
        if (pstates->find("y") == pstates->end()) {
            cerr << "TSM-loss: pstates does not contain y -> fatal!" << endl;
        }
        MatrixN y = *((*pstates)["y"]);
        MatrixN probs=*((*pcache)["probs"]);
        MatrixN mask;
        // XXX: int N=probs.rows()/T;

        int N=y.rows();

        if (pcache->find("mask")==pcache->end()) {
            mask=MatrixN(N,T);
            mask.setOnes();
        } else {
            mask=*((*pcache)["mask"]);
        }
        /*
            if (y.rows() != probs.rows() || y.cols() != 1) {
            cerr << layerName << ": "  << "Loss, dimension mismatch in Softmax(x), Probs: ";
            cerr << shape(probs) << " y:" << shape(y) << " y.cols=" << y.cols() << "(should be 1)" << endl;
            return 1000.0;
        }
        */
        //if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));

        floatN loss=0.0;
        for (int n=0; n<N; n++) {
            for (int t=0; t<T; t++) {
                floatN pi = probs(n*T+t,y(n,t));
                if (pi==0.0) {
                    cerr << "Invalid zero log-probability at n=" << n << "t=" << t << endl;
                    loss += 10000.0;
                }
                else {
                    // cerr << "[" << pi << "," << mask(n,t) << "]";
                    loss -= log(pi) * mask(n,t);
                }
            }
        }
        loss /= N; // probs.rows();
        return loss;
    }
    virtual MatrixN backward(const MatrixN& dy, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) override {
        MatrixN probs=*((*pcache)["probs"]);
        MatrixN mask;
        // int N=probs.rows()/T;
        if (pstates->find("y") == pstates->end()) {
            cerr << "TSM-bw: pstates does not contain y -> fatal!" << endl;
        }
        MatrixN y = *((*pstates)["y"]);
        int N=y.rows();
        if (pcache->find("mask")==pcache->end()) {
            mask=MatrixN(N,T);
            mask.setOnes();
        } else {
            mask=*((*pcache)["mask"]);
        }

        MatrixN dx(probs);

        for (int n=0; n<N; n++) {
            for (int t=0; t<T; t++) {
                dx(n*T+t,y(n,t)) -= 1.0;
            }
        }

        dx /= N;  // dx.rows();

        // dx: [(N * T), D)] -> [N, (T, D)]
        MatrixN dxr(N, T*D);
        for (int n=0; n<N; n++) {
            for (int t=0; t<T; t++) {
                for (int d=0; d<D; d++) {
                    dxr(n,t*D+d)=dx(n*T+t,d) * mask(n,t);
                }
            }
        }

        return dxr;
    }
};

#endif
