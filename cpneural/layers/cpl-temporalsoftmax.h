#ifndef _CPL_TEMPORALSOFTMAX_H
#define _CPL_TEMPORALSOFTMAX_H

#include "../cp-layers.h"

class  TemporalSoftmax : public Layer {
private:
    int T,D,M;
    void setup(const CpParams& cx) {
        int allOk=true;
        layerName="TemporalSoftmax";
        inputShapeRang=1;
        layerType=LayerType::LT_LOSS;
        cp=cx;
        vector<int> inputShape=cp.getPar("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        D=inputShape[0]; // cp.getPar("D",128);
        T=inputShape[1]; // cp.getPar("T",128);

        outputShape={T};

        layerInit=allOk;
    }
public:
     TemporalSoftmax(const CpParams& cx) {
        setup(cx);
    }
     TemporalSoftmax(const string conf) {
        setup(CpParams(conf));
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
    - x: Input scores, of shape (N, (T, V))
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    */
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache, int id=0) override {
        if (x.cols() != T*D) {
            cerr << layerName << ": " << "Forward: dimension mismatch TemporalAFfine in x*W: x(cols):" << x.cols() << " T*D:" << T*D << endl;
            MatrixN probs(0,0);
            return probs;
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

        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(xt));
        if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        VectorN mxc = xt.rowwise().maxCoeff();
        MatrixN xn = xt;
        xn.colwise() -=  mxc;
        MatrixN xne = xn.array().exp().matrix();
        VectorN xnes = xne.rowwise().sum();
        for (unsigned int i=0; i<xne.rows(); i++) { // XXX broadcasting?
            xne.row(i) = xne.row(i) / xnes(i);
        }
        //MatrixN probs = xne;
        //if (pcache!=nullptr) cppl_set(pcache, "probs", new MatrixN(probs));

        MatrixN probst=xne;

        MatrixN probs(N,T*D);
        for (int n=0; n<N; n++) {
            for (int d=0; d<D; d++) {
                for (int t=0; t<T; t++) {
                    probs(n,d*T+t)=probst(n*D+d,t);
                }
            }
        }
        if (pcache!=nullptr) cppl_set(pcache, "probs", new MatrixN(probs));
        return probs;
    }
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) override {
        MatrixN probs=*((*pcache)["probs"]);
/*        if (y.rows() != probs.rows() || y.cols() != 1) {
            cerr << layerName << ": "  << "Loss, dimension mismatch in Softmax(x), Probs: ";
            cerr << shape(probs) << " y:" << shape(y) << " y.cols=" << y.cols() << "(should be 1)" << endl;
            return 1000.0;
        }
*/        //if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        floatN loss=0.0;
        for (unsigned int i=0; i<probs.rows(); i++) {
            if (y(i,0)>=probs.cols()) {
                cerr << "internal error: y(" << i << ",0) >= " << probs.cols() << endl;
                return -10000.0;
            }
            floatN pi = probs(i,y(i,0));
            if (pi==0.0) cerr << "Invalid zero log-probability at " << i << endl;
            else loss -= log(pi);
        }
        loss /= probs.rows();
        return loss;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
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
