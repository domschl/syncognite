#ifndef _CP_LAYERS_H
#define _CP_LAYERS_H

#include "cp-layer.h"

class Nonlinearities {
public:
    MatrixN sigmoid(MatrixN& m) {
        ArrayN ar = m.array();
        return (1/(1+(ar.maxCoeff() - ar).exp())).matrix();
    }
    MatrixN tanh(MatrixN& m) {
        ArrayN ar = m.array();
        return (ar.tanh()).matrix();
    }
    MatrixN Relu(MatrixN& m) {
        ArrayN ar = m.array();
        return (ar.max(0)).matrix();
    }
};

class Affine : public Layer {
public:
    Affine(t_layer_topo topo) {
        assert (topo.size()==2);
        layerName="Affine";
        layerType=LayerType::LT_NORMAL;
        cppl_set(&params, "W", new MatrixN(topo[0],topo[1])); // W
        cppl_set(&params, "b", new MatrixN(1,topo[1])); // b

        params["W"]->setRandom();
        floatN xavier = 1.0/(floatN)(topo[0]+topo[1]); // (setRandom is [-1,1]-> fakt 0.5, xavier is 2/(ni+no))
        *params["W"] *= xavier;
        params["b"]->setRandom();
        *params["b"] *= xavier;
    }
    ~Affine() {
        cppl_delete(params);
    }
    virtual MatrixN forward(MatrixN& x, t_cppl* pcache) override {
        if (params["W"]->rows() != x.cols()) {
            cout << layerName << ": " << "Forward: dimension mismatch in x*W: x:" << shape(x) << " W:"<< shape(*params["W"]) << endl;
            MatrixN y(0,0);
            return y;
        }
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        MatrixN y = x * (*params["W"]);
        RowVectorN b = *params["b"];
        y.rowwise() += b;
        return y;
    }
    virtual MatrixN backward(MatrixN& dchain, t_cppl* pcache, t_cppl* pgrads) override {
        MatrixN dx = dchain * (*params["W"]).transpose(); // dx
        cppl_set(pgrads, "W", new MatrixN((*(*pcache)["x"]).transpose() * dchain)); //dW
        cppl_set(pgrads, "b", new MatrixN(dchain.colwise().sum())); //db
        return dx;
    }
};

class Relu : public Layer {
public:
    Relu(t_layer_topo topo) {
        assert (topo.size()==1);
        layerName="Relu";
        layerType=LayerType::LT_NORMAL;
    }
    ~Relu() {
        cppl_delete(params);
    }
    virtual MatrixN forward(MatrixN& x, t_cppl *pcache) override {
        if (pcache!=nullptr) (*pcache)["x"] = new MatrixN(x);
        MatrixN ych=x;
        for (unsigned int i=0; i<ych.size(); i++) {
            if (x(i)<0.0) {
                ych(i)=0.0;
            }
        }
        return ych;
    }
    virtual MatrixN backward(MatrixN& dchain, t_cppl *pcache, t_cppl *pgrads) override {
        MatrixN y=*((*pcache)["x"]);
        for (unsigned int i=0; i<y.size(); i++) {
            if (y(i)>0.0) y(i)=1.0;
            else y(i)=0.0;
        }
        MatrixN dx = y.cwiseProduct(dchain); // dx
        return dx;
    }
};
/*
class AffineRelu : public Layer {
public:
    Affine *af;
    Relu *rl;
    AffineRelu(t_layer_topo topo) {
        assert (topo.size()==2);
        layerName="AffineRelu";
        layerType=LayerType::LT_NORMAL;
        names=vector<string>{"x"};
        af=new Affine({topo[0],topo[1]});
        rl=new Relu({topo[1]});
        params=vector<MatrixN *>(1);
        params[0]=new MatrixN(1,topo[0]); // x
        grads=vector<MatrixN *>(1);
        grads[0]=new MatrixN(1,topo[0]); // dx
    }
    ~AffineRelu() {
        delete af;
        af=nullptr;
        delete rl;
        rl=nullptr;
        for (unsigned int i=0; i<params.size(); i++) {
            delete params[i];
            params[i]=nullptr;
            delete grads[i];
            grads[i]=nullptr;
        }
    }
    virtual MatrixN forward(MatrixN& x) override {
        *(params[0])=x;
        MatrixN y0=af->forward(x);
        MatrixN y=rl->forward(y0);
        return y;
    }
    virtual MatrixN backward(MatrixN& dchain) override {
        MatrixN dx0=rl->backward(dchain);
        MatrixN dx=af->backward(dx0);
        *(grads[0])=dx;
        return dx;
    }
    virtual bool update(Optimizer *popti) override {
        af->update(popti);
        rl->update(popti);
        return true;
    }
};


class Softmax : public Layer {
public:
    Softmax(t_layer_topo topo) {
        assert (topo.size()==1);
        layerName="Softmax";
        layerType=LayerType::LT_LOSS;
        names=vector<string>{"x"};
        params=vector<MatrixN *>(1);
        params[0]=new MatrixN(1,topo[0]); // x

        grads=vector<MatrixN *>(1);
        grads[0]=new MatrixN(1,topo[0]); // dx

        cache=vector<MatrixN *>(2);
        cache[0]=new MatrixN(1,topo[0]); // probs
        cache[1]=new MatrixN(1,1); // y
    }
    ~Softmax() {
        for (unsigned int i=0; i<params.size(); i++) {
            delete params[i];
            params[i]=nullptr;
            delete grads[i];
            grads[i]=nullptr;
        }
        for (unsigned int i=0; i<cache.size(); i++) {
            delete cache[i];
            cache[i]=nullptr;
        }
    }
    virtual MatrixN forward(MatrixN& x) override {
        MatrixN *px=params[0];
        if (px->cols() != x.cols()) {
            cout << layerName << ": " << "Sm forward: dimension mismatch in Softmax(x): Rx:" << shape(*px) << " x:"<< shape(x) << endl;
            return MatrixN(0,0);
        }
        if (params[0]->rows() != x.rows() || params[0]->cols() != x.cols()) {
            params[0]->resize(x.rows(), x.cols());
            params[0]->setZero();
            grads[0]->resize(x.rows(), x.cols());
            grads[0]->setZero();
            cache[0]->resize(x.rows(), x.cols());
            cache[0]->setZero();
        }
        if (cache[1]->rows() != x.rows()) {
            cache[1]->resize(x.rows(),1);
            cache[1]->setZero();
            for (int i=0; i<(*(cache[1])).size(); i++) (*(cache[1]))(i)= (-1.0); // XXX: for error testing
        }
        *params[0]=x;
        VectorN mxc = x.rowwise().maxCoeff();
        MatrixN xn = x;
        xn.colwise() +=  mxc;
        MatrixN xne = xn.array().exp().matrix();
        VectorN xnes = xne.rowwise().sum();
        for (unsigned int i=0; i<xne.rows(); i++) { // XXX broadcasting?
            xne.row(i) /= xnes(i);
        }
        MatrixN probs = xne;
        *cache[0]=probs;

        //MatrixN logprobs = xne;
        //for (unsigned int i=0; i<probs.rows(); i++) {
        //    logprobs(i,1) = -log(probs(i,y(i,0)));
        //}


        return probs;
    }
    virtual floatN loss(MatrixN& y) override {
        MatrixN probs=*cache[0];
        if (y.rows() != probs.rows() || y.cols() != 1) {
            cout << layerName << ": " << "Loss: dimension mismatch in Softmax(x): Probs:" << shape(probs) << " y:" << shape(y) << " y.cols=" << y.cols() << "(should be 1)" << endl;
            //cout << "y:" << endl << y << endl;
            return 1000.0;
        }
        if ((*cache[1]).rows() !=y.rows()) {
            cout << layerName << ": Internal error, cache[1] wrong size: " << (*cache[1]).rows() << "!=" << y.rows() << endl;
            return 1000.0;
        }
        if ((*cache[1]).cols() !=1) {
            cout << layerName << ": Internal error, cache[1] wrong size: " << (*cache[1]).cols() << "!= 1" << endl;
            return 1000.0;
        }
        if (y.cols()!=1) {
            cout << "Internal Error when setting y-cache: cols=" << y.cols() << " <- should be 1!" << endl;
        }
        *cache[1]=y;
        if ((*cache[1]).cols() !=1) {
            cout << layerName << ": Internal error (2), cache[1] wrong size: " << (*cache[1]).cols() << "!= 1" << endl;
            return 1000.0;
        }
        floatN loss=0.0;
        for (unsigned int i=0; i<probs.rows(); i++) {
            floatN pi = probs(i,y(i,0));
            if (pi==0.0) cout << "Invalid zero probability at " << i << endl;
            else loss -= log(pi);
        }
        loss /= probs.rows();
        return loss;
    }
    virtual MatrixN backward(MatrixN& y) override {
        MatrixN probs=*cache[0];

        MatrixN dx=probs;
        for (unsigned int i=0; i<probs.rows(); i++) {
            dx(i,y(i,0)) -= 1.0;
        }
        dx /= dx.rows();
        *grads[0] = dx;
        // dx = probs.copy()
        // dx[np.arange(N), y] -= 1
        // dx /= N
        return *grads[0];
    }
};


class TwoLayerNet : public Layer {
public:
    Affine *af1;
    Relu *rl;
    Affine *af2;
    Softmax *sm;
    TwoLayerNet(t_layer_topo topo) {
        assert (topo.size()==3);
        layerName="TwoLayerNet";
        layerType=LayerType::LT_LOSS;
        names=vector<string>{"x"};
        af1=new Affine({topo[0],topo[1]}); // XXX pointers to sub-objects?!
        rl=new Relu({topo[1]});
        af2=new Affine({topo[1],topo[2]});
        sm=new Softmax({topo[2]});

        params=vector<MatrixN *>(1);
        params[0]=new MatrixN(1,topo[0]); // x
        grads=vector<MatrixN *>(1);
        grads[0]=new MatrixN(1,topo[0]); // dx

        cache=vector<MatrixN *>(2);   // XXX redundant with softwmax-layer?!
        cache[0]=new MatrixN(1,topo[1]); // probs
        cache[1]=new MatrixN(1,1); // y
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
        for (unsigned int i=0; i<params.size(); i++) {
            delete params[i];
            params[i]=nullptr;
            delete grads[i];
            grads[i]=nullptr;
        }
        for (unsigned int i=0; i<cache.size(); i++) {
            delete cache[i];
            cache[i]=nullptr;
        }
    }
    virtual MatrixN forward(MatrixN& x) override {
        if (params[0]->rows() != x.rows() || params[0]->cols() != x.cols()) {
            params[0]->resize(x.rows(), x.cols());
            params[0]->setZero();
            grads[0]->resize(x.rows(), x.cols());
            grads[0]->setZero();
            cache[0]->resize(x.rows(), x.cols());
            cache[0]->setZero();
        }
        if (cache[1]->rows() != x.rows()) {
            cache[1]->resize(x.rows(),1);
            cache[1]->setZero();
            for (int i=0; i<(*(cache[1])).size(); i++) (*(cache[1]))(i)= (-1.0); // XXX: for error testing
        }
        //cout << "reshape-2LN:" << shape(*(params[0])) << shape(x) << endl;
        *(params[0])=x;
        MatrixN y0=af1->forward(x);
        MatrixN y1=rl->forward(y0);
        MatrixN y=af2->forward(y1);
        MatrixN yu=sm->forward(y);
        *cache[0]=yu;
        return y;
    }
    virtual floatN loss(MatrixN& y) override {
        *cache[1]=y;
        return sm->loss(y);
    }
    virtual MatrixN backward(MatrixN& dchain) override {
        MatrixN dx3=sm->backward(dchain);
        MatrixN dx2=af2->backward(dx3);
        MatrixN dx1=rl->backward(dx2);
        MatrixN dx=af1->backward(dx1);
        *(grads[0])=dx;
        return dx;
    }
    virtual bool update(Optimizer *popti) override {
        af1->update(popti);
        rl->update(popti);
        af2->update(popti);
        sm->update(popti);
        return true;
    }

};
*/
void registerLayers() {
    REGISTER_LAYER("Affine", Affine, 2)
    REGISTER_LAYER("Relu", Relu, 1)
/*
    REGISTER_LAYER("AffineRelu", AffineRelu, 2)
    REGISTER_LAYER("Softmax", Softmax, 1)
    REGISTER_LAYER("TwoLayerNet", TwoLayerNet, 3)
*/}
#endif
