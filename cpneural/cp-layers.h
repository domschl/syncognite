#ifndef _CP_LAYERS_H
#define _CP_LAYERS_H

#include "cp-layer.h"

/*template<typename T> Layer * createInstance() { return new T; }

typedef std::map<std::string, Layer*(*)()> map_type;

map_type layermap;
layermap["Affine"] = &createInstance<Affine>;
layermap["Relu"] = &createInstance<Relu>;
layermap["AffineRelu"] = &createInstance<AffineRelu>;
layermap["Softmax"] = &createInstance<Softmax>;
*/

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
        layername="Affine";
        lt=LayerType::LT_NORMAL;
        names=vector<string>{"x", "W", "b"};
        params=vector<MatrixN *>(3);
        params[0]=new MatrixN(1,topo[0]); // x
        params[1]=new MatrixN(topo[0],topo[1]); // W
        params[2]=new MatrixN(1,topo[1]); // b

        grads=vector<MatrixN *>(3);
        grads[0]=new MatrixN(1,topo[0]); // dx
        grads[1]=new MatrixN(topo[0],topo[1]); // dW
        grads[2]=new MatrixN(1,topo[1]); // db

        MatrixN *pW = params[1];
        MatrixN *pb = params[2];
        pW->setRandom();
        floatN xavier = 2.0/(topo[0]+topo[1]);
        *pW *= xavier;
        pb->setRandom();
        *pb *= xavier;
    }
    ~Affine() {
        for (unsigned int i=0; i<params.size(); i++) {
            delete params[i];
            params[i]=nullptr;
            delete grads[i];
            grads[i]=nullptr;
        }
    }
    virtual MatrixN forward(MatrixN& x) override {
        MatrixN *px=params[0];
        MatrixN *pW=params[1];
        MatrixN *pb=params[2];
        if (pW->rows() != x.cols()) {
            cout << layername << ": " << "Forward: dimension mismatch in x*W: x:" << shape(*px) << " W:"<< shape(*pW) << endl;
            MatrixN y(0,0);
            return y;
        }
        if (params[0]->rows() != x.rows() || params[0]->cols() != x.cols()) {
            params[0]->resize(x.rows(), x.cols());
            grads[0]->resize(x.rows(), x.cols());
        }
        *params[0] = x;
        MatrixN y = (*px) * (*pW);
        RowVectorN b = *pb;
        y.rowwise() += b;
        return y;
    }
    virtual MatrixN backward(MatrixN& dchain) override {
        MatrixN *px=params[0];
        MatrixN *pW=params[1];
        *grads[0] = dchain * (*pW).transpose(); // dx
        *grads[1] = (*px).transpose() * dchain; //dW
        *grads[2] = dchain.colwise().sum(); //db
        return *grads[0];
    }
};

class Relu : public Layer {
public:
    Relu(t_layer_topo topo) {
        assert (topo.size()==1);
        layername="Relu";
        lt=LayerType::LT_NORMAL;
        names=vector<string>{"x"};
        params=vector<MatrixN *>(1);
        params[0]=new MatrixN(1,topo[0]); // x

        grads=vector<MatrixN *>(1);
        grads[0]=new MatrixN(1,topo[0]); // dx
    }
    ~Relu() {
        for (unsigned int i=0; i<params.size(); i++) {
            delete params[i];
            params[i]=nullptr;
            delete grads[i];
            grads[i]=nullptr;
        }
    }
    virtual MatrixN forward(MatrixN& x) override {
        MatrixN *px=params[0];
        if (px->cols() != x.cols()) {
            cout << layername << ": " << "Forward: dimension mismatch in Relu(x): Rx:" << shape(*px) << " x:"<< shape(x) << endl;
            return MatrixN(0,0);
        }
        if (params[0]->rows() != x.rows() || params[0]->cols() != x.cols()) {
            params[0]->resize(x.rows(), x.cols());
            grads[0]->resize(x.rows(), x.cols());
        }

        *params[0]=x;
        MatrixN ych=x;

        for (unsigned int i=0; i<ych.size(); i++) {
            if (x(i)<0.0) {
                ych(i)=0.0;
            }
        }
        return ych;
    }
    virtual MatrixN backward(MatrixN& dchain) override {
        MatrixN *px=params[0];
        MatrixN y=*px;
        for (unsigned int i=0; i<y.size(); i++) {
            if (y(i)>0.0) y(i)=1.0;
            else y(i)=0.0;
        }
        *grads[0]=y;
        *grads[0] = y.cwiseProduct(dchain); // dx
        return *grads[0];
    }
};

class AffineRelu : public Layer {
public:
    Affine *af;
    Relu *rl;
    AffineRelu(t_layer_topo topo) {
        assert (topo.size()==2);
        layername="AffineRelu";
        lt=LayerType::LT_NORMAL;
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
};


class Softmax : public Layer {
public:
    Softmax(t_layer_topo topo) {
        assert (topo.size()==1);
        layername="Softmax";
        lt=LayerType::LT_LOSS;
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
            cout << layername << ": " << "Sm forward: dimension mismatch in Softmax(x): Rx:" << shape(*px) << " x:"<< shape(x) << endl;
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
        /*
        MatrixN logprobs = xne;
        for (unsigned int i=0; i<probs.rows(); i++) {
            logprobs(i,1) = -log(probs(i,y(i,0)));
        }
        */

        return probs;
    }
    virtual floatN loss(MatrixN& y) override {
        MatrixN probs=*cache[0];
        if (y.rows() != probs.rows() || y.cols() != 1) {
            cout << layername << ": " << "Loss: dimension mismatch in Softmax(x): Probs:" << shape(probs) << " y:" << shape(y) << " y.cols=" << y.cols() << "(should be 1)" << endl;
            cout << "y:" << endl << y << endl;
            return 1000.0;
        }
        if ((*cache[1]).rows() !=y.rows()) {
            cout << layername << ": Internal error, cache[1] wrong size: " << (*cache[1]).rows() << "!=" << y.rows() << endl;
            return 1000.0;
        }
        if ((*cache[1]).cols() !=1) {
            cout << layername << ": Internal error, cache[1] wrong size: " << (*cache[1]).cols() << "!= 1" << endl;
            return 1000.0;
        }
        if (y.cols()!=1) {
            cout << "Internal Error when setting y-cache: cols=" << y.cols() << " <- should be 1!" << endl;
        }
        *cache[1]=y;
        if ((*cache[1]).cols() !=1) {
            cout << layername << ": Internal error (2), cache[1] wrong size: " << (*cache[1]).cols() << "!= 1" << endl;
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
        layername="TwoLayerNet";
        lt=LayerType::LT_LOSS;
        names=vector<string>{"x"};
        af1=new Affine({topo[0],topo[1]});
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
};

template<typename T>
Layer* createInstance(t_layer_topo t) {
    return new T(t);
}

typedef std::map<std::string, Layer*(*)(t_layer_topo)> t_layermap;

t_layermap mapl;
t_layer_topo tl{0,0};
mapl["Affine"] = &(createInstance<Affine>(tl));
/*layermap["Relu"] = &createInstance<Relu>;
layermap["AffineRelu"] = &createInstance<AffineRelu>;
layermap["Softmax"] = &createInstance<Softmax>;*/



#endif
