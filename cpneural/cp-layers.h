#ifndef _CP_LAYERS_H
#define _CP_LAYERS_H

#include "cp-math.h"
#include "cp-layer.h"
#include "cp-timer.h"

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
private:
    int numGpuThreads;
    int numCpuThreads;
    void setup(const CpParams& cx) {
        layerName="Affine";
        topoParams=2;
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        vector<int> topo=cp.getPar("topo",vector<int>{0});
        assert (topo.size()==2);
        outTopo={topo[1]};
        //cout << "CrAff:" << topo[0] <<"/" << topo[1] << endl;
        cppl_set(&params, "W", new MatrixN(topo[0],topo[1])); // W
        cppl_set(&params, "b", new MatrixN(1,topo[1])); // b
        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();

        params["W"]->setRandom();
        floatN xavier = 1.0/std::sqrt((floatN)(topo[0]+topo[1])); // (setRandom is [-1,1]-> fakt 0.5, xavier is 2/(ni+no))
        *params["W"] *= xavier;
        params["b"]->setRandom();
        *params["b"] *= xavier;
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
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id=0) override {
        if (params["W"]->rows() != x.cols()) {
            cout << layerName << ": " << "Forward: dimension mismatch in x*W: x:" << shape(x) << " W:" << shape(*params["W"]) << endl;
            MatrixN y(0,0);
            return y;
        }
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));

        #ifdef USE_GPU
        int algo=1;
        #else
        int algo=0;
        #endif
        MatrixN y(x.rows(), (*params["W"]).cols());
        if (algo==0 || id>=numGpuThreads) {
            // cout << "C" << id << " ";
            /*
            y = x * (*params["W"]);
            RowVectorN b = *params["b"];
            y.rowwise() += b;
            */
            y=(x * (*params["W"])).rowwise() + RowVectorN(*params["b"]);
        } else {
            #ifdef USE_GPU
            // cout << "G" << id << "/" << numGpuThreads << " ";
            MatrixN x1(x.rows(),x.cols()+1);
            MatrixN xp1(x.rows(),1);
            xp1.setOnes();
            x1 << x, xp1;
            MatrixN Wb((*params["W"]).rows()+1,(*params["W"]).cols());
            Wb<<*params["W"], *params["b"];
            MatrixN y2;
            y=matmul(&x1,&Wb,id);
/*            viennacl::context ctx(viennacl::ocl::get_context(static_cast<long>(id)));
            viennacl::matrix<float>vi_Wb(Wb.rows(), Wb.cols(), ctx);
            viennacl::matrix<float>vi_x1(x1.rows(), x1.cols(), ctx);
            viennacl::matrix<float>vi_y(x1.rows(), Wb.cols(), ctx);
            viennacl::copy(Wb, vi_Wb);
            viennacl::copy(x1, vi_x1);
            vi_y = viennacl::linalg::prod(vi_x1, vi_Wb);
            viennacl::copy(vi_y, y);
*/
            //MatrixN yc = x1 * Wb;
            //matCompare(y,yc,"consistency");
            #endif
        }
        return y;
    //return (x* *params["W"]).rowwise() + RowVectorN(*params["b"]);
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
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

/*            Timer t;
            viennacl::context ctx(viennacl::ocl::get_context(id)); //static_cast<long>(id)
            viennacl::matrix<float>vi_Wt(W.cols(), W.rows(),ctx);
            viennacl::matrix<float>vi_dW(W.rows(), W.cols(),ctx);
            viennacl::matrix<float>vi_dchain(dchain.rows(), dchain.cols(), ctx);
            viennacl::matrix<float>vi_xt(x.cols(), x.rows(), ctx);
            viennacl::matrix<float>vi_dx(x.rows(), x.cols(), ctx);
            viennacl::copy(Wt, vi_Wt);
            viennacl::copy(dchain, vi_dchain);
            viennacl::copy(xt, vi_xt);
            vi_dx=viennacl::linalg::prod(vi_dchain,vi_Wt);
            vi_dW=viennacl::linalg::prod(vi_xt, vi_dchain);
            viennacl::copy(vi_dx, dx);
            viennacl::copy(vi_dW, dW);
            cppl_set(pgrads, "W", new MatrixN(dW));
*/
            //MatrixN dx2 = dchain * (*params["W"]).transpose(); // dx
            //MatrixN dW2 = (*(*pcache)["x"]).transpose() * dchain; //dW
            //matCompare(dx,dx2,"dx");
            //matCompare(dW,dW2,"dW");

            cppl_set(pgrads, "b", new MatrixN(dchain.colwise().sum())); //db
            #endif
        }
        return dx;
    }
};

class Relu : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="Relu";
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        topoParams=1;
        vector<int> topo=cp.getPar("topo", vector<int>{0});
        outTopo={topo[0]};
        layerInit=true;
    }
public:
    Relu(const CpParams& cx) {
        setup(cx);
    }
    Relu(string conf) {
        setup(CpParams(conf));
    }
    ~Relu() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl *pcache, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        MatrixN ych=x;
        for (unsigned int i=0; i<ych.size(); i++) {
            if (x(i)<0.0) {
                ych(i)=0.0;
            }
        }
        return ych;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl *pcache, t_cppl *pgrads, int id=0) override {
        MatrixN y=*((*pcache)["x"]);
        for (unsigned int i=0; i<y.size(); i++) {
            if (y(i)>0.0) y(i)=1.0;
            else y(i)=0.0;
        }
        MatrixN dx = y.cwiseProduct(dchain); // dx
        return dx;
    }
};

void mlPush(string prefix, t_cppl *src, t_cppl *dst) {
    if (dst!=nullptr) {
        for (auto pi : *src) {
            cppl_set(dst, prefix+"-"+pi.first, pi.second);
        }
    } else {
        cppl_delete(src);
    }
}

void mlPop(string prefix, t_cppl *src, t_cppl *dst) {
    for (auto ci : *src) {
        if (ci.first.substr(0,prefix.size()+1)==prefix+"-") cppl_set(dst, ci.first.substr(prefix.size()+1), ci.second);
    }
}

class AffineRelu : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="AffineRelu";
        layerType=LayerType::LT_NORMAL;
        topoParams=2;
        cp=cx;
        vector<int> topo=cp.getPar("topo", vector<int>{0});
        assert (topo.size()==2);
        outTopo={topo[1]};
        CpParams ca;
        ca.setPar("topo", vector<int>{topo[0],topo[1]});
        af=new Affine(ca);
        mlPush("af", &(af->params), &params);
        CpParams cl;
        cl.setPar("topo", vector<int>{topo[1]});
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


// Batch normalization
class BatchNorm : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="BatchNorm";
        layerType=LayerType::LT_NORMAL;
        topoParams=1;
        cp=cx;
        eps = cp.getPar("eps", (floatN)1e-5);
        momentum = cp.getPar("momentum", (floatN)0.9);
        trainMode = cp.getPar("train", (bool)false);
        vector<int> topo=cp.getPar("topo", vector<int>{0});
        outTopo={topo[0]};

        MatrixN *pgamma=new MatrixN(1,topo[0]);
        pgamma->setOnes();
        cppl_set(&params,"gamma",pgamma);
        MatrixN *pbeta=new MatrixN(1,topo[0]);
        pbeta->setZero();
        cppl_set(&params,"beta",pbeta);
        layerInit=true;
    }
public:
    floatN eps;
    floatN momentum;
    bool trainMode;

    BatchNorm(const CpParams& cx) {
        setup(cx);
    }
    BatchNorm(string conf) {
        setup(CpParams(conf));
    }
    ~BatchNorm() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id=0) override {
        MatrixN *prm, *prv;
        MatrixN *pbeta, *pgamma;
        MatrixN xout;
        trainMode = cp.getPar("train", false);
        if (pcache==nullptr || pcache->find("running_mean")==pcache->end()) {
            prm=new MatrixN(1,shape(x)[1]);
            prm->setZero();
            if (pcache!=nullptr) cppl_set(pcache,"running_mean",prm);
        } else {
            prm=(*pcache)["running_mean"];
        }
        if (pcache==nullptr || pcache->find("running_var")==pcache->end()) {
            prv=new MatrixN(1,shape(x)[1]);
            prv->setZero(); //setOnes();
            if (pcache != nullptr) cppl_set(pcache,"running_var",prv);
        } else {
            prv=(*pcache)["running_var"];
        }
        pgamma=params["gamma"];
        pbeta=params["beta"];

        floatN N=shape(x)[0];
        MatrixN mean=(x.colwise().sum()/N); //.row(0);
        RowVectorN meanv=mean.row(0);
        MatrixN xme=x.rowwise()-meanv;
        MatrixN sqse=(xme.array() * xme.array()).colwise().sum()/N+eps;
        MatrixN stdv=sqse.array().sqrt(); //.row(0);
        RowVectorN stdvv=stdv.row(0);

        if (pcache!=nullptr) {
            *(*pcache)["running_mean"] = *prm * momentum + mean * (1.0-momentum);
            *(*pcache)["running_var"]  = *prv * momentum + stdv * (1.0-momentum);
        } else {
            delete prm;
            prm=nullptr;
            delete prv;
            prv=nullptr;
        }

        if (trainMode) {
            MatrixN x2 = xme.array().rowwise() / RowVectorN(stdv.row(0)).array();
            xout = x2.array().rowwise() * RowVectorN((*pgamma).row(0)).array();
            xout.rowwise() += RowVectorN((*pbeta).row(0));

            if (pcache != nullptr) {
                cppl_update(pcache,"sqse",&sqse);
                cppl_update(pcache,"xme",&xme);
                cppl_update(pcache, "x2", &x2);

                if (momentum==1.0) {
                    cout << "ERROR: momentum should never be 1" << endl;
                }
            }
        } else { // testmode
            MatrixN xot = x.rowwise() - meanv;
            MatrixN xot2 = xot.array().rowwise() / stdvv.array();
            MatrixN xot3 = xot2.array().rowwise() * RowVectorN((*pgamma).row(0)).array();
            xout = xot3.rowwise() + RowVectorN((*pbeta).row(0));
        }
        return xout;
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        if (pcache->find("sqse")==pcache->end()) cout << "Bad: no cache entry for sqse!" << endl;
        MatrixN sqse=*((*pcache)["sqse"]);
        if (pcache->find("xme")==pcache->end()) cout << "Bad: no cache entry for xme!" << endl;
        MatrixN xme=*((*pcache)["xme"]);
        if (pcache->find("x2")==pcache->end()) cout << "Bad: no cache entry for x2!" << endl;
        MatrixN x2=*((*pcache)["x2"]);
        if (params.find("gamma")==params.end()) cout << "Bad: no params entry for gamma!" << endl;
        MatrixN gamma=*(params["gamma"]);
        MatrixN dbeta=y.colwise().sum();

        MatrixN dgamma=(x2.array() * y.array()).colwise().sum();
        if (shape(gamma) != shape(dgamma)) cout << "bad: dgamma has wrong shape: " << shape(gamma) << shape(dgamma) << endl;

        floatN N=y.rows();
        MatrixN d1=MatrixN(y).setOnes();
        MatrixN dx0 = gamma.array() * (sqse.array().pow(-0.5)) / N;
        MatrixN dx1 = (y*N).rowwise() - RowVectorN(y.colwise().sum());
        MatrixN iv = sqse.array().pow(-1.0);
        MatrixN dx21 = (xme.array().rowwise() * RowVectorN(iv).array());
        MatrixN dx22 = (y.array() * xme.array()).colwise().sum();
        MatrixN dx2 = dx21.array().rowwise() * RowVectorN(dx22).array();
        // cout << "d1:" << shape(d1) << ", dx0:" << shape(dx0) << endl;
        //MatrixN dx=(dx1 - dx2).array().rowwise() * RowVectorN(d1*dx0).array() ;
        MatrixN dx=(dx1 - dx2).array().rowwise() * RowVectorN(dx0).array() ;
        cppl_set(pgrads,"gamma",new MatrixN(dgamma));
        cppl_set(pgrads,"beta",new MatrixN(dbeta));

        return dx;
    }

};


// Dropout: with probability drop, a neuron's value is dropped. (1-p)?
class Dropout : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="Dropout";
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        topoParams=1;
        vector<int> topo=cp.getPar("topo", vector<int>{0});
        outTopo={topo[0]};
        drop = cp.getPar("drop", (floatN)0.5);
        trainMode = cp.getPar("train", (bool)false);
        freeze = cp.getPar("freeze", (bool)false);
        layerInit=true;
    }
public:
    floatN drop;
    bool trainMode;
    bool freeze;

    Dropout(const CpParams& cx) {
        setup(cx);
    }
    Dropout(string conf) {
        setup(CpParams(conf));
    }
    ~Dropout() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        drop = cp.getPar("drop", (floatN)0.5);
        if (drop==1.0) return x;
        trainMode = cp.getPar("train", false);
        freeze = cp.getPar("freeze", false);
        MatrixN xout;
        if (trainMode) {
            MatrixN* pmask;
            freeze = cp.getPar("freeze", false);
            if (freeze && pcache!=nullptr && pcache->find("dropmask")!=pcache->end()) {
                pmask=(*pcache)["dropmask"];
                xout = x.array() * (*pmask).array();
            } else {
                pmask=new MatrixN(x);
                if (freeze) srand(123);
                pmask->setRandom();
                for (int i=0; i<x.size(); i++) {
                    if (((*pmask)(i)+1.0)/2.0 < drop) (*pmask)(i)=1.0;
                    else (*pmask)(i)=0.0;
                }
                xout = x.array() * (*pmask).array();
                if (pcache!=nullptr) cppl_set(pcache, "dropmask", pmask);
                else delete pmask;
            }
            //cout << "drop:" << drop << endl << xout << endl;
        } else {
            xout = x * drop;
        }
        return xout;
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        MatrixN dx;
        trainMode = cp.getPar("train", false);
        if (trainMode && drop!=1.0) {
            MatrixN* pmask=(*pcache)["dropmask"];
            dx=y.array() * (*pmask).array();
        } else {
            dx=y;
        }
        return dx;
    }
};

// Convolution layer
class Convolution : public Layer {
    // N: number of data points;  input is N x (C x W x H)
    // C: color-depth
    // W: width of input
    // H: height of input
    // F: number of filter-kernels;   kernel is F x (C x WW x HH)
    // C: identical number of color-depth    // XXX: move kernel sizes to params?
    // WW: filter-kernel depth
    // HH: filter-kernel height
    // params:
    //   stride
    //   pad
    // Output: N x (F x WO x HO)
    //   WO: output width
    //   HO: output height
private:
    int numGpuThreads;
    int numCpuThreads;
    int C, H, W, F, HH, WW;
    int HO, WO;
    int pad, stride;
    bool mlverbose=false;
    void setup(const CpParams& cx) {
        layerName="Convolution";
        topoParams=6;
        bool retval=true;
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        vector<int> topo=cp.getPar("topo",vector<int>{0});
        assert (topo.size()==6);
        // TOPO: C, H, W, F, HH, WW
        C=topo[0]; H=topo[1]; W=topo[2];
        F=topo[3]; HH=topo[4]; WW=topo[5];

        // W: F, C, HH, WW
        //cppl_set(&params, "Wb", new MatrixN(F,C*HH*WW+1)); // Wb, b= +1!
        cppl_set(&params, "W", new MatrixN(F,C*HH*WW));
        cppl_set(&params, "b", new MatrixN(F,1));
        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();

        stride = cp.getPar("stride", 1);
        pad = cp.getPar("pad", (int)((HH-1)/2));
        if (pad>=HH || pad>=WW) {
            cout << "bad  configuration, pad:" << pad << ">=" << " HH,WW:" << HH << "," << WW << endl;
            retval=false;
        }
        if ((H + 2 * pad - HH) % stride != 0) {
            int r=(H + 2 * pad - HH) % stride;
            if (r>pad) {
                cout << "H <-> stride does not fit! r=" << r << ", pad=" << pad << endl;
                retval=false;
            }
        }
        if ((W + 2 * pad - WW) % stride != 0) {
            int r=(W + 2 * pad - WW) % stride;
            if (r>pad) {
                cout << "w <-> stride does not fit! r=" << r << ", pad=" << pad << endl;
                retval=false;
            }
        }
        HO = 1 + (H + 2 * pad - HH) / stride;
        WO = 1 + (W + 2 * pad - WW) / stride;

        if (HO*stride+HH-stride < H+pad) {
            cout << "H: current stride:" << stride << ", pad:" << pad << " combination does not cover input-field" << endl;
            retval=false;
        }
        if (WO*stride+WW-stride < W+pad) {
            cout << "W: current stride:" << stride << ", pad:" << pad << " combination does not cover input-field" << endl;
            retval=false;
        }

        outTopo={topo[3],WO,HO};

        params["W"]->setRandom();
        floatN xavier = 1.0/std::sqrt((floatN)(C*H*W + F*HO*WO)) / 10.0;
        *params["W"] = *params["W"] * xavier;

        params["b"]->setRandom();
        *params["b"] = *params["b"] * xavier;
        layerInit=retval;
    }
public:
    Convolution(const CpParams& cx) {
        setup(cx);
    }
    Convolution(const string conf) {
        setup(CpParams(conf));
    }
    ~Convolution() {
        cppl_delete(&params);
    }
/*
    void im2col(MatrixN xx, MatrixN *px2c) {
        int N=shape(xx)[0];
        // add padding and b-caused 1s
        //      p p x x x x x x x p p
        int xd, yd;
        int xs, ys;
        int x0, y0;
        int err=0;
        floatN pix;
        for (int n=0; n<N; n++) {
            for (int y=0; y<HO; y++) {
                for (int x=0; x<WO; x++) {
                    y0=y*stride-pad;
                    x0=x*stride-pad;
                    for (int cc=0; cc<C; cc++) {
                        for (int cy=0; cy<HH; cy++) {
                            for (int cx=0; cx<WW; cx++) {
                                if (err>5) {
                                    cout << "FATAL i2c abort." << endl;
                                    return;
                                }
                                ys=y0+cy;
                                xs=x0+cx;
                                if (xs<0 || xs>=W || ys<0 || ys>=H) { //pad
                                    pix=0.0;
                                } else {
                                    unsigned int xxs=cc*H*W+ys*W+xs;
                                    if (xxs>=shape(xx)[1]) {
                                        cout << "i2c xxs illegal: " << shape(xx) << xxs << "=" << cc << "," << ys <<"," << xs << endl;
                                        ++err;
                                        continue;
                                    }
                                    pix=xx(n,xxs);
                                }
                                yd=cc*HH*WW+cy*WW+cx;
                                xd=n*(HO*WO)+y*WO+x;
                                if (yd<0 || yd>=C*HH*WW) {
                                    cout << "i2c yd illegal: " << yd << endl;
                                    ++err;
                                    continue;
                                }
                                if (xd<0 || xd>=N*WO*HO) {
                                    cout << "i2c xd illegal: " << xd << endl;
                                    ++err;
                                    continue;
                                }
                                (*px2c)(yd,xd)=pix;
                            }
                        }
                    }
                }
            }
        }
        //cout << "x2c:" << endl << *px2c << endl;
    }
*/
    void im2col(MatrixN xx, MatrixN *px2c) {
        int N=shape(xx)[0];
        // add padding and b-caused 1s
        //      p p x x x x x x x p p
        int xd, yd;
        int xs, ys;
        int x0, y0;
        int n,x,y,cx,cy,cc;
        int cchhww,cchw,cchhwwcyww;
        int xxso;
        int HOWO=HO*WO;
        int HHWW=HH*WW;
        int HW=H*W;
        unsigned int xxs;
        floatN pix;
        for (n=0; n<N; n++) {
            for (y=0; y<HO; y++) {
                y0=y*stride-pad;
                for (x=0; x<WO; x++) {
                    x0=x*stride-pad;
                    xd=n*HOWO+y*WO+x;
                    for (cc=0; cc<C; cc++) {
                        cchhww=cc*HHWW;
                        cchw=cc*HW;
                        for (cy=0; cy<HH; cy++) {
                            ys=y0+cy;
                            xxso=cchw+ys*W;
                            cchhwwcyww=cchhww+cy*WW;
                            for (cx=0; cx<WW; cx++) {
                                xs=x0+cx;
                                if (xs<0 || xs>=W || ys<0 || ys>=H) { //pad
                                    pix=0.0;
                                } else {
                                    xxs=xxso+xs;
                                    pix=xx(n,xxs);
                                }
                                yd=cchhwwcyww+cx;
                                (*px2c)(yd,xd)=pix;
                            }
                        }
                    }
                }
            }
        }
        //cout << "x2c:" << endl << *px2c << endl;
    }
/*
    MatrixN iim2col(MatrixN x2c, int N) {
        MatrixN dx(N,C*W*H);
        dx.setZero();

        int xd, yd;
        int xs, ys;
        int x0, y0;
        int err=0;
//        floatN pix;
        for (int n=0; n<N; n++) {
            for (int y=0; y<HO; y++) {
                for (int x=0; x<WO; x++) {
                    y0=y*stride-pad;
                    x0=x*stride-pad;
                    for (int cc=0; cc<C; cc++) {
                        for (int cy=0; cy<HH; cy++) {
                            for (int cx=0; cx<WW; cx++) {
                                if (err>5) {
                                    cout << "FATAL ii2c abort" << endl;
                                    return dx;
                                }
                                ys=y0+cy;
                                xs=x0+cx;
                                if (xs<0 || xs>=W || ys<0 || ys>=H) { //pad
//                                    pix=0.0;
                                } else {
                                    unsigned int xxs=cc*H*W+ys*W+xs;
                                    if (xxs>=shape(dx)[1]) {
                                        cout << "ii2c xxs illegal: " << xxs << endl;
                                        ++err;
                                        continue;
                                    }
                                    yd=cc*HH*WW+cy*WW+cx;
                                    xd=n*(HO*WO)+y*WO+x;
                                    if (yd<0 || yd>=C*HH*WW) {
                                        cout << "ii2c yd illegal: " << yd << endl;
                                        ++err;
                                        continue;
                                    }
                                    if (xd<0 || xd>=N*WO*HO) {
                                        cout << "ii2c xd illegal: " << xd << endl;
                                        ++err;
                                        continue;
                                    }
                                    dx(n,xxs) += x2c(yd,xd);
                                }
//                                (*px2c)(yd,xd)=pix;
                            }
                        }
                    }
                }
            }
        }

        return dx;
    }
*/
    MatrixN iim2col(MatrixN x2c, int N) {
        MatrixN dx(N,C*W*H);
        dx.setZero();

        int xd, yd;
        int xs, ys;
        int x0, y0;
        int n,x,y,cc,cy,cx;
        unsigned int xxs;
        int cchw, cchhww, xxso, cchhwwcyww;
        int HOWO=HO*WO;
        int HHWW=HH*WW;
        int HW=H*W;
        for (n=0; n<N; n++) {
            for (y=0; y<HO; y++) {
                y0=y*stride-pad;
                for (x=0; x<WO; x++) {
                    x0=x*stride-pad;
                    xd=n*HOWO+y*WO+x;
                    for (cc=0; cc<C; cc++) {
                        cchw=cc*HW;
                        cchhww=cc*HHWW;
                        for (cy=0; cy<HH; cy++) {
                            ys=y0+cy;
                            xxso=cchw+ys*W;
                            cchhwwcyww=cchhww+cy*WW;
                            if (ys<0 || ys>=H) continue;
                            for (cx=0; cx<WW; cx++) {
                                xs=x0+cx;
                                if (xs<0 || xs>=W) continue;
                                xxs=xxso+xs;
                                yd=cchhwwcyww+cx;
                                dx(n,xxs) += x2c(yd,xd);
                            }
                        }
                    }
                }
            }
        }

        return dx;
    }

    MatrixN col2im(MatrixN y2c, int N) {
        MatrixN xx(N,F*WO*HO);
//        int err=0;
        int WHO=WO*HO;
        for (int n=0; n<N; n++) {
            for (int x=0; x<F*WHO; x++) {
                int p=n*F*WHO+x;
                int ox=p%(N*WHO);
                int py=(p/(WHO))%F;
                int px=ox%(WHO)+n*(WHO);
/*                if (py>=y2c.rows() || px>=y2c.cols()) {
                    cout << "c2i Illegal rows/cols in col2im:" << shape(y2c)<< py << "," <<px<<endl;
                    ++err;
                    if (err>5) {
                        cout << "FATAL c2i abort." << endl;
                        return xx;
                    }
                    continue;
                }
*/                xx(n,x)=y2c(py,px);
            }
        }
        return xx;
    }

    MatrixN icol2im(MatrixN dy, int N) {
//        MatrixN iy(F,N*H*W);
        MatrixN iy(F,N*HO*WO);
//        int err=0;
        int HWO=WO*HO;
        for (int f=0; f<F; f++) {
            for (int x=0; x<N*HWO; x++) {
                int p=f*N*HWO+x;
                int ox=p%(F*HWO);
                int py=(p/(HWO))%N;
                int px=ox%(HWO)+f*(HWO);
/*                if (py>=dy.rows() || px>=dy.cols()) {
                    cout << "ic2i Illegal rows/cols in col2im:" << shape(dy)<< py << "," <<px<<endl;
                    ++err;
                    if (err>5) {
                        cout << "FATAL ic2i abort." << endl;
                        return iy;
                    }
                    continue;
                }
*/                iy(f,x)=dy(py,px);
            }
        }
        return iy;
    }

    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id=0) override {
        // XXX cache x2c and use allocated memory for im2col call!
        auto N=shape(x)[0];
        MatrixN *px2c = new MatrixN(C*HH*WW, N*HO*WO);
        px2c->setZero();
        int algo=0;
        Timer t;
        if (shape(x)[1]!=(unsigned int)C*W*H) {
            cout << "ConvFw: Invalid input data x: expected C*H*W=" << C*H*W << ", got: " << shape(x)[1] << endl;
            return MatrixN(0,0);
        }

        // x: N, C, H, W;  w: F, C, HH, WW
        if (mlverbose) t.startWall();
        im2col(x, px2c);
        if (mlverbose) cout << "im2col:"<<t.stopWallMicro()<<"µs"<<endl;

        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x)); // XXX where do we need x?
        if (pcache!=nullptr) cppl_set(pcache, "x2c", px2c);

/*        cout <<"x:"<<shape(x)<<endl;
        cout <<"px2c:"<<shape(*px2c)<<endl;
        cout << "W:"<<shape(*params["W"]) << endl;
        cout << "b:"<<shape(*params["b"]) << endl;
*/
        if (mlverbose) t.startWall();
        MatrixN y2c;
        #ifdef USE_GPU
        algo=1;
        #endif
        if (algo==0 || id>=numGpuThreads) {
            y2c=((*params["W"]) * (*px2c)).colwise() + ColVectorN(*params["b"]);
        } else {
            y2c=matmul(params["W"], px2c, id, mlverbose).colwise() + ColVectorN(*params["b"]);
        }
        if (mlverbose) cout << "matmul:"<<t.stopWallMicro()<<"µs"<<endl;
        if (mlverbose) t.startWall();
        MatrixN y=col2im(y2c, N);
        if (mlverbose) cout << "col2im:"<<t.stopWallMicro()<<"µs"<<endl;
        // cout <<"col2im y2c:"<<shape(y2c)<<"->y:"<<shape(y)<<endl;
        if (pcache==nullptr) delete px2c;
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        int N=shape(dchain)[0];
        if (shape(dchain)[1]!=(unsigned int)F*HO*WO) {
            cout << "ConvBw: Invalid input data dchain: expected F*HO*WO=" << F*HO*WO << ", got: " << shape(dchain)[1] << endl;
            return MatrixN(0,0);
        }
        int algo=0;
        #ifdef  USE_GPU
        algo=1;
        #endif
        /*
        cout << "dchain:" << shape(dchain) << endl;
        cout << "W:" << shape(*params["W"]) << endl;
        cout << "x:" << shape(*(*pcache)["x"]) << endl;
        cout << "x2c:" << shape(*(*pcache)["x2c"]) << endl;
        cout << "WO:" << WO << "," << "HO:" << HO << endl;
        */
        MatrixN dc2=icol2im(dchain,N);
        // cout << "dc2:" << shape(dc2) << endl;

        MatrixN dx;
        if (algo==0 || id>=numGpuThreads) {
            MatrixN dx2c = dc2.transpose() * (*params["W"]); // dx
            dx=iim2col(dx2c.transpose(), N);
            cppl_set(pgrads, "W", new MatrixN(dc2 * (*(*pcache)["x2c"]).transpose())); //dW
            cppl_set(pgrads, "b", new MatrixN(dc2.rowwise().sum())); //db
        } else {
            MatrixN dc2t;
            dc2t=dc2.transpose();
            MatrixN W=*params["W"];
            MatrixN dx2c=matmul(&dc2t,&W,id,mlverbose);
            dx=iim2col(dx2c.transpose(), N);

            MatrixN x2ct=(*(*pcache)["x2c"]).transpose();
            MatrixN dW=matmul(&dc2,&x2ct,id,mlverbose);
            cppl_set(pgrads, "W", new MatrixN(dW));
            cppl_set(pgrads, "b", new MatrixN(dc2.rowwise().sum())); //db
        }
        return dx;
    }
};

// Pooling layer
class Pooling : public Layer {
    // N: number of data points;  input is N x (C x W x H)
    // C: color-depth
    // W: width of input
    // H: height of input
    // C: identical number of color-depth
    // WW: pooling-kernel depth
    // HH: pooling-kernel height
    // params:
    //   stride
    // Output: N x (C x WO x HO)
    //   WO: output width
    //   HO: output height
private:
    int numGpuThreads;
    int numCpuThreads;
    int C, H, W, HH, WW;
    int HO, WO;
    int stride;
    void setup(const CpParams& cx) {
        layerName="Pooling";
        topoParams=3;  // XXX: move kernel sizes to params?
        bool retval=true;
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        vector<int> topo=cp.getPar("topo",vector<int>{0});
        assert (topo.size()==3);
        stride = cp.getPar("stride", 2);
        // TOPO: C, H, W        // XXX: we don't need HH und WW, they have to be equal to stride anyway!
        C=topo[0]; H=topo[1]; W=topo[2];
        HH=stride; WW=stride;  // XXX: Simplification, our algo doesn't work for HH or WW != stride.
        if (HH!=stride || WW!=stride) {
            cout << "Implementation only supports stride equal to HH and WW!";
            retval=false;
        }
        // W: F, C, HH, WW
        //cppl_set(&params, "Wb", new MatrixN(F,C*HH*WW+1)); // Wb, b= +1!
        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();

        HO = (H-HH)/stride+1;
        WO = (W-WW)/stride+1;

        outTopo={C,WO,HO};

        layerInit=retval;
    }
public:
    Pooling(const CpParams& cx) {
        setup(cx);
    }
    Pooling(const string conf) {
        setup(CpParams(conf));
    }
    ~Pooling() {
        cppl_delete(&params);
    }

    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id=0) override {
        // XXX cache x2c and use allocated memory for im2col call!
        auto N=shape(x)[0];
        if (shape(x)[1]!=(unsigned int)C*W*H) {
            cout << "PoolFw: Invalid input data x: expected C*H*W=" << C*H*W << ", got: " << shape(x)[1] << endl;
            return MatrixN(0,0);
        }
        MatrixN *pmask = new MatrixN(N, C*H*W);
        pmask->setZero();
        MatrixN y(N,C*WO*HO);
        y.setZero();
        floatN mx;
        int chw,chowo,px0,iy0;
        int xs, ys,px, mxi, myi;
        for (int n=0; n<(int)N; n++) {
            for (int c=0; c<C; c++) {
                chw=c*H*W;
                chowo=c*HO*WO;
                for (int iy=0; iy<HO; iy++) {
                    iy0=chowo+iy*WO;
                    for (int ix=0; ix<WO; ix++) {
                        mx=0.0; mxi= (-1); myi= (-1);
                        for (int cy=0; cy<HH; cy++) {
                            ys=iy*stride+cy;
                            px0=chw+ys*W;
                            if (ys>=H) continue;
                            for (int cx=0; cx<WW; cx++) {
                                xs=ix*stride+cx;
                                if (xs>=W) continue;
                                px=px0+xs;
                                if (cx==0 && cy==0) {
                                    mx=x(n,px);
                                    myi=n;
                                    mxi=px;
                                } else {
                                    if (x(n,px)>mx) {
                                        mx=x(n,px);
                                        myi=n;
                                        mxi=px;
                                    }
                                }
                            }
                        }
                        y(n,iy0+ix)=mx;
                        if (mxi!=(-1) && myi!=(-1)) (*pmask)(myi,mxi) = 1.0;
                    }
                }
            }
        }

        //cout << "x:" << endl << x << endl;
        //cout << "mask:" << endl << *pmask << endl;

        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x)); // XXX where do we need x?
        if (pcache!=nullptr) cppl_set(pcache, "mask", pmask);

        if (pcache==nullptr) delete pmask;
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        int N=shape(dchain)[0];
        if (shape(dchain)[1]!=(unsigned int)C*HO*WO) {
            cout << "PoolBw: Invalid input data dchain: expected C*HO*WO=" << C*HO*WO << ", got: " << shape(dchain)[1] << endl;
            return MatrixN(0,0);
        }
        MatrixN dx(N,C*W*H);
        dx.setZero();
        int chw,chowo,px0,py0,ix0;
        int xs, ys, px, py;
        for (int n=0; n<(int)N; n++) {
            for (int c=0; c<C; c++) {
                chw=c*H*W;
                chowo=c*WO*HO;
                for (int iy=0; iy<HO; iy++) {
                    py0=chowo+iy*WO;
                    for (int ix=0; ix<WO; ix++) {
                        ix0=ix*stride;
                        for (int cy=0; cy<HH; cy++) {
                            ys=iy*stride+cy;
                            px0=chw+ys*W;
                            if (ys>=H) continue;
                            for (int cx=0; cx<WW; cx++) {
                                xs=ix0+cx;
                                if (xs>=W) continue;
                                px=px0+xs;
                                py=py0+ix;
                                MatrixN *pmask=(*pcache)["mask"];
                                dx(n, px) += dchain(n,py) * (*pmask)(n, px);
                            }
                        }
                    }
                }
            }
        }
        return dx;
    }
};


// SpatialBatchNorm layer
class SpatialBatchNorm : public Layer {
    // N: number of data points;  input is N x (C x W x H)
    // C: color-depth
    // W: width of input
    // H: height of input
    // C: identical number of color-depth
    // WW: SpatialBatchNorm-kernel depth
    // HH: SpatialBatchNorm-kernel height
    // params:
    //   stride
    // Output: N x (C x WO x HO)
    //   WO: output width
    //   HO: output height
private:
    int C, H, W;
    BatchNorm *pbn;
    void setup(const CpParams& cx) {
        layerName="SpatialBatchNorm";
        topoParams=3;  // XXX: move kernel sizes to params?
        bool retval=true;
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        vector<int> topo=cp.getPar("topo",vector<int>{0});
        assert (topo.size()==3);
        // TOPO: C, H, W        // XXX: we don't need HH und WW, they have to be equal to stride anyway!
        C=topo[0]; H=topo[1]; W=topo[2];
        outTopo={C,H,W};

        CpParams cs(cp);
        cs.setPar("topo",vector<int>{C*H*W});
        pbn = new BatchNorm(cs);
        mlPush("bn", &(pbn->params), &params);

        layerInit=retval;
    }
public:
    SpatialBatchNorm(const CpParams& cx) {
        setup(cx);
    }
    SpatialBatchNorm(const string conf) {
        setup(CpParams(conf));
    }
    ~SpatialBatchNorm() {
        //cppl_delete(&params);
        delete pbn;
        pbn=nullptr;
    }

    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id=0) override {
        // XXX cache x2c and use allocated memory for im2col call!
        int N=shape(x)[0];
        if (shape(x)[1]!=(unsigned int)C*W*H) {
            cout << "SpatialBatchNorm Fw: Invalid input data x: expected C*H*W=" << C*H*W << ", got: " << shape(x)[1] << endl;
            return MatrixN(0,0);
        }
        MatrixN xs(C,N*H*W);
        for (int n=0; n<N; n++) {  // Uhhh..
            for (int c=0; c<C; c++) {
                for (int h=0; h<H; h++) {
                    for (int w=0; w<W; w++) {
                        xs(c,n*H*W+h*W+w)=x(n,c*H*W+h*W+w);
                    }
                }
            }
        }

        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));

        t_cppl tcachebn;
        MatrixN ys=pbn->forward(x, &tcachebn, id);
        mlPush("bn", &tcachebn, pcache);

        MatrixN y;
        for (int n=0; n<N; n++) {  // Uhhh..
            for (int c=0; c<C; c++) {
                for (int h=0; h<H; h++) {
                    for (int w=0; w<W; w++) {
                        y(n,c*H*W+h*W+w)=ys(c,n*H*W+h*W+w);
                    }
                }
            }
        }
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        //int N=shape(dchain)[0];
        if (shape(dchain)[1]!=(unsigned int)C*H*W) {
            cout << "SpatialBatchNorm Bw: Invalid input data dchain: expected C*H*W=" << C*H*W << ", got: " << shape(dchain)[1] << endl;
            return MatrixN(0,0);
        }
        MatrixN dx;
        return dx;
    }
};


// Multiclass support vector machine Svm
class Svm : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="Svm";
        layerType=LayerType::LT_LOSS;
        cp=cx;
        topoParams=1;
        vector<int> topo=cp.getPar("topo", vector<int>{0});
        outTopo={topo[0]};
        layerInit=true;
    }
public:
    Svm(const CpParams& cx) {
        setup(cx);
    }
    Svm(string conf) {
        setup(CpParams(conf));
    }
    ~Svm() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        VectorN correctClassScores(x.rows());
        for (unsigned int i=0; i<x.rows(); i++) {
            correctClassScores(i)=x(i,(int)y(i));
        }
        MatrixN margins=x;
        margins.colwise() -= correctClassScores;
        margins = (margins.array() + 1.0).matrix();
        for (unsigned int i=0; i < margins.size(); i++) {
            if (margins(i)<0.0) margins(i)=0.0;
        }
        for (unsigned int i=0; i<margins.rows(); i++) {
            margins(i,y(i,0)) = 0.0;
        }

        if (pcache!=nullptr) cppl_set(pcache, "margins", new MatrixN(margins));
        return margins;
    }
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) override {
        MatrixN margins=*((*pcache)["margins"]);
        floatN loss = margins.sum() / margins.rows();
        return loss;
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        MatrixN margins=*((*pcache)["margins"]);
        MatrixN x=*((*pcache)["x"]);
        VectorN numPos(x.rows());
        MatrixN dx=x;
        dx.setZero();
        for (unsigned int i=0; i<margins.rows(); i++) {
            int num=0;
            for (unsigned int j=0; j<margins.cols(); j++) {
                if (margins(i,j)>0.0) ++num;
            }
            numPos(i) = (floatN)num;
        }
        for (unsigned int i=0; i<dx.size(); i++) {
            if (margins(i) > 0.0) dx(i)=1.0;
        }
        for (unsigned int i=0; i<dx.rows(); i++) {
            dx(i,y(i,0)) -= numPos(i);
        }
        dx /= dx.rows();
        return dx;
    }
};

class Softmax : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="Softmax";
        layerType=LayerType::LT_LOSS;
        cp=cx;
        topoParams=1;
        vector<int> topo=cp.getPar("topo", vector<int>{0});
        outTopo={topo[0]};
        layerInit=true;
    }
public:
    Softmax(const CpParams& cx) {
        setup(cx);
    }
    Softmax(string conf) {
        setup(CpParams(conf));
    }
    ~Softmax() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        VectorN mxc = x.rowwise().maxCoeff();
        MatrixN xn = x;
        xn.colwise() -=  mxc;
        MatrixN xne = xn.array().exp().matrix();
        VectorN xnes = xne.rowwise().sum();
        for (unsigned int i=0; i<xne.rows(); i++) { // XXX broadcasting?
            xne.row(i) = xne.row(i) / xnes(i);
        }
        MatrixN probs = xne;
        if (pcache!=nullptr) cppl_set(pcache, "probs", new MatrixN(probs));
        return probs;
    }
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) override {
        MatrixN probs=*((*pcache)["probs"]);
        if (y.rows() != probs.rows() || y.cols() != 1) {
            cout << layerName << ": "  << "Loss, dimension mismatch in Softmax(x), Probs: ";
            cout << shape(probs) << " y:" << shape(y) << " y.cols=" << y.cols() << "(should be 1)" << endl;
            return 1000.0;
        }
        //if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        floatN loss=0.0;
        for (unsigned int i=0; i<probs.rows(); i++) {
            if (y(i,0)>=probs.cols()) {
                cout << "internal error: y(" << i << ",0) >= " << probs.cols() << endl;
                return -10000.0;
            }
            floatN pi = probs(i,y(i,0));
            if (pi==0.0) cout << "Invalid zero log-probability at " << i << endl;
            else loss -= log(pi);
        }
        loss /= probs.rows();
        return loss;
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        MatrixN probs=*((*pcache)["probs"]);

        MatrixN dx=probs;
        for (unsigned int i=0; i<probs.rows(); i++) {
            dx(i,y(i,0)) -= 1.0;
        }
        dx /= dx.rows();
        return dx;
    }
};


class TwoLayerNet : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="TwoLayerNet";
        layerType=LayerType::LT_LOSS;
        topoParams=3;
        cp=cx;
        vector<int> topo=cp.getPar("topo",vector<int>{0});
        outTopo={topo[2]};

        CpParams c1,c2,c3,c4;
        c1.setPar("topo",vector<int>{topo[0],topo[1]});
        c2.setPar("topo",vector<int>{topo[1]});
        c3.setPar("topo",vector<int>{topo[1],topo[2]});
        c4.setPar("topo",vector<int>{topo[2]});
        af1=new Affine(c1);
        mlPush("af1", &(af1->params), &params);
        rl=new Relu(c2);
        mlPush("rl", &(rl->params), &params);
        af2=new Affine(c3);
        mlPush("af2", &(af2->params), &params);
        sm=new Softmax(c4);
        mlPush("sm", &(sm->params), &params);
        layerInit=true;
    }
public:
    Affine *af1;
    Relu *rl;
    Affine *af2;
    Softmax *sm;
    TwoLayerNet(CpParams cx) {
        setup(cx);
    }
    TwoLayerNet(string conf) {
        setup(CpParams(conf));
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
    }
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        t_cppl c1;
        MatrixN y0=af1->forward(x,&c1, id);
        mlPush("af1",&c1,pcache);
        t_cppl c2;
        MatrixN y1=rl->forward(y0,&c2, id);
        mlPush("rl",&c2,pcache);
        t_cppl c3;
        MatrixN yo=af2->forward(y1,&c3, id);
        mlPush("af2",&c3,pcache);
        t_cppl c4;
        MatrixN yu=sm->forward(yo,y,&c4, id);
        mlPush("sm",&c4,pcache);
        return yo;
    }
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) override {
        t_cppl c4;
        mlPop("sm",pcache,&c4);
        return sm->loss(y, &c4);
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        t_cppl c4;
        t_cppl g4;
        mlPop("sm",pcache,&c4);
        MatrixN dx3=sm->backward(y, &c4, &g4, id);
        mlPush("sm",&g4,pgrads);

        t_cppl c3;
        t_cppl g3;
        mlPop("af2",pcache,&c3);
        MatrixN dx2=af2->backward(dx3,&c3,&g3, id);
        mlPush("af2", &g3, pgrads);

        t_cppl c2;
        t_cppl g2;
        mlPop("rl",pcache,&c2);
        MatrixN dx1=rl->backward(dx2, &c2, &g2, id);
        mlPush("rl", &g2, pgrads);

        t_cppl c1;
        t_cppl g1;
        mlPop("af1",pcache,&c1);
        MatrixN dx=af1->backward(dx1, &c1, &g1, id);
        mlPush("af1", &g1, pgrads);

        return dx;
    }
    virtual bool update(Optimizer *popti, t_cppl *pgrads, string var, t_cppl *pocache) override {
        t_cppl g1;
        mlPop("af1",pgrads,&g1);
        af1->update(popti,&g1, var+"2l1", pocache); // XXX push/pop pocache?
        t_cppl g2;
        mlPop("rl",pgrads,&g2);
        rl->update(popti,&g2, var+"2l2", pocache);
        t_cppl g3;
        mlPop("af2",pgrads,&g3);
        af2->update(popti,&g3, var+"2l3", pocache);
        t_cppl g4;
        mlPop("sm",pgrads,&g4);
        sm->update(popti,&g4, var+"2l4", pocache);
        return true;
    }

};


void registerLayers() {
    REGISTER_LAYER("Affine", Affine, 2)
    REGISTER_LAYER("Relu", Relu, 1)
    REGISTER_LAYER("AffineRelu", AffineRelu, 2)
    REGISTER_LAYER("BatchNorm", BatchNorm, 1)
    REGISTER_LAYER("Dropout", Dropout, 1)
    REGISTER_LAYER("Convolution", Convolution, 6) // XXX: adapt to 3 + params?
    REGISTER_LAYER("Pooling", Pooling, 3)
    REGISTER_LAYER("SpatialBatchNorm", SpatialBatchNorm, 3)
    REGISTER_LAYER("Softmax", Softmax, 1)
    REGISTER_LAYER("Svm", Svm, 1)
    REGISTER_LAYER("TwoLayerNet", TwoLayerNet, 3)
}

class MultiLayer : public Layer {
private:
    void setup(const CpParams& cx) {
        cp=cx;
        layerName="multi"; //cp.getPar("name","multi");
        lossLayer="";
        layerType=LayerType::LT_NORMAL;
        trainMode = cp.getPar("train", false);
        checked=false;
    }
public:
    map<string, Layer*> layerMap;
    map<string, vector<string>> layerInputs;
    string lossLayer;
    bool checked;
    bool trainMode;

    MultiLayer(const CpParams& cx) {
        setup(cx);
    }
    MultiLayer(string conf) {
        setup(CpParams(conf));
        layerInit=true;
    }
    ~MultiLayer() {
    }
    void addLayer(string name, Layer* player, vector<string> inputLayers) {
        layerMap[name]=player;
        if (player->layerType==LayerType::LT_LOSS) {
            if (lossLayer!="") {
                cout << "ERROR: a loss layer with name: " << lossLayer << "has already been defined, and is now overwritten by: " << name << endl;
            }
            layerType=LayerType::LT_LOSS;
            lossLayer=name;
        }
        if (player->layerInit==false) {
            cout << "Attempt to add layer " << name << " with bad initialization." << endl;
        }
        layerInputs[name]=inputLayers;
        mlPush(name, &(player->params), &params);
        checked=false;
    }
    bool checkTopology(bool verbose=false) {
        if (lossLayer=="") {
            cout << "No loss layer defined!" << endl;
            return false;
        }
        vector<string> lyr;
        lyr=getLayerFromInput("input");
        if (lyr.size()!=1) {
            cout << "One (1) layer with name >input< needed, got: " << lyr.size() << endl;
        }
        bool done=false;
        vector<string> lst;
        while (!done) {
            string cl=lyr[0];
            for (auto li : lst) if (li==cl) {
                cout << "recursion with layer: " << cl << endl;
                return false;
            }
            lst.push_back(cl);
            if (cl==lossLayer) done=true;
            else {
                lyr=getLayerFromInput(cl);
                if (lyr.size()!=1) {
                    cout << "One (1) layer that uses " << cl << " as input needed, got: " << lyr.size() << endl;
                    return false;
                }
            }
        }
        if (verbose) {
            bool done=false;
            string cLay="input";
            vector<string>nLay;
            while (!done) {
                nLay=getLayerFromInput(cLay);
                string name=nLay[0];
                Layer *p=layerMap[name];
                cout << name << ": " << p->cp.getPar("topo", vector<int>{}) << " -> " << p->oTopo() << endl;
                if (p->layerInit==false) cout << "  " << name << ": bad initialization!" << endl;
                cLay=nLay[0];
                if (p->layerType==LayerType::LT_LOSS) done=true;
            }
        }
        checked=true;
        return true;
    }
    vector<string> getLayerFromInput(string input) {
        vector<string> lys;
        for (auto li : layerInputs) {
            for (auto lii : li.second) {
                if (lii==input) lys.push_back(li.first);
            }
        }
        return lys;
    }
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache, int id=0) override {
        string cLay="input";
        vector<string> nLay;
        bool done=false;
        MatrixN x0=x;
        MatrixN xn;
        trainMode = cp.getPar("train", false);
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        while (!done) {
            nLay=getLayerFromInput(cLay);
            if (nLay.size()!=1) {
                cout << "Unexpected topology: "<< nLay.size() << " layer follow layer " << cLay << " 1 expected.";
                return x;
            }
            string name=nLay[0];
            Layer *p = layerMap[name];
            t_cppl cache;
            //cache.clear();
            if (p->layerType==LayerType::LT_NORMAL) xn=p->forward(x0,&cache, id);
            else xn=p->forward(x0,y,&cache, id);
            if (pcache!=nullptr) {
                mlPush(name, &cache, pcache);
            } else {
                cppl_delete(&cache);
            }
            if (p->layerType==LayerType::LT_LOSS) done=true;
            cLay=name;
            int oi=-10;
            int fi=-10;
            bool cont=false;
            bool inferr=false;
            for (int i=0; i<xn.size(); i++) {
                if (std::isnan(xn(i)) || std::isinf(xn(i))) {
                    if (i-1==oi) {
                        if (!cont) {
                            cont=true;
                        }
                    } else {
                        cout << "[" << i;
                        if (std::isnan(xn(i))) cout << "N"; else cout <<"I";
                        fi=i;
                        cont=false;
                    }
                    oi=i;
                    inferr=true;
                } else {
                    if (fi==i-1) {
                        cout << "]";
                        cont=false;
                    } else if (oi==i-1) {
                        cont=false;
                        cout << ".." << oi;
                        if (std::isnan(xn(oi))) cout << "N"; else cout <<"I";
                        cout << "]";
                    }
                }
            }
            if (inferr) {
                cout << endl << "Internal error, layer " << name << " resulted in NaN/Inf values! ABORT." << endl;
                //cout << "x:" << x0 << endl;
                cout << "y=" << name << "(x):" << shape(x0) << "->" << shape(xn) << endl;
                peekMat("x:", x0);
                cout << "y=" << name << "(x):";
                peekMat("", xn);
                exit(-1);
            }
            x0=xn;
        }
        return xn;
    }
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) override {
        t_cppl cache;
        if (lossLayer=="") {
            cout << "Invalid configuration, no loss layer defined!" << endl;
            return 1000.0;
        }
        Layer* pl=layerMap[lossLayer];
        mlPop(lossLayer, pcache, &cache);
        floatN ls=pl->loss(y, &cache);
        return ls;
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        if (lossLayer=="") {
            cout << "Invalid configuration, no loss layer defined!" << endl;
            return y;
        }
        bool done=false;
        MatrixN dxn;
        string cl=lossLayer;
        MatrixN dx0=y;
        trainMode = cp.getPar("train", false);
        while (!done) {
            t_cppl cache;
            t_cppl grads;
            //cache.clear();
            //grads.clear();
            Layer *pl=layerMap[cl];
            mlPop(cl,pcache,&cache);
            dxn=pl->backward(dx0, &cache, &grads, id);
            mlPush(cl,&grads,pgrads);
            vector<string> lyr=layerInputs[cl];
            if (lyr[0]=="input") {
                done=true;
            } else {
                cl=lyr[0];
                dx0=dxn;
            }
        }
        return dxn;
    }
    virtual bool update(Optimizer *popti, t_cppl *pgrads, string var, t_cppl *pocache) override {
        for (auto ly : layerMap) {
            t_cppl grads;
            string cl=ly.first;
            Layer *pl=ly.second;
            mlPop(cl, pgrads, &grads);
            pl->update(popti,&grads, var+layerName+cl, pocache);  // XXX push/pop pocache?
        }
        return true;
    }
    virtual void setFlag(string name, bool val) override {
        cp.setPar(name,val);
        for (auto ly : layerMap) {
            ly.second->setFlag(name, val);
        }
    }

};

class LayerBlock : public Layer {
private:
    void setup(const CpParams& cx) {
        cp=cx;
        layerName="block"; //cp.getPar("name","block");
        lossLayer="";
        layerType=LayerType::LT_NORMAL;
        trainMode = cp.getPar("train", false);
        checked=false;
    }
public:
    map<string, Layer*> layerMap;
    map<string, vector<string>> layerInputs;
    string lossLayer;
    bool checked;
    bool trainMode;

    LayerBlock(const CpParams& cx) {
        setup(cx);
    }
    LayerBlock(const string conf) {
        setup(CpParams(conf));
        layerInit=true;
    }
    ~LayerBlock() {
        for (auto pli : layerMap) {
            if (pli.second != nullptr) {
                free(pli.second);
                pli.second=nullptr;
            }
        }
        layerMap.clear();
    }
    bool removeLayer(const string name) {
        auto fi=layerMap.find(name);
        if (fi == layerMap.end()) {
            cout << "Cannot remove layer: " << name << ", a layer with this name does not exist in block " << layerName << endl;
            return false;
        }
        free(fi->second);
        layerMap.erase(fi);
        return true;
    }
    bool addLayer(const string name, const CpParams& cp, const vector<string> inputLayers) {
        if (layerMap.find(name) != layerMap.end()) {
            cout << "Cannot add layer: " << name << ", a layer with this name is already part of block " << layerName << endl;
            return false;
        }
        if (_syncogniteLayerFactory.mapl.find(name) == _syncogniteLayerFactory.mapl.end()) {
            cout << "Cannot add layer: " << name << ", a layer with this name is not defined." << endl;
            return false;
        }
        layerMap[name]=CREATE_LAYER(name, cp)   // Macro!
        Layer *pLayer = layerMap[name];
        if (pLayer->layerInit==false) {
            cout << "Attempt to add layer " << name << " failed: Bad initialization." << endl;
            removeLayer(name);
            return false;
        }
        if (pLayer->layerType==LayerType::LT_LOSS) {
            if (lossLayer!="") {
                cout << "ERROR: a loss layer with name: " << lossLayer << "has already been defined, cannot add new loss layer: " << name << " to " << layerName << endl;
                removeLayer(name);
                return false;
            }
            layerType=LayerType::LT_LOSS;
            lossLayer=name;
        }
        layerInputs[name]=inputLayers;
        mlPush(name, &(pLayer->params), &params);
        checked=false;
        return true;
    }
    bool addLayer(string name, string params, vector<string> inputLayers) {
        return addLayer(name, CpParams(params), inputLayers);
    }

    bool checkTopology(bool verbose=false) {
        if (lossLayer=="") {
            cout << "No loss layer defined!" << endl;
            return false;
        }
        vector<string> lyr;
        lyr=getLayerFromInput("input");
        if (lyr.size()!=1) {
            cout << "One (1) layer with name >input< needed, got: " << lyr.size() << endl;
        }
        bool done=false;
        vector<string> lst;
        while (!done) {
            string cl=lyr[0];
            for (auto li : lst) if (li==cl) {
                cout << "recursion with layer: " << cl << endl;
                return false;
            }
            lst.push_back(cl);
            if (cl==lossLayer) done=true;
            else {
                lyr=getLayerFromInput(cl);
                if (lyr.size()!=1) {
                    cout << "One (1) layer that uses " << cl << " as input needed, got: " << lyr.size() << endl;
                    return false;
                }
            }
        }
        if (verbose) {
            bool done=false;
            string cLay="input";
            vector<string>nLay;
            while (!done) {
                nLay=getLayerFromInput(cLay);
                string name=nLay[0];
                Layer *p=layerMap[name];
                cout << name << ": " << p->cp.getPar("topo", vector<int>{}) << " -> " << p->oTopo() << endl;
                if (p->layerInit==false) cout << "  " << name << ": bad initialization!" << endl;
                cLay=nLay[0];
                if (p->layerType==LayerType::LT_LOSS) done=true;
            }
        }
        checked=true;
        return true;
    }
    vector<string> getLayerFromInput(string input) {
        vector<string> lys;
        for (auto li : layerInputs) {
            for (auto lii : li.second) {
                if (lii==input) lys.push_back(li.first);
            }
        }
        return lys;
    }
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache, int id=0) override {
        string cLay="input";
        vector<string> nLay;
        bool done=false;
        MatrixN x0=x;
        MatrixN xn;
        trainMode = cp.getPar("train", false);
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        while (!done) {
            nLay=getLayerFromInput(cLay);
            if (nLay.size()!=1) {
                cout << "Unexpected topology: "<< nLay.size() << " layer follow layer " << cLay << " 1 expected.";
                return x;
            }
            string name=nLay[0];
            Layer *p = layerMap[name];
            t_cppl cache;
            //cache.clear();
            if (p->layerType==LayerType::LT_NORMAL) xn=p->forward(x0,&cache, id);
            else xn=p->forward(x0,y,&cache, id);
            if (pcache!=nullptr) {
                mlPush(name, &cache, pcache);
            } else {
                cppl_delete(&cache);
            }
            if (p->layerType==LayerType::LT_LOSS) done=true;
            cLay=name;
            int oi=-10;
            int fi=-10;
            bool cont=false;
            bool inferr=false;
            for (int i=0; i<xn.size(); i++) {
                if (std::isnan(xn(i)) || std::isinf(xn(i))) {
                    if (i-1==oi) {
                        if (!cont) {
                            cont=true;
                        }
                    } else {
                        cout << "[" << i;
                        if (std::isnan(xn(i))) cout << "N"; else cout <<"I";
                        fi=i;
                        cont=false;
                    }
                    oi=i;
                    inferr=true;
                } else {
                    if (fi==i-1) {
                        cout << "]";
                        cont=false;
                    } else if (oi==i-1) {
                        cont=false;
                        cout << ".." << oi;
                        if (std::isnan(xn(oi))) cout << "N"; else cout <<"I";
                        cout << "]";
                    }
                }
            }
            if (inferr) {
                cout << endl << "Internal error, layer " << name << " resulted in NaN/Inf values! ABORT." << endl;
                //cout << "x:" << x0 << endl;
                cout << "y=" << name << "(x):" << shape(x0) << "->" << shape(xn) << endl;
                peekMat("x:", x0);
                cout << "y=" << name << "(x):";
                peekMat("", xn);
                exit(-1);
            }
            x0=xn;
        }
        return xn;
    }
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) override {
        t_cppl cache;
        if (lossLayer=="") {
            cout << "Invalid configuration, no loss layer defined!" << endl;
            return 1000.0;
        }
        Layer* pl=layerMap[lossLayer];
        mlPop(lossLayer, pcache, &cache);
        floatN ls=pl->loss(y, &cache);
        return ls;
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        if (lossLayer=="") {
            cout << "Invalid configuration, no loss layer defined!" << endl;
            return y;
        }
        bool done=false;
        MatrixN dxn;
        string cl=lossLayer;
        MatrixN dx0=y;
        trainMode = cp.getPar("train", false);
        while (!done) {
            t_cppl cache;
            t_cppl grads;
            //cache.clear();
            //grads.clear();
            Layer *pl=layerMap[cl];
            mlPop(cl,pcache,&cache);
            dxn=pl->backward(dx0, &cache, &grads, id);
            mlPush(cl,&grads,pgrads);
            vector<string> lyr=layerInputs[cl];
            if (lyr[0]=="input") {
                done=true;
            } else {
                cl=lyr[0];
                dx0=dxn;
            }
        }
        return dxn;
    }
    virtual bool update(Optimizer *popti, t_cppl *pgrads, string var, t_cppl *pocache) override {
        for (auto ly : layerMap) {
            t_cppl grads;
            string cl=ly.first;
            Layer *pl=ly.second;
            mlPop(cl, pgrads, &grads);
            pl->update(popti,&grads, var+layerName+cl, pocache);  // XXX push/pop pocache?
        }
        return true;
    }
    virtual void setFlag(string name, bool val) override {
        cp.setPar(name,val);
        for (auto ly : layerMap) {
            ly.second->setFlag(name, val);
        }
    }
};
#endif
