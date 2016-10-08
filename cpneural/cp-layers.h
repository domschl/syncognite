#ifndef _CP_LAYERS_H
#define _CP_LAYERS_H

#include "cp-math.h"
#include "cp-layer.h"
#include "cp-timer.h"
#include <time.h>

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
    int hidden;
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
        outputShape={hidden};

        cppl_set(&params, "W", new MatrixN(inputShapeFlat,hidden)); // W
        cppl_set(&params, "b", new MatrixN(1,hidden)); // b
        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();

        params["W"]->setRandom();
        floatN xavier = 1.0/std::sqrt((floatN)(inputShapeFlat+hidden)); // (setRandom is [-1,1]-> fakt 0.5, xavier is 2/(ni+no))
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
        inputShapeRang=1;
        vector<int> inputShape=cp.getPar("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        outputShape=inputShape;
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
        outputShape={hidden};
        CpParams ca;
        ca.setPar("inputShape", vector<int>{inputShapeFlat});
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


// Batch normalization
class BatchNorm : public Layer {
private:
    void setup(const CpParams& cx) {
        layerName="BatchNorm";
        layerType=LayerType::LT_NORMAL;
        inputShapeRang=1;
        cp=cx;
        eps = cp.getPar("eps", (floatN)1e-5);
        momentum = cp.getPar("momentum", (floatN)0.9);
        trainMode = cp.getPar("train", (bool)false);
        vector<int> inputShape=cp.getPar("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        outputShape={inputShape};

        MatrixN *pgamma=new MatrixN(1,inputShapeFlat);
        pgamma->setOnes();
        cppl_set(&params,"gamma",pgamma);
        MatrixN *pbeta=new MatrixN(1,inputShapeFlat);
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
    bool freeze;
    void setup(const CpParams& cx) {
        layerName="Dropout";
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        inputShapeRang=1;
        vector<int> inputShape=cp.getPar("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        outputShape={inputShape};
        drop = cp.getPar("drop", (floatN)0.5);
        trainMode = cp.getPar("train", (bool)false);
        freeze = cp.getPar("freeze", (bool)false);
        if (freeze) srand(123);
        else srand(time(nullptr));
        layerInit=true;
    }
public:
    floatN drop;
    bool trainMode;

    Dropout(const CpParams& cx) {
        setup(cx);
    }
    Dropout(string conf) {
        setup(CpParams(conf));
    }
    ~Dropout() {
        cppl_delete(&params);
    }

    /* not thread safe
    unsigned long fastrand(void) {          //period 2^96-1
    static unsigned long x=123456789, y=362436069, z=521288629;
    unsigned long t;
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

       t = x;
       x = y;
       y = z;
       z = t ^ x ^ y;

      return z;
    } */
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id=0) override {
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        drop = cp.getPar("drop", (floatN)0.5);
        if (drop==1.0) return x;
        trainMode = cp.getPar("train", false);
        freeze = cp.getPar("freeze", false);
        if (freeze) srand(123);
        MatrixN xout;
        if (trainMode) {
            MatrixN* pmask;
            if (freeze && pcache!=nullptr && pcache->find("dropmask")!=pcache->end()) {
                pmask=(*pcache)["dropmask"];
                xout = x.array() * (*pmask).array();
            } else {
                pmask=new MatrixN(x);
                // pmask->setRandom();
                //pmask->setZero();
                int dr=(int)(drop*1000.0);
                for (int i=0; i<x.size(); i++) {
                    //if (((*pmask)(i)+1.0)/2.0 < drop) (*pmask)(i)=1.0;
                    //else (*pmask)(i)=0.0;
                    //if (fastrand()%1000 < dr) (*pmask)(i)=1.0;
                    if (rand()%1000 < dr) (*pmask)(i)=1.0;
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
    bool mlverbose;
    void setup(const CpParams& cx) {
        layerName="Convolution";
        inputShapeRang=3;
        bool retval=true;
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        vector<int> inputShape=cp.getPar("inputShape",vector<int>{});
        if (inputShape.size()!=3) {
            retval=false;
        } else { // inputShape: C, H, W;
            C=inputShape[0]; H=inputShape[1]; W=inputShape[2];
        }

        vector<int> kernel=cp.getPar("kernel", vector<int>{});
        if (kernel.size()!=3) {
            retval=false;
            F=0; HH=0; WW=0;
        } else { // Kernel: F, HH, WW
            F=kernel[0]; HH=kernel[1]; WW=kernel[2];
        }
        if (F*HH*WW==0) {
            F=16; HH=3; WW=3;
        }

        // W: F, C, HH, WW
        //cppl_set(&params, "Wb", new MatrixN(F,C*HH*WW+1)); // Wb, b= +1!
        cppl_set(&params, "W", new MatrixN(F,C*HH*WW));
        cppl_set(&params, "b", new MatrixN(F,1));
        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();

        stride = cp.getPar("stride", 1);
        mlverbose = cp.getPar("verbose", false);
        pad = cp.getPar("pad", (int)((HH-1)/2));
        if (pad>=HH || pad>=WW) {
            cout << "bad configuration, pad:" << pad << ">=" << " HH,WW:" << HH << "," << WW << endl;
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

        outputShape={F,WO,HO};

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

    void im2col(const MatrixN &xx, MatrixN *px2c) {
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
                            if (ys<0 || ys>=H) continue; // pad -> zero
                            xxso=cchw+ys*W;
                            cchhwwcyww=cchhww+cy*WW;
                            for (cx=0; cx<WW; cx++) {
                                xs=x0+cx;
                                if (xs<0 || xs>=W) continue;
                                xxs=xxso+xs;
                                pix=xx(n,xxs);
                                yd=cchhwwcyww+cx;
                                (*px2c)(yd,xd)=pix;
                            }
                        }
                    }
                }
            }
        }
    }

    MatrixN iim2col(const MatrixN &x2c, int N) {
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
                            if (ys<0 || ys>=H) continue;
                            xxso=cchw+ys*W;
                            cchhwwcyww=cchhww+cy*WW;
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

    MatrixN col2imx(const MatrixN& y2c, int N) {
        MatrixN xx(N,F*WO*HO);
//        int err=0;
        int WHO=WO*HO;
        int NWHO=N*WHO;
        int p,ox,px,py;
        int nfwho;
        int nwho;
        for (int n=0; n<N; n++) {
            nfwho=n*F*WHO;
            nwho=n*WHO;
            for (int x=0; x<F*WHO; x++) {
                p=nfwho+x;
                ox=p%NWHO;
                py=(p/WHO)%F;
                px=ox%WHO+nwho;
                xx(n,x)=y2c(py,px);
            }
        }
        cout << "col2im-in :" << endl << y2c << endl << endl;
        cout << "col2im-out:" << endl << xx << endl << endl;
//        cout << "." << endl;
        return xx;
    }

    MatrixN col2im(const MatrixN& y2c, int N) {
        //cout << N << "," << F << "," << HO << "," << WO << endl;
        MatrixN xx(N,F*HO*WO);
        int py=0,px=0;
        int sx=0,sy=0;
        int MX=y2c.cols();
        int NX=HO*WO;
        int c=0;
        for (int i=0; i<y2c.size(); i++) {
            xx(py,px)=y2c(sy,sx);
            ++sx;
            if (sx==MX) {
                sx=0; ++sy;
            }
            ++c;
            ++px;
            if (c==NX) {
                c=0;
                px-=NX;
                ++py;
                if (py==N) {
                    py=0;
                    px+=NX;
                }
            }
        }
        return xx;
    }

    MatrixN col2imB(const MatrixN& y2c, int N) {
        //cout << N << "," << F << "," << HO << "," << WO << endl;
        MatrixN y=y2c.transpose();
        Eigen::Map<MatrixN>xx0(y.data(),HO*WO,N*F);
        MatrixN xxt=xx0.transpose();
        MatrixN xx(N,F*HO*WO);
//        cout << endl << xxt << endl;
        for (int i=0; i<F; i++) {
            xx.block(0,HO*WO*i,N,HO*WO)=xxt.block(N*i,0,N,HO*WO);
        }

//        cout << "col2im-in :" << endl << y2c << endl << endl;
//        cout << "col2im-out:" << endl << xx << endl << endl;
        return xx;
    }

    MatrixN icol2imx(const MatrixN& dy, int N) {
        MatrixN iy(F,N*HO*WO);
        int p,ox,py,px;
        int fhwo;
        int HWO=WO*HO;
        int FHWO=F*HWO;
        int f,x;
        for (f=0; f<F; f++) {
            fhwo=f*HWO;
            for (x=0; x<N*HWO; x++) {
                p=f*N*HWO+x;
                ox=p%FHWO;
                py=(p/HWO)%N;
                px=ox%HWO+fhwo;
/*                if (f>=iy.rows() || x>=iy.cols()) {
                    cout << "iy:" << f << "," << x << endl;
                }
                if (py>=dy.rows() || px>=dy.cols()) {
                    cout << "dy:" << py << "," << px << endl;
                }
*/
                iy(f,x)=dy(py,px);
            }
        }
        cout << "icol2imx:" << shape(dy) << shape(iy) << endl;
        cout << "icol2im-in :" << endl << dy << endl << endl;
        cout << "icol2im-out:" << endl << iy << endl << endl;
        return iy;
    }

    MatrixN icol2im(const MatrixN& dy, int N) {
        MatrixN iy(F,N*HO*WO);
        int py=0,px=0;
        int sx=0,sy=0;
        int MX=dy.cols();
        int NX=HO*WO;
        int c=0;

        for (int i=0; i<dy.size(); i++) {
/*            if (py>=iy.rows() || px>=iy.cols()) {
                cout << "Bad index " << i << ":" << py << "," << px << endl;
                exit(-1);
            }
            if (sy>=dy.rows() || sx>=dy.cols()) {
                cout <<  "Bad s-index " << i << ":" << sy << "," << sx << endl;
                exit(-1);
            }
*/            iy(py,px)=dy(sy,sx);
            ++sx;
            if (sx==MX) {
                sx=0; ++sy;
            }
            ++c;
            ++px;
            if (c==NX) {
                c=0;
                px-=NX;
                ++py;
                if (py==F) {
                    py=0;
                    px+=NX;
                }
            }
        }
/*        cout << "icol2im:" << shape(dy) << shape(iy) << endl;
        cout << "icol2im-in :" << endl << dy << endl << endl;
        cout << "icol2im-out:" << endl << iy << endl << endl;
*/
        return iy;
    }

    MatrixN dummy(MatrixN d) {return d;}

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
        if (mlverbose) {
            y2c=dummy(y2c);
            cout << "matmul:"<<t.stopWallMicro()<<"µs"<<endl;
        }
        if (mlverbose) t.startWall();
        MatrixN y=col2im(y2c, N);
        if (mlverbose) cout << "col2im:"<<t.stopWallMicro()<<"µs" << shape(y2c) << "->" << shape(y)<<endl;
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
        Timer t;
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
        if (mlverbose) t.startWall();
        MatrixN dc2=icol2im(dchain,N);
        if (mlverbose) cout << "icol2im:"<<t.stopWallMicro()<<"µs"<<endl;

        MatrixN dx;
        if (algo==0 || id>=numGpuThreads) {
            if (mlverbose) t.startWall();
            MatrixN dx2c = dc2.transpose() * (*params["W"]); // dx
            if (mlverbose) cout << "bw-m1:"<<t.stopWallMicro()<<"µs"<<endl;
            if (mlverbose) t.startWall();
            dx=iim2col(dx2c.transpose(), N);
            if (mlverbose) cout << "iim2col:"<<t.stopWallMicro()<<"µs"<<endl;
            if (mlverbose) t.startWall();
            cppl_set(pgrads, "W", new MatrixN(dc2 * (*(*pcache)["x2c"]).transpose())); //dW
            cppl_set(pgrads, "b", new MatrixN(dc2.rowwise().sum())); //db
            if (mlverbose) cout << "bw-m2:"<<t.stopWallMicro()<<"µs"<<endl;
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
        inputShapeRang=3;  // XXX: move kernel sizes to params?
        bool retval=true;
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        vector<int> inputShape=cp.getPar("inputShape",vector<int>{0});
        assert (inputShape.size()==3);
        stride = cp.getPar("stride", 2);
        // inputShape: C, H, W        // XXX: we don't need HH und WW, they have to be equal to stride anyway!
        C=inputShape[0]; H=inputShape[1]; W=inputShape[2];
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

        outputShape={C,WO,HO};

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
    int C, H, W, N0;
    BatchNorm *pbn;
    void setup(const CpParams& cx) {
        layerName="SpatialBatchNorm";
        inputShapeRang=3;
        bool retval=true;
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        vector<int> inputShape=cp.getPar("inputShape",vector<int>{});
        assert (inputShape.size()==3);
        // inputShape: C, H, W
        C=inputShape[0]; H=inputShape[1]; W=inputShape[2];
        N0=cp.getPar("batch_size",100);   // Unusual: we need to know the batch_size for creation of the BN layer!
        outputShape={C,H,W};

        CpParams cs(cp);
        cs.setPar("inputShape",vector<int>{N0*H*W});
        pbn = new BatchNorm(cs);
        pbn->cp.setPar("train",cp.getPar("train",false));
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

    MatrixN nchw2cnhw(const MatrixN &x, const int N) {
        MatrixN xs(C,N*H*W);
        int nhw,chw,h0,h1;
        int n,c,h,w;
        int HW=H*W;
        for (n=0; n<N; n++) {  // Uhhh..
            nhw=n*HW;
            for (c=0; c<C; c++) {
                chw=c*HW;
                for (h=0; h<H; h++) {
                    h0=nhw+h*W;
                    h1=chw+h*W;
                    for (w=0; w<W; w++) {
                        xs(c,h0+w)=x(n,h1+w);
                    }
                }
            }
        }
        return xs;
    }

    MatrixN cnhw2nchw(const MatrixN& ys, const int N) {
        MatrixN y(N,C*H*W);
        int nhw,chw,h0,h1;
        int n,c,h,w;
        int HW=H*W;
        for (n=0; n<N; n++) {  // Uhhh..
            nhw=n*HW;
            for (c=0; c<C; c++) {
                chw=c*HW;
                for (h=0; h<H; h++) {
                    h0=nhw+h*W;
                    h1=chw+h*W;
                    for (w=0; w<W; w++) {
                        y(n,h1+w)=ys(c,h0+w);
                    }
                }
            }
        }
        return y;
    }

    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id=0) override {
        // XXX cache x2c and use allocated memory for im2col call!
        int N=shape(x)[0];
        pbn->cp.setPar("train",cp.getPar("train",false));
        if (shape(x)[1]!=(unsigned int)C*W*H) {
            cout << "SpatialBatchNorm Fw: Invalid input data x: expected C*H*W=" << C*H*W << ", got: " << shape(x)[1] << endl;
            return MatrixN(0,0);
        }
        if (N>N0) {
            cout << "SpatialBatchNorm Fw: batch_size at forward time" << N << " must be <= batch_size init value:" << N0 << endl;
            return MatrixN(0,0);
        }

        MatrixN xs=nchw2cnhw(x, N);
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));

        MatrixN ys;
        if (pcache != nullptr) {
            t_cppl tcachebn;
            ys=pbn->forward(xs, &tcachebn, id);
            mlPush("bn", &tcachebn, pcache);
        } else {
            ys=pbn->forward(xs, nullptr, id);
        }

        MatrixN y=cnhw2nchw(ys, N);
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        int N=shape(dchain)[0];
        pbn->cp.setPar("train",cp.getPar("train",false));
        if (shape(dchain)[1]!=(unsigned int)C*H*W) {
            cout << "SpatialBatchNorm Bw: Invalid input data dchain: expected C*H*W=" << C*H*W << ", got: " << shape(dchain)[1] << endl;
            return MatrixN(0,0);
        }

        MatrixN dcn=nchw2cnhw(dchain, N);
        t_cppl tcachebn;
        t_cppl tgradsbn;
        mlPop("bn",pcache,&tcachebn);
        MatrixN dxc=pbn->backward(dcn,&tcachebn,&tgradsbn,id);
        mlPush("bn",&tgradsbn,pgrads);

        MatrixN dx=cnhw2nchw(dxc,N);

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
        inputShapeRang=1;
        vector<int> inputShape=cp.getPar("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        outputShape={inputShapeFlat};
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
        inputShapeRang=1;
        vector<int> inputShape=cp.getPar("inputShape", vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        outputShape={inputShapeFlat};
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
    vector<int> hidden;
    void setup(const CpParams& cx) {
        bool retval=true;
        layerName="TwoLayerNet";
        layerType=LayerType::LT_LOSS;
        inputShapeRang=1;
        cp=cx;
        vector<int> inputShape=cp.getPar("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        hidden=cp.getPar("hidden",vector<int>{1024,1024});

        if (hidden.size()!=2) retval=false;

        outputShape={hidden[1]};

        CpParams c1,c2,c3,c4;
        c1.setPar("inputShape",vector<int>{inputShapeFlat});
        c1.setPar("hidden",hidden[0]);
        c2.setPar("inputShape",vector<int>{hidden[0]});
        c3.setPar("inputShape",vector<int>{hidden[0]});
        c3.setPar("hidden",hidden[1]);
        c4.setPar("inputShape",vector<int>{hidden[1]});
        af1=new Affine(c1);
        mlPush("af1", &(af1->params), &params);
        rl=new Relu(c2);
        mlPush("rl", &(rl->params), &params);
        af2=new Affine(c3);
        mlPush("af2", &(af2->params), &params);
        sm=new Softmax(c4);
        mlPush("sm", &(sm->params), &params);
        layerInit=retval;
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

class RNN : public Layer {
private:
    int numGpuThreads;
    int numCpuThreads;
    int hidden;
    void setup(const CpParams& cx) {
        layerName="RNN";
        inputShapeRang=1;
        layerType=LayerType::LT_NORMAL;
        cp=cx;
        vector<int> inputShape=cp.getPar("inputShape",vector<int>{});
        int inputShapeFlat=1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        hidden=cp.getPar("hidden",1024);
        outputShape={hidden};

        cppl_set(&params, "Wxh", new MatrixN(inputShapeFlat,hidden));
        cppl_set(&params, "Whh", new MatrixN(hidden,hidden));
        cppl_set(&params, "bh", new MatrixN(1,hidden));
        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();

        params["Wxh"]->setRandom();
        params["Whh"]->setRandom();
        floatN xavier = 1.0/std::sqrt((floatN)(inputShapeFlat+hidden)); // (setRandom is [-1,1]-> fakt 0.5, xavier is 2/(ni+no))
        *params["Wxh"] *= xavier;
        *params["Whh"] *= xavier;
        params["bh"]->setRandom();
        *params["bh"] *= xavier;
        layerInit=true;
    }
public:
    RNN(const CpParams& cx) {
        setup(cx);
    }
    RNN(const string conf) {
        setup(CpParams(conf));
    }
    ~RNN() {
        cppl_delete(&params);
    }
    /*def rnn_step_forward(x, prev_h, Wx, Wh, b):
        """
        Run the forward pass for a single timestep of a vanilla RNN that uses a
        tanh activation function.

        The input data has dimension D, the hidden state has dimension H, and we
        use a minibatch size of N.

        Inputs:
        - x: Input data for this timestep, of shape (N, D).
        - prev_h: Hidden state from previous timestep, of shape (N, H)
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases of shape (H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - cache: Tuple of values needed for the backward pass.
        """
        next_h, cache = None, None

        phh = np.dot(prev_h, Wh)
        phx = np.dot(x, Wx)
        pshx = phx + b
        ps = phh + pshx
        next_h = np.tanh(ps)
        cache = (next_h, Wx, Wh, prev_h, x)

        return next_h, cache */
        virtual MatrixN forward_step(const MatrixN& x, t_cppl* pcache, int id=0) {
            // h(t)=tanh(Whh·h(t-1) + Wxh·x(t) + bh)
            if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
            int N=shape(x)[0];
            MatrixN *ph;
            if (pcache==nullptr || pcache->find("h")==pcache->end()) {
                ph=new MatrixN(N,hidden);
                ph->setZero();
                if (pcache!=nullptr) cppl_set(pcache,"h",ph);
            } else {
                ph=(*pcache)["h"];
                if (pcache->find("ho")==pcache->end()) cppl_set(pcache,"ho",new MatrixN(*ph));
                else cppl_update(pcache,"ho",ph);
            }
            MatrixN hn = ((*ph * *params["Whh"] + x * *params["Wxh"]).rowwise() + RowVectorN(*params["bh"])).array().tanh();
            *ph=hn;
            if (pcache==nullptr) free(ph);
            return hn;
        }

    /*def rnn_forward(x, h0, Wx, Wh, b):
        """
        Run a vanilla RNN forward on an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The RNN uses a hidden
        size of H, and we work over a minibatch containing N sequences. After
        running the RNN forward, we return the hidden states for all timesteps.

        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D).
        - h0: Initial hidden state, of shape (N, H)
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases of shape (H,)

        Returns a tuple of:
        - h: Hidden states for the entire timeseries, of shape (N, T, H).
        - cache: Values needed in the backward pass
        """
        h, cache = None, None

        N, T, D = x.shape
        H = h0.shape[1]

        cai = []
        h = np.zeros((N, T, H))
        ht = h0
        for i in range(T):
            xi = x[:, i, :]
            ht, ci = rnn_step_forward(xi, ht, Wx, Wh, b)
            h[:, i, :] = ht
            cai.append(ci)
        cache = (h, h0, Wx, Wh, x, cai)
        return h, cache
*/
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id=0) override {
        if (params["Wxh"]->rows() != x.cols()) {
            cout << layerName << ": " << "Forward: dimension mismatch in x*Wxh: x:" << shape(x) << " Wxh:" << shape(*params["Wxh"]) << endl;
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

            #endif
        }
        return y;
    //return (x* *params["W"]).rowwise() + RowVectorN(*params["b"]);
    }

/*
def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (N, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.
    #                                                                        #
    # HINT: For the tanh function, you can compute the local derivative in
    # terms of the output value from tanh.                                   #
    ##########################################################################
    next_h, Wx, Wh, prev_h, x = cache

    algo = 1
    if algo == 0:
        # next_h = np.tanh(ps)
        # dps = # np.dot((np.ones(next_h.shape) - next_h ** 2), dnext_h.T)
        dps = (np.ones(next_h.shape) - next_h ** 2) * dnext_h
        # ps = phh + pshx
        dphh = dps
        dpshx = dps
        # pshx = phx + b
        dphx = dpshx.T
        db = np.sum(dpshx, axis=0)
        # phx = np.dot(x, Wx)
        dx = np.dot(Wx, dphx).T
        dWx = np.dot(dphx, x).T
        # phh = np.dot(prev_h, Wh)
        dprev_h = np.dot(Wh, dphh.T).T
        print(prev_h.shape, dprev_h.shape)
        dWh = np.dot(prev_h.T, dphh)
    elif algo == 1:
        # next_h = np.tanh(ps)
        # dps = # np.dot((np.ones(next_h.shape) - next_h ** 2), dnext_h.T)
        dps = (np.ones(next_h.shape) - next_h ** 2) * dnext_h
        dpst = dps.T
        # ps = phh + pshx
        # pshx = phx + b
        db = np.sum(dps, axis=0)
        # phx = np.dot(x, Wx)
        dx = np.dot(Wx, dpst).T
        dWx = np.dot(dpst, x).T
        # phh = np.dot(prev_h, Wh)
        dprev_h = np.dot(Wh, dpst).T
        dWh = np.dot(prev_h.T, dps)
    ##########################################################################
    #                               END OF YOUR CODE                         #
    ##########################################################################
    return dx, dprev_h, dWx, dWh, db
*/
virtual MatrixN backward_step(const MatrixN& dchain, t_cppl* pcache, t_cppl* pgrads, int id=0) {
    if (pcache->find("x") == pcache->end()) {
        cout << "cache does not contain x -> fatal!" << endl;
    }
    MatrixN x(*(*pcache)["x"]);
    MatrixN dx(x);
    dx.setZero();
    MatrixN Wxh(*params["Wxh"]);
    MatrixN Whh(*params["Whh"]);
    MatrixN bh(*params["bh"]);
    MatrixN h(*(*pcache)["h"]);
    MatrixN ho(*(*pcache)["ho"]);

    MatrixN dWxh(Wxh);
    dWxh.setZero();
    MatrixN dWhh(Whh);
    dWhh.setZero();
    MatrixN dbh(bh);
    dbh.setZero();
    MatrixN dh(h);
    dh.setZero();

    MatrixN hsq = h.array() * dchain.array();
    MatrixN hone = MatrixN(h);
    h.setOnes();
    MatrixN t1=(hone-hsq).array() * h.array();
    MatrixN t1t=t1.transpose();
    dbh=t1.colwise().sum();
    dx=(Wxh * t1t).transpose();
    dWxh=(t1t * x).transpose();
    dh=(Whh * t1t).transpose();
    dWhh=ho.transpose() * t1;

    (*pgrads)["Wxh"] = new MatrixN(dWxh);
    (*pgrads)["Whh"] = new MatrixN(dWhh);
    (*pgrads)["bh"] = new MatrixN(dbh);
    (*pgrads)["h"] = new MatrixN(dh);

    return dx;
}


/*    """
    Compute the backward pass for a vanilla RNN over an entire sequence of
    data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None

    h, h0, Wx, Wh, x, cai = cache
    N, T, H = dh.shape
    _, _, D = x.shape
    init = 0

    dx = np.zeros((N, T, D))
    dh0 = np.zeros(dh.shape)
    # prev_h = h0
    dph = 0.0
    for i in reversed(range(T)):
        dph = dh[:, i, :] + dph
        # next_h = h[:, i, :]
        ca = cai[i]
        dxi, dph, dWxi, dWhi, dbi = rnn_step_backward(dph, ca)
        dx[:, i, :] = dxi
        dh0 = dph
        if init == 0:
            init = 1
            dWx = dWxi
            dWh = dWhi
            db = dbi
        else:
            dph = dh0
            dWx += dWxi
            dWh += dWhi
            db += dbi

    return dx, dh0, dWx, dWh, db
*/
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

            cppl_set(pgrads, "b", new MatrixN(dchain.colwise().sum())); //db
            #endif
        }
        return dx;
    }
};


void registerLayers() {
    REGISTER_LAYER("Affine", Affine, 1)
    REGISTER_LAYER("Relu", Relu, 1)
    REGISTER_LAYER("AffineRelu", AffineRelu, 1)
    REGISTER_LAYER("BatchNorm", BatchNorm, 1)
    REGISTER_LAYER("Dropout", Dropout, 1)
    REGISTER_LAYER("Convolution", Convolution, 3)
    REGISTER_LAYER("Pooling", Pooling, 3)
    REGISTER_LAYER("SpatialBatchNorm", SpatialBatchNorm, 3)
    REGISTER_LAYER("Softmax", Softmax, 1)
    REGISTER_LAYER("Svm", Svm, 1)
    REGISTER_LAYER("TwoLayerNet", TwoLayerNet, 1)
}


class LayerBlock : public Layer {
private:
    bool bench;
    void setup(const CpParams& cx) {
        cp=cx;
        layerName=cp.getPar("name",(string)"block");
        bench=cp.getPar("bench",false);
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
                delete pli.second;
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
        delete fi->second;
        layerMap.erase(fi);
        return true;
    }
    bool addLayer(const string layerclass, const string name, CpParams& cp, const vector<string> inputLayers) {
        if (layerMap.find(name) != layerMap.end()) {
            cout << "Cannot add layer: " << name << ", a layer with this name is already part of block " << layerName << endl;
            return false;
        }
        if (_syncogniteLayerFactory.mapl.find(layerclass) == _syncogniteLayerFactory.mapl.end() and layerclass!="Input") {
            cout << "Cannot add layer: " << layerclass << ", layer class is not defined." << endl;
            return false;
        }
        string firstInput="";  // XXX multiple input layers!
        for (auto li : inputLayers) {
            if (li!="input") {
                if (layerMap.find(li) == layerMap.end()) {
                    cout << "Cannot add layer: " << name << ", it depends on an input layer " << li << ", which is not defined." << endl;
                    return false;
                } else {
                    if (firstInput=="") firstInput=li;
                }
            } else {
                firstInput="input";
            }
        }

        if (firstInput!="" && firstInput!="input") {
            auto lP=layerMap.find(firstInput);
            if (lP==layerMap.end()) {
                cout << "Can't find input-layer: " << firstInput << " internal error in layer defintion of " << name << endl;
                return false;
            }
            vector<int> inputShape, prevOutputShape;
            inputShape=cp.getPar("inputShape", vector<int>{});
            prevOutputShape=lP->second->getOutputShape();
            if (prevOutputShape.size()==0) {
                cout << "Missing outputShape defintion for inputLayer " << firstInput << endl;
                return false;
            }
            if (inputShape.size()<prevOutputShape.size()) {
                inputShape=prevOutputShape;
            }
            for (unsigned int i=0; i<prevOutputShape.size(); i++) {
                inputShape[i]=prevOutputShape[i];
            }
            cp.setPar("inputShape",inputShape);
        }
        layerMap[name]=CREATE_LAYER(layerclass, cp)   // Macro!
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
    bool addLayer(string layerclass, string name, string params, vector<string> inputLayers) {
        CpParams cp(params);
        return addLayer(layerclass, name, cp, inputLayers);
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

                int inputShapeFlat=1;
                for (int j : p->cp.getPar("inputShape", vector<int>{})) {
                    inputShapeFlat *= j;
                }
                int outputShapeFlat=1;
                for (int j : p->getOutputShape()) {
                    outputShapeFlat *= j;
                }

                cout << name << ": " << p->cp.getPar("inputShape", vector<int>{}) << "[" << inputShapeFlat << "]";
                cout << " -> " << p->getOutputShape() << "[" << outputShapeFlat << "]" << endl;

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
        Timer t;
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
            if (bench) t.startWall();
            if (p->layerType==LayerType::LT_NORMAL) xn=p->forward(x0,&cache, id);
            else xn=p->forward(x0,y,&cache, id);
            if (bench) cout << name << "-fw:\t" << t.stopWallMicro() << endl;
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
        Timer t;
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
            if (bench) t.startWall();
            dxn=pl->backward(dx0, &cache, &grads, id);
            if (bench) cout << cl << "-bw:\t" << t.stopWallMicro() << endl;
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
