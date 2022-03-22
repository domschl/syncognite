#ifndef _CPL_POOLING_H
#define _CPL_POOLING_H

#include "../cp-layers.h"


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
    int numCpuThreads;
    int C, H, W, HH, WW;
    int HO, WO;
    int stride;
    void setup(const json& jx) {
        j=jx;
        layerName=j.value("name",(string)"Pooling");
        layerClassName="Pooling";
        inputShapeRang=3;  // XXX: move kernel sizes to params?
        bool retval=true;
        layerType=LayerType::LT_NORMAL;
        vector<int> inputShape=j.value("inputShape",vector<int>{0});
        assert (inputShape.size()==3);
        stride = j.value("stride", 2);
        // inputShape: C, H, W        // XXX: we don't need HH und WW, they have to be equal to stride anyway!
        C=inputShape[0]; H=inputShape[1]; W=inputShape[2];
        HH=stride; WW=stride;  // XXX: Simplification, our algo doesn't work for HH or WW != stride.
        if (HH!=stride || WW!=stride) {
            cerr << "Implementation only supports stride equal to HH and WW!";
            retval=false;
        }
        // W: F, C, HH, WW
        //cppl_set(&params, "Wb", new MatrixN(F,C*HH*WW+1)); // Wb, b= +1!
        numCpuThreads=cpGetNumCpuThreads();

        HO = (H-HH)/stride+1;
        WO = (W-WW)/stride+1;

        outputShape={C,WO,HO};

        layerInit=retval;
    }
public:
    Pooling(const json& jx) {
        setup(jx);
    }
    Pooling(const string conf) {
        setup(json::parse(conf));
    }
    ~Pooling() {
        cppl_delete(&params);
    }

    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        // XXX cache x2c and use allocated memory for im2col call!
        auto N=shape(x)[0];
        if (shape(x)[1]!=(unsigned int)C*W*H) {
            cerr << "PoolFw: Invalid input data x: expected C*H*W=" << C*H*W << ", got: " << shape(x)[1] << endl;
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

        //cerr << "x:" << endl << x << endl;
        //cerr << "mask:" << endl << *pmask << endl;

        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x)); // XXX where do we need x?
        if (pcache!=nullptr) cppl_set(pcache, "mask", pmask);

        if (pcache==nullptr) delete pmask;
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) override {
        int N=shape(dchain)[0];
        if (shape(dchain)[1]!=(unsigned int)C*HO*WO) {
            cerr << "PoolBw: Invalid input data dchain: expected C*HO*WO=" << C*HO*WO << ", got: " << shape(dchain)[1] << endl;
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

#endif
