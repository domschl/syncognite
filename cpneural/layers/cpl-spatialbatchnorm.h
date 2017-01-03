#ifndef _CPL_SPATIALBATCHNORM_H
#define _CPL_SPATIALBATCHNORM_H

#include "../cp-layers.h"


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
        N0=cp.getPar("N",100);   // Unusual: we need to know the batch_size for creation of the BN layer!
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

    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        // XXX cache x2c and use allocated memory for im2col call!
        int N=shape(x)[0];
        pbn->cp.setPar("train",cp.getPar("train",false));
        if (shape(x)[1]!=(unsigned int)C*W*H) {
            cerr << "SpatialBatchNorm Fw: Invalid input data x: expected C*H*W=" << C*H*W << ", got: " << shape(x)[1] << endl;
            return MatrixN(0,0);
        }
        if (N>N0) {
            cerr << "SpatialBatchNorm Fw: batch_size at forward time" << N << " must be <= batch_size init value:" << N0 << endl;
            return MatrixN(0,0);
        }

        MatrixN xs=nchw2cnhw(x, N);
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));

        MatrixN ys;
        if (pcache != nullptr) {
            t_cppl tcachebn;
            ys=pbn->forward(xs, &tcachebn, pstates, id);
            mlPush("bn", &tcachebn, pcache);
        } else {
            ys=pbn->forward(xs, nullptr, pstates, id);
        }

        MatrixN y=cnhw2nchw(ys, N);
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) override {
        int N=shape(dchain)[0];
        pbn->cp.setPar("train",cp.getPar("train",false));
        if (shape(dchain)[1]!=(unsigned int)C*H*W) {
            cerr << "SpatialBatchNorm Bw: Invalid input data dchain: expected C*H*W=" << C*H*W << ", got: " << shape(dchain)[1] << endl;
            return MatrixN(0,0);
        }

        MatrixN dcn=nchw2cnhw(dchain, N);
        t_cppl tcachebn;
        t_cppl tgradsbn;
        mlPop("bn",pcache,&tcachebn);
        MatrixN dxc=pbn->backward(dcn,&tcachebn,pstates,&tgradsbn,id);
        mlPush("bn",&tgradsbn,pgrads);

        MatrixN dx=cnhw2nchw(dxc,N);

        return dx;
    }
};

#endif
