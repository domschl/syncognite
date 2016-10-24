#ifndef _CPL_CONVOLUTION_H
#define _CPL_CONVOLUTION_H

#include "../cp-layers.h"

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
            cerr << "bad configuration, pad:" << pad << ">=" << " HH,WW:" << HH << "," << WW << endl;
            retval=false;
        }
        if ((H + 2 * pad - HH) % stride != 0) {
            int r=(H + 2 * pad - HH) % stride;
            if (r>pad) {
                cerr << "H <-> stride does not fit! r=" << r << ", pad=" << pad << endl;
                retval=false;
            }
        }
        if ((W + 2 * pad - WW) % stride != 0) {
            int r=(W + 2 * pad - WW) % stride;
            if (r>pad) {
                cerr << "w <-> stride does not fit! r=" << r << ", pad=" << pad << endl;
                retval=false;
            }
        }
        HO = 1 + (H + 2 * pad - HH) / stride;
        WO = 1 + (W + 2 * pad - WW) / stride;

        if (HO*stride+HH-stride < H+pad) {
            cerr << "H: current stride:" << stride << ", pad:" << pad << " combination does not cover input-field" << endl;
            retval=false;
        }
        if (WO*stride+WW-stride < W+pad) {
            cerr << "W: current stride:" << stride << ", pad:" << pad << " combination does not cover input-field" << endl;
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
        cerr << "col2im-in :" << endl << y2c << endl << endl;
        cerr << "col2im-out:" << endl << xx << endl << endl;
//        cerr << "." << endl;
        return xx;
    }

    MatrixN col2im(const MatrixN& y2c, int N) {
        //cerr << N << "," << F << "," << HO << "," << WO << endl;
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
        //cerr << N << "," << F << "," << HO << "," << WO << endl;
        MatrixN y=y2c.transpose();
        Eigen::Map<MatrixN>xx0(y.data(),HO*WO,N*F);
        MatrixN xxt=xx0.transpose();
        MatrixN xx(N,F*HO*WO);
//        cerr << endl << xxt << endl;
        for (int i=0; i<F; i++) {
            xx.block(0,HO*WO*i,N,HO*WO)=xxt.block(N*i,0,N,HO*WO);
        }

//        cerr << "col2im-in :" << endl << y2c << endl << endl;
//        cerr << "col2im-out:" << endl << xx << endl << endl;
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
                    cerr << "iy:" << f << "," << x << endl;
                }
                if (py>=dy.rows() || px>=dy.cols()) {
                    cerr << "dy:" << py << "," << px << endl;
                }
*/
                iy(f,x)=dy(py,px);
            }
        }
        cerr << "icol2imx:" << shape(dy) << shape(iy) << endl;
        cerr << "icol2im-in :" << endl << dy << endl << endl;
        cerr << "icol2im-out:" << endl << iy << endl << endl;
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
                cerr << "Bad index " << i << ":" << py << "," << px << endl;
                exit(-1);
            }
            if (sy>=dy.rows() || sx>=dy.cols()) {
                cerr <<  "Bad s-index " << i << ":" << sy << "," << sx << endl;
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
/*        cerr << "icol2im:" << shape(dy) << shape(iy) << endl;
        cerr << "icol2im-in :" << endl << dy << endl << endl;
        cerr << "icol2im-out:" << endl << iy << endl << endl;
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
            cerr << "ConvFw: Invalid input data x: expected C*H*W=" << C*H*W << ", got: " << shape(x)[1] << endl;
            return MatrixN(0,0);
        }

        // x: N, C, H, W;  w: F, C, HH, WW
        if (mlverbose) t.startWall();
        im2col(x, px2c);
        if (mlverbose) cerr << "im2col:"<<t.stopWallMicro()<<"µs"<<endl;

        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x)); // XXX where do we need x?
        if (pcache!=nullptr) cppl_set(pcache, "x2c", px2c);

/*        cerr <<"x:"<<shape(x)<<endl;
        cerr <<"px2c:"<<shape(*px2c)<<endl;
        cerr << "W:"<<shape(*params["W"]) << endl;
        cerr << "b:"<<shape(*params["b"]) << endl;
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
            cerr << "matmul:"<<t.stopWallMicro()<<"µs"<<endl;
        }
        if (mlverbose) t.startWall();
        MatrixN y=col2im(y2c, N);
        if (mlverbose) cerr << "col2im:"<<t.stopWallMicro()<<"µs" << shape(y2c) << "->" << shape(y)<<endl;
        // cerr <<"col2im y2c:"<<shape(y2c)<<"->y:"<<shape(y)<<endl;
        if (pcache==nullptr) delete px2c;
        return y;
    }
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        int N=shape(dchain)[0];
        if (shape(dchain)[1]!=(unsigned int)F*HO*WO) {
            cerr << "ConvBw: Invalid input data dchain: expected F*HO*WO=" << F*HO*WO << ", got: " << shape(dchain)[1] << endl;
            return MatrixN(0,0);
        }
        int algo=0;
        Timer t;
        #ifdef  USE_GPU
        algo=1;
        #endif
        /*
        cerr << "dchain:" << shape(dchain) << endl;
        cerr << "W:" << shape(*params["W"]) << endl;
        cerr << "x:" << shape(*(*pcache)["x"]) << endl;
        cerr << "x2c:" << shape(*(*pcache)["x2c"]) << endl;
        cerr << "WO:" << WO << "," << "HO:" << HO << endl;
        */
        if (mlverbose) t.startWall();
        MatrixN dc2=icol2im(dchain,N);
        if (mlverbose) cerr << "icol2im:"<<t.stopWallMicro()<<"µs"<<endl;

        MatrixN dx;
        if (algo==0 || id>=numGpuThreads) {
            if (mlverbose) t.startWall();
            MatrixN dx2c = dc2.transpose() * (*params["W"]); // dx
            if (mlverbose) cerr << "bw-m1:"<<t.stopWallMicro()<<"µs"<<endl;
            if (mlverbose) t.startWall();
            dx=iim2col(dx2c.transpose(), N);
            if (mlverbose) cerr << "iim2col:"<<t.stopWallMicro()<<"µs"<<endl;
            if (mlverbose) t.startWall();
            cppl_set(pgrads, "W", new MatrixN(dc2 * (*(*pcache)["x2c"]).transpose())); //dW
            cppl_set(pgrads, "b", new MatrixN(dc2.rowwise().sum())); //db
            if (mlverbose) cerr << "bw-m2:"<<t.stopWallMicro()<<"µs"<<endl;
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

#endif
