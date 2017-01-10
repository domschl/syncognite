#ifndef _CPL_AFFINE_H
#define _CPL_AFFINE_H

#include "../cp-layers.h"

class Affine : public Layer {
private:
    int numGpuThreads;
    int numCpuThreads;
    int hidden;
    floatN initfactor;
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
        XavierMode inittype=xavierInitType(cp.getPar("init",(string)"standard"));
        initfactor=cp.getPar("initfactor",(floatN)1.0);
        outputShape={hidden};

        MatrixN W = xavierInit(MatrixN(inputShapeFlat,hidden),inittype,initfactor);
        MatrixN b = xavierInit(MatrixN(1,hidden),inittype,initfactor);
        cppl_set(&params, "W", new MatrixN(W)); // W
        cppl_set(&params, "b", new MatrixN(b)); // b
        numGpuThreads=cpGetNumGpuThreads();
        numCpuThreads=cpGetNumCpuThreads();


        /*
        params["W"]->setRandom();
        floatN xavier = 1.0/std::sqrt((floatN)(inputShapeFlat+hidden)); // (setRandom is [-1,1]-> fakt 0.5, xavier is 2/(ni+no))
        *params["W"] *= xavier;
        params["b"]->setRandom();
        *params["b"] *= xavier;
        */
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
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id=0) override {
        if (params["W"]->rows() != x.cols()) {
            cerr << layerName << ": " << "Forward: dimension mismatch in x*W: x:" << shape(x) << " W:" << shape(*params["W"]) << endl;
            MatrixN y(0,0);
            return y;
        }
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));

        #ifdef USE_GPU
        int algo=1;
        #else
        int algo=0;
        #endif
        MatrixN y; // (x.rows(), params["W"]->cols());
        // y.setZero();
        if (algo==0 || id>=numGpuThreads) {
            // cerr << "C" << id << " ";
            /*
            y = x * (*params["W"]);
            RowVectorN b = *params["b"];
            y.rowwise() += b;
            */
            //cerr << "X:" << shape(x) << " W:" << shape(*(params["W"])) << " b:" << shape(*(params["b"])) << endl;
            //cerr << "x:" << x << endl;
            //cerr << "w:" << *(params["W"]) << endl;
            //cerr << "b:" << *(params["b"]) << endl;
            y=(x * (*(params["W"]))).rowwise() + RowVectorN(*params["b"]);

            //cerr << "AFy:" << shape(y) << endl << y << endl;
            //std::flush(cerr);

        } else {
            #ifdef USE_GPU
            // cerr << "G" << id << "/" << numGpuThreads << " ";
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
    virtual MatrixN backward(const MatrixN& dchain, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id=0) override {
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

#endif
