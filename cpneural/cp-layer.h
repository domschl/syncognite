#ifndef _CP_LAYER_H
#define _CP_LAYER_H

#include "cp-neural.h"

//#define USE_DOUBLE
#ifndef USE_DOUBLE
#ifndef USE_FLOAT
//pragma message("Please define either USE_FLOAT or USE_DOUBLE")
#define USE_FLOAT
#endif
#endif

#ifdef USE_DOUBLE
#ifdef USE_FLOAT
#error CONFIGURATION MESS: either USE_DOUBLE or USE_FLOAT, not both!
#endif
using MatrixN=Eigen::MatrixXd;
using VectorN=Eigen::VectorXd;
using RowVectorN=Eigen::RowVectorXd;
using ColVectorN=Eigen::VectorXd;
using ArrayN=Eigen::ArrayXd;
using floatN=double;
#define CP_DEFAULT_NUM_H (1.e-6)
#define CP_DEFAULT_NUM_EPS (1.e-9)
#endif
#ifdef USE_FLOAT
using MatrixN=Eigen::MatrixXf;
using VectorN=Eigen::VectorXf;
using RowVectorN=Eigen::RowVectorXf;
using ColVectorN=Eigen::VectorXf;
using ArrayN=Eigen::ArrayXf;
using floatN=float;
#define CP_DEFAULT_NUM_H ((float)1.e-4)
#define CP_DEFAULT_NUM_EPS ((float)1.e-3)
#endif

using CpParams=ParamParser<floatN>;

#if defined (USE_VIENNACL) || (USE_CUDA)
#define USE_GPU
#endif

#ifdef USE_VIENNACL
#define VIENNACL_HAVE_EIGEN
#ifdef USE_OPENCL
#define VIENNACL_WITH_OPENCL
//#pragma message("Eigen is active with ViennaCl and OpenCL")
#else
#error "VIENNACL currently requires WITH_OPENCL Cmake option to be set."
#endif
#ifdef USE_CUDA
#define VIENNACL_WITH_CUDA
#error "CUDA option with ViennaCL currently does not work!"
#endif
#endif

#include "cp-math.h"
#include "cp-layer.h"

#ifdef USE_VIENNACL
#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/linalg/prod.hpp>
#endif

#ifdef USE_VIENNACL
#include <viennacl/ocl/device.hpp>
#include <viennacl/ocl/platform.hpp>
#include <viennacl/ocl/backend.hpp>
#endif

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

// for cpInitCompute():
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>


//using t_cppl = cp_t_params<MatrixN *>;
typedef t_param_parser<MatrixN *> t_cppl;

vector<unsigned int> shape(const MatrixN& m) {
    vector<unsigned int> s(2);
    s[0]=(unsigned int)(m.rows());
    s[1]=(unsigned int)(m.cols());
    return s;
}

bool matCompare(MatrixN& m0, MatrixN& m1, string msg="", floatN eps=1.e-6) {
    if (m0.cols() != m1.cols() || m0.rows() != m1.rows()) {
        cerr << msg << ": Incompatible shapes " << shape(m0) << "!=" << shape(m1) << endl;
        return false;
    }
    MatrixN d = m0 - m1;
    floatN dif = d.cwiseProduct(d).sum();
    if (dif < eps) {
        if (msg!="") cerr << msg << " err=" << dif << endl;
        return true;
    } else {
        if (msg!="") {
            //IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
            //cerr << msg << " m0:" << endl << m0.format(CleanFmt) << endl;
            //cerr << msg << " m1:" << endl << m1.format(CleanFmt) << endl;
            cerr << "err=" << dif << endl;
        }
        return false;
    }
}

void peekMat(const string label, const MatrixN& m) {
    cerr << label << " ";
    if (m.size()<10) cerr << m << endl;
    else {
        for (int j=0; j<m.size(); j++) {
            if (j<4 || m.size()-j < 4) cerr << m(j) << " ";
            else if (j==4) cerr << " ... ";
        }
        cerr << endl;
    }
}
#ifdef USE_CUDA
#define MAX_GPUTHREADS 64
cublasHandle_t *cuHandles;
float *cuScratch1[MAX_GPUTHREADS];
float *cuScratch2[MAX_GPUTHREADS];
float *cuScratch3[MAX_GPUTHREADS];
long maxCuScratch=0;

#define CUDA_THRESHOLD 30
#define CUDA_SCRATCH_SIZE 330000000

void checkScratch(long n, bool verbose=false) {
    if (n > maxCuScratch) {
        maxCuScratch=n;
        if (verbose) cerr << "maxCuScratch:" << maxCuScratch << endl;
        if (maxCuScratch > CUDA_SCRATCH_SIZE) {
            cerr << "Internal error, CUDA_SCRATCH_SIZE exceeded: " << maxCuScratch << " > " << CUDA_SCRATCH_SIZE << endl;
            exit(-1);
        }
    }
}

#endif

#ifdef USE_VIENNACL
#define MAX_GPUTHREADS 64
#define VIENNACL_THRESHOLD 600
#endif

MatrixN matmul(MatrixN *a, MatrixN *b, int contextId, bool verbose=false) {
    Timer t,t1;
    #ifdef USE_CUDA
    // Create a handle for CUBLAS

    if (a->rows() + a->cols() + b->cols()<CUDA_THRESHOLD) {
        if (verbose) t.startWall();
        MatrixN y= *a * (*b);
        if (verbose) cerr << "Eigen matmul " << shape(*a) << shape(*b) << "->" << t.stopWallMicro() << endl;
        return y;
    } else {

        const floatN alpha=1;
        const floatN beta=0;

        #ifdef USE_FLOAT
        if (verbose) t.startWall();
        MatrixN c(a->rows(), b->cols());
        float *ca, *cb, *cc;
        //cerr << shape(*a) << " x " << shape(*b) << endl;
        //t.startWall();
/*        cudaMalloc((void **)&ca,a->rows()*(a->cols())*sizeof(float));
        cudaMalloc((void **)&cb,b->rows()*(b->cols())*sizeof(float));
        cudaMalloc((void **)&cc,a->rows()*(b->cols())*sizeof(float));
*/      //  cerr << "  cAlloc:" << t.stopWallMicro() << endl;

        if (verbose) t1.startWall();
        checkScratch(a->rows() * (a->cols()) * sizeof(float), verbose);
        checkScratch(b->rows() * (b->cols()) * sizeof(float), verbose);
        checkScratch(a->rows() * (b->cols()) * sizeof(float), verbose);

        cudaError_t cudaStat;
        cudaStat=cudaMemcpy(cuScratch1[contextId],a->data(),a->rows()*(a->cols())*sizeof(float),cudaMemcpyHostToDevice);
        if (cudaStat != cudaSuccess) {
            cerr << "cudaMemcpy1 failed: " << cudaStat << endl;
            exit(-1);
        }
        int sz=b->rows()*(b->cols())*sizeof(float);
        cudaStat=cudaMemcpy(cuScratch2[contextId],b->data(),sz,cudaMemcpyHostToDevice);
        if (cudaStat != cudaSuccess) {
            cerr << "cudaMemcpy2 failed:" << cudaStat << "Size: " << sz << endl;
            switch (cudaStat) {
                case cudaErrorInvalidValue:
                    cerr << "InvVal";
                    break;
                case cudaErrorInvalidPitchValue:
                    cerr << "InvPitch";
                    break;
                case cudaErrorInvalidDevicePointer:
                    cerr << "InvDevPtr";
                    break;
                case cudaErrorInvalidMemcpyDirection:
                    cerr << "InvDir";
                    break;
                default:
                    cerr << "Undocumented error";
                    break;
            }
            cerr << endl;
            exit(-1);
        }
        if (verbose) cerr << "  cMemcpy:" << t1.stopWallMicro() << endl;

        if (verbose) t1.startWall();
        if (cublasSgemm(cuHandles[contextId], CUBLAS_OP_N, CUBLAS_OP_N, a->rows(), b->cols(), a->cols(), &alpha,
                    cuScratch1[contextId], a->rows(), cuScratch2[contextId], b->rows(), &beta,
                    cuScratch3[contextId], c.rows()) != CUBLAS_STATUS_SUCCESS) {
            cerr << "cublasSgemm failed!" << endl;
            exit(-1);

        }
        if (verbose) cerr << "  cMathML:" << t1.stopWallMicro() << endl;

        if (verbose) t1.startWall();
        cudaStat=cudaMemcpy(c.data(),cuScratch3[contextId],a->rows()*b->cols()*sizeof(float),cudaMemcpyDeviceToHost);
        if (cudaStat != cudaSuccess) {
            cerr << "cudaMemcpy3 failed:" << cudaStat << endl;
            exit(-1);
        }
        if (verbose) cerr << "  cMemcp2:" << t1.stopWallMicro() << endl;

        //t.startWall();
/*        cudaFree(ca);
        cudaFree(cb);
        cudaFree(cc);
*/       // cerr << "  cFree:" << t.stopWallMicro() << endl;

        if (verbose) cerr << "Cuda matmul " << shape(*a) << shape(*b) << ": " << t.stopWallMicro() << endl;
        #else
        #error "USE_DOUBLE not supported with USE_CUDA"
        #endif

        return c;
    }

    #else
    #ifdef USE_VIENNACL
    if (a->rows()+a->cols()+b->cols() < VIENNACL_THRESHOLD) {
        t.startWall();
        MatrixN y= *a* (*b);
        if (verbose) cerr << "Eigen matmul " << shape(*a) << shape(*b) << "->" << t.stopWallMicro() << endl;
        return y;
    } else {
        viennacl::context ctx(viennacl::ocl::get_context(static_cast<long>(contextId)));
        viennacl::matrix<float>vi_b(b.rows(), b.cols(), ctx);
        viennacl::matrix<float>vi_a(a.rows(), a.cols(), ctx);
        viennacl::matrix<float>vi_y(a.rows(), b.cols(), ctx);
        viennacl::copy(b, vi_b);
        viennacl::copy(a, vi_a);
        vi_y = viennacl::linalg::prod(vi_a, vi_b);
        viennacl::copy(vi_y, y);
        return y;
    }
    #else
    t.startWall();
    MatrixN y= *a* (*b);
    if (verbose) cerr << "Eigen matmul " << shape(*a) << shape(*b) << "->" << t.stopWallMicro() << endl;
    return y;
    #endif
    #endif
}

int cpNumGpuThreads=1;
int cpNumEigenThreads=1;
int cpNumCpuThreads=1;

bool threadContextInit(unsigned int numThreads) {
    #ifdef USE_VIENNACL
    if (numThreads > MAX_GPUTHREADS) numThreads=MAX_GPUTHREADS;
    if (viennacl::ocl::get_platforms().size() == 0) {
        std::cerr << "Error: No ViennaClplatform found!" << std::endl;
        return false;
    }
    viennacl::ocl::platform pf = viennacl::ocl::get_platforms()[0];
    std::vector<viennacl::ocl::device> const & devices = pf.devices();
    int nrDevs = pf.devices().size();
    cerr << nrDevs << " devices found." << endl;
    for (unsigned int i=0; i<numThreads; i++) {
        viennacl::ocl::setup_context(i, devices[i%nrDevs]); // XXX support for multiple devices is a bit basic.
        cerr << "Context " << i << " on: " << viennacl::ocl::get_context(i).devices()[0].name() << endl;
    }
    // Set context to 0 for main program, 1-numThreads for threads
    //viennacl::context ctx(viennacl::ocl::get_context(static_cast<long>(0)));
    //cerr << "Contexts created, got context 0 for main program." << endl;
    #else
    #ifdef USE_CUDA
    cuHandles=(cublasContext **)malloc(sizeof(cublasHandle_t) * cpNumGpuThreads);
    for (int i=0; i<cpNumGpuThreads; i++) {
        cublasCreate(&(cuHandles[i]));
        cerr << "Context " << i << " on: cublas" << endl;
        //cudaHostAlloc((void **)&(cuScratch1[i]), CUDA_SCRATCH_SIZE, cudaHostAllocDefault);
        //cudaHostAlloc((void **)&(cuScratch2[i]), CUDA_SCRATCH_SIZE, cudaHostAllocDefault);
        //cudaHostAlloc((void **)&(cuScratch3[i]), CUDA_SCRATCH_SIZE, cudaHostAllocDefault);

        // cudaMalloc((void **)&(cuScratch1[i]), CUDA_SCRATCH_SIZE); //, cudaHostAllocWriteCombined);
        // cudaMalloc((void **)&(cuScratch2[i]), CUDA_SCRATCH_SIZE); //, cudaHostAllocWriteCombined);
        // cudaMalloc((void **)&(cuScratch3[i]), CUDA_SCRATCH_SIZE); //, cudaHostAllocDefault);
        if (cudaMalloc((void **)&(cuScratch1[i]), CUDA_SCRATCH_SIZE)!=cudaSuccess) {
            cerr << "cudaMallocHost failed!" << endl;
            exit(-1);
        }
        if (cudaMalloc((void **)&(cuScratch2[i]), CUDA_SCRATCH_SIZE)!=cudaSuccess) {
            cerr << "cudaMallocHost failed!" << endl;
            exit(-1);
        }
        if (cudaMalloc((void **)&(cuScratch3[i]), CUDA_SCRATCH_SIZE)!=cudaSuccess) {
            cerr << "cudaMallocHost failed!" << endl;
            exit(-1);
        }
    }
    cudaSetDeviceFlags(cudaDeviceScheduleAuto); //BlockingSync); //ScheduleYield); //Spin); //cudaDeviceScheduleBlockingSync
    #endif
    #endif
    return true;
}

bool threadContextDestroy() {
    #ifdef USE_CUDA
    for (int i=0; i<cpNumGpuThreads; i++) {
        cudaFreeHost(&(cuScratch1[i]));
        cudaFreeHost(&(cuScratch2[i]));
        cudaFreeHost(&(cuScratch3[i]));
        cublasDestroy(cuHandles[i]);
    }
    free(cuHandles);
    cuHandles=nullptr;
    #endif
    return true;
}

int cpGetNumGpuThreads() {
    return cpNumGpuThreads;
}
int cpGetNumEigenThreads() {
    return cpNumEigenThreads;
}
int cpGetNumCpuThreads() {
    return cpNumCpuThreads;
}

bool cpInitCompute(string name, CpParams* poptions=nullptr) {
    CpParams cp;
    string options="";
    struct passwd *pw = getpwuid(getuid());
    const char *homedir = pw->pw_dir;
    string conffile = string(homedir) + "/.syncognite";
    std::ifstream cfile(conffile);
    if (cfile.is_open()) {
        string line;
        string conf;
        while (std::getline(cfile,line)) {
            conf+=line;
        }
        cp.setString(conf);
        cfile.close();
    } else {
        cerr << "New configureation, '" << conffile << "' not found." << endl;
    }
    // omp_set_num_threads(n)
    // Eigen::setNbThreads(n);
    // n=Eigen::nbThreads();
// myfile << "Writing this to a file.\n";


    #ifdef USE_GPU
    cpNumGpuThreads=cp.getPar("NumGpuThreads", 8);
    #else
    cpNumGpuThreads=0;
    #endif
    cpNumEigenThreads=cp.getPar("NumEigenThreads", 1);
    int numHWThreads=std::thread::hardware_concurrency();
    cpNumCpuThreads=cp.getPar("NumCpuThreads", numHWThreads);
    if (poptions!=nullptr) {
        *poptions=cp;
    }

    // Eigen::initParallel();
    Eigen::setNbThreads(cpNumEigenThreads);

    #ifdef USE_VIENNACL
    options += "VIENNACL ";
    threadContextInit(cpNumGpuThreads);
    #ifdef USE_OPENCL
    options += "OPENCL ";
    #endif
    #endif

    #ifdef USE_CUDA
    options += "CUDA ";
    threadContextInit(cpNumGpuThreads);
    #endif

    #ifdef USE_FLOAT
    options+="FLOAT ";
    #endif
    #ifdef USE_AVX
    options+="AVX ";
    #endif
    #ifdef USE_SSE2
    options+="SSE2 ";
    #endif
    #ifdef USE_SSE4
    options+="SSE4 ";
    #endif
    #ifdef USE_FMA
    options += "FMA ";
    #endif
    #ifdef USE_OPENMP
    options += "OPENMP ";
    #endif


    std::ofstream c2file(conffile);
    if (c2file.is_open()) {
        string line;
        line=cp.getString();
        c2file << line << endl;
        c2file.close();
    }
    cerr << "Compile-time options: " << options << endl;
    cerr << "Eigen is using:      " << cpNumEigenThreads << " threads." << endl;
    cerr << "CpuPool is using:    " << cpNumCpuThreads << " threads." << endl;
    cerr << "Cpu+GpuPool is using:    " << cpNumGpuThreads << " threads." << endl;
    return true;
}

void cpExitCompute() {
    #ifdef USE_GPU
    threadContextDestroy();
    #endif
}

class Optimizer {
public:
    CpParams cp;
    virtual ~Optimizer() {}; // Otherwise destructor of derived classes is never called!
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl *pcache) {return x;};
};

enum LayerType { LT_UNDEFINED, LT_NORMAL, LT_LOSS};

typedef int t_layer_props_entry;
typedef std::map<string, t_layer_props_entry> t_layer_props;

class Layer;

template<typename T>
Layer* createLayerInstance(const CpParams& cp) {
    return new T(cp);
}

//void cppl_delete(t_cppl p);
void cppl_delete(t_cppl *p) {
    int nr=0;
    if (p->size()==0) {
        return;
    }
    for (auto it : *p) {
        if (it.second != nullptr) delete it.second;
        (*p)[it.first]=nullptr;
        ++nr;
    }
    p->erase(p->begin(),p->end());
}

void cppl_set(t_cppl *p, string key, MatrixN *val) {
    auto it=p->find(key);
    if (it != p->end()) {
        cerr << "MEM! Override condition for " << key << " update prevented, freeing previous pointer..." << endl;
        delete it->second;
    }
    (*p)[key]=val;
}

void cppl_update(t_cppl *p, string key, MatrixN *val) {
    if (p->find(key)==p->end()) {
        MatrixN *pm=new MatrixN(*val);
        cppl_set(p, key, pm);
    } else {
        *((*p)[key])=*val;
    }
}

void cppl_remove(t_cppl *p, string key) {
    auto it=p->find(key);
    if (it!=p->end()) {
        if (it->second != nullptr) delete it->second;
        p->erase(it);
    }
}

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

// XXX: dubious:
void mlPopX(string prefix, t_cppl *src, t_cppl *dst) {
    for (auto ci=src->cbegin(); ci!=src->cend(); ci++) {
        if (ci->first.substr(0,prefix.size()+1)==prefix+"-") {
            cppl_set(dst, ci->first.substr(prefix.size()+1), ci->second);
            src->erase(ci);
        }
    }
}


typedef std::map<std::string, Layer*(*)(const CpParams&)> t_layer_creator_map;

class LayerFactory {
public:
    t_layer_creator_map mapl;
    t_layer_props mapprops;
    void registerInstanceCreator(std::string name, Layer*(sub)(const CpParams&), t_layer_props_entry lprops ) {
        auto it=mapl.find(name);
        if (it!=mapl.end()) {
            cerr << "Layer " << name << " is already registered, preventing additional registration." << endl;
        } else {
            mapl[name] = sub;
            mapprops[name] = lprops;
        }
    }
    Layer* createLayerInstance(std::string name, const CpParams& cp) {
        return mapl[name](cp);
    }
};

LayerFactory _syncogniteLayerFactory;

#define REGISTER_LAYER(LayerName, LayerClass, props) _syncogniteLayerFactory.registerInstanceCreator(LayerName,&createLayerInstance<LayerClass>, props);
#define CREATE_LAYER(LayerName, cp) _syncogniteLayerFactory.createLayerInstance(LayerName, cp);


class Layer {
public:
    string layerName;
    LayerType layerType;
    int inputShapeRang;
    vector<int>outputShape;
    CpParams cp;
    t_cppl params;
    bool layerInit;

    virtual ~Layer() {}; // Otherwise destructor of derived classes is never called!
    virtual vector<int> getOutputShape() { return outputShape;}
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id)  { MatrixN d(0,0); return d;}
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache, int id)  { MatrixN d(0,0); return d;}
    virtual MatrixN backward(const MatrixN& dtL, t_cppl* pcache, t_cppl* pgrads, int id) { MatrixN d(0,0); return d;}
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) { return 1001.0; }
    virtual bool update(Optimizer *popti, t_cppl* pgrads, string var, t_cppl* pocache) {
        /*for (int i=0; i<params.size(); i++) {
            *params[i] = popti->update(*params[i],*grads[i]);
        }*/
        for (auto it : params) {
            string key = it.first;
            if (pgrads->find(key)==pgrads->end()) {
                cerr << "Internal error on update of layer: " << layerName << " at key: " << key << endl;
                cerr << "Grads-vars: ";
                for (auto gi : *pgrads) cerr << gi.first << " ";
                cerr << endl;
                cerr << "Params-vars: ";
                for (auto pi : params) cerr << pi.first << " ";
                cerr << endl;
                cerr << "Irrecoverable internal error, ABORT";
                exit(-1);
            } else {
                *params[key] = popti->update(*params[key],*((*pgrads)[key]), var+key, pocache);
            }
        }
        return true;
    }
    floatN train(const MatrixN& x, const MatrixN& y, const MatrixN &xv, const MatrixN &yv,
                        string optimizer, const CpParams& cp);
    t_cppl workerThread(const MatrixN& xb, const MatrixN& yb, floatN *pl, int id);
    floatN test(const MatrixN& x, const MatrixN& y, int batchsize=100)  {
        setFlag("train",false);
        int bs=batchsize;
        int N=shape(x)[0];
        MatrixN xb,yb;
        int co=0;
        for (int ck=0; ck<(N+bs-1)/bs; ck++) {
            int x0=ck*bs;
            int dl=bs;
            if (x0+dl>N) dl=N-x0;
            xb=x.block(x0,0,dl,x.cols());
            yb=y.block(x0,0,dl,y.cols());
            MatrixN yt=forward(xb, yb, nullptr, 0);
            if (yt.rows() != yb.rows()) {
                cerr << "test: incompatible row count!" << endl;
                return -1000.0;
            }
            for (int i=0; i<yt.rows(); i++) {
                int ji=-1;
                floatN pr=-10000;
                for (int j=0; j<yt.cols(); j++) {
                    if (yt(i,j)>pr) {
                        pr=yt(i,j);
                        ji=j;
                    }
                }
                if (ji==(-1)) {
                    cerr << "Internal: at " << layerName << "could not identify max-index for y-row-" << i << ": " << yt.row(i) << endl;
                    return -1000.0;
                }
                if (ji==yb(i,0)) ++co;
            }
        }
        floatN err=1.0-(floatN)co/(floatN)y.rows();
        return err;
    }
    bool selfTest(const MatrixN& x, const MatrixN& y, floatN h, floatN eps);
    virtual void setFlag(string name, bool val) {
        cp.setPar(name,val);
    }

private:
    bool checkForward(const MatrixN& x, floatN eps);
    bool checkForward(const MatrixN& x, const MatrixN &y, floatN eps);
    bool checkBackward(const MatrixN& x, const MatrixN& y, t_cppl *pcache, floatN eps);
    bool calcNumGrads(const MatrixN& xorg, const MatrixN& dchain, t_cppl *pcache, t_cppl *pgrads, t_cppl* pnumGrads, floatN h, bool lossFkt);
    MatrixN calcNumGrad(const MatrixN& x, const MatrixN& dchain, t_cppl* pcachem, string var, floatN h);
    MatrixN calcNumGradLoss(const MatrixN& x, t_cppl* pcache, string var, floatN h);
    bool checkGradients(const MatrixN& x, const MatrixN& y, const MatrixN& dchain, t_cppl *pcache, floatN h, floatN eps, bool lossFkt);
    bool checkLayer(const MatrixN& x, const MatrixN& y, const MatrixN& dchain, t_cppl *cache, floatN h, floatN eps, bool lossFkt);
};

#endif
