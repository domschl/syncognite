
#ifndef _CP_TOOLS_H
#define _CP_TOOLS_H

#include "cp-neural.h"

// for cpInitCompute():
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

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
        if (ci.first.substr(0,prefix.size()+1)==prefix+"-") {
            cppl_set(dst, ci.first.substr(prefix.size()+1), ci.second);
            //src->erase(ci.first); // XXX for rnn-ho inits! DANGEROUS!
        }
    }
}

// XXX: dubious:
void mlPopX(string prefix, t_cppl *src, t_cppl *dst) {
    for (auto ci=src->cbegin(); ci!=src->cend(); ci++) {
        if (ci->first.substr(0,prefix.size()+1)==prefix+"-") {
            cppl_set(dst, ci->first.substr(prefix.size()+1), ci->second);
            src->erase(ci); // Did not work
        }
    }
}

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
        viennacl::matrix<float>vi_b(b->rows(), b->cols(), ctx);
        viennacl::matrix<float>vi_a(a->rows(), a->cols(), ctx);
        viennacl::matrix<float>vi_y(a->rows(), b->cols(), ctx);
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

#endif
