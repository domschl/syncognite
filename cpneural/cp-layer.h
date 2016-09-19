#ifndef _CP_LAYER_H
#define _CP_LAYER_H

#include <iostream>
#include <fstream>
#include <cctype>
#include <string>
#include <algorithm>
#include <sstream>
#include <vector>
#include <map>
//#include <mutex>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "cp-math.h"

using Eigen::IOFormat;
using std::cout; using std::endl;
using std::vector; using std::string; using std::map;

//#define USE_DOUBLE
#ifndef USE_DOUBLE
#ifndef USE_FLOAT
#pragma "Please define either USE_FLOAT or USE_DOUBLE"
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
using ArrayN=Eigen::ArrayXd;
using floatN=double;
#define CP_DEFAULT_NUM_H (1.e-6)
#define CP_DEFAULT_NUM_EPS (1.e-9)
#endif
#ifdef USE_FLOAT
using MatrixN=Eigen::MatrixXf;
using VectorN=Eigen::VectorXf;
using RowVectorN=Eigen::RowVectorXf;
using ArrayN=Eigen::ArrayXf;
using floatN=float;
#define CP_DEFAULT_NUM_H ((float)1.e-4)
#define CP_DEFAULT_NUM_EPS ((float)1.e-3)
#endif

using Tensor4=Eigen::Tensor<floatN, 4>;

#ifdef USE_VIENNACL
#define VIENNACL_HAVE_EIGEN
#ifdef USE_OPENCL
#define VIENNACL_WITH_OPENCL
#pragma info("Eigen is active with ViennaCl and OpenCL")
#else
#error "VIENNACL currently requires WITH_OPENCL Cmake option to be set."
#endif
#ifdef USE_CUDA
#define VIENNACL_WITH_CUDA
#pragma "CUDA and ViennaCL current does not work!"
#endif
#endif

#include "cp-math.h"
#include "cp-layer.h"


#ifdef USE_VIENNACL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#endif

#ifdef USE_VIENNACL
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"
#endif

// for cpInitCompute():
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

template <typename T>
using cp_t_params = map<string, T>;
//using t_cppl = cp_t_params<MatrixN *>;
typedef cp_t_params<MatrixN *> t_cppl;
typedef cp_t_params<Tensor4 *> t_cppl4;

vector<unsigned int> shape(const MatrixN& m) {
    vector<unsigned int> s(2);
    s[0]=(unsigned int)(m.rows());
    s[1]=(unsigned int)(m.cols());
    return s;
}

bool matCompare(MatrixN& m0, MatrixN& m1, string msg="", floatN eps=1.e-6) {
    if (m0.cols() != m1.cols() || m0.rows() != m1.rows()) {
        cout << msg << ": Incompatible shapes " << shape(m0) << "!=" << shape(m1) << endl;
        return false;
    }
    MatrixN d = m0 - m1;
    floatN dif = d.cwiseProduct(d).sum();
    if (dif < eps) {
        if (msg!="") cout << msg << " err=" << dif << endl;
        return true;
    } else {
        if (msg!="") {
            //IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
            //cout << msg << " m0:" << endl << m0.format(CleanFmt) << endl;
            //cout << msg << " m1:" << endl << m1.format(CleanFmt) << endl;
            cout << "err=" << dif << endl;
        }
        return false;
    }
}

void peekMat(const string label, const MatrixN& m) {
    cout << label << " ";
    if (m.size()<10) cout << m << endl;
    else {
        for (int j=0; j<m.size(); j++) {
            if (j<4 || m.size()-j < 4) cout << m(j) << " ";
            else if (j==4) cout << " ... ";
        }
        cout << endl;
    }
}

class CpParams {
    cp_t_params<int> iparams;
    cp_t_params<vector<int>> viparams;
    cp_t_params<vector<floatN>> vfparams;
    cp_t_params<floatN> fparams;
    cp_t_params<bool> bparams;
    cp_t_params<string> sparams;
//    std::mutex parMutex;
public:
    CpParams() {}
    CpParams(string init) {
        setString(init);
    }
    string trim(const string &s) {
        auto wsfront=std::find_if_not(s.begin(),s.end(),[](int c){return std::isspace(c);});
        //auto wsfront=std::find_if_not(s.begin(),s.end(),std::isspace);
        auto wsback=std::find_if_not(s.rbegin(),s.rend(),[](int c){return std::isspace(c);}).base();
        //auto wsback=std::find_if_not(s.rbegin(),s.end(),std::isspace).base();
        return (wsback<=wsfront ? string() : string(wsfront,wsback));
    }
    void split(const string &s, char delim, vector<string> &elems) {
        std::stringstream ss;
        ss.str(s);
        string item;
        while (getline(ss, item, delim)) {
            elems.push_back(trim(item));
        }
    }
    string getBlock(string str, string del1, string del2) {
        auto p1=str.find(del1);
        auto p2=str.rfind(del2);
        string r;
        if (p2<=p1) r=string(); else r=str.substr(p1,p2-p1);
        if (r.size()>=del1.size()+del2.size()) r=r.substr(del1.size(),str.size()-del1.size()-del2.size());
        return r;
    }
    void setString(string str) {
        string bl=getBlock(str,"{","}");
        vector<string> tk;
        split(bl,';',tk);
        for (auto t : tk) {
            vector<string> par;
            string p1=getBlock(t,"","=");
            string p2=getBlock(t,"=","");
            if (p1.size()>0) {
                if (p2[0]=='[') { // array
                    string tm=getBlock(p2,"[","]");
                    vector<string> ar;
                    split(tm,',',ar);
                    if (tm.find(".")!=tm.npos || tm.find("e")!=tm.npos) { //float array
                        vector<floatN> vf;
                        for (auto af : ar) {
                            try {
                                vf.push_back((floatN)stod(af));
                            } catch (...) {
                                cout << "EXCEPTION in CpParams 3" << endl;
                            }
                        }
                        setPar(p1,vf);
                    } else { // int array
                        vector<int> vi;
                        for (auto ai : ar) {
                            try {
                                vi.push_back(stoi(ai));
                            } catch (...) {
                                cout << "EXCEPTION in CpParams 4" << endl;
                            }
                        }
                        setPar(p1,vi);
                    }
                } else if (p2=="true") { // boolean
                    setPar(p1,true);
                } else if(p2=="false") { // boolean
                    setPar(p1,false);
                } else if (p2[0]=='\'') { //string
                    // XXX: (de/)encode escape stuff:   ;{}
                    string st=getBlock(p2,"'", "'");
                    setPar(p1,st);
                } else if (p2.find(".")!=p2.npos || p2.find("e")!=p2.npos) { //float
                    try {
                        setPar(p1,stof(p2));
                    } catch (...) {
                        cout << "EXCEPTION in CpParams 1" << endl;
                    }
                } else { //assume int
                    try {
                        setPar(p1,stoi(p2));
                    } catch (...) {
                        cout << "EXCEPTION in CpParams 2" << endl;
                    }
                }
            }
        }
    }
    string getString(bool pretty=true) {
        string ind,sep,asep, ter, tnl;
        string qt=""; // or: "\"";

        if (pretty) {
            ind="  "; sep="="; asep=", "; ter=";\n"; tnl="\n";
        } else {
            ind=""; sep="="; asep=","; ter=";"; tnl="";
        }
        string sdl="'";
        string str="{"+tnl;
        for (auto it : fparams) str += ind + qt+it.first + qt+sep + std::to_string(it.second) + ter;
        for (auto it : iparams) str += ind + qt+it.first + qt+sep + std::to_string(it.second) + ter;
        for (auto it : viparams) {
            str += ind + qt+it.first + qt+sep + "[";
            bool is=false;
            for (auto i : it.second) {
                if (is) str+=asep;
                is=true;
                str+=std::to_string(i);
            }
            str+="]"+ter;
        }
        for (auto it : vfparams) {
            str += ind + qt+it.first + qt+sep + "[";
            bool is=false;
            for (auto f : it.second) {
                if (is) str+=asep;
                is=true;
                str+=std::to_string(f);
            }
            str+="]"+ter;
        }
        for (auto it : bparams) {
            str += ind + qt+it.first + qt+sep;
            if (it.second) str+="true";
            else str+="false";
            str+=ter;
        }
        for (auto it : sparams) {
            // XXX: decode escape stuff:   ;{}
            str += ind + qt+it.first + qt+sep +sdl+it.second +sdl+ ter;
        }
        str+="}"+tnl;
        return str;
    }
    bool isDefined(string par) {
        if (iparams.find(par)!=iparams.end()) return true;
        if (viparams.find(par)!=viparams.end()) return true;
        if (vfparams.find(par)!=vfparams.end()) return true;
        if (fparams.find(par)!=fparams.end()) return true;
        if (bparams.find(par)!=bparams.end()) return true;
        // XXX Warning: Strings lake even rudimentary escape sequence handling!
        if (sparams.find(par)!=sparams.end()) return true;
        return false;
    }
    void erase(string par) {
        if (fparams.find(par)!=fparams.end()) fparams.erase(par);
        if (iparams.find(par)!=iparams.end()) iparams.erase(par);
        if (viparams.find(par)!=viparams.end()) viparams.erase(par);
        if (vfparams.find(par)!=vfparams.end()) vfparams.erase(par);
        if (bparams.find(par)!=bparams.end()) bparams.erase(par);
        // XXX Warning: Strings lake even rudimentary escape sequence handling!
        if (sparams.find(par)!=sparams.end()) sparams.erase(par);
    }
    floatN getPar(string par, floatN def=0.0) {
        auto it=fparams.find(par);
        if (it==fparams.end()) {
            fparams[par]=def;
        }
        return fparams[par];
    }
    int getPar(string par, int def=0) {
        auto it=iparams.find(par);
        if (it==iparams.end()) {
            iparams[par]=def;
        }
        return iparams[par];
    }
    vector<int> getPar(string par, vector<int> def={}) {
        auto it=viparams.find(par);
        if (it==viparams.end()) {
            viparams[par]=def;
        }
        return viparams[par];
    }
    vector<floatN> getPar(string par, vector<floatN> def={}) {
        auto it=vfparams.find(par);
        if (it==vfparams.end()) {
            vfparams[par]=def;
        }
        return vfparams[par];
    }
    bool getPar(string par, bool def=false) {
        auto it=bparams.find(par);
        if (it==bparams.end()) {
            bparams[par]=def;
        }
        return bparams[par];
    }
    string getPar(string par, string def="") {
        auto it=sparams.find(par);
        if (it==sparams.end()) {
            // XXX: decode escape stuff:   ;{}
            sparams[par]=def;
        }
        return sparams[par];
    }
    void setPar(string par, int val) {
//        parMutex.lock();
        erase(par);
        iparams[par]=val;
//        parMutex.unlock();
    }
    void setPar(string par, vector<int> val) {
//        parMutex.lock();
        erase(par);
        viparams[par]=val;
//        parMutex.unlock();
    }
    void setPar(string par, vector<floatN> val) {
//        parMutex.lock();
        erase(par);
        vfparams[par]=val;
//        parMutex.unlock();
    }
    void setPar(string par, floatN val) {
//        parMutex.lock();
        erase(par);
        fparams[par]=val;
//        parMutex.unlock();
    }
    void setPar(string par, bool val) {
//        parMutex.lock();
        erase(par);
        bparams[par]=val;
//        parMutex.unlock();
    }
    void setPar(string par, string val) {
//        parMutex.lock();
        erase(par);
        // XXX: encode escape stuff:   ;{}
        sparams[par]=val;
//        parMutex.unlock();
    }
};

int cpNumGpuThreads=1;
int cpNumEigenThreads=1;
int cpNumCpuThreads=1;
#define MAX_GPUTHREADS 64

bool threadViennaClContextinit(unsigned int numThreads) {
    #ifdef USE_VIENNACL
    if (numThreads > MAX_GPUTHREADS) numThreads=MAX_GPUTHREADS;
    if (viennacl::ocl::get_platforms().size() == 0) {
        std::cerr << "Error: No ViennaClplatform found!" << std::endl;
        return false;
    }
    viennacl::ocl::platform pf = viennacl::ocl::get_platforms()[0];
    std::vector<viennacl::ocl::device> const & devices = pf.devices();
    int nrDevs = pf.devices().size();
    cout << nrDevs << " devices found." << endl;
    for (unsigned int i=0; i<numThreads; i++) {
        viennacl::ocl::setup_context(i, devices[i%nrDevs]); // XXX support for multiple devices is a bit basic.
        cout << "Context " << i << " on: " << viennacl::ocl::get_context(i).devices()[0].name() << endl;
    }
    // Set context to 0 for main program, 1-numThreads for threads
    //viennacl::context ctx(viennacl::ocl::get_context(static_cast<long>(0)));
    //cout << "Contexts created, got context 0 for main program." << endl;
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

bool cpInitCompute(CpParams* poptions=nullptr) {
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
        cout << "New configureation, '" << conffile << "' not found." << endl;
    }
    // omp_set_num_threads(n)
    // Eigen::setNbThreads(n);
    // n=Eigen::nbThreads();
// myfile << "Writing this to a file.\n";


    cpNumGpuThreads=cp.getPar("NumGpuThreads", 1);
    cpNumEigenThreads=cp.getPar("NumEigenThreads", 1);
    cpNumCpuThreads=cp.getPar("NumCpuThreads", 1);
    if (poptions!=nullptr) {
        *poptions=cp;
    }

    Eigen::initParallel();
    Eigen::setNbThreads(cpNumEigenThreads);

    #ifdef USE_VIENNACL
    options += "VIENNACL ";
    threadViennaClContextinit(cpNumGpuThreads);
    #ifdef USE_OPENCL
    options += "OPENCL ";
    #endif
    #ifdef USE_CUDA
    options += "CUDA ";
    #endif
    #endif

    #ifdef USE_FLOAT
    options+="FLOAT ";
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
    cout << "Compile-time options: " << options << endl;
    cout << "Eigen is using:      " << cpNumEigenThreads << " threads." << endl;
    cout << "CpuPool is using:    " << cpNumCpuThreads << " threads." << endl;
    cout << "GpuPool is using:    " << cpNumGpuThreads << " threads." << endl;
    return true;
}

class Optimizer {
public:
    CpParams cp;
    virtual ~Optimizer() {}; // Otherwise destructor of derived classes is never called!
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl *pcache) {return x;};
};

enum LayerType { LT_UNDEFINED, LT_NORMAL, LT_LOSS};
/*
typedef struct LayerParams {
    map<string, int> iParams;
    map<string, floatN> fParams;
} t_layer_params;

*///typedef std::vector<int> t_layer_topo;
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

void cppl_delete(t_cppl4 *p) {
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
        cout << "MEM! Override condition for " << key << " update prevented, freeing previous pointer..." << endl;
        delete it->second;
    }
    (*p)[key]=val;
}

void cppl_set(t_cppl4 *p, string key, Tensor4 *val) {
    auto it=p->find(key);
    if (it != p->end()) {
        cout << "MEM! Override condition for " << key << " update prevented, freeing previous pointer..." << endl;
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

void cppl_update(t_cppl4 *p, string key, Tensor4 *val) {
    if (p->find(key)==p->end()) {
        Tensor4 *pm=new Tensor4(*val);
        cppl_set(p, key, pm);
    } else {
        *((*p)[key])=*val;
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
            cout << "Layer " << name << " is already registered, prevent additional registration." << endl;
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
    int topoParams;
    CpParams cp;
    t_cppl params;

    virtual ~Layer() {}; // Otherwise destructor of derived classes is never called!
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, int id)  { MatrixN d(0,0); return d;}
    virtual Tensor4 forward(const Tensor4& x, t_cppl4* pcache, int id)  { Tensor4 d(0,0,0,0); return d;}
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache, int id)  { MatrixN d(0,0); return d;}
    virtual MatrixN backward(const MatrixN& dtL, t_cppl* pcache, t_cppl* pgrads, int id) { MatrixN d(0,0); return d;}
    virtual Tensor4 backward(const Tensor4& dtL, t_cppl4* pcache, t_cppl4* pgrads, int id) { Tensor4 d(0,0,0,0); return d;}
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) { return 1001.0; }
    virtual bool update(Optimizer *popti, t_cppl* pgrads, string var, t_cppl* pocache) {
        /*for (int i=0; i<params.size(); i++) {
            *params[i] = popti->update(*params[i],*grads[i]);
        }*/
        for (auto it : params) {
            string key = it.first;
            if (pgrads->find(key)==pgrads->end()) {
                cout << "Internal error on update of layer: " << layerName << " at key: " << key << endl;
                cout << "Grads-vars: ";
                for (auto gi : *pgrads) cout << gi.first << " ";
                cout << endl;
                cout << "Params-vars: ";
                for (auto pi : params) cout << pi.first << " ";
                cout << endl;
                cout << "Irrecoverable internal error, ABORT";
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
    floatN test(const MatrixN& x, const MatrixN& y)  {
        setFlag("train",false);
        MatrixN yt=forward(x, y, nullptr, 0);
        if (yt.rows() != y.rows()) {
            cout << "test: incompatible row count!" << endl;
            return -1000.0;
        }
        int co=0;
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
                cout << "Internal: at " << layerName << "could not identify max-index for y-row-" << i << ": " << yt.row(i) << endl;
                return -1000.0;
            }
            if (ji==y(i,0)) ++co;
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
