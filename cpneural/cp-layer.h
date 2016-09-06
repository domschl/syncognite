#ifndef _CP_LAYER_H
#define _CP_LAYER_H

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <Eigen/Dense>

using Eigen::IOFormat;
using std::cout; using std::endl;
using std::vector; using std::string; using std::map;

#define USE_DOUBLE
//#define USE_FLOAT

#ifdef USE_DOUBLE
using MatrixN=Eigen::MatrixXd;
using VectorN=Eigen::VectorXd;
using RowVectorN=Eigen::RowVectorXd;
using ArrayN=Eigen::ArrayXd;
using floatN=double;
#define CP_DEFAULT_NUM_H (1.e-6)
#define CP_DEFAULT_NUM_EPS (1.e-10)
#else
#ifdef USE_FLOAT
using MatrixN=Eigen::MatrixXf;
using VectorN=Eigen::VectorXf;
using RowVectorN=Eigen::RowVectorXf;
using ArrayN=Eigen::ArrayXf;
using floatN=float;
#define CP_DEFAULT_NUM_H (1.e-3)
#define CP_DEFAULT_NUM_EPS (1.e-6)
#endif
#endif

#define NULL_MAT (MatrixN(0,0))

template <typename T>
using cp_t_params = map<string, T>;
//using t_cppl = cp_t_params<MatrixN *>;
typedef cp_t_params<MatrixN *> t_cppl;

vector<unsigned int> shape(const MatrixN& m) {
    vector<unsigned int> s(2);
    s[0]=(unsigned int)(m.rows());
    s[1]=(unsigned int)(m.cols());
    return s;
}

class CpParams {
    cp_t_params<int> iparams;
    cp_t_params<vector<int>> viparams;
    cp_t_params<floatN> fparams;
    cp_t_params<bool> bparams;
    cp_t_params<string> sparams;
public:
/*    CpParams(string init) {

    }
*/
    void split(const string &s, char delim, vector<string> &elems) {
        std::stringstream ss;
        ss.str(s);
        string item;
        while (getline(ss, item, delim)) {
            elems.push_back(item);
        }
    }
    void setString(string jsonpars) {

    }
    string getString(bool pretty=true) {
        string str="{";
        string ind,sep,asep, ter;
        if (pretty) {
            ind="  "; sep=": "; asep=", "; ter=";\n";
        } else {
            ind=""; sep=":"; asep=","; ter=";";
        }
        for (auto it : fparams) str += ind + "\""+it.first + "\""+sep + std::to_string(it.second) + ter;
        for (auto it : iparams) str += ind + "\""+it.first + "\""+sep + std::to_string(it.second) + ter;
        for (auto it : viparams) {
            str += ind + "\""+it.first + "\""+sep + "[";
            bool is=false;
            for (auto i : it.second) {
                if (is) str+=asep;
                is=true;
                str+=std::to_string(i);
            }
            str+="]"+ter;
        }
        for (auto it : bparams) {
            str += ind + "\""+it.first + "\""+sep;
            if (it.second) str+="true";
            else str+="false";
            str+=ter;
        }
        for (auto it : sparams) str += ind + "\""+it.first + "\""+sep +"\""+it.second +"\""+ ter;
        return str;
    }
    bool isDefined(string par) {
        if (iparams.find(par)!=iparams.end()) return true;
        if (viparams.find(par)!=viparams.end()) return true;
        if (fparams.find(par)!=fparams.end()) return true;
        if (bparams.find(par)!=bparams.end()) return true;
        if (sparams.find(par)!=sparams.end()) return true;
        return false;
    }
    void erase(string par) {
        if (fparams.find(par)!=fparams.end()) fparams.erase(par);
        if (iparams.find(par)!=iparams.end()) iparams.erase(par);
        if (viparams.find(par)!=viparams.end()) viparams.erase(par);
        if (bparams.find(par)!=bparams.end()) bparams.erase(par);
        if (sparams.find(par)!=sparams.end()) sparams.erase(par);
    }
    floatN getPar(string par, floatN def) {
        auto it=fparams.find(par);
        if (it==fparams.end()) {
            fparams[par]=def;
        }
        return fparams[par];
    }
    int getPar(string par, int def) {
        auto it=iparams.find(par);
        if (it==iparams.end()) {
            iparams[par]=def;
        }
        return iparams[par];
    }
    vector<int> getPar(string par, vector<int> def) {
        auto it=viparams.find(par);
        if (it==viparams.end()) {
            viparams[par]=def;
        }
        return viparams[par];
    }
    bool getPar(string par, bool def) {
        auto it=bparams.find(par);
        if (it==bparams.end()) {
            bparams[par]=def;
        }
        return bparams[par];
    }
    string getPar(string par, string def) {
        auto it=sparams.find(par);
        if (it==sparams.end()) {
            sparams[par]=def;
        }
        return sparams[par];
    }
    void setPar(string par, int val) {
        erase(par);
        iparams[par]=val;
    }
    void setPar(string par, vector<int> val) {
        erase(par);
        viparams[par]=val;
    }
    void setPar(string par, floatN val) {
        erase(par);
        fparams[par]=val;
    }
    void setPar(string par, bool val) {
        erase(par);
        bparams[par]=val;
    }
    void setPar(string par, string val) {
        erase(par);
        sparams[par]=val;
    }
};

class Optimizer : public CpParams {
public:
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl *pcache) {return x;};
};

enum LayerType { LT_UNDEFINED, LT_NORMAL, LT_LOSS};

typedef struct LayerParams {
    map<string, int> iParams;
    map<string, floatN> fParams;
} t_layer_params;

typedef std::vector<int> t_layer_topo;
typedef int t_layer_props_entry;
typedef std::map<string, t_layer_props_entry> t_layer_props;

class Layer;

template<typename T>
Layer* createLayerInstance(t_layer_topo tp) {
    return new T(tp);
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
        cout << "MEM! Override condition for " << key << " update prevented, freeing previous pointer..." << endl;
        delete it->second;
    }
    (*p)[key]=val;
}

typedef std::map<std::string, Layer*(*)(t_layer_topo)> t_layer_creator_map;

class LayerFactory {
public:
    t_layer_creator_map mapl;
    t_layer_props mapprops;
    void registerInstanceCreator(std::string name, Layer*(sub)(t_layer_topo), t_layer_props_entry lprops ) {
        auto it=mapl.find(name);
        if (it!=mapl.end()) {
            cout << "Layer " << name << " is already registered, prevent additional registration." << endl;
        } else {
            mapl[name] = sub;
            mapprops[name] = lprops;
        }
    }
    Layer* createLayerInstance(std::string name, t_layer_topo tp) {
        return mapl[name](tp);
    }
};

LayerFactory _syncogniteLayerFactory;

#define REGISTER_LAYER(LayerName, LayerClass, props) _syncogniteLayerFactory.registerInstanceCreator(LayerName,&createLayerInstance<LayerClass>, props);
#define CREATE_LAYER(LayerName, topo) _syncogniteLayerFactory.createLayerInstance(LayerName, topo);


class Layer {
public:
    string layerName;
    LayerType layerType;
    t_cppl params;

    virtual ~Layer() {}; // Otherwise destructor of derived classes is never called!
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache)  { MatrixN d(0,0); return d;}
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache)  { MatrixN d(0,0); return d;}
    virtual MatrixN backward(const MatrixN& dtL, t_cppl* pcache, t_cppl* pgrads) { MatrixN d(0,0); return d;}
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) { return 1001.0; }
    virtual bool update(Optimizer *popti, t_cppl* pgrads, string var, t_cppl* pocache) {
        /*for (int i=0; i<params.size(); i++) {
            *params[i] = popti->update(*params[i],*grads[i]);
        }*/
        for (auto it : params) {
            string key = it.first;
            *params[key] = popti->update(*params[key],*((*pgrads)[key]), var+key, pocache);
        }
        return true;
    }
    floatN train(const MatrixN& x, const MatrixN& y, const MatrixN &xv, const MatrixN &yv,
                        string optimizer, cp_t_params<int> ipars, cp_t_params<floatN> fpars);
    t_cppl workerThread(const MatrixN& xb, const MatrixN& yb, floatN *pl);
    floatN test(const MatrixN& x, const MatrixN& y)  {
        MatrixN yt=forward(x, y, nullptr);
        if (yt.rows() != y.rows()) {
            return -1000.0;
        }
        int co=0;
        for (int i=0; i<yt.rows(); i++) {
            int ji=-1;
            floatN pr=-100;
            for (int j=0; j<yt.cols(); j++) {
                if (yt(i,j)>pr) {
                    pr=yt(i,j);
                    ji=j;
                }
            }
            if (ji==y(i,0)) ++co;
        }
        floatN err=1.0-(floatN)co/(floatN)y.rows();
        return err;
    }
    bool selfTest(const MatrixN& x, const MatrixN& y, floatN h, floatN eps);

private:
    bool checkForward(const MatrixN& x, floatN eps);
    bool checkForward(const MatrixN& x, const MatrixN &y, floatN eps);
    bool checkBackward(const MatrixN& x, const MatrixN& y, t_cppl *pcache, floatN eps);
    bool calcNumGrads(const MatrixN& dchain, t_cppl *pcache, t_cppl *pgrads, t_cppl* pnumGrads, floatN h, bool lossFkt);
    MatrixN calcNumGrad(const MatrixN& dchain, t_cppl* pcachem, string var, floatN h);
    MatrixN calcNumGradLoss(t_cppl* pcache, string var, floatN h);
    bool checkGradients(const MatrixN& x, const MatrixN& y, const MatrixN& dchain, t_cppl *pcache, floatN h, floatN eps, bool lossFkt);
    bool checkLayer(const MatrixN& x, const MatrixN& y, const MatrixN& dchain, t_cppl *cache, floatN h, floatN eps, bool lossFkt);
};

#endif
