#ifndef _CP_LAYER_H
#define _CP_LAYER_H

#include "cp-neural.h"

enum XavierMode { XAV_STANDARD, XAV_NORMAL, XAV_ORTHONORMAL, XAV_ORTHOGONAL};

XavierMode xavierInitType(string stype) {
    XavierMode type=XavierMode::XAV_STANDARD;
    string ltype(stype);
    std::transform(ltype.begin(), ltype.end(), ltype.begin(), ::tolower);

    if (ltype=="standard") type=XavierMode::XAV_STANDARD;
    else if (ltype=="normal") type=XavierMode::XAV_NORMAL;
    else if (ltype=="orthonormal") type=XavierMode::XAV_ORTHONORMAL;
    else if (ltype=="orthogonal") type=XavierMode::XAV_ORTHOGONAL;
    return type;
}

MatrixN xavierInit(const MatrixN &w, XavierMode xavMode=XavierMode::XAV_STANDARD, floatN initfactor=1.0) {
    if (initfactor == 0.0) {
        cerr << "Initfactor=0.0 is invalid! Setting to 1.0" << endl;
        initfactor=1.0;
    }
    floatN xavier = 2.0/std::sqrt((floatN)(w.cols()+w.rows())) * initfactor;
    float mean=0.0;
    float std=xavier / 2.0;
    MatrixN wo(w);
    MatrixN wot(w);
    std::default_random_engine rde;
    std::normal_distribution<float> distn(mean, std);
    switch (xavMode) {
        case XavierMode::XAV_NORMAL:
            for (int i=0; i<wo.size(); i++) wo(i)=distn(rde);
        break;
        case XavierMode::XAV_ORTHONORMAL:
            for (int i=0; i<wot.size(); i++) wot(i)=distn(rde);
            if (wot.rows() == wot.cols()) {
                wo = wot.householderQr().householderQ();
            } else {
                wo=wot;  // we can only orthonormalize square matrices!
            }
        break;
        case XavierMode::XAV_ORTHOGONAL:
            for (int i=0; i<wot.size(); i++) wot(i)=distn(rde);
            if (wot.rows() == wot.cols()) {
                wo = wot.householderQr().householderQ();
                wo *= xavier; // orthogonal instead of orthonormal!
            } else {
                wo=wot;  // we can only orthonormalize square matrices!
            }
        break;
        case XavierMode::XAV_STANDARD:
        default:
            wo.setRandom(); // [-1,1]
            wo = wo * xavier/2.0;  // (setRandom is [-1,1]-> fakt 0.5, xavier is 2/(ni+no))
        break;
    }
    return wo;
}

class Optimizer {
public:
    CpParams cp;
    virtual ~Optimizer() {}; // Otherwise destructor of derived classes is never called!
    virtual MatrixN update(MatrixN& x, MatrixN& dx, string var, t_cppl *pcache) {return x;};
};

enum LayerType {
    LT_UNDEFINED = 0,
    LT_NORMAL = 1,
    LT_EXTERNALSTATE = 2,
    LT_LOSS = 4
};
inline LayerType operator&(LayerType a, LayerType b) {
    return static_cast<LayerType>(static_cast<int>(a) & static_cast<int>(b));
}
bool layerHasType(LayerType lt, LayerType lt2) {
    return ((lt & lt2) == lt2);
}

typedef int t_layer_props_entry;
typedef std::map<string, t_layer_props_entry> t_layer_props;

class Layer;

template<typename T>
Layer* createLayerInstance(const CpParams& cp) {
    return new T(cp);
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
    std::mutex lossQueueMutex;
    std::queue<floatN> lossQueue;

    virtual ~Layer() {}; // Otherwise destructor of derived classes is never called!
    virtual vector<int> getOutputShape() { return outputShape;}
    virtual void genZeroStates(t_cppl* pstates) { return; }
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id) { MatrixN d(0,0); return d;}
    virtual MatrixN backward(const MatrixN& dy, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id) { MatrixN d(0,0); return d;}
    virtual floatN loss(t_cppl* pcache, t_cppl* pstates) { return 1001.0; }
    virtual bool update(Optimizer *popti, t_cppl* pgrads, string var, t_cppl* pocache) {
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

    virtual void setFlag(string name, bool val) { cp.setPar(name,val); }

    floatN train(const MatrixN& x, t_cppl* pstates, const MatrixN &xv, t_cppl* pstatesv,
                        string optimizer, const CpParams& cp);
    floatN train(const MatrixN& x, const MatrixN& y, const MatrixN &xv, const MatrixN& yv,
                        string optimizer, const CpParams& cp);
    t_cppl workerThread(MatrixN *pxb, t_cppl* pstates, int id);
    floatN test(const MatrixN& x, t_cppl* pstates, int batchsize);
    floatN test(const MatrixN& x, const MatrixN& y, int batchsize);
    bool selfTest(const MatrixN& x, t_cppl *pstates, floatN h, floatN eps);

private:
    bool checkForward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, floatN eps);
    bool checkBackward(const MatrixN& x, t_cppl *pcache, t_cppl* pstates, floatN eps);
    MatrixN calcNumGrad(const MatrixN& xorg, const MatrixN& dchain, t_cppl* pcache, t_cppl* pstates, string var, floatN h);
    MatrixN calcNumGradLoss(const MatrixN& xorg, t_cppl *pcache, t_cppl* pstates, string var, floatN h);
    bool calcNumGrads(const MatrixN& x, const MatrixN& dchain, t_cppl *pcache, t_cppl* pstates, t_cppl *pgrads, t_cppl *pnumGrads, floatN h, bool lossFkt);
    bool checkGradients(const MatrixN& x, const MatrixN& y, const MatrixN& dchain, t_cppl *pcache, t_cppl *pstates, floatN h, floatN eps, bool lossFkt);
    bool checkLayer(const MatrixN& x, const MatrixN& y, const MatrixN& dchain, t_cppl *pcache, t_cppl* pstates, floatN h, floatN eps, bool lossFkt);
};

#endif
