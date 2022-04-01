#pragma once

#include "cp-neural.h"

/** Modes for Xavier initialization */
enum XavierMode { XAV_STANDARD, /** default Xavier with linear distribution */ 
    XAV_NORMAL, /** Xavier with normal distribution */
    XAV_ORTHONORMAL, /** Xavier with orthonormal distribution */
    XAV_ORTHOGONAL, /** Xavier with orthogonal distribution */
};

XavierMode xavierInitType(string xavierStyle) {
    /** @brief Returns the Xavier initialization type.
     * @param xavierStyle Name of Xavier style (e.g. "standard", "normal", "orthonormal", "orthogonal")
     * @return Xavier initialization type \ref XavierMode
     */
    XavierMode type=XavierMode::XAV_STANDARD;
    string trXavierStyle(xavierStyle);
    std::transform(trXavierStyle.begin(), trXavierStyle.end(), trXavierStyle.begin(), ::tolower);

    if (trXavierStyle=="standard") type=XavierMode::XAV_STANDARD;
    else if (trXavierStyle=="normal") type=XavierMode::XAV_NORMAL;
    else if (trXavierStyle=="orthonormal") type=XavierMode::XAV_ORTHONORMAL;
    else if (trXavierStyle=="orthogonal") type=XavierMode::XAV_ORTHOGONAL;
    return type;
}

MatrixN xavierInit(const MatrixN &w, XavierMode xavierMode=XavierMode::XAV_STANDARD, floatN initfactor=1.0) {
    /** @brief Initializes weights with Xavier initialization.
     * Xavier init: 2.0/sqrt(w.cols+w.rows) * initfactor
     * @param w Matrix to initialize
     * @param xavierMode \ref XavierMode
     * @param initfactor Initialization factor >0.0, used to scale the initialization values.
     * @return initialized MatrixN
     */
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
    switch (xavierMode) {
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

class Loss {
public:
    json j;
    virtual ~Loss() {}; // Otherwise destructor of derived classes is never called!
    virtual floatN loss(MatrixN& yhat, MatrixN& y, t_cppl *pParams) {return 1001.0;};
};

class Optimizer {
public:
    json j;
    virtual ~Optimizer() {}; // Otherwise destructor of derived classes is never called!
    virtual void updateOptimizerParameters(const json &opt_params) {
        for (auto& el : opt_params.items()) {
            j[el.key()] = el.value();
        }
        //j = opt_params;
    };
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
inline LayerType operator|(LayerType a, LayerType b) {
    return static_cast<LayerType>(static_cast<int>(a) | static_cast<int>(b));
}
bool layerHasType(LayerType lt, LayerType lt2) {
    return ((lt & lt2) == lt2);
}

typedef int t_layer_props_entry;
typedef std::map<string, t_layer_props_entry> t_layer_props;

class Layer;

template<typename T>
Layer* createLayerInstance(const json& j) {
    return new T((const json &)j);
}

typedef std::map<std::string, Layer*(*)(const json&)> t_layer_creator_map;

class LayerFactory {
public:
    t_layer_creator_map mapl;
    t_layer_props mapprops;
    void registerInstanceCreator(std::string name, Layer*(sub)(const json&), t_layer_props_entry lprops ) {
        auto it=mapl.find(name);
        if (it!=mapl.end()) {
            cerr << "Layer " << name << " is already registered, preventing additional registration." << endl;
        } else {
            mapl[name] = sub;
            mapprops[name] = lprops;
        }
    }
    Layer* createLayerInstance(std::string name, const json& j) {
        return mapl[name](j);
    }
};

LayerFactory _syncogniteLayerFactory;

#define REGISTER_LAYER(LayerClassName, LayerClass, props) _syncogniteLayerFactory.registerInstanceCreator(LayerClassName,&createLayerInstance<LayerClass>, props);
#define CREATE_LAYER(LayerClassName, jprops) _syncogniteLayerFactory.createLayerInstance(LayerClassName, jprops);

typedef map<string, t_cppl> retdict;

class Layer {
public:
    string layerName;
    string layerClassName;
    LayerType layerType;
    int inputShapeRang;
    vector<int>outputShape;
    json j;
    t_cppl params;
    bool layerInit;
    std::mutex lossQueueMutex;
    std::queue<floatN> lossQueue;

    virtual ~Layer() {}; // Otherwise destructor of derived classes is never called!
    virtual vector<int> getOutputShape() { return outputShape;}
    virtual void genZeroStates(t_cppl* pstates, int N) { return; }
    virtual MatrixN forward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, int id) { MatrixN d(0,0); return d;}
    virtual MatrixN backward(const MatrixN& dy, t_cppl* pcache, t_cppl* pstates, t_cppl* pgrads, int id) { MatrixN d(0,0); return d;}
    // XXX TODO: remove
    //virtual floatN loss(t_cppl* pcache, t_cppl* pstates) { return 1001.0; }
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

    virtual void setFlag(string name, bool val) { j[name]=val; }

    virtual void getLayerConfiguration(json &jlc) {
        jlc=j;
        jlc["layerclassname"]=layerClassName;
        jlc["layername"]=layerName;
    }

    virtual bool saveLayerConfiguration(H5::H5File* pfile) {
        hsize_t		dims1[] = {1};
        int rank{1};
        H5::DataSpace sid1(rank, dims1);
        H5::StrType tid1(0, H5T_VARIABLE);
        
        if(H5T_STRING!=H5Tget_class(tid1.getId()) || !H5Tis_variable_str(tid1.getId())) {
            cerr << "saveLayerConfiguration: not a variable length string type." << endl;
            return false;
        }
        string dsname;
        dsname="Configuration:"+layerName;
        H5::DataSet dataset = pfile->createDataSet(dsname, tid1, sid1);

        string conf=j.dump();
        const char *pc[1];
        pc[0]=conf.c_str();
        dataset.write((void *)pc, tid1);
        dataset.close();

        return true;
    }
        
    virtual bool saveParameters(H5::H5File* pfile) {
        string prefix=layerName+"-";
        for (auto pi : params) {
            string name=prefix+pi.first;
            MatrixN *pm=params[pi.first];

            hsize_t fdim[2];
            fdim[0]=pm->rows();
            fdim[1]=pm->cols();
            int rank=2;
            H5::DataSpace fspace( rank, fdim );

            H5::DataSet dataset=pfile->createDataSet(name, H5_FLOATN, fspace);
            dataset.write(pm->data(), H5_FLOATN);
        }
        return true;
    }

    virtual bool loadParameters(H5::H5File* pfile) {
        string prefix=layerName+"-";
        bool ok=true;
        for (auto pi : params) {
            string name=prefix+pi.first;
            MatrixN *pm=params[pi.first];
            H5::DataSet dataset;
            try {
                dataset=pfile->openDataSet(name);
            } catch (...) {
                cerr << "Dataset " << name << " does not exist, can't restore parameter." << endl;
                return false;
            }
            H5::DataSpace filespace = dataset.getSpace();
            int rank = filespace.getSimpleExtentNdims();
            hsize_t dims[rank];    // dataset dimensions
            rank = filespace.getSimpleExtentDims( dims );
            if (rank!=2) {
                cerr << "Currently, rank of datasets " << name << " needs to be 2 for MatrixN, its: " << rank << endl;
                cerr << "Don't know how to restore this." << endl;
                return false;
            }
            if (dims[0] != pm->rows() || dims[1] != pm->cols()) {
                cerr << "Incompabile dataset " << name << ", dims: (" << dims[0] << "," << dims[1] << "), model: " << shape(*pm) << endl;
                cerr << "Don't know how to restore this." << endl;
                return false;         
            }
            H5::DataSpace mspace1(rank, dims);
            auto dataClass = dataset.getTypeClass();
            if (dataClass==H5T_FLOAT) {
                if (dataset.getFloatType().getSize()==H5_FLOATN_SIZE) {
                    dataset.read(pm->data(), H5_FLOATN, mspace1, filespace );
                } else {
                    cerr << "Bad float type (float vs double) for " << name << endl;
                    ok=false;
                }
            } else {
                cerr << "Bad format for " << name << endl;
                ok=false;
            }
        }
        return ok;
    }

    //floatN train(const MatrixN& x, t_cppl* pstates, const MatrixN &xv, t_cppl* pstatesv,
    //                    string optimizer, const json& j);
    floatN train(const MatrixN& x, t_cppl* pstates, const MatrixN &xv, t_cppl* pstatesv,
                        Optimizer* pOptimizer, t_cppl* pOptimizerState, Loss* pLoss, const json& j);
    //floatN train(const MatrixN& x, const MatrixN& y, const MatrixN &xv, const MatrixN& yv,
    //                    string optimizer, const json& j);
    floatN train(const MatrixN& x, const MatrixN& y, const MatrixN &xv, const MatrixN& yv,
                        Optimizer *pOptimizer, t_cppl* pOptimizerState, Loss *pLoss, const json& j);
    retdict workerThread(MatrixN *pxb, t_cppl* pstates, int id, Loss *pLoss);
    floatN test(const MatrixN& x, t_cppl* pstates, int batchsize);
    floatN test(const MatrixN& x, const MatrixN& y, int batchsize);
    bool selfTest(const MatrixN& x, t_cppl *pstates, floatN h, floatN eps, int verbose, Loss *pLoss);

private:
    bool checkForward(const MatrixN& x, t_cppl* pcache, t_cppl* pstates, floatN eps, int verbose);
    bool checkBackward(const MatrixN& x, t_cppl *pcache, t_cppl* pstates, floatN eps, int verbose);
    MatrixN calcNumGrad(const MatrixN& xorg, const MatrixN& dchain, t_cppl* pcache, t_cppl* pstates, string var, floatN h, int verbose);
    MatrixN calcNumGradLoss(const MatrixN& xorg, t_cppl *pcache, t_cppl* pstates, string var, floatN h, int verbose,Loss *pLoss);
    // XXX: rework lossFkt parameter
    bool calcNumGrads(const MatrixN& x, const MatrixN& dchain, t_cppl *pcache, t_cppl* pstates, t_cppl *pgrads, t_cppl *pnumGrads, floatN h, bool lossFkt, int verbose, Loss *pLoss);
    bool checkGradients(const MatrixN& x, const MatrixN& y, const MatrixN& dchain, t_cppl *pcache, t_cppl *pstates, floatN h, floatN eps, bool lossFkt, int verbose, Loss *pLoss);
    bool checkLayer(const MatrixN& x, const MatrixN& y, const MatrixN& dchain, t_cppl *pcache, t_cppl* pstates, floatN h, floatN eps, bool lossFkt, int verbose, Loss *pLoss);
};

