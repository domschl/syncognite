#ifndef _CP_LAYER_H
#define _CP_LAYER_H

#include <iostream>
#include <string>
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

vector<unsigned int> shape(MatrixN& m) {
    vector<unsigned int> s(2);
    s[0]=(unsigned int)(m.rows());
    s[1]=(unsigned int)(m.cols());
    return s;
}

class Optimizer {
public:
    cp_t_params<int> iparams;
    cp_t_params<floatN> fparams;
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
    virtual MatrixN update(MatrixN& x, MatrixN& dx) {return x;};
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

typedef std::map<std::string, Layer*(*)(t_layer_topo)> t_layer_creator_map;

class LayerFactory {
public:
    t_layer_creator_map mapl;
    t_layer_props mapprops;
    void registerInstanceCreator(std::string name, Layer*(sub)(t_layer_topo), t_layer_props_entry lprops ) {
        mapl[name] = sub;
        mapprops[name] = lprops;
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
    vector<MatrixN *> params;
    vector<MatrixN *> grads;
    vector<MatrixN *> cache;
    vector<string> names;

    virtual MatrixN forward(MatrixN& lL)  { MatrixN d(0,0); return d;}
    virtual MatrixN backward(MatrixN& dtL) { MatrixN d(0,0); return d;}
    virtual floatN loss(MatrixN& y) { return 1001.0; }
    virtual bool update(Optimizer *popti) {
        for (int i=1; i<params.size(); i++) {
            *params[i] = popti->update(*params[i],*grads[i]);
        }
        return true;
    }

    floatN train(MatrixN& x, MatrixN& y, string optimizer, cp_t_params<int> ipars, cp_t_params<floatN> fpars);
    bool selfTest(MatrixN& x, MatrixN& y, floatN h, floatN eps);
private:
    bool checkForward(MatrixN& x, floatN eps);
    bool checkBackward(MatrixN& dchain, floatN eps);
    bool calcNumGrads(MatrixN& dchain, vector<MatrixN *> numGrads, floatN h, bool lossFkt);
    MatrixN calcNumGrad(MatrixN& dchain, unsigned int ind, floatN h);
    MatrixN calcNumGradLoss(MatrixN& dchain, unsigned int ind, floatN h);
    bool checkGradients(MatrixN& x, MatrixN& dchain, floatN h, floatN eps, bool lossFkt);
    bool checkLayer(MatrixN& x, MatrixN& dchain, floatN h, floatN eps, bool lossFkt);
};

#endif
