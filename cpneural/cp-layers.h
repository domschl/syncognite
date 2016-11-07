#ifndef _CP_LAYERS_H
#define _CP_LAYERS_H

#include "cp-neural.h"

#include "layers/cpl-affine.h"
#include "layers/cpl-relu.h"
#include "layers/cpl-affinerelu.h"
#include "layers/cpl-batchnorm.h"
#include "layers/cpl-dropout.h"
#include "layers/cpl-convolution.h"
#include "layers/cpl-pooling.h"
#include "layers/cpl-spatialbatchnorm.h"
#include "layers/cpl-svm.h"
#include "layers/cpl-softmax.h"
#include "layers/cpl-twolayernet.h"
#include "layers/cpl-rnn.h"
#include "layers/cpl-wordembedding.h"
#include "layers/cpl-temporalaffine.h"
#include "layers/cpl-nonlinearity.h"


void registerLayers() {
    REGISTER_LAYER("Affine", Affine, 1)
    REGISTER_LAYER("Relu", Relu, 1)
    REGISTER_LAYER("Nonlinearity", Nonlinearity, 1)
    REGISTER_LAYER("AffineRelu", AffineRelu, 1)
    REGISTER_LAYER("BatchNorm", BatchNorm, 1)
    REGISTER_LAYER("Dropout", Dropout, 1)
    REGISTER_LAYER("Convolution", Convolution, 3)
    REGISTER_LAYER("Pooling", Pooling, 3)
    REGISTER_LAYER("SpatialBatchNorm", SpatialBatchNorm, 3)
    REGISTER_LAYER("RNN", RNN, 1)
    REGISTER_LAYER("WordEmbedding", WordEmbedding, 1)
    REGISTER_LAYER("TemporalAffine", TemporalAffine, 1)
    REGISTER_LAYER("Softmax", Softmax, 1)
    REGISTER_LAYER("Svm", Svm, 1)
    REGISTER_LAYER("TwoLayerNet", TwoLayerNet, 1)
}


class LayerBlock : public Layer {
private:
    bool bench;
    void setup(const CpParams& cx) {
        cp=cx;
        layerName=cp.getPar("name",(string)"block");
        bench=cp.getPar("bench",false);
        lossLayer="";
        layerType=LayerType::LT_NORMAL;
        trainMode = cp.getPar("train", false);
        checked=false;
    }
public:
    map<string, Layer*> layerMap;
    map<string, vector<string>> layerInputs;
    string lossLayer;
    bool checked;
    bool trainMode;

    LayerBlock(const CpParams& cx) {
        setup(cx);
    }
    LayerBlock(const string conf) {
        setup(CpParams(conf));
        layerInit=true;
    }
    ~LayerBlock() {
        for (auto pli : layerMap) {
            if (pli.second != nullptr) {
                delete pli.second;
                pli.second=nullptr;
            }
        }
        layerMap.clear();
    }
    bool removeLayer(const string name) {
        auto fi=layerMap.find(name);
        if (fi == layerMap.end()) {
            cerr << "Cannot remove layer: " << name << ", a layer with this name does not exist in block " << layerName << endl;
            return false;
        }
        delete fi->second;
        layerMap.erase(fi);
        return true;
    }
    bool addLayer(const string layerclass, const string name, CpParams& cp, const vector<string> inputLayers) {
        if (layerMap.find(name) != layerMap.end()) {
            cerr << "Cannot add layer: " << name << ", a layer with this name is already part of block " << layerName << endl;
            return false;
        }
        if (_syncogniteLayerFactory.mapl.find(layerclass) == _syncogniteLayerFactory.mapl.end() and layerclass!="Input") {
            cerr << "Cannot add layer: " << layerclass << ", layer class is not defined." << endl;
            return false;
        }
        string firstInput="";  // XXX multiple input layers!
        for (auto li : inputLayers) {
            if (li!="input") {
                if (layerMap.find(li) == layerMap.end()) {
                    cerr << "Cannot add layer: " << name << ", it depends on an input layer " << li << ", which is not defined." << endl;
                    return false;
                } else {
                    if (firstInput=="") firstInput=li;
                }
            } else {
                firstInput="input";
            }
        }

        if (firstInput!="" && firstInput!="input") {
            auto lP=layerMap.find(firstInput);
            if (lP==layerMap.end()) {
                cerr << "Can't find input-layer: " << firstInput << " internal error in layer defintion of " << name << endl;
                return false;
            }
            vector<int> inputShape, prevOutputShape;
            inputShape=cp.getPar("inputShape", vector<int>{});
            prevOutputShape=lP->second->getOutputShape();
            if (prevOutputShape.size()==0) {
                cerr << "Missing outputShape defintion for inputLayer " << firstInput << endl;
                return false;
            }
            if (inputShape.size()<prevOutputShape.size()) {
                inputShape=prevOutputShape;
            }
            for (unsigned int i=0; i<prevOutputShape.size(); i++) {
                inputShape[i]=prevOutputShape[i];
            }
            cp.setPar("inputShape",inputShape);
        }
        layerMap[name]=CREATE_LAYER(layerclass, cp)   // Macro!
        Layer *pLayer = layerMap[name];
        if (pLayer->layerInit==false) {
            cerr << "Attempt to add layer " << name << " failed: Bad initialization." << endl;
            removeLayer(name);
            return false;
        }
        if (pLayer->layerType==LayerType::LT_LOSS) {
            if (lossLayer!="") {
                cerr << "ERROR: a loss layer with name: " << lossLayer << "has already been defined, cannot add new loss layer: " << name << " to " << layerName << endl;
                removeLayer(name);
                return false;
            }
            layerType=LayerType::LT_LOSS;
            lossLayer=name;
        }
        layerInputs[name]=inputLayers;
        mlPush(name, &(pLayer->params), &params);
        checked=false;
        return true;
    }
    bool addLayer(string layerclass, string name, string params, vector<string> inputLayers) {
        CpParams cp(params);
        return addLayer(layerclass, name, cp, inputLayers);
    }

    bool checkTopology(bool verbose=false) {
        if (lossLayer=="") {
            cerr << "No loss layer defined!" << endl;
            return false;
        }
        vector<string> lyr;
        lyr=getLayerFromInput("input");
        if (lyr.size()!=1) {
            cerr << "One (1) layer with name >input< needed, got: " << lyr.size() << endl;
        }
        bool done=false;
        vector<string> lst;
        while (!done) {
            string cl=lyr[0];
            for (auto li : lst) if (li==cl) {
                cerr << "recursion with layer: " << cl << endl;
                return false;
            }
            lst.push_back(cl);
            if (cl==lossLayer) done=true;
            else {
                lyr=getLayerFromInput(cl);
                if (lyr.size()!=1) {
                    cerr << "One (1) layer that uses " << cl << " as input needed, got: " << lyr.size() << endl;
                    return false;
                }
            }
        }
        if (verbose) {
            bool done=false;
            string cLay="input";
            vector<string>nLay;
            while (!done) {
                nLay=getLayerFromInput(cLay);
                string name=nLay[0];
                Layer *p=layerMap[name];

                int inputShapeFlat=1;
                for (int j : p->cp.getPar("inputShape", vector<int>{})) {
                    inputShapeFlat *= j;
                }
                int outputShapeFlat=1;
                for (int j : p->getOutputShape()) {
                    outputShapeFlat *= j;
                }

                cerr << name << ": " << p->cp.getPar("inputShape", vector<int>{}) << "[" << inputShapeFlat << "]";
                cerr << " -> " << p->getOutputShape() << "[" << outputShapeFlat << "]" << endl;

                if (p->layerInit==false) cerr << "  " << name << ": bad initialization!" << endl;
                cLay=nLay[0];
                if (p->layerType==LayerType::LT_LOSS) done=true;
            }
        }
        checked=true;
        return true;
    }
    vector<string> getLayerFromInput(string input) {
        vector<string> lys;
        for (auto li : layerInputs) {
            for (auto lii : li.second) {
                if (lii==input) lys.push_back(li.first);
            }
        }
        return lys;
    }
    virtual MatrixN forward(const MatrixN& x, const MatrixN& y, t_cppl* pcache, int id=0) override {
        string cLay="input";
        vector<string> nLay;
        bool done=false;
        MatrixN x0=x;
        MatrixN xn;
        Timer t;
        trainMode = cp.getPar("train", false);
        if (pcache!=nullptr) cppl_set(pcache, "x", new MatrixN(x));
        if (pcache!=nullptr) cppl_set(pcache, "y", new MatrixN(y));
        while (!done) {
            nLay=getLayerFromInput(cLay);
            if (nLay.size()!=1) {
                cerr << "Unexpected topology: "<< nLay.size() << " layer follow layer " << cLay << " 1 expected.";
                return x;
            }
            string name=nLay[0];
            Layer *p = layerMap[name];
            t_cppl cache;
            //cache.clear();
            if (bench) t.startWall();
            if (p->layerType==LayerType::LT_NORMAL) xn=p->forward(x0,&cache, id);
            else xn=p->forward(x0,y,&cache, id);
            if (bench) cerr << name << "-fw:\t" << t.stopWallMicro() << endl;
            if (pcache!=nullptr) {
                mlPush(name, &cache, pcache);
            } else {
                cppl_delete(&cache);
            }
            if (p->layerType==LayerType::LT_LOSS) done=true;
            cLay=name;
            int oi=-10;
            int fi=-10;
            bool cont=false;
            bool inferr=false;
            for (int i=0; i<xn.size(); i++) {
                if (std::isnan(xn(i)) || std::isinf(xn(i))) {
                    if (i-1==oi) {
                        if (!cont) {
                            cont=true;
                        }
                    } else {
                        cerr << "[" << i;
                        if (std::isnan(xn(i))) cerr << "N"; else cerr <<"I";
                        fi=i;
                        cont=false;
                    }
                    oi=i;
                    inferr=true;
                } else {
                    if (fi==i-1) {
                        cerr << "]";
                        cont=false;
                    } else if (oi==i-1) {
                        cont=false;
                        cerr << ".." << oi;
                        if (std::isnan(xn(oi))) cerr << "N"; else cerr <<"I";
                        cerr << "]";
                    }
                }
            }
            if (inferr) {
                cerr << endl << "Internal error, layer " << name << " resulted in NaN/Inf values! ABORT." << endl;
                //cerr << "x:" << x0 << endl;
                cerr << "y=" << name << "(x):" << shape(x0) << "->" << shape(xn) << endl;
                peekMat("x:", x0);
                cerr << "y=" << name << "(x):";
                peekMat("", xn);
                exit(-1);
            }
            x0=xn;
        }
        return xn;
    }
    virtual floatN loss(const MatrixN& y, t_cppl* pcache) override {
        t_cppl cache;
        if (lossLayer=="") {
            cerr << "Invalid configuration, no loss layer defined!" << endl;
            return 1000.0;
        }
        Layer* pl=layerMap[lossLayer];
        mlPop(lossLayer, pcache, &cache);
        floatN ls=pl->loss(y, &cache);
        return ls;
    }
    virtual MatrixN backward(const MatrixN& y, t_cppl* pcache, t_cppl* pgrads, int id=0) override {
        if (lossLayer=="") {
            cerr << "Invalid configuration, no loss layer defined!" << endl;
            return y;
        }
        bool done=false;
        Timer t;
        MatrixN dxn;
        string cl=lossLayer;
        MatrixN dx0=y;
        trainMode = cp.getPar("train", false);
        while (!done) {
            t_cppl cache;
            t_cppl grads;
            //cache.clear();
            //grads.clear();
            Layer *pl=layerMap[cl];
            mlPop(cl,pcache,&cache);
            if (bench) t.startWall();
            dxn=pl->backward(dx0, &cache, &grads, id);
            if (bench) cerr << cl << "-bw:\t" << t.stopWallMicro() << endl;
            mlPush(cl,&grads,pgrads);
            vector<string> lyr=layerInputs[cl];
            if (lyr[0]=="input") {
                done=true;
            } else {
                cl=lyr[0];
                dx0=dxn;
            }
        }
        return dxn;
    }
    virtual bool update(Optimizer *popti, t_cppl *pgrads, string var, t_cppl *pocache) override {
        for (auto ly : layerMap) {
            t_cppl grads;
            string cl=ly.first;
            Layer *pl=ly.second;
            mlPop(cl, pgrads, &grads);
            pl->update(popti,&grads, var+layerName+cl, pocache);  // XXX push/pop pocache?
        }
        return true;
    }
    virtual void setFlag(string name, bool val) override {
        cp.setPar(name,val);
        for (auto ly : layerMap) {
            ly.second->setFlag(name, val);
        }
    }
};
#endif
