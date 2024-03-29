#pragma once

#include "cp-neural.h"

class LayerBlock : public Layer {
  private:
    bool bench;
    string inittype{"standard"};
    floatN initfactor = 1.0;
    void setup(const json &jx) {
        j = jx;
        layerName = j.value("name", "block");
        layerClassName = "LayerBlock";
        bench = j.value("bench", false);
        firstLayer = "";
        lastLayer = "";
        trainMode = j.value("train", false);
        inittype = j.value("init", "standard");
        initfactor = j.value("initfactor", (floatN)1.0);
        checked = false;
    }

  public:
    map<string, Layer *> layerMap;
    map<string, vector<string>> layerInputs;
    string firstLayer;
    string lastLayer;
    bool checked;
    bool trainMode;

    LayerBlock(const json &jx) {
        setup(jx);
        layerInit = true;
    }
    LayerBlock(const string conf) {
        setup(json::parse(conf));
        layerInit = true;
    }
    ~LayerBlock() {
        for (auto pli : layerMap) {
            if (pli.second != nullptr) {
                delete pli.second;
                pli.second = nullptr;
            }
        }
        layerMap.clear();
    }
    bool removeLayer(const string name) {
        auto fi = layerMap.find(name);
        if (fi == layerMap.end()) {
            cerr << "Cannot remove layer: " << name
                 << ", a layer with this name does not exist in block "
                 << layerName << endl;
            return false;
        }
        delete fi->second;
        layerMap.erase(fi);
        return true;
    }
    bool addLayer(const string layerclass, const string name, json &jl,
                  const vector<string> inputLayers) {
        jl["layerclass"] = layerclass;
        jl["name"] = name;
        jl["inputlayers"] = inputLayers;

        if (layerMap.find(name) != layerMap.end()) {
            cerr << "Cannot add layer: " << name
                 << ", a layer with this name is already part of block "
                 << layerName << endl;
            return false;
        }
        if (_syncogniteLayerFactory.mapl.find(layerclass) ==
                _syncogniteLayerFactory.mapl.end() and
            layerclass != "Input") {
            cerr << "Cannot add layer: " << layerclass
                 << ", layer class is not defined." << endl;
            return false;
        }
        string firstInput = "";  // XXX multiple input layers!
        for (auto li : inputLayers) {
            if (li != "input") {
                if (layerMap.find(li) == layerMap.end()) {
                    cerr << "Cannot add layer: " << name
                         << ", it depends on an input layer " << li
                         << ", which is not defined." << endl;
                    return false;
                } else {
                    if (firstInput == "")
                        firstInput = li;
                }
            } else {
                firstInput = "input";
            }
        }

        if (firstInput != "" && firstInput != "input") {
            auto lP = layerMap.find(firstInput);
            if (lP == layerMap.end()) {
                cerr << "Can't find input-layer: " << firstInput
                     << " internal error in layer defintion of " << name
                     << endl;
                return false;
            }
            vector<int> inputShape, prevOutputShape;
            inputShape = jl.value("inputShape", vector<int>{});
            prevOutputShape = lP->second->getOutputShape();
            if (prevOutputShape.size() == 0) {
                cerr << "Missing outputShape defintion for inputLayer "
                     << firstInput << endl;
                return false;
            }
            if (inputShape.size() < prevOutputShape.size()) {
                inputShape = prevOutputShape;
            }
            for (unsigned int i = 0; i < prevOutputShape.size(); i++) {
                inputShape[i] = prevOutputShape[i];
            }
            jl["inputShape"] = inputShape;
        }

        jl["name"] = name;

        string itp = jl.value("init", inittype);
        jl["init"] = itp;  // jl.value("init", itp); // set init to global block
                           // value, if not set for the specific layer.
        floatN ifc = jl.value("initfactor", initfactor);
        jl["initfactor"] = ifc;

        layerMap[name] = CREATE_LAYER(layerclass, jl)  // Macro!
            Layer *pLayer = layerMap[name];
        if (pLayer->layerInit == false) {
            cerr << "Attempt to add layer " << name
                 << " failed: Bad initialization." << endl;
            removeLayer(name);
            return false;
        }
        /*
        if (pLayer->layerType & LayerType::LT_LOSS) {
            if (lossLayer != "") {
                cerr << "ERROR: a loss layer with name: " << lossLayer
                     << "has already been defined, cannot add new loss layer: "
                     << name << " to " << layerName << endl;
                removeLayer(name);
                return false;
            }
            layerType = layerType | LayerType::LT_LOSS;
            lossLayer = name;
        }
        */
        layerInputs[name] = inputLayers;
        mlPush(name, &(pLayer->params), &params);
        checked = false;
        return true;
    }
    bool addLayer(string layerclass, string name, string params,
                  vector<string> inputLayers) {
        json jl = json::parse(params);
        // cerr << layerclass << ", " << name << ", " << params << endl;
        return addLayer(layerclass, name, jl, inputLayers);
    }
    bool addLayer(json &jl) {
        const string layerclass = jl["layerclass"];
        const string name = jl["name"];
        const vector<string> inputLayers = jl["inputlayers"];
        return addLayer(layerclass, name, jl, inputLayers);
    }
    bool checkTopology(bool verbose = false) {
        if (lastLayer == "") {
            cerr << "No loss layer defined!" << endl;
            return false;
        }
        vector<string> lyr;
        lyr = getNextLayers("input");
        if (lyr.size() != 1) {
            cerr << "One (1) layer with name >input< needed, got: "
                 << lyr.size() << endl;
        }
        bool done = false;
        vector<string> lst;
        while (!done) {
            string cl = lyr[0];
            for (auto li : lst)
                if (li == cl) {
                    cerr << "recursion with layer: " << cl << endl;
                    return false;
                }
            lst.push_back(cl);
            if (cl == lastLayer)
                done = true;
            else {
                lyr = getNextLayers(cl);
                if (lyr.size() != 1) {
                    cerr << "One (1) layer that uses " << cl
                         << " as input needed, got: " << lyr.size() << endl;
                    return false;
                }
            }
        }
        if (verbose) {
            bool done = false;
            string currentLayer = "input";
            vector<string> nextLayers;
            while (!done) {
                nextLayers = getNextLayers(currentLayer);
                string name = nextLayers[0];
                Layer *p = layerMap[name];

                int inputShapeFlat = 1;
                for (int j : p->j.value("inputShape", vector<int>{})) {
                    inputShapeFlat *= j;
                }
                int outputShapeFlat = 1;
                for (int j : p->getOutputShape()) {
                    outputShapeFlat *= j;
                }
                // string intype=p->cp.getPar("init",(string)"not defined");

                cerr << name << ": " << p->j.value("inputShape", vector<int>{})
                     << "[" << inputShapeFlat << "]";
                cerr << " -> " << p->getOutputShape() << "[" << outputShapeFlat
                     << "]" << endl;

                if (p->layerInit == false)
                    cerr << "  " << name << ": bad initialization!" << endl;
                currentLayer = nextLayers[0];
                if (p->layerType & LayerType::LT_LOSS)
                    done = true;
            }
        }
        checked = true;
        return true;
    }
    vector<string> getNextLayers(string input) {
        vector<string> lys;
        for (auto li : layerInputs) {
            for (auto lii : li.second) {
                if (lii == input)
                    lys.push_back(li.first);
            }
        }
        return lys;
    }
    virtual MatrixN forward(const MatrixN &x, t_cppl *pcache, t_cppl *pstates,
                            int id = 0) override {
        string currentLayer = "input";
        vector<string> nextLayers;
        bool done = false;
        MatrixN x0 = x;
        MatrixN xn;
        Timer t;
        trainMode = j.value("train", false);
        while (!done) {
            nextLayers = getNextLayers(currentLayer);
            if (nextLayers.size() != 1) {
                cerr << "Unexpected topology: " << nextLayers.size()
                     << " layer follow layer " << currentLayer << " 1 expected.";
                return x;
            }
            string name = nextLayers[0];
            Layer *p = layerMap[name];
            t_cppl cache;
            if (pcache != nullptr)
                mlPop(name, pcache, &cache);
            if (bench)
                t.startWall();
            xn = p->forward(x0, &cache, pstates, id);
            if (bench)
                cerr << name << "-fw:\t" << t.stopWallMicro() << endl;
            if (pcache != nullptr) {
                mlPush(name, &cache, pcache);
            } else {
                cppl_delete(&cache);
            }
            if (p->layerType & LayerType::LT_LOSS)
                done = true;
            currentLayer = name;
            int oi = -10;
            int fi = -10;
            bool cont = false;
            bool inferr = false;
            for (int i = 0; i < xn.size(); i++) {
                if (std::isnan(xn(i)) || std::isinf(xn(i))) {
                    if (i - 1 == oi) {
                        if (!cont) {
                            cont = true;
                        }
                    } else {
                        cerr << endl << "[" << i;
                        if (std::isnan(xn(i)))
                            cerr << "N";
                        else
                            cerr << "I";
                        fi = i;
                        cont = false;
                    }
                    oi = i;
                    inferr = true;
                } else {
                    if (fi == i - 1) {
                        cerr << "]";
                        cont = false;
                    } else if (oi == i - 1) {
                        cont = false;
                        cerr << ".." << oi;
                        if (std::isnan(xn(oi)))
                            cerr << "N";
                        else
                            cerr << "I";
                        cerr << "]";
                    }
                }
            }
            if (inferr) {
                cerr << endl
                     << "Internal error, layer " << name
                     << " resulted in NaN/Inf values! ABORT." << endl;
                // cerr << "x:" << x0 << endl;
                cerr << "y=" << name << "(x):" << shape(x0) << "->" << shape(xn)
                     << endl;
                peekMat("x:", x0);
                cerr << "y=" << name << "(x):";
                peekMat("", xn);
                exit(-1);
            }
            x0 = xn;
        }
        return xn;
    }
    virtual MatrixN backward(const MatrixN &dy, t_cppl *pcache, t_cppl *pstates,
                             t_cppl *pgrads, int id = 0) override {
        if (lastLayer == "") {
            cerr << "Invalid configuration, no loss layer defined!" << endl;
            return dy;
        }
        bool done = false;
        Timer t;
        MatrixN dxn;
        string cl = lastLayer;
        MatrixN dx0 = dy;
        trainMode = j.value("train", false);
        while (!done) {
            t_cppl cache;
            t_cppl grads;
            Layer *pl = layerMap[cl];
            mlPop(cl, pcache, &cache);
            if (bench)
                t.startWall();
            dxn = pl->backward(dx0, &cache, pstates, &grads, id);
            if (bench)
                cerr << cl << "-bw:\t" << t.stopWallMicro() << endl;
            mlPush(cl, &grads, pgrads);
            vector<string> lyr = layerInputs[cl];
            if (lyr[0] == "input") {
                done = true;
            } else {
                cl = lyr[0];
                dx0 = dxn;
            }
        }
        return dxn;
    }
    virtual bool update(Optimizer *popti, t_cppl *pgrads, string var,
                        t_cppl *pocache) override {
        for (auto ly : layerMap) {
            t_cppl grads;
            string cl = ly.first;
            Layer *pl = ly.second;
            mlPop(cl, pgrads, &grads);
            pl->update(popti, &grads, var + layerName + cl,
                       pocache);  // XXX push/pop pocache?
        }
        return true;
    }
    virtual void setFlag(string name, bool val) override {
        j[name] = val;
        for (auto ly : layerMap) {
            ly.second->setFlag(name, val);
        }
    }
    virtual void getLayerConfiguration(json &jlc) override {
        string currentLayer = "input";
        vector<string> nextLayers;
        bool done = false;
        vector<json> jlayers{};
        while (!done) {
            nextLayers = getNextLayers(currentLayer);
            if (nextLayers.size() != 1) {
                cerr << "Unexpected topology: " << nextLayers.size()
                     << " layer follow layer " << currentLayer << " 1 expected.";
                break;
            }
            string name = nextLayers[0];
            Layer *p = layerMap[name];
            json ji;
            p->getLayerConfiguration(ji);
            jlayers.push_back(ji);
            if (p->layerType & LayerType::LT_LOSS)
                done = true;
            else
                currentLayer = name;
        }
        jlc[layerName] = j;
        jlc["layerclassname"] = layerClassName;
        jlc["layername"] = layerName;
        jlc["sublayers"] = jlayers;
        return;
    }

    virtual bool saveLayerConfiguration(H5::H5File *pfile) override {
        json lc;
        getLayerConfiguration(lc);

        hsize_t dims1[] = {1};
        int rank{1};
        H5::DataSpace sid1(rank, dims1);
        H5::StrType tid1(0, H5T_VARIABLE);

        if (H5T_STRING != H5Tget_class(tid1.getId()) ||
            !H5Tis_variable_str(tid1.getId())) {
            cerr << "saveLayerConfiguration: not a variable length string type."
                 << endl;
            return false;
        }
        string dsname;
        dsname = "Configuration:" + layerName;
        H5::DataSet dataset = pfile->createDataSet(dsname, tid1, sid1);

        string conf = lc.dump();
        const char *pc[1];
        pc[0] = conf.c_str();
        dataset.write((void *)pc, tid1);
        dataset.close();

        return true;
    }

    virtual bool saveParameters(H5::H5File *pfile) override {
        bool done = false;
        string currentLayer = "input";
        vector<string> nextLayers;
        while (!done) {
            nextLayers = getNextLayers(currentLayer);
            string name = nextLayers[0];
            Layer *p = layerMap[name];
            if (!p->saveParameters(pfile)) {
                cerr << "Saving parameters of layerblock " << layerName
                     << " failed when trying to save sublayer " << name
                     << ", aborting!" << endl;
                return false;
            }
            currentLayer = nextLayers[0];
            if (p->layerType & LayerType::LT_LOSS)
                done = true;
        }
        return true;
    }
    virtual bool loadParameters(H5::H5File *pfile) override {
        bool done = false;
        string currentLayer = "input";
        vector<string> nextLayers;
        while (!done) {
            nextLayers = getNextLayers(currentLayer);
            string name = nextLayers[0];
            Layer *p = layerMap[name];
            if (!p->loadParameters(pfile)) {
                cerr << "Loading parameters of layerblock " << layerName
                     << " failed when trying to load sublayer " << name
                     << ", aborting!" << endl;
                return false;
            }
            currentLayer = nextLayers[0];
            if (p->layerType & LayerType::LT_LOSS)
                done = true;
        }
        return true;
    }
};
