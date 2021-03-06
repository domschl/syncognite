#ifndef _CPL_ONEHOT_H
#define _CPL_ONEHOT_H

#include "../cp-layers.h"

class OneHot : public Layer {
  private:
    int T, V;
    MatrixN wrVect;
    void setup(const json &jx) {
        j = jx;
        layerName = j.value("name", (string) "OneHot");
        layerClassName = "OneHot";
        inputShapeRang = 1;
        layerType = LayerType::LT_NORMAL;
        vector<int> inputShape = j.value("inputShape", vector<int>{});
        int inputShapeFlat = 1;
        for (int j : inputShape) {
            inputShapeFlat *= j;
        }
        T = inputShape[0];
        V = j.value("V", 1024);
        outputShape = {V, T};

        wrVect = MatrixN(V, V);
        wrVect.setIdentity();  // Simple one-hot word vectors.

        layerInit = true;
    }

  public:
    OneHot(const json &jx) {
        setup(jx);
    }
    OneHot(const string conf) {
        setup(json::parse(conf));
    }
    ~OneHot() {
        cppl_delete(&params);
    }
    virtual MatrixN forward(const MatrixN &x, t_cppl *pcache, t_cppl *pstates,
                            int id = 0) override {
        if (x.cols() != T) {
            cerr << layerName << ": "
                 << "Forward: dimension mismatch in x:" << shape(x)
                 << " T:" << T << endl;
            MatrixN y(0, 0);
            return y;
        }
        int N = shape(x)[0];

        MatrixN y(N, V * T);
        int n, v, t;
        for (n = 0; n < N; n++) {
            for (v = 0; v < V; v++) {
                for (t = 0; t < T; t++) {
                    y(n, t * V + v) = wrVect((int)x(n, t), v);
                    /*
                    if (x(n,t)>=V) {
                        cerr << "OneHot: Internal error: " << x(n,t) << "
                    exceeds V:" << V << endl; } else { y(n,t*V+v) =
                    wrVect(x(n,t),v);
                    }
                    */
                }
            }
        }
        return y;
    }
    virtual MatrixN backward(const MatrixN &dchain, t_cppl *pcache,
                             t_cppl *pstates, t_cppl *pgrads,
                             int id = 0) override {
        MatrixN dx;
        return dx;  // Matrix of size(0,0): there is no dx, since x is discrete
                    // integers (indices)
    }
};

#endif
