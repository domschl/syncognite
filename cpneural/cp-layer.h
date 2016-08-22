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
#else
#ifdef USE_FLOAT
using MatrixN=Eigen::MatrixXf;
using VectorN=Eigen::VectorXf;
using RowVectorN=Eigen::RowVectorXf;
using ArrayN=Eigen::ArrayXf;
using floatN=float;
#endif
#endif

typedef map<string, floatN> t_params;

vector<unsigned int> shape(MatrixN& m) {
    vector<unsigned int> s(2);
    s[0]=(unsigned int)(m.rows());
    s[1]=(unsigned int)(m.cols());
    return s;
}

class Layer {
public:
    string layername;
    vector<MatrixN *> params;
    vector<MatrixN *> grads;
    vector<MatrixN *> cache;
    vector<string> names;

    virtual MatrixN forward(MatrixN& lL)  { MatrixN d(0,0); return d;}
    virtual MatrixN backward(MatrixN& dtL) { MatrixN d(0,0); return d;}
    virtual floatN loss(MatrixN& y) { return 1001.0; }

    floatN learn(MatrixN& x, MatrixN& y, string optimizer, t_params pars);

    bool checkLayer(MatrixN& x, MatrixN& dchain, floatN h, floatN eps, bool lossFkt);
    bool checkAll(MatrixN& x);
    bool checkLoss(MatrixN& x, MatrixN& y0);
private:
    bool checkForward(MatrixN& x, floatN eps);
    bool checkBackward(MatrixN& dchain, floatN eps);
    bool calcNumGrads(MatrixN& dchain, vector<MatrixN *> numGrads, floatN h, bool lossFkt);
    MatrixN calcNumGrad(MatrixN& dchain, unsigned int ind, floatN h);
    MatrixN calcNumGradLoss(MatrixN& dchain, unsigned int ind, floatN h);
    bool checkGradients(MatrixN& x, MatrixN& dchain, floatN h, floatN eps, bool lossFkt);
};

#endif
