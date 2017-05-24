#ifndef _CP_TESTNEURAL_H
#define _CP_TESTNEURAL_H

#include <cp-neural.h>

bool matCompT(MatrixN &m0, MatrixN &m1, string msg, floatN eps, int verbose);
bool matComp(MatrixN &m0, MatrixN &m1, string msg, floatN eps) {
    return matCompT(m0, m1, msg, eps, 3);
}
void registerTestResult(string testcase, string subtest, bool result, string message);

#include "layertests/cplt-affine.h"
#include "layertests/cplt-affinerelu.h"
#include "layertests/cplt-batchnorm.h"
#include "layertests/cplt-convolution.h"
#include "layertests/cplt-dropout.h"
#include "layertests/cplt-lstm.h"
#include "layertests/cplt-nonlinearity.h"
#include "layertests/cplt-pooling.h"
#include "layertests/cplt-relu.h"
#include "layertests/cplt-rnn.h"
#include "layertests/cplt-ran.h"
#include "layertests/cplt-softmax.h"
#include "layertests/cplt-spatialbatchnorm.h"
#include "layertests/cplt-svm.h"
#include "layertests/cplt-temporalaffine.h"
#include "layertests/cplt-temporalsoftmax.h"
#include "layertests/cplt-twolayernet.h"
#include "layertests/cplt-wordembedding.h"

using std::cerr;
using std::endl;


const vector<string> additionaltestcases{
    "LayerBlock", "TrainTwoLayerNet",
};

#endif
