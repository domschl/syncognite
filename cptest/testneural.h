#ifndef _CP_TESTNEURAL_H
#define _CP_TESTNEURAL_H

#include <cp-neural.h>

bool matComp(MatrixN& m0, MatrixN& m1, string msg="", floatN eps=1.e-6);

#include "layertests/cplt-affine.h"
#include "layertests/cplt-relu.h"
#include "layertests/cplt-nonlinearity.h"
#include "layertests/cplt-affinerelu.h"
/*
#include "layertests/cplt-batchnorm.h"
#include "layertests/cplt-dropout.h"
#include "layertests/cplt-convolution.h"
#include "layertests/cplt-pooling.h"
#include "layertests/cplt-spatialbatchnorm.h"
#include "layertests/cplt-svm.h"
#include "layertests/cplt-softmax.h"
#include "layertests/cplt-twolayernet.h"
#include "layertests/cplt-rnn.h"
#include "layertests/cplt-wordembedding.h"
#include "layertests/cplt-temporalaffine.h"
#include "layertests/cplt-temporalsoftmax.h"
*/
using std::cerr; using std::endl;


#endif
