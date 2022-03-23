#pragma once

/*! \mainpage Syncognite neural network library in C++
\section Introduction

Syncognite is a header-only library for neural networks in C++.using Eigen3 for matrix operations.

Layers are implemented with manual forward and backward propagation (so no autodiff).

This is the main header file for the library.

In addition to this include file, you may want to include the required layers.

\section Reference
<a href="https://github.com/domschl/syncognite">syncognite github repository</a>
*/

#include <iostream>
#include <fstream>
#include <cctype>
#include <string>
#include <functional>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <queue>
#include <future>
#include <list>
#include <set>
#include <chrono>
#include <iomanip>

#include "nlohmann_json/json.hpp"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using std::cerr; using std::endl;
using std::vector; using std::string;
using std::map;

using json = nlohmann::json;

using Eigen::IOFormat;

#include "cp-math.h"
#include "cp-util.h"
#include "cp-timer.h"

//#define USE_DOUBLE
#ifndef USE_DOUBLE
 #ifndef USE_FLOAT
  #define USE_FLOAT
 #endif
#endif

#ifdef USE_DOUBLE
 #ifdef USE_FLOAT
  #error CONFIGURATION MESS: either USE_DOUBLE or USE_FLOAT, not both!
 #endif
 using MatrixN=Eigen::MatrixXd;
 using VectorN=Eigen::VectorXd;
 using RowVectorN=Eigen::RowVectorXd;
 using ColVectorN=Eigen::VectorXd;
 using ArrayN=Eigen::ArrayXd;
 using floatN=double;
 #define CP_DEFAULT_NUM_H (1.e-6)
 #define CP_DEFAULT_NUM_EPS (1.e-9)
 #define H5_FLOATN (H5::PredType::NATIVE_DOUBLE)
 #define H5_FLOATN_SIZE 8
#endif
#ifdef USE_FLOAT
 using MatrixN=Eigen::MatrixXf;
 using VectorN=Eigen::VectorXf;
 using RowVectorN=Eigen::RowVectorXf;
 using ColVectorN=Eigen::VectorXf;
 using ArrayN=Eigen::ArrayXf;
 using floatN=float;
 #define CP_DEFAULT_NUM_H ((float)1.e-4)
 #define CP_DEFAULT_NUM_EPS ((float)1.e-3)
 #define H5_FLOATN (H5::PredType::NATIVE_FLOAT)
 #define H5_FLOATN_SIZE 4
#endif

using CpParams=ParamParser<floatN>;
typedef t_param_parser<MatrixN *> t_cppl;

#include <H5Cpp.h>
#include <H5File.h>
#include <H5DataSet.h>

#include "cp-tools.h"
#include "cp-layer.h"
#include "cp-layers.h"
#include "cp-layer-tests.h"
#include "cp-optim.h"
