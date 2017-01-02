#ifndef _CP_NEURAL_H
#define _CP_NEURAL_H

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
#include <future>
#include <list>
#include <set>
#include <chrono>
#include <iomanip>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using std::cerr; using std::endl;
using std::vector; using std::string;
using std::map;

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
#endif

using CpParams=ParamParser<floatN>;
typedef t_param_parser<MatrixN *> t_cppl;

#if defined (USE_VIENNACL) || (USE_CUDA)
#define USE_GPU
#endif

#ifdef USE_VIENNACL
#define VIENNACL_HAVE_EIGEN
#ifdef USE_OPENCL
#define VIENNACL_WITH_OPENCL
//#pragma message("Eigen is active with ViennaCl and OpenCL")
#else
#error "VIENNACL currently requires WITH_OPENCL Cmake option to be set."
#endif
#ifdef USE_CUDA
#define VIENNACL_WITH_CUDA
#error "CUDA option with ViennaCL currently does not work!"
#endif
#endif

#ifdef USE_VIENNACL
#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/linalg/prod.hpp>
#endif

#ifdef USE_VIENNACL
#include <viennacl/ocl/device.hpp>
#include <viennacl/ocl/platform.hpp>
#include <viennacl/ocl/backend.hpp>
#endif

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#include "cp-tools.h"
#include "cp-layer.h"
#include "cp-layers.h"
#include "cp-layer-tests.h"
#include "cp-optim.h"

#endif
