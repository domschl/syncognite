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
#include "cp-layer.h"
#include "cp-layers.h"
#include "cp-layer-tests.h"
#include "cp-optim.h"

#endif
