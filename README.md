# syncognite - A neural network library inspired by Stanford's CS231n course

This library implements some of the assignments from [CS231n 2016 course by Andrej Karpathy, Fei-Fei Li, Justin Johnson](http://cs231n.stanford.edu/index.html) as C++ framework.

Current state: **pre-alpha**:
* work of RNNs very much unfinished and not yet adapted to API-changes for state-tensor-fields.
* data-source and build system need work.

## Dependencies:
* C++ 11 compiler (on Linux or Mac, Intel or ARM)
* Eigen v3.3
* Hdf5 for sample data. (optional, for data-sources MNIST and CIFAR10)
* OpenMP (optional, for threading)
* Cuda, OpenCL, ViennaCL (experimental, optional for BLAS speedups)
* CMake build system.