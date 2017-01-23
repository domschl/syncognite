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

## Build
syncognite uses the CMake build system. You will need [Eigen 3.3](http://eigen.tuxfamily.org/index.php?title=Main_Page) and HDF5.

Clone the repository:
```
git clone git://github.com/domschl/syncognite
```
Create a ```Build``` directory within the syncognite directory and configure the build:
```
# in sycognite/Build:
cmake ..
# use ccmake to configure options paths:
ccmake ..
```
Build the project:
```
make
```
Things that should work:

Subprojects:
* testneural (performance tests for all layers using testdata and numerical integration)
* mnisttest (MNIST recognition with a convolutional network)
