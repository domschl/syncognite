# syncognite - A neural network library inspired by Stanford's CS231n course

[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://domschl.github.io/syncognite/index.html)
[![Cmake](https://github.com/domschl/syncognite/workflows/CMake/badge.svg)](https://github.com/domschl/syncognite/actions)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)

A neural network library for convolutional, fully connected nets and RNNs in C++.

This library implements some of the assignments from Stanfords's [CS231n](http://cs231n.stanford.edu/index.html) 2016 course by Andrej Karpathy, Fei-Fei Li, Justin Johnson and [CS224d](http://cs224d.stanford.edu/index.html) by Richard Socher as C++ framework.

The current `v2`-version of the project has the following objectives:
* implement full support for graphs (not only sequential)
* cleanup & documentation
* This will be work-in-progress for considerable time. The previous version is archived in branch [`v1`](https://github.com/domschl/syncognite/tree/v1).
* CUDA support and other external graphics card libs have been removed (since for good performance they need to rely on blackbox-libs)

Current state: **beta**

### Features

* Fully connected networks
* Convolutional layers
* Recurrent nets (RNNs)
* Long-term short-term memory nets (LSTMs)
* ReLu, Sigmoid, TanH, SELU<sup>(1)</sup>, resilu<sup>(2)</sup> nonlinearities
* BatchNorm, SpatialBatchNorm, Dropout layers
* Softmax, SVM loss
* TemporalAffine and TemporalSoftmax layers for RNNs

[1]: "scaled exponential linear units" (SELUs), https://arxiv.org/abs/1706.02515

[2]: "resilu residual & relu nonlinearity + linearity" (linear skip connection combined with non-linearity) ([s.b.](https://github.com/domschl/syncognite#resilu-non--linearity))

## Sample

### Model

Example: C++ definition of a deep convolutional net with batch-norm, dropout and fully connected layers:

```cpp
LayerBlock lb(R"({"name":"DomsNet","bench":false,"init":"orthonormal"})"_json);

lb.addLayer("Convolution", "cv1", R"({"inputShape":[1,28,28],"kernel":[48,5,5],"stride":1,"pad":2})",{"input"});
lb.addLayer("BatchNorm","sb1","{}",{"cv1"});
lb.addLayer("Relu","rl1","{}",{"sb1"});
lb.addLayer("Dropout","doc1",R"({"drop":0.8})",{"rl1"});
lb.addLayer("Convolution", "cv2", R"({"kernel":[48,3,3],"stride":1,"pad":1})",{"doc1"});
lb.addLayer("Relu","rl2","{}",{"cv2"});
lb.addLayer("Convolution", "cv3", R"({"kernel":[64,3,3],"stride":2,"pad":1})",{"rl2"});
lb.addLayer("BatchNorm","sb2","{}",{"cv3"});
lb.addLayer("Relu","rl3","{}",{"sb2"});
lb.addLayer("Dropout","doc2",R"({"drop":0.8})",{"rl3"});
lb.addLayer("Convolution", "cv4", R"({"kernel":[64,3,3],"stride":1,"pad":1})",{"doc2"});
lb.addLayer("Relu","rl4","{}",{"cv4"});
lb.addLayer("Convolution", "cv5", R"({"kernel":[128,3,3],"stride":2,"pad":1})",{"rl4"});
lb.addLayer("BatchNorm","sb3","{}",{"cv5"});
lb.addLayer("Relu","rl5","{}",{"sb3"});
lb.addLayer("Dropout","doc3",R"({"drop":0.8})",{"rl5"});
lb.addLayer("Convolution", "cv6", R"({"kernel":[128,3,3],"stride":1,"pad":1})",{"doc3"});
lb.addLayer("Relu","rl6","{}",{"cv6"});

lb.addLayer("Affine","af1",R"({"hidden":1024})",{"rl6"});
lb.addLayer("BatchNorm","bn1","{}",{"af1"});
lb.addLayer("Relu","rla1","{}",{"bn1"});
lb.addLayer("Dropout","do1",R"({"drop":0.7})",{"rla1"});
lb.addLayer("Affine","af2",R"({"hidden":512})",{"do1"});
lb.addLayer("BatchNorm","bn2","{}",{"af2"});
lb.addLayer("Relu","rla2","{}",{"bn2"});
lb.addLayer("Dropout","do2",R"({"drop":0.7})",{"rla2"});
lb.addLayer("Affine","af3",R"({"hidden":10})",{"do2"});
lb.addLayer("Softmax","sm1","{}",{"af3"});
```

### Training

```cpp
json jo(R"({"verbose":true,"shuffle":true,"lr_decay":0.95,"epsilon":1e-8})"_json);
jo["epochs"]=(floatN)40.0;
jo["batch_size"]=50;
jo["learning_rate"]=(floatN)5e-4;
jo["regularization"]=(floatN)1e-8;

lb.train(X, y, Xv, yv, "Adam", jo);

floatN train_err, val_err, test_err;
train_err=lb.test(X, y, jo.value("batch_size", 50));
val_err=lb.test(Xv, yv, jo.value("batch_size", 50));
test_err=lb.test(Xt, yt, jo.value("batch_size", 50));

cerr << "Final results on MNIST after " << jo.value("epochs",(floatN)0.0) << " epochs:" << endl;
cerr << "      Train-error: " << train_err << " train-acc: " << 1.0-train_err << endl;
cerr << " Validation-error: " << val_err <<   "   val-acc: " << 1.0-val_err << endl;
cerr << "       Test-error: " << test_err <<  "  test-acc: " << 1.0-test_err << endl;
```

see [mnisttest](cpmnist/) or [cifar10test](cpcifar10/) for complete examples.

### A model that generates text via LSTMs can be defined with:

```cpp
json j0;
string oName{"OH0"};
j0["inputShape"]=vector<int>{T};
j0["V"]=VS;
lb.addLayer("OneHot",oName,j0,{"input"});

int layer_depth=4;
string nName;
json j1;
j1["inputShape"]=vector<int>{VS,T};
j1["N"]=BS;
j1["H"]=H;
j1["forgetgateinitones"]=true;
j1["forgetbias"]=1.0;
j1["clip"]=clip;
for (auto l=0; l<layer_depth; l++) {
	nName="lstm"+std::to_string(l);
	lb.addLayer(rnntype,nName,j1,{oName});
	oName=nName;
}

json j11;
j11["inputShape"]=vector<int>{VS,T};
lb.addLayer("TemporalSoftmax","sm1",j11,{"af1"});
```
see [rnnreader](rnnreader/) for a complete example.

## Dependencies:

* C++ 11 compiler (on Linux (tested: clang, gcc, Intel icpc) or macOS (clang x86-64 and Apple silicon (clang 12, 13)), Raspberry ARM(gcc))
* CMake build system.
* [Hdf5](https://support.hdfgroup.org/HDF5/) [C++ API](https://support.hdfgroup.org/HDF5/doc/cpplus_RM/) for model saving and sample data, `hdf5` or `libhdf5-dev`.

### Apple silicon notes

* use `ccmake` to configure `USE_SYSTEM_BLAS` to `ON`, which instructs eigen to use M1's hardware accelerators. `rnnreader` sees dramatic 3x-6x speedup, single thread benchmarks in `bench` see 200%-400% improvements! [Testet on macOS 12 beta 3 - 2021-07-19]
* Memory: macOS simply doesn't give processes all available memory. Expect swapping (and significant speed decrease) when allocating more than 4-5GB, even on 16GB M1 machines.
* The hdf5 libraries are available for ARM64 (`brew install hdf5`).

### External libraries that are included in the source tree:

* [Eigen](http://eigen.tuxfamily.org/) v3.4 `eigen3`, already (in default configuration) included in the source tree as submodule. 
* [nlohmann_json](https://github.com/nlohmann/json), already included in source tree (cpneural/nlohmann_json).

## Build

syncognite uses the CMake build system.

Clone the repository:

```bash
git clone git://github.com/domschl/syncognite
git submodule init
git submodule update    # This gets the in-tree Eigen3
```

Create a ```build``` directory within the syncognite directory and configure the build:

```bash
# in sycognite/build, default is make-build-system, but Ninja can also be used:
cmake [-G Ninja] ..
# optionally use ccmake to configure options and paths:
ccmake ..
```

To configure your editor / ide for include paths use (in `build`):

```
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=YES ..
```
or simply execute the helper `create_compile_commands.sh`.

macOS users might want to configure for building with Xcode:

```
cmake -G Xcode ..
```

Build the project:

```bash
make
# or
ninja
# or (macOS) start Xcode and load the generated project file, or:
xcodebuild -configuration Release
```

## History

* 2022-04-01: nlohmann_json updated to latest
* 2022-03-24: Serious bug fixed in stateful optimizers (incl. Adam): state was lost on each call, causing slow convergence.
* 2022-03-22: Started v2-branch Removed CUDA and other external graphics libs.
* 2021-10-10: Moved CI from travis (defunct) to github workflows. Valgrind currently disabled.
* 2021-08-21: eigen update to 3.4 release
* 2021-07-19: eigen update to 3.4rc1
* 2021-07-19: Dramatic speed improvements when configuring eigen to use system blas (using `ccmake`) with Apple M1, seems to use M1's magic hardware accelerators.
* 2020-11-12: Switched eigen3 submodule to [gitlab](https://gitlab.com/libeigen/eigen), tracks 3.3 branch
* 2020-07-31: Apple ARM tested ok.
* 2020-07-05: Tests with resilu (non-)linearity
* 2018-03-02: Removed faulty RAN layer, switched to official eigen3 github-mirror at: [Github eigen3](https://github.com/eigenteam/eigen-git-mirror), fixes for eigen-dev stricted type-checking.

## Subprojects:

Things that should work:

* [testneural](cptest/) (cptest subproject, consistency tests for all layers using testdata and numerical differentials)
* [bench](bench/) (benchmark subproject, benchmarks for all layers)
* [mnisttest](cpmnist/) (cpmnist subproject, MNIST handwritten digit recognition with a convolutional network, requires [dataset download](datasets/).)
* [cifar10test](cpcifar10/) (cpcifar10 subproject, cifar10 image recognition with a convolutional network, requires [dataset download](datasets/).)
* [rnnreader](rnnreader/) (rnnreader subproject, text generation via RNN/LSTMs, similar to char-rnn.)


## Appendix

### Resilu (non-) linearity

See [jupyter notebook](https://github.com/domschl/syncognite/blob/master/doc/resilu-linearity.ipynb) for visualization and more discussions of resilu function.

<!-- the folloing uses the hack from https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b to display latex. Seriously. -->
<!-- Good code generator latex -> github: https://jsfiddle.net/8ndx694g/ -->

(1) $\quad rsi(x)=\frac{x}{1-e^{-x}}$

$rsi(x)$ can be rewritten as:

(2) $\quad rsi(x)=\frac{x}{e^{x}-1}+x$

thus can be interpreted as a residual combination of linearity and non-linearity via addition.

Since $rsi(x)$ shows a phase-transition instability at $x=0$, a taylor $O(4)$ approximation is used for $rsi(x)$ and $\nabla rsi(x)$ for $-h\lt 0\lt h$.

Both $e^x$ quotients (1) and (2) have as limit $ReLU(x)$ or, in case of (2): $-ReLU(x)$, if $e^x$ is replaced by $e^{\frac{x}{a}}$  for small constants $a$.
