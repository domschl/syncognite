# syncognite - A neural network library inspired by Stanford's CS231n course

A neural network library for convolutional, fully connected nets and RNNs in C++.

This library implements some of the assignments from Stanfords's [CS231n](http://cs231n.stanford.edu/index.html) 2016 course by Andrej Karpathy, Fei-Fei Li, Justin Johnson and [CS224d](http://cs224d.stanford.edu/index.html) by Richard Socher as C++ framework.

Current state: **pre-alpha**
* documentation and pointers to samples incomplete.
* work of RNNs very much unfinished and not yet adapted to API-changes for state-tensor-fields.
* image processing (deep convolutions etc.) should work.

## Sample
### Model
A deep convolutional net with batch-norm, dropout and fully connected layers:
```cpp
  LayerBlock lb("{name='DomsNet';bench=false;init='orthonormal'}");

	lb.addLayer("Convolution", "cv1", "{inputShape=[1,28,28];kernel=[48,5,5];stride=1;pad=2}",{"input"});
	lb.addLayer("BatchNorm","sb1","",{"cv1"});
	lb.addLayer("Relu","rl1","",{"sb1"});
	lb.addLayer("Dropout","doc1","{drop=0.8}",{"rl1"});
	lb.addLayer("Convolution", "cv2", "{kernel=[48,3,3];stride=1;pad=1}",{"doc1"});
	lb.addLayer("Relu","rl2","",{"cv2"});
	lb.addLayer("Convolution", "cv3", "{kernel=[64,3,3];stride=2;pad=1}",{"rl2"});
	lb.addLayer("BatchNorm","sb2","",{"cv3"});
	lb.addLayer("Relu","rl3","",{"sb2"});
	lb.addLayer("Dropout","doc2","{drop=0.8}",{"rl3"});
	lb.addLayer("Convolution", "cv4", "{kernel=[64,3,3];stride=1;pad=1}",{"doc2"});
	lb.addLayer("Relu","rl4","",{"cv4"});
	lb.addLayer("Convolution", "cv5", "{kernel=[128,3,3];stride=2;pad=1}",{"rl4"});
	lb.addLayer("BatchNorm","sb3","",{"cv5"});
	lb.addLayer("Relu","rl5","",{"sb3"});
	lb.addLayer("Dropout","doc3","{drop=0.8}",{"rl5"});
	lb.addLayer("Convolution", "cv6", "{kernel=[128,3,3];stride=1;pad=1}",{"doc3"});
	lb.addLayer("Relu","rl6","",{"cv6"});

	lb.addLayer("Affine","af1","{hidden=1024}",{"rl6"});
	lb.addLayer("BatchNorm","bn1","",{"af1"});
	lb.addLayer("Relu","rla1","",{"bn1"});
	lb.addLayer("Dropout","do1","{drop=0.7}",{"rla1"});
	lb.addLayer("Affine","af2","{hidden=512}",{"do1"});
	lb.addLayer("BatchNorm","bn2","",{"af2"});
	lb.addLayer("Relu","rla2","",{"bn2"});
	lb.addLayer("Dropout","do2","{drop=0.7}",{"rla2"});
	lb.addLayer("Affine","af3","{hidden=10}",{"do2"});
	lb.addLayer("Softmax","sm1","",{"af3"});
```
### Training
```cpp
  CpParams cpo("{verbose=true;shuffle=true;lr_decay=0.95;epsilon=1e-8}");
	cpo.setPar("epochs",(floatN)40.0);
	cpo.setPar("batch_size",50);
	cpo.setPar("learning_rate", (floatN)5e-4);
	cpo.setPar("regularization", (floatN)1e-8);

	lb.train(X, y, Xv, yv, "Adam", cpo);

	floatN train_err, val_err, test_err;
	train_err=lb.test(X, y, cpo.getPar("batch_size", 50));
	val_err=lb.test(Xv, yv, cpo.getPar("batch_size", 50));
	test_err=lb.test(Xt, yt, cpo.getPar("batch_size", 50));

	cerr << "Final results on MNIST after " << cpo.getPar("epochs",(floatN)0.0) << " epochs:" << endl;
	cerr << "      Train-error: " << train_err << " train-acc: " << 1.0-train_err << endl;
	cerr << " Validation-error: " << val_err <<   "   val-acc: " << 1.0-val_err << endl;
	cerr << "       Test-error: " << test_err <<  "  test-acc: " << 1.0-test_err << endl;
```

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
```bash
git clone git://github.com/domschl/syncognite
```
Create a ```Build``` directory within the syncognite directory and configure the build:
```bash
# in sycognite/Build:
cmake ..
# optionally use ccmake to configure options and paths:
ccmake ..
```
Build the project:
```bash
make
```
Things that should work:

Subprojects:
* testneural (cptest subproject, performance tests for all layers using testdata and numerical integration)
* [mnisttest](cpmnist/README.md) (cpmnist subproject, MNIST handwritten digit recognition with a convolutional network, requires [dataset download](datasets/README.md).)
* [cifar10test](cpcifar10/README.md) (cpcifar10 subproject, cifar10 image recognition with a convolutional network, requires [dataset download](datasets/README.md).)
* ...
