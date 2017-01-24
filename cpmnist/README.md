# Training with MNIST handwritten digits
## Requirements
Check the [main page](../README.md) for build requirements.
## Build
After configuration of CMake, the mnist part can be built directly by:
```
make mnisttest
```
## Training
From build directory:
```
cpmnist/mnisttest ../datasets/mnist.h5
```
## Logging
For live logging of the training progress, use gnuplot:
```
gnuplot ../plots/liveplot.gnu
```
Note: Currently, this works only after 1. episode is complete.
![after 6 episodes](../doc/images/mnist6.png)
