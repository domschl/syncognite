# Benchmarks for all network layers
This benchmarks all currently implemented neural network layers.

Affine, Relu, Nonlinearity, AffineRelu, BatchNorm, Dropout, Convolution, Pooling,
SpatialBatchNorm, RNN, LSTM, WordEmbedding, TemporalAffine, TemporalSoftmax, Softmax, Svm, TwoLayerNet.

## Requirements
Check the [main page](../../..) for build requirements. Additionally, `curses` is needed.

## Build
After configuration of CMake, the mnist part can be built directly by:
```bash
make bench
```
The code in [`cp-bench.cpp`](cp-bench.cpp) shows one-liner layer definitions for all network layers within `benchRecipes`, which can be used as examples of layer definitions.

## Extending the benchmarks
New benchmarks can be added by extending `benchRecipes`:
```cpp
map<string, string> benchRecipes = {
    {"Affine", "{benchIdx=1;benchName='Affine';benchN=100;inputShape=[1024];hidden=1024}"},
    {"Convolution", "{benchIdx=10;benchName='Convolution';benchN=100;inputShape=[3,32,32];kernel=[64,5,5];stride=1;pad=2}"},
    {"LSTM", "{benchIdx=12;benchName='LSTM';benchN=100;inputShape=[100,80];N=100;H=256}"},
// ...
};
```

## Usage
```bash
bench/bench
```
Press 'q' to stop the Benchmarks.

## Output
```bash

```
