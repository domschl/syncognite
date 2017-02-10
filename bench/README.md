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
The code in [`bench.cpp`](bench.cpp) shows one-liner layer definitions for all network layers within `benchRecipes`, which can be used as examples of layer definitions.

New benchmarks can be added by extending `benchRecipes`.
## Usage
```bash
bench/bench
```
Press 'q' to stop the Benchmarks.

## Output
```bash

```
