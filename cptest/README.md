# Tests for all network layers

This tests all currently implemented neural network layers.

Affine, Relu, Nonlinearity, AffineRelu, BatchNorm, Dropout, Convolution, Pooling,
SpatialBatchNorm, RNN, LSTM, WordEmbedding, TemporalAffine, TemporalSoftmax, Softmax, Svm,
TwoLayerNet.

## Requirements

Check the [main page](../../..) for build requirements.

## Build

After configuration of CMake, the mnist part can be built directly by:

```bash
make testneural
```

## Usage

```bash
cptest/testneural [-v[v[v]]] <optional-list-of-module-names>
```

* `-v`, `-vv`, `-vvv` verbose to very, very verbose output
* If no module name(s) are given, all modules are tested. Possible modules are (displayed, when called with no arguments): `Affine AffineRelu BatchNorm Convolution Dropout LSTM Nonlinearity OneHot Pooling RNN Relu Softmax SpatialBatchNorm Svm TemporalAffine TemporalSoftmax TwoLayerNet WordEmbedding LayerBlock TrainTwoLayerNet`.

## Output

```bash
$ cptest/testneural -vv


Syncognite layer tests
Active tests: Affine AffineRelu BatchNorm Convolution Dropout LSTM Nonlinearity OneHot Pooling RNN Relu Softmax SpatialBatchNorm Svm TemporalAffine TemporalSoftmax TwoLayerNet WordEmbedding LayerBlock TrainTwoLayerNet 
Affine Layer: 
  SelfTest for: Affine
  Forward vectorizer OK, err=1.69104e-13
  Affine: Forward vectorizing test OK!
    Backward vectorizer dx OK, err=7.6614e-14
    Backward vectorizer dW OK, err=3.22011e-12
    Backward vectorizer db OK, err=3.16192e-13
  Affine: Backward vectorizing test OK!
    Affine: ∂/∂W OK, err=8.54962e-06
    Affine: ∂/∂b OK, err=2.21182e-07
    Affine: ∂/∂x OK, err=8.42131e-06
  Affine: Gradient numerical test OK!
  Affine: checkLayer: Numerical gradient check tests ok!
  Affine, Numerical gradient: Ok 
    AffineForward, err=5.13478e-16
  Affine, Forward (with test-data): Ok 
    AffineBackward dx, err=1.13687e-13
    AffineBackward dW, err=8.90399e-14
    AffineBackward bx, err=7.10543e-14
  Affine, Backward (with test-data): Ok 
Relu Layer: 
  SelfTest for: Relu
  Forward vectorizer OK, err=0
  Relu: Forward vectorizing test OK!
    Backward vectorizer dx OK, err=0
  Relu: Backward vectorizing test OK!
    Relu: ∂/∂x OK, err=6.47668e-07
  Relu: Gradient numerical test OK!
  Relu: checkLayer: Numerical gradient check tests ok!
  Relu, Numerical gradient: Ok 
    ReluForward, err=0
  Relu, Forward (with test-data): Ok 
    ReluBackward dx, err=0
  Relu, Backward (with test-data): Ok 
Nonlinearity Layers: 
  SelfTest for: Nonlinearity-relu
  Forward vectorizer OK, err=0
  Nonlinearity-relu: Forward vectorizing test OK!
    Backward vectorizer dx OK, err=0
  Nonlinearity-relu: Backward vectorizing test OK!
    Nonlinearity-relu: ∂/∂x OK, err=5.73272e-07
  Nonlinearity-relu: Gradient numerical test OK!
  Nonlinearity-relu: checkLayer: Numerical gradient check tests ok!
  Nonlinearity, relu, numerical gradient: Ok 
  SelfTest for: Nonlinearity-sigmoid
  Forward vectorizer OK, err=0
  Nonlinearity-sigmoid: Forward vectorizing test OK!
    Backward vectorizer dx OK, err=0
  Nonlinearity-sigmoid: Backward vectorizing test OK!
    Nonlinearity-sigmoid: ∂/∂x OK, err=1.59529e-06
  Nonlinearity-sigmoid: Gradient numerical test OK!
  Nonlinearity-sigmoid: checkLayer: Numerical gradient check tests ok!
  Nonlinearity, sigmoid, numerical gradient: Ok 
  SelfTest for: Nonlinearity-tanh
  Forward vectorizer OK, err=0
  Nonlinearity-tanh: Forward vectorizing test OK!
    Backward vectorizer dx OK, err=0
  Nonlinearity-tanh: Backward vectorizing test OK!
    Nonlinearity-tanh: ∂/∂x OK, err=2.43863e-06
  Nonlinearity-tanh: Gradient numerical test OK!
  Nonlinearity-tanh: checkLayer: Numerical gradient check tests ok!
  Nonlinearity, tanh, numerical gradient: Ok 
    NonlinearityForwardRelu, err=0
    NonlinearityForwardSigmoid, err=8.88178e-15
    NonlinearityForwardTanh, err=1.77636e-15
  Nonlinearity, Forward (with test-data): Ok 
    NonlinearityBackward (relu) dx, err=0
    NonlinearityBackward (sigmoid) dx, err=2.48412e-15
    NonlinearityBackward (tanh) dx, err=4.20219e-14
  Nonlinearity, Backward (with test-data): Ok 
AffineRelu Layer: 
  SelfTest for: AffineRelu
  Forward vectorizer OK, err=0
  AffineRelu: Forward vectorizing test OK!
    Backward vectorizer dx OK, err=0
    Backward vectorizer daf-W OK, err=0
    Backward vectorizer daf-b OK, err=0
  AffineRelu: Backward vectorizing test OK!
    AffineRelu: ∂/∂af-W OK, err=2.92675e-07
    AffineRelu: ∂/∂af-b OK, err=4.45456e-08
    AffineRelu: ∂/∂x OK, err=2.58245e-07
  AffineRelu: Gradient numerical test OK!
  AffineRelu: checkLayer: Numerical gradient check tests ok!
  AffineRelu, Numerical gradient: Ok 
    AffineRelu, err=5.06262e-14
    AffineRelu dx, err=5.15143e-13
    AffineRelu dW, err=3.03313e-13
    AffineRelu db, err=1.42109e-14
  AffineRelu, Forward/Backward (with test-data): Ok 
BatchNorm Layer: 
  SelfTest for: BatchNorm
    BatchNorm: ∂/∂beta OK, err=1.77087e-06
    BatchNorm: ∂/∂gamma OK, err=2.06069e-06
    BatchNorm: ∂/∂x OK, err=0.000187067
  BatchNorm: Gradient numerical test OK!
  BatchNorm: checkLayer: Numerical gradient check tests ok!
  BatchNorm, Numerical gradient: Ok 
    Mean:-3.57628e-08 -4.47035e-09 -5.96046e-09
    StdDev:0.999998 0.999995 0.999993
    BatchNormForward, err=9.2642e-14
  BatchNorm forward ok.
    BatchNorm run-mean, err=3.33067e-16
  BatchNorm running mean ok.
    BatchNorm run-var, err=2.27596e-15
  BatchNorm running var ok.
    Mean:11 12 13
    StdDev:0.999998  1.99999  2.99998
    BatchNormForward, err=9.09495e-13
  BatchNorm beta/gamma forward ok.
    BatchNormForward, err=3.33067e-16
  BatchNorm running mean2 ok.
    BatchNormForward, err=2.27596e-15
  BatchNorm running var2 ok.
    Running mean after 200 cycl: -0.00770193  0.00501021 -0.00927578
    Running stdvar after 200 cycl: 0.582115 0.575163  0.57423
  switching test
    Mean:-4.76837e-09           -1            4
    Batchnorm train/test sequence: mean, err=2.27374e-17
    StdDev:0.999985  1.99997  2.99996
    Batchnorm train/test sequence: stdderi, err=3.30893e-09
  BatchNorm, Forward (with test-data): Ok 
    BatchNormBackward dx, err=9.78476e-15
    BatchNormBackward dgamma, err=1.04416e-13
    BatchNormBackward dbeta, err=2.84564e-13
  BatchNorm, Backward (with test-data): Ok 
Dropout Layer: 
  SelfTest for: Dropout
    Dropout: ∂/∂x OK, err=6.89784e-08
  Dropout: Gradient numerical test OK!
  Dropout: checkLayer: Numerical gradient check tests ok!
  Dropout, Numerical gradient: Ok 
  Dropout: x-mean:9.99955
    y-mean:7.9936
    yt-mean:7.99963
    drop:0.8
    offs:10
  Dropout: statistics tests ok, err1: 0.00603485 err2: 0.000451088 err3: 0.00640202
  Dropout, Forward (with test-data): Ok 
Convolution Layer: 
  SelfTest for: Convolution
  Forward vectorizer OK, err=0
  Convolution: Forward vectorizing test OK!
    Backward vectorizer dx OK, err=0
    Backward vectorizer dW OK, err=1.51879e-12
    Backward vectorizer db OK, err=1.03739e-12
  Convolution: Backward vectorizing test OK!
    Convolution: ∂/∂W OK, err=8.0253e-10
    Convolution: ∂/∂b OK, err=5.54792e-11
    Convolution: ∂/∂x OK, err=6.09593e-10
  Convolution: Gradient numerical test OK!
  Convolution: checkLayer: Numerical gradient check tests ok!
  Convolution, Numerical gradient: Ok 
    ConvolutionForward, err=2.26996e-12
  Convolution, Forward (with test-data): Ok 
    ConvolutionBackward dx, err=1.53643e-11
    ConvolutionBackward dW, err=5.04814e-11
    ConvolutionBackward bx, err=1.47793e-11
  Convolution, Backward (with test-data): Ok 
Pooling Layer: 
  SelfTest for: Pooling
  Forward vectorizer OK, err=0
  Pooling: Forward vectorizing test OK!
    Backward vectorizer dx OK, err=0
  Pooling: Backward vectorizing test OK!
    Pooling: ∂/∂x OK, err=1.8559e-06
  Pooling: Gradient numerical test OK!
  Pooling: checkLayer: Numerical gradient check tests ok!
  Pooling, Numerical gradient: Ok 
    PoolingForward, err=0
  Pooling, Forward (with test-data): Ok 
    PoolingBackward dx, err=0
  Pooling, Backward (with test-data): Ok 
SpatialBatchNorm Layer: 
  SelfTest for: SpatialBatchNorm
    SpatialBatchNorm: ∂/∂bn-beta OK, err=7.68454e-07
    SpatialBatchNorm: ∂/∂bn-gamma OK, err=3.277e-06
    SpatialBatchNorm: ∂/∂x OK, err=8.45516e-05
  SpatialBatchNorm: Gradient numerical test OK!
  SpatialBatchNorm: checkLayer: Numerical gradient check tests ok!
  SpatialBatchNorm, Numerical gradient: Ok 
  There is currently no additional forward/backward test
    with test-data implemented.
SVM Layer: 
  SelfTest for: Svm
  Forward vectorizer OK, err=0
  Svm: Forward vectorizing test OK!
    Backward vectorizer dx OK, err=0
  Svm: Backward vectorizing test OK!
    Svm: ∂/∂x OK, err=6.58882e-07
  Svm: Gradient numerical test OK!
  Svm: checkLayer: Numerical gradient check tests ok!
  SVM, Numerical gradient: Ok 
    Svm probabilities, err=2.84217e-14
  Loss ok, loss=4.00208 (ref: 4.00208), err=4.76837e-07
    Softmax dx, err=0
  SVM, Check (with test-data): Ok 
Softmax Layer: 
  SelfTest for: Softmax
  Forward vectorizer OK, err=0
  Softmax: Forward vectorizing test OK!
    Backward vectorizer dx OK, err=0
  Softmax: Backward vectorizing test OK!
    Softmax: ∂/∂x OK, err=2.45815e-08
  Softmax: Gradient numerical test OK!
  Softmax: checkLayer: Numerical gradient check tests ok!
  Softmax, Numerical gradient: Ok 
    Softmax probabilities, err=6.66134e-15
  Loss ok, loss=1.6092 (ref: 1.6092), err=2.38419e-07
    Softmax dx, err=5.23886e-16
  Softmax, Check (with test-data): Ok 
TwoLayerNet Layer: 
  SelfTest for: TwoLayerNet
  Forward vectorizer OK, err=0
  TwoLayerNet: Forward vectorizing test OK!
    Backward vectorizer dx OK, err=7.51418e-23
    Backward vectorizer daf1-W OK, err=3.69171e-17
    Backward vectorizer daf1-b OK, err=1.05181e-16
    Backward vectorizer daf2-W OK, err=1.13895e-16
    Backward vectorizer daf2-b OK, err=9.23761e-13
  TwoLayerNet: Backward vectorizing test OK!
    TwoLayerNet: ∂/∂af1-W OK, err=2.66674e-07
    TwoLayerNet: ∂/∂af1-b OK, err=1.95608e-07
    TwoLayerNet: ∂/∂af2-W OK, err=9.01057e-08
    TwoLayerNet: ∂/∂af2-b OK, err=3.93722e-08
    TwoLayerNet: ∂/∂x OK, err=2.90555e-08
  TwoLayerNet: Gradient numerical test OK!
  TwoLayerNet: checkLayer: Numerical gradient check tests ok!
  TwoLayerNet, Numerical gradient: Ok 
    TwoLayerNetScores, err=7.81597e-14
  TwoLayerNet: loss-err: 1.19209e-07 for reg=0 OK.
    TwoLayerNet dW1, err=1.29931e-14
    TwoLayerNet db1, err=1.97758e-16
    TwoLayerNet dW2, err=4.26326e-13
    TwoLayerNet db2, err=8.88178e-16
  TwoLayerNet, Check (with test-data): Ok 
RNN Layer: 
  SelfTest for: rnn
    rnn: ∂/∂Whh OK, err=3.18309e-05
    rnn: ∂/∂Wxh OK, err=3.27241e-05
    rnn: ∂/∂bh OK, err=3.78659e-06
    rnn: ∂/∂rnn-h0 OK, err=6.17938e-06
    rnn: ∂/∂x OK, err=2.1104e-05
  rnn: Gradient numerical test OK!
  rnn: checkLayer: Numerical gradient check tests ok!
  RNN, Numerical gradient: Ok 
    RNNForwardStep, err=3.01981e-14
  RNN, StepForward (with test-data): Ok 
    RNNStepBackward dx, err=1.53038e-12
    RNNStepBackward dWxh, err=1.87741e-12
    RNNStepBackward dWhh, err=8.81343e-13
    RNNStepBackward bh, err=5.39151e-13
    RNNStepBackward h0, err=2.61745e-12
  RNN, StepBackward (with test-data): Ok 
    RNNForward, err=2.86854e-14
  RNN, Forward (with test-data): Ok 
    RNNBackward dx, err=2.68238e-11
    RNNBackward dWxh, err=3.31291e-11
    RNNBackward dWhh, err=1.78926e-11
    RNNBackward bh, err=3.87246e-13
    RNNBackward h0, err=8.74733e-12
  RNN, Backward (with test-data): Ok 
LSTM Layer: 
  SelfTest for: lstm
    lstm: ∂/∂Whh OK, err=1.06849e-05
    lstm: ∂/∂Wxh OK, err=1.44986e-05
    lstm: ∂/∂bh OK, err=3.39602e-06
    lstm: ∂/∂lstm-h0 OK, err=5.83338e-07
    lstm: ∂/∂x OK, err=3.10257e-06
  lstm: Gradient numerical test OK!
  lstm: checkLayer: Numerical gradient check tests ok!
  LSTM, Numerical gradient: Ok 
    LSTMForwardStep, err=8.28226e-14
    LSTMForwardStep, err=3.9968e-14
  LSTM, StepForward (with test-data): Ok 
    internal LSTM state-test, err=0
    LSTMStepBackward dx, err=1.57008e-12
    LSTMStepBackward dWxh, err=7.6005e-13
    LSTMStepBackward dWhh, err=1.4087e-12
    LSTMStepBackward bh, err=2.17649e-13
    LSTMStepBackward h0, err=8.54431e-13
    LSTMStepBackward c0, err=7.49541e-14
  LSTM, StepBackward (with test-data): Ok 
    LSTMForward, err=7.82499e-14
  LSTM, Forward (with test-data): Ok 
    LSTMBackward dx, err=1.81453e-13
    LSTMBackward dWxh, err=3.68338e-13
    LSTMBackward dWhh, err=1.17496e-13
    LSTMBackward bh, err=4.93251e-14
    LSTMBackward h0, err=4.996e-14
  LSTM, Backward (with test-data): Ok 
WordEmbedding Layer: 
  SelfTest for: WordEmbedding
    WordEmbedding: ∂/∂W OK, err=8.88534e-12
  WordEmbedding: Gradient numerical test OK!
  WordEmbedding: checkLayer: Numerical gradient check tests ok!
  WordEmbedding, Numerical gradient: Ok 
    WordEmbeddingForward, err=0
  WordEmbedding, Forward (with test-data): Ok 
    WordEmbeddingBackward forward consistency check, err=0
    WordEmbeddingBackward dW, err=4.81805e-12
  WordEmbedding, Backward (with test-data): Ok 
TemporalAffine Layer: 
  SelfTest for: TemporalAffine
  Forward vectorizer OK, err=1.11022e-15
  TemporalAffine: Forward vectorizing test OK!
    Backward vectorizer dx OK, err=1.77636e-15
    Backward vectorizer dW OK, err=3.87157e-12
    Backward vectorizer db OK, err=5.11591e-13
  TemporalAffine: Backward vectorizing test OK!
    TemporalAffine: ∂/∂W OK, err=5.49355e-06
    TemporalAffine: ∂/∂b OK, err=7.36359e-07
    TemporalAffine: ∂/∂x OK, err=6.46227e-06
  TemporalAffine: Gradient numerical test OK!
  TemporalAffine: checkLayer: Numerical gradient check tests ok!
  TemporalAffine, Numerical gradient: Ok 
    TemporalAffineForward, err=7.8515e-13
  TemporalAffine, Forward (with test-data): Ok 
    TemporalAffineBackward dx, err=1.42469e-12
    TemporalAffineBackward dW, err=3.1819e-13
    TemporalAffineBackward bx, err=6.03961e-14
  TemporalAffine, Backward (with test-data): Ok 
TemporalSoftmax Layer: 
  SelfTest for: TemporalSoftmax
    TemporalSoftmax: ∂/∂x OK, err=2.38695e-08
  TemporalSoftmax: Gradient numerical test OK!
  TemporalSoftmax: checkLayer: Numerical gradient check tests ok!
  TemporalSoftmax, Numerical gradient: Ok 
  Checking TemporalSoftmaxLoss:
    TemporalSMLoss check OK for ex (1): 2.30271, theoretical: 2.3
    TemporalSMLoss check OK for ex (2): 23.0259, theoretical: 23
    TemporalSMLoss check OK for ex (3): 2.34867, theoretical: 2.3
  TemporalSoftmax, Loss (with test-data): Ok 
    TemporalSoftmax dx, err=3.31224e-15
  TemporalSoftmax, Check (with test-data): Ok 
LayerBlock Layer (Affine-ReLu-Affine-Softmax): 
  LayerName for lb: testblock
  LayerBlock, Topology check: Ok 
  SelfTest for: testblock
  Forward vectorizer OK, err=0
  testblock: Forward vectorizing test OK!
    Backward vectorizer dx OK, err=5.18191e-18
    Backward vectorizer daf1-W OK, err=2.89076e-14
    Backward vectorizer daf1-b OK, err=7.0635e-15
    Backward vectorizer daf2-W OK, err=1.35139e-14
    Backward vectorizer daf2-b OK, err=1.68754e-14
  testblock: Backward vectorizing test OK!
    testblock: ∂/∂af1-W OK, err=0.000501083
    testblock: ∂/∂af1-b OK, err=0.000158899
    testblock: ∂/∂af2-W OK, err=6.95978e-05
    testblock: ∂/∂af2-b OK, err=5.1601e-08
    testblock: ∂/∂x OK, err=2.61467e-07
  testblock: Gradient numerical test OK!
  testblock: checkLayer: Numerical gradient check tests ok!
  LayerBlock, Numerical self-test: Ok 
TwoLayerNet training test: 
  Training net: data-size: 100, chunks: 5, batch_size: 20, threads: 4 (bz*ch): 100
  Train-test, train-err=0.27
         validation-err=0.225
         final test-err=0.225
  TrainTest, TwoLayerNet training: Ok 
All tests ok.
```
