# Tests for all network layers
This tests all currently implemented neural network layers.

Affine, Relu, Nonlinearity, AffineRelu, BatchNorm, Dropout, Convolution, Pooling,
SpatialBatchNorm, RNN, WordEmbedding, TemporalAffine, TemporalSoftmax, Softmax, Svm,
TwoLayerNet.

## Requirements
Check the [main page](../../..) for build requirements.
## Build
After configuration of CMake, the mnist part can be built directly by:
```bash
make testneural
```
## Output
```bash
$ cptest/testneural
Compile-time options: FLOAT AVX FMA OPENMP
Eigen is using:      1 threads.
CpuPool is using:    8 threads.
Cpu+GpuPool is using:    0 threads.
=== 0.: Init: registering layers
=== 1.: Numerical gradient tests
SelfTest for: Affine -----------------
CheckLayer
  check forward vectorizer Affine...
Forward vectorizer OK, err=1.25067e-13
Affine: Forward vectorizing test OK!
  check backward vectorizer Affine...
Backward vectorizer dx OK, err=8.42752e-14
Backward vectorizer dW OK, err=3.44334e-12
Backward vectorizer db OK, err=3.97904e-13
Affine: Backward vectorizing test OK!
  check numerical gradients Affine...
  checking numerical gradient for W...
  checking numerical gradient for b...
  checking numerical gradient for x...
Affine: ∂/∂W OK, err=1.17778e-05
Affine: ∂/∂b OK, err=1.37792e-07
Affine: ∂/∂x OK, err=1.07419e-05
Affine: Gradient numerical test OK!
Affine: checkLayer: Numerical gradient check tests ok!
SelfTest for: Relu -----------------
CheckLayer
  check forward vectorizer Relu...
Forward vectorizer OK, err=0
Relu: Forward vectorizing test OK!
  check backward vectorizer Relu...
Backward vectorizer dx OK, err=0
Relu: Backward vectorizing test OK!
  check numerical gradients Relu...
  checking numerical gradient for x...
Relu: ∂/∂x OK, err=5.01606e-07
Relu: Gradient numerical test OK!
Relu: checkLayer: Numerical gradient check tests ok!
SelfTest for: Nonlinearity-relu -----------------
CheckLayer
  check forward vectorizer Nonlinearity-relu...
Forward vectorizer OK, err=0
Nonlinearity-relu: Forward vectorizing test OK!
  check backward vectorizer Nonlinearity-relu...
Backward vectorizer dx OK, err=0
Nonlinearity-relu: Backward vectorizing test OK!
  check numerical gradients Nonlinearity-relu...
  checking numerical gradient for x...
Nonlinearity-relu: ∂/∂x OK, err=5.6392e-07
Nonlinearity-relu: Gradient numerical test OK!
Nonlinearity-relu: checkLayer: Numerical gradient check tests ok!
SelfTest for: Nonlinearity-sigmoid -----------------
CheckLayer
  check forward vectorizer Nonlinearity-sigmoid...
Forward vectorizer OK, err=0
Nonlinearity-sigmoid: Forward vectorizing test OK!
  check backward vectorizer Nonlinearity-sigmoid...
Backward vectorizer dx OK, err=0
Nonlinearity-sigmoid: Backward vectorizing test OK!
  check numerical gradients Nonlinearity-sigmoid...
  checking numerical gradient for x...
Nonlinearity-sigmoid: ∂/∂x OK, err=1.34359e-06
Nonlinearity-sigmoid: Gradient numerical test OK!
Nonlinearity-sigmoid: checkLayer: Numerical gradient check tests ok!
SelfTest for: Nonlinearity-tanh -----------------
CheckLayer
  check forward vectorizer Nonlinearity-tanh...
Forward vectorizer OK, err=0
Nonlinearity-tanh: Forward vectorizing test OK!
  check backward vectorizer Nonlinearity-tanh...
Backward vectorizer dx OK, err=0
Nonlinearity-tanh: Backward vectorizing test OK!
  check numerical gradients Nonlinearity-tanh...
  checking numerical gradient for x...
Nonlinearity-tanh: ∂/∂x OK, err=3.01089e-06
Nonlinearity-tanh: Gradient numerical test OK!
Nonlinearity-tanh: checkLayer: Numerical gradient check tests ok!
SelfTest for: AffineRelu -----------------
CheckLayer
  check forward vectorizer AffineRelu...
Forward vectorizer OK, err=0
AffineRelu: Forward vectorizing test OK!
  check backward vectorizer AffineRelu...
Backward vectorizer dx OK, err=0
Backward vectorizer daf-W OK, err=1.28841e-13
Backward vectorizer daf-b OK, err=0
AffineRelu: Backward vectorizing test OK!
  check numerical gradients AffineRelu...
  checking numerical gradient for af-W...
  checking numerical gradient for af-b...
  checking numerical gradient for x...
AffineRelu: ∂/∂af-W OK, err=2.71084e-07
AffineRelu: ∂/∂af-b OK, err=4.12491e-08
AffineRelu: ∂/∂x OK, err=6.035e-07
AffineRelu: Gradient numerical test OK!
AffineRelu: checkLayer: Numerical gradient check tests ok!
SelfTest for: BatchNorm -----------------
CheckLayer
  check numerical gradients BatchNorm...
  checking numerical gradient for beta...
  checking numerical gradient for gamma...
  checking numerical gradient for x...
BatchNorm: ∂/∂beta OK, err=5.66084e-06
BatchNorm: ∂/∂gamma OK, err=4.66735e-06
BatchNorm: ∂/∂x OK, err=0.000339093
BatchNorm: Gradient numerical test OK!
BatchNorm: checkLayer: Numerical gradient check tests ok!
SelfTest for: Dropout -----------------
CheckLayer
  check numerical gradients Dropout...
  checking numerical gradient for x...
Dropout: ∂/∂x OK, err=6.89784e-08
Dropout: Gradient numerical test OK!
Dropout: checkLayer: Numerical gradient check tests ok!
SelfTest for: Convolution -----------------
CheckLayer
  check forward vectorizer Convolution...
Forward vectorizer OK, err=8.88178e-16
Convolution: Forward vectorizing test OK!
  check backward vectorizer Convolution...
Backward vectorizer dx OK, err=1.66533e-16
Backward vectorizer dW OK, err=1.51434e-12
Backward vectorizer db OK, err=2.41585e-13
Convolution: Backward vectorizing test OK!
  check numerical gradients Convolution...
  checking numerical gradient for W...
  checking numerical gradient for b...
  checking numerical gradient for x...
Convolution: ∂/∂W OK, err=8.8004e-10
Convolution: ∂/∂b OK, err=3.55271e-13
Convolution: ∂/∂x OK, err=1.0801e-09
Convolution: Gradient numerical test OK!
Convolution: checkLayer: Numerical gradient check tests ok!
SelfTest for: Pooling -----------------
CheckLayer
  check forward vectorizer Pooling...
Forward vectorizer OK, err=0
Pooling: Forward vectorizing test OK!
  check backward vectorizer Pooling...
Backward vectorizer dx OK, err=0
Pooling: Backward vectorizing test OK!
  check numerical gradients Pooling...
  checking numerical gradient for x...
Pooling: ∂/∂x OK, err=1.78114e-06
Pooling: Gradient numerical test OK!
Pooling: checkLayer: Numerical gradient check tests ok!
SelfTest for: SpatialBatchNorm -----------------
CheckLayer
  check numerical gradients SpatialBatchNorm...
  checking numerical gradient for bn-beta...
  checking numerical gradient for bn-gamma...
  checking numerical gradient for x...
SpatialBatchNorm: ∂/∂bn-beta OK, err=7.03705e-07
SpatialBatchNorm: ∂/∂bn-gamma OK, err=2.60908e-06
SpatialBatchNorm: ∂/∂x OK, err=8.78724e-05
SpatialBatchNorm: Gradient numerical test OK!
SpatialBatchNorm: checkLayer: Numerical gradient check tests ok!
SelfTest for: Svm -----------------
CheckLayer
  check forward vectorizer Svm...
Forward vectorizer OK, err=0
Svm: Forward vectorizing test OK!
  check backward vectorizer Svm...
Backward vectorizer dx OK, err=0
Svm: Backward vectorizing test OK!
  check numerical gradients Svm...
Svm: ∂/∂x OK, err=8.54885e-07
Svm: Gradient numerical test OK!
Svm: checkLayer: Numerical gradient check tests ok!
SelfTest for: Softmax -----------------
CheckLayer
  check forward vectorizer Softmax...
Forward vectorizer OK, err=1.33227e-15
Softmax: Forward vectorizing test OK!
  check backward vectorizer Softmax...
Backward vectorizer dx OK, err=1.73472e-17
Softmax: Backward vectorizing test OK!
  check numerical gradients Softmax...
Softmax: ∂/∂x OK, err=3.0876e-08
Softmax: Gradient numerical test OK!
Softmax: checkLayer: Numerical gradient check tests ok!
SelfTest for: TwoLayerNet -----------------
CheckLayer
  check forward vectorizer TwoLayerNet...
Forward vectorizer OK, err=2.1684e-18
TwoLayerNet: Forward vectorizing test OK!
  check backward vectorizer TwoLayerNet...
WARNING: x is not in cache!
Backward vectorizer dx OK, err=7.36395e-22
Backward vectorizer daf1-W OK, err=1.21864e-15
Backward vectorizer daf1-b OK, err=1.30451e-15
Backward vectorizer daf2-W OK, err=2.93136e-15
Backward vectorizer daf2-b OK, err=8.99836e-14
TwoLayerNet: Backward vectorizing test OK!
  check numerical gradients TwoLayerNet...
TwoLayerNet: ∂/∂af1-W OK, err=2.31902e-07
TwoLayerNet: ∂/∂af1-b OK, err=9.23293e-07
TwoLayerNet: ∂/∂af2-W OK, err=3.90354e-07
TwoLayerNet: ∂/∂af2-b OK, err=5.7047e-08
TwoLayerNet: ∂/∂x OK, err=6.31138e-08
TwoLayerNet: Gradient numerical test OK!
TwoLayerNet: checkLayer: Numerical gradient check tests ok!
SelfTest for: RNN -----------------
CheckLayer
  check numerical gradients RNN...
  checking numerical gradient for Whh...
  checking numerical gradient for Wxh...
  checking numerical gradient for bh...
  checking numerical gradient for h0...
Numerical check, mapping gradient h0 to state -> h
pm set
  checking numerical gradient for x...
RNN: ∂/∂Whh OK, err=3.71331e-05
RNN: ∂/∂Wxh OK, err=2.20368e-05
RNN: ∂/∂bh OK, err=8.64702e-06
RNN: ∂/∂h0 OK, err=5.02865e-06
RNN: ∂/∂x OK, err=2.62612e-05
RNN: Gradient numerical test OK!
RNN: checkLayer: Numerical gradient check tests ok!
SelfTest for: WordEmbedding -----------------
CheckLayer
  check numerical gradients WordEmbedding...
  checking numerical gradient for W...
WordEmbedding: ∂/∂W OK, err=8.64375e-12
WordEmbedding: Gradient numerical test OK!
WordEmbedding: checkLayer: Numerical gradient check tests ok!
SelfTest for: TemporalAffine -----------------
CheckLayer
  check forward vectorizer TemporalAffine...
Forward vectorizer OK, err=8.88178e-16
TemporalAffine: Forward vectorizing test OK!
  check backward vectorizer TemporalAffine...
Backward vectorizer dx OK, err=1.16573e-15
Backward vectorizer dW OK, err=4.3473e-12
Backward vectorizer db OK, err=7.42517e-13
TemporalAffine: Backward vectorizing test OK!
  check numerical gradients TemporalAffine...
  checking numerical gradient for W...
  checking numerical gradient for b...
  checking numerical gradient for x...
TemporalAffine: ∂/∂W OK, err=4.03043e-06
TemporalAffine: ∂/∂b OK, err=9.30543e-07
TemporalAffine: ∂/∂x OK, err=2.30975e-06
TemporalAffine: Gradient numerical test OK!
TemporalAffine: checkLayer: Numerical gradient check tests ok!
SelfTest for: TemporalSoftmax -----------------
CheckLayer
  check numerical gradients TemporalSoftmax...
TemporalSoftmax: ∂/∂x OK, err=5.50926e-08
TemporalSoftmax: Gradient numerical test OK!
TemporalSoftmax: checkLayer: Numerical gradient check tests ok!
LayerName for lb: testblock
af1: (10)[10] -> (1024)[1024]
rl1: (1024)[1024] -> (1024)[1024]
af2: (1024)[1024] -> (10)[10]
sm1: (10)[10] -> (1)[1]
Topology-check for LayerBlock: ok.
SelfTest for: testblock -----------------
CheckLayer
  check forward vectorizer testblock...
Forward vectorizer OK, err=0
testblock: Forward vectorizing test OK!
  check backward vectorizer testblock...
WARNING: x is not in cache!
Backward vectorizer dx OK, err=2.29064e-18
Backward vectorizer daf1-W OK, err=2.16861e-14
Backward vectorizer daf1-b OK, err=7.80435e-15
Backward vectorizer daf2-W OK, err=1.4172e-14
Backward vectorizer daf2-b OK, err=9.05942e-14
testblock: Backward vectorizing test OK!
  check numerical gradients testblock...
testblock: ∂/∂af1-W OK, err=0.000266998
testblock: ∂/∂af1-b OK, err=0.000100093
testblock: ∂/∂af2-W OK, err=5.08095e-05
testblock: ∂/∂af2-b OK, err=9.96668e-08
testblock: ∂/∂x OK, err=2.37594e-07
testblock: Gradient numerical test OK!
testblock: checkLayer: Numerical gradient check tests ok!
=== 2.: Test-data tests
AffineForward err=5.13478e-16
AffineForward (Affine) with test data: OK.
AffineBackward dx err=1.13687e-13
AffineBackward dW err=8.90399e-14
AffineBackward bx err=7.10543e-14
AffineBackward (Affine) with test data: OK.
ReluForward err=0
ReluForward with test data: OK.
ReluBackward dx err=0
ReluBackward (Affine) with test data: OK.
NonlinearityForwardRelu err=0
NonlinearityForwardSigmoid err=8.88178e-15
NonlinearityForwardTanh err=1.77636e-15
NonlinearityForward with test data: OK.
NonlinearityBackward (relu) dx err=0
NonlinearityBackward (sigmoid) dx err=2.48412e-15
NonlinearityBackward (tanh) dx err=4.20219e-14
NonlinearityBackward with test data: OK.
AffineRelu err=5.06262e-14
AffineRelu dx err=5.15143e-13
AffineRelu dW err=3.03313e-13
AffineRelu db err=1.42109e-14
AffineRelu with test data: OK.
Mean:-3.57628e-08 -4.47035e-09 -5.96046e-09
StdDev:0.999998 0.999995 0.999993
BatchNormForward err=9.2642e-14
  BatchNorm forward ok.
 err=3.33067e-16
  BatchNorm running mean ok.
 err=2.27596e-15
  BatchNorm running var ok.
Mean:11 12 13
StdDev:0.999998  1.99999  2.99998
BatchNormForward err=9.09495e-13
  BatchNorm beta/gamma forward ok.
 err=3.33067e-16
  BatchNorm running mean2 ok.
 err=2.27596e-15
  BatchNorm running var2 ok.
  Running mean after 200 cycl:  -0.0119702 -0.00531782 -0.00854186
  Running stdvar after 200 cycl: 0.579566 0.576194 0.575204
switching test
  Mean:-1.43051e-08           -1            4
Batchnorm train/test sequence: mean err=1.44155e-14
  StdDev:0.999985  1.99997  2.99995
Batchnorm train/test sequence: stdderi err=3.16448e-09
BatchNormForward with test data: OK.
BatchNormBackward dx err=9.78476e-15
BatchNormBackward dgamma err=1.04416e-13
BatchNormBackward dbeta err=2.84564e-13
BatchNormBackward with test data: OK.
Dropout: x-mean:10.0011
  y-mean:8.01151
  yt-mean:8.00091
  drop:0.8
  offs:10
Dropout: statistics tests ok, err1:0.0105972 err2:0.0011301 err3:0.0115051
Dropout with test data: OK.
ConvolutionForward err=2.26996e-12
ConvolutionForward (Convolution) with test data: OK.
ConvolutionBackward dx err=1.53643e-11
ConvolutionBackward dW err=5.04814e-11
ConvolutionBackward bx err=1.47793e-11
ConvolutionBackward (Convolution) with test data: OK.
PoolingForward err=0
PoolingForward with test data: OK.
PoolingBackward dx err=0
PoolingBackward with test data: OK.
Svm probabilities err=2.84217e-14
Loss ok, loss=4.00208 (ref: 4.00208), err=4.76837e-07
Softmax dx err=0
Svm with test data: OK.
Softmax probabilities err=6.66134e-15
Loss ok, loss=1.6092 (ref: 1.6092), err=2.38419e-07
Softmax dx err=5.23886e-16
Softmax with test data: OK.
TwoLayerNetScores err=7.81597e-14
TwoLayerNet: loss-err: 1.19209e-07 for reg=0 OK.
Got grads: af1-W af1-b af2-W af2-b
TwoLayerNet dW1 err=1.29931e-14
TwoLayerNet db1 err=1.97758e-16
TwoLayerNet dW2 err=4.26326e-13
TwoLayerNet db2 err=8.88178e-16
TwoLayerNet with test data: OK.
RNNForwardStep err=3.01981e-14
RNNForwardStep with test data: OK.
RNNStepBackward dx err=1.53038e-12
RNNStepBackward dWxh err=1.87741e-12
RNNStepBackward dWhh err=8.81343e-13
RNNStepBackward bh err=5.39151e-13
RNNStepBackward h0 err=2.61745e-12
RNNBackwardStep with test data: OK.
RNNForward err=2.86854e-14
RNNForward with test data: OK.
RNNBackward dx err=2.68238e-11
RNNBackward dWxh err=3.31291e-11
RNNBackward dWhh err=1.78926e-11
RNNBackward bh err=3.87246e-13
RNNBackward h0 err=8.74733e-12
RNNBackward with test data: OK.
WordEmbeddingForward err=0
WordEmbeddingForward with test data: OK.
WordEmbeddingBackward forward consistency check err=0
WordEmbeddingBackward dW err=4.81805e-12
WordEmbeddingBackward with test data: OK.
TemporalAffineForward err=7.8515e-13
TemporalAffineForward with test data: OK.
TemporalAffineBackward dx err=1.42469e-12
TemporalAffineBackward dW err=3.1819e-13
TemporalAffineBackward bx err=6.03961e-14
TemporalAffineBackward with test data: OK.
Checking TemporalSoftmaxLoss:
  TemporalSMLoss check OK for ex (1): 2.30265, theoretical: 2.3
  TemporalSMLoss check OK for ex (2): 23.0258, theoretical: 23
  TemporalSMLoss check OK for ex (3): 2.32376, theoretical: 2.3
TemporalSoftmaxLoss with test data: OK.
TemporalSoftmax dx err=3.31224e-15
TemporalSoftmax with test data: OK.
Training net: data-size: 100, chunks: 5, batch_size: 20, threads: 4 (bz*ch): 100
Train-test, train-err=0.06
       validation-err=0.075
       final test-err=0.075
TrainTest: OK.
All tests ok.
```
