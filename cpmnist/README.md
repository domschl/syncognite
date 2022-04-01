# Training with MNIST handwritten digits
## Requirements
Check the [main page](../../..) for build requirements.
## Build
After configuration of CMake, the mnist part can be built directly by:
```bash
make mnisttest
```
## Dataset
Use the [script download_mnist.py](../datasets/) to download the MNIST training data in HDF5 format.

## Training
From `Build` directory:
```bash
cpmnist/mnisttest ../datasets/mnist.h5
```
## Logging
For live logging of the training progress, use gnuplot:
```bash
gnuplot ../plot/liveplot.gnu
```
![40 epochs](../doc/images/mnist-graph.png)

Note: Currently, gnuplot can only show graphs after 1. episode is complete.

## Output
```bash
$ cpmnist/mnisttest ../datasets/mnist.h5
Reading: t_test t_train t_valid x_test x_train x_valid
t_test (10000, 1)
t_train (50000, 1)
t_valid (10000, 1)
x_test (10000, 784)
x_train (50000, 784)
x_valid (10000, 784)
Compile-time options: FLOAT
Eigen is using:      1 threads.
CpuPool is using:    8 threads.
Checking multi-layer topology...
cv1: (1, 28, 28)[784] -> (48, 28, 28)[37632]
sb1: (48, 28, 28)[37632] -> (48, 28, 28)[37632]
rl1: (48, 28, 28)[37632] -> (48, 28, 28)[37632]
doc1: (48, 28, 28)[37632] -> (48, 28, 28)[37632]
cv2: (48, 28, 28)[37632] -> (48, 28, 28)[37632]
rl2: (48, 28, 28)[37632] -> (48, 28, 28)[37632]
cv3: (48, 28, 28)[37632] -> (64, 14, 14)[12544]
sb2: (64, 14, 14)[12544] -> (64, 14, 14)[12544]
rl3: (64, 14, 14)[12544] -> (64, 14, 14)[12544]
doc2: (64, 14, 14)[12544] -> (64, 14, 14)[12544]
cv4: (64, 14, 14)[12544] -> (64, 14, 14)[12544]
rl4: (64, 14, 14)[12544] -> (64, 14, 14)[12544]
cv5: (64, 14, 14)[12544] -> (128, 7, 7)[6272]
sb3: (128, 7, 7)[6272] -> (128, 7, 7)[6272]
rl5: (128, 7, 7)[6272] -> (128, 7, 7)[6272]
doc3: (128, 7, 7)[6272] -> (128, 7, 7)[6272]
cv6: (128, 7, 7)[6272] -> (128, 7, 7)[6272]
rl6: (128, 7, 7)[6272] -> (128, 7, 7)[6272]
af1: (128, 7, 7)[6272] -> (1024)[1024]
bn1: (1024)[1024] -> (1024)[1024]
rla1: (1024)[1024] -> (1024)[1024]
do1: (1024)[1024] -> (1024)[1024]
af2: (1024)[1024] -> (512)[512]
bn2: (512)[512] -> (512)[512]
rla2: (512)[512] -> (512)[512]
do2: (512)[512] -> (512)[512]
af3: (512)[512] -> (10)[10]
sm1: (10)[10] -> (1)[1]
Topology-check for MultiLayer: ok.

Training net: data-size: 50000, chunks: 1000, batch_size: 50, threads: 8 (bz*ch): 50000
Ep: 1, Time: 161s, (20s test) loss:0.0924 err(val):0.0203 acc(val):0.9797
Ep: 2, Time: 161s, (20s test) loss:0.0623 err(val):0.0162 acc(val):0.9838
Ep: 3, Time: 160s, (21s test) loss:0.0488 err(val):0.0146 acc(val):0.9854
Ep: 4, Time: 160s, (20s test) loss:0.0367 err(val):0.0119 acc(val):0.9881
Ep: 5, Time: 160s, (20s test) loss:0.0335 err(val):0.0114 acc(val):0.9886
Ep: 6, Time: 159s, (20s test) loss:0.0195 err(val):0.0113 acc(val):0.9887
Ep: 7, Time: 159s, (20s test) loss:0.0232 err(val):0.0090 acc(val):0.9910
Ep: 8, Time: 161s, (20s test) loss:0.0177 err(val):0.0096 acc(val):0.9904
Ep: 9, Time: 165s, (25s test) loss:0.0151 err(val):0.0100 acc(val):0.9900
Ep: 10, Time: 159s, (20s test) loss:0.0156 err(val):0.0086 acc(val):0.9914
Ep: 11, Time: 160s, (20s test) loss:0.0096 err(val):0.0092 acc(val):0.9908
Ep: 12, Time: 160s, (20s test) loss:0.0080 err(val):0.0081 acc(val):0.9919
Ep: 13, Time: 163s, (23s test) loss:0.0088 err(val):0.0084 acc(val):0.9916
Ep: 14, Time: 160s, (20s test) loss:0.0065 err(val):0.0085 acc(val):0.9915
Ep: 15, Time: 162s, (21s test) loss:0.0048 err(val):0.0087 acc(val):0.9913
Ep: 16, Time: 160s, (20s test) loss:0.0033 err(val):0.0080 acc(val):0.9920
Ep: 17, Time: 158s, (20s test) loss:0.0047 err(val):0.0084 acc(val):0.9916
Ep: 18, Time: 159s, (20s test) loss:0.0027 err(val):0.0081 acc(val):0.9919
Ep: 19, Time: 165s, (25s test) loss:0.0053 err(val):0.0076 acc(val):0.9924
Ep: 20, Time: 160s, (20s test) loss:0.0068 err(val):0.0082 acc(val):0.9918
Ep: 21, Time: 160s, (20s test) loss:0.0039 err(val):0.0077 acc(val):0.9923
Ep: 22, Time: 160s, (20s test) loss:0.0041 err(val):0.0079 acc(val):0.9921
Ep: 23, Time: 158s, (19s test) loss:0.0029 err(val):0.0082 acc(val):0.9918
Ep: 24, Time: 158s, (20s test) loss:0.0023 err(val):0.0075 acc(val):0.9925
Ep: 25, Time: 160s, (20s test) loss:0.0024 err(val):0.0071 acc(val):0.9929
Ep: 26, Time: 159s, (20s test) loss:0.0018 err(val):0.0072 acc(val):0.9928
Ep: 27, Time: 159s, (20s test) loss:0.0012 err(val):0.0073 acc(val):0.9927
Ep: 28, Time: 159s, (20s test) loss:0.0014 err(val):0.0070 acc(val):0.9930
Ep: 29, Time: 158s, (20s test) loss:0.0024 err(val):0.0073 acc(val):0.9927
Ep: 30, Time: 160s, (20s test) loss:0.0014 err(val):0.0070 acc(val):0.9930
Ep: 31, Time: 160s, (20s test) loss:0.0027 err(val):0.0072 acc(val):0.9928
Ep: 32, Time: 159s, (20s test) loss:0.0021 err(val):0.0075 acc(val):0.9925
Ep: 33, Time: 160s, (20s test) loss:0.0009 err(val):0.0070 acc(val):0.9930
Ep: 34, Time: 160s, (20s test) loss:0.0011 err(val):0.0075 acc(val):0.9925
Ep: 35, Time: 159s, (20s test) loss:0.0015 err(val):0.0079 acc(val):0.9921
Ep: 36, Time: 161s, (21s test) loss:0.0016 err(val):0.0075 acc(val):0.9925
Ep: 37, Time: 161s, (20s test) loss:0.0016 err(val):0.0076 acc(val):0.9924
Ep: 38, Time: 162s, (20s test) loss:0.0016 err(val):0.0073 acc(val):0.9927
Ep: 39, Time: 161s, (20s test) loss:0.0013 err(val):0.0074 acc(val):0.9926
Ep: 40, Time: 162s, (21s test) loss:0.0007 err(val):0.0074 acc(val):0.9926
Final results on MNIST after 40.0000 epochs:
      Train-error: 0.0000 train-acc: 1.0000
 Validation-error: 0.0074   val-acc: 0.9926
       Test-error: 0.0080  test-acc: 0.9920
```
