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

Note: `gnuplot` can only show graphs after 1. episode is complete.

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
Ep: 1, Time: 155s, (16s test) loss:0.1003 err(val):0.0197 acc(val):0.9803
Ep: 2, Time: 155s, (17s test) loss:0.0697 err(val):0.0164 acc(val):0.9836
Ep: 3, Time: 157s, (17s test) loss:0.0522 err(val):0.0130 acc(val):0.9870
Ep: 4, Time: 155s, (16s test) loss:0.0395 err(val):0.0117 acc(val):0.9883
Ep: 5, Time: 156s, (17s test) loss:0.0335 err(val):0.0110 acc(val):0.9890
Ep: 6, Time: 154s, (16s test) loss:0.0234 err(val):0.0085 acc(val):0.9915
Ep: 7, Time: 156s, (17s test) loss:0.0255 err(val):0.0095 acc(val):0.9905
Ep: 8, Time: 156s, (17s test) loss:0.0158 err(val):0.0095 acc(val):0.9905
Ep: 9, Time: 195s, (18s test) loss:0.0160 err(val):0.0096 acc(val):0.9904
Ep: 10, Time: 162s, (17s test) loss:0.0176 err(val):0.0076 acc(val):0.9924
Ep: 11, Time: 159s, (18s test) loss:0.0104 err(val):0.0088 acc(val):0.9912
Ep: 12, Time: 168s, (17s test) loss:0.0095 err(val):0.0080 acc(val):0.9920
Ep: 13, Time: 160s, (17s test) loss:0.0089 err(val):0.0068 acc(val):0.9932
Ep: 14, Time: 156s, (16s test) loss:0.0122 err(val):0.0078 acc(val):0.9922
Ep: 15, Time: 156s, (17s test) loss:0.0058 err(val):0.0079 acc(val):0.9921
Ep: 16, Time: 158s, (16s test) loss:0.0083 err(val):0.0074 acc(val):0.9926
Ep: 17, Time: 158s, (17s test) loss:0.0041 err(val):0.0076 acc(val):0.9924
Ep: 18, Time: 158s, (16s test) loss:0.0032 err(val):0.0069 acc(val):0.9931
Ep: 19, Time: 158s, (17s test) loss:0.0046 err(val):0.0075 acc(val):0.9925
Ep: 20, Time: 156s, (16s test) loss:0.0050 err(val):0.0075 acc(val):0.9925
Ep: 21, Time: 158s, (17s test) loss:0.0035 err(val):0.0063 acc(val):0.9937
Ep: 22, Time: 159s, (16s test) loss:0.0051 err(val):0.0064 acc(val):0.9936
Ep: 23, Time: 156s, (17s test) loss:0.0022 err(val):0.0066 acc(val):0.9934
Ep: 24, Time: 165s, (17s test) loss:0.0034 err(val):0.0064 acc(val):0.9936
Ep: 25, Time: 164s, (16s test) loss:0.0030 err(val):0.0057 acc(val):0.9943
Ep: 26, Time: 160s, (16s test) loss:0.0025 err(val):0.0066 acc(val):0.9934
Ep: 27, Time: 160s, (16s test) loss:0.0024 err(val):0.0059 acc(val):0.9941
Ep: 28, Time: 156s, (16s test) loss:0.0024 err(val):0.0060 acc(val):0.9940
Ep: 29, Time: 156s, (16s test) loss:0.0038 err(val):0.0059 acc(val):0.9941
Ep: 30, Time: 155s, (16s test) loss:0.0024 err(val):0.0061 acc(val):0.9939
Ep: 31, Time: 156s, (16s test) loss:0.0011 err(val):0.0060 acc(val):0.9940
Ep: 32, Time: 155s, (17s test) loss:0.0025 err(val):0.0055 acc(val):0.9945
Ep: 33, Time: 155s, (17s test) loss:0.0006 err(val):0.0054 acc(val):0.9946
Ep: 34, Time: 155s, (16s test) loss:0.0015 err(val):0.0060 acc(val):0.9940
Ep: 35, Time: 156s, (17s test) loss:0.0008 err(val):0.0055 acc(val):0.9945
Ep: 36, Time: 156s, (16s test) loss:0.0010 err(val):0.0056 acc(val):0.9944
Ep: 37, Time: 155s, (16s test) loss:0.0007 err(val):0.0063 acc(val):0.9937
Ep: 38, Time: 155s, (16s test) loss:0.0008 err(val):0.0059 acc(val):0.9941
Ep: 39, Time: 155s, (16s test) loss:0.0010 err(val):0.0064 acc(val):0.9936
Ep: 40, Time: 155s, (16s test) loss:0.0005 err(val):0.0059 acc(val):0.9941
Final results on MNIST after 40.0000 epochs:
      Train-error: 0.0001 train-acc: 0.9999
 Validation-error: 0.0059   val-acc: 0.9941
       Test-error: 0.0066  test-acc: 0.9934
```
