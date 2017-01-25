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
gnuplot ../plots/liveplot.gnu
```
![after 6 episodes](../doc/images/mnist6.png)
Note: Currently, this works only after 1. episode is complete.

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
Compile-time options: FLOAT AVX FMA OPENMP
Eigen is using:      1 threads.
CpuPool is using:    8 threads.
Cpu+GpuPool is using:    0 threads.
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
Ep: 1, Time: 554s, (41s test) loss:0.377919 err(val):0.067300 acc(val):0.932700377919798
Ep: 2, Time: 548s, (41s test) loss:0.225277 err(val):0.040600 acc(val):0.959400225277823
Ep: 3, Time: 546s, (41s test) loss:0.155357 err(val):0.029700 acc(val):0.970300155357854
Ep: 4, Time: 549s, (41s test) loss:0.113462 err(val):0.025500 acc(val):0.974500113462917
Ep: 5, Time: 546s, (41s test) loss:0.104645 err(val):0.022500 acc(val):0.977500104645560
Ep: 6, Time: 544s, (41s test) loss:0.087063 err(val):0.020900 acc(val):0.979100087063650
Ep: 7, Time: 562s, (41s test) loss:0.069253 err(val):0.018900 acc(val):0.981100069253870
Ep: 8, Time: 569s, (43s test) loss:0.075011 err(val):0.018100 acc(val):0.981900075011740
Ep: 9, Time: 573s, (42s test) loss:0.060439 err(val):0.017000 acc(val):0.983000060439804
Ep: 10, Time: 567s, (41s test) loss:0.056876 err(val):0.016400 acc(val):0.983600056876576
Ep: 11, Time: 567s, (41s test) loss:0.049419 err(val):0.015700 acc(val):0.984300049419315
Ep: 12, Time: 569s, (41s test) loss:0.055270 err(val):0.015500 acc(val):0.984500055270021
Ep: 13, Time: 572s, (42s test) loss:0.057783 err(val):0.015400 acc(val):0.984600057783070
...
```
