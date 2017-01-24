# Training with CIFAR10 images
## Requirements
Check the [main page](../README.md) for build requirements.
## Build
After configuration of CMake, the cifar10 part can be built directly by:
```
make cifar10test
```
## Dataset
Use the [script download_cifar10.py](../datasets/README.md) to download the CIFAR10 training data in HDF5 format.

## Training
From build directory:
```
cpcifar10/cifar10test ../datasets/cifar10.h5
```
## Logging
For live logging of the training progress, use gnuplot:
```
gnuplot ../plots/liveplot.gnu
```
![after 7 episodes](../doc/images/cifar10-7.png)
Note: Currently, this works only after 1. episode is complete.

## output
```
$ cpcifar10/cifar10test ../datasets/cifar10.h5
Reading: test-data dataset rank = 4, dimensions; 10000 x 3 x 32 x 32 int
test-labels dataset rank = 1, dimensions; 10000 int
train-data dataset rank = 4, dimensions; 50000 x 3 x 32 x 32 int
train-labels dataset rank = 1, dimensions; 50000 int

test-data (10000, 3072)
test-labels (10000, 1)
train-data (50000, 3072)
train-labels (50000, 1)
test-data tensor-4
train-data tensor-4
Compile-time options: FLOAT AVX FMA OPENMP
Eigen is using:      1 threads.
CpuPool is using:    8 threads.
Cpu+GpuPool is using:    0 threads.
Checking LayerBlock topology...
cv1: (3, 32, 32)[3072] -> (64, 32, 32)[65536]
sb1: (64, 32, 32)[65536] -> (64, 32, 32)[65536]
rl1: (64, 32, 32)[65536] -> (64, 32, 32)[65536]
doc1: (64, 32, 32)[65536] -> (64, 32, 32)[65536]
cv2: (64, 32, 32)[65536] -> (64, 32, 32)[65536]
rl2: (64, 32, 32)[65536] -> (64, 32, 32)[65536]
cv3: (64, 32, 32)[65536] -> (128, 16, 16)[32768]
sb2: (128, 16, 16)[32768] -> (128, 16, 16)[32768]
rl3: (128, 16, 16)[32768] -> (128, 16, 16)[32768]
doc2: (128, 16, 16)[32768] -> (128, 16, 16)[32768]
cv4: (128, 16, 16)[32768] -> (128, 16, 16)[32768]
rl4: (128, 16, 16)[32768] -> (128, 16, 16)[32768]
cv5: (128, 16, 16)[32768] -> (256, 8, 8)[16384]
sb3: (256, 8, 8)[16384] -> (256, 8, 8)[16384]
rl5: (256, 8, 8)[16384] -> (256, 8, 8)[16384]
doc3: (256, 8, 8)[16384] -> (256, 8, 8)[16384]
cv6: (256, 8, 8)[16384] -> (256, 8, 8)[16384]
rl6: (256, 8, 8)[16384] -> (256, 8, 8)[16384]
doc4: (256, 8, 8)[16384] -> (256, 8, 8)[16384]
cv7: (256, 8, 8)[16384] -> (512, 4, 4)[8192]
rl7: (512, 4, 4)[8192] -> (512, 4, 4)[8192]
doc5: (512, 4, 4)[8192] -> (512, 4, 4)[8192]
cv8: (512, 4, 4)[8192] -> (512, 4, 4)[8192]
rl8: (512, 4, 4)[8192] -> (512, 4, 4)[8192]
af1: (512, 4, 4)[8192] -> (1024)[1024]
bn1: (1024)[1024] -> (1024)[1024]
rla1: (1024)[1024] -> (1024)[1024]
do1: (1024)[1024] -> (1024)[1024]
af2: (1024)[1024] -> (512)[512]
bn2: (512)[512] -> (512)[512]
rla2: (512)[512] -> (512)[512]
do2: (512)[512] -> (512)[512]
af3: (512)[512] -> (10)[10]
sm1: (10)[10] -> (1)[1]
Topology-check for LayerBLock: ok.
Training net: data-size: 49000, chunks: 980, batch_size: 50, threads: 8 (bz*ch): 49000
Ep: 1, Time: 1369s, (13s test) loss:1.773042 err(val):0.609000 acc(val):0.391000.773101453
Ep: 2, Time: 1361s, (14s test) loss:1.597675 err(val):0.516000 acc(val):0.484000.603656972
Ep: 3, Time: 1362s, (12s test) loss:1.433608 err(val):0.479000 acc(val):0.521000.436751770
Ep: 4, Time: 1359s, (12s test) loss:1.359592 err(val):0.457000 acc(val):0.543000.369401933
Ep: 5, Time: 1343s, (12s test) loss:1.286471 err(val):0.436000 acc(val):0.564000.273972529
Ep: 6, Time: 1346s, (13s test) loss:1.223408 err(val):0.424000 acc(val):0.576000.225793357
Ep: 7, Time: 1356s, (12s test) loss:1.178642 err(val):0.397000 acc(val):0.603000.177728710
At:   39% of epoch 8, 27.831942 ms/data, ett: 1363s, eta: -830s, loss: 1.106684, 1.1311514

...
```
