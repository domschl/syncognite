# Helper scripts to download the original test-data

## MNIST: download_mnist.py
Downloads the original handwritten mnist digits from [Yan LeCun's website](http://yann.lecun.com/exdb/mnist/)
The data is transformed into a single [HDF5 database](https://support.hdfgroup.org/HDF5/) file.

## Requirements:
* HDF5 OS drivers (Arch linux: ```hdf5-cpp-fortran```, Mac homebrew: ```hdf5```)
* Python HDF5 support: [python-h5py](http://docs.h5py.org/en/latest/index.html), or ```pip install h5py```.

## Download required data:
```
python download_mnist.py
```

## Result:
```mnist.h5``` in directory ```datasets```.

Then use:
```
mnisttest <path-to-database>/mnist.h5
```
