#!/usr/bin/python
import numpy as np
import h5py
import gzip
import os
try:
    # Python 3.x
    from urllib.request import urlopen
    from urllib.error import URLError
except:
    # Python 2.x
    from urllib2 import urlopen
    from urllib2 import URLError

# see: http://yann.lecun.com/exdb/mnist/
mnist_urls = \
    [{"name": "train-images-idx3-ubyte.gz",
      "size": 9912422,
      "url": 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
      "offset": 16,
      "entrysize": 784,
      "format": "float",
      "norm": 256.0,
      "datasets": {"x_train": 50000, "x_valid": 10000}},
     {"name": "train-labels-idx1-ubyte.gz",
      "size": 28881,
      "url": 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
      "offset": 8,
      "entrysize": 1,
      "format": "int",
      "datasets": {"t_train": 50000, "t_valid": 10000}},
     {"name": "t10k-images-idx3-ubyte.gz",
      "size": 1648877,
      "url": 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
      "offset": 16,
      "entrysize": 784,
      "format": "float",
      "norm": 256.0,
      "datasets": {"x_test": 10000}},
     {"name": "t10k-labels-idx1-ubyte.gz",
      "size": 4542,
      "url": 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
      "offset": 8,
      "entrysize": 1,
      "format": "int",
      "datasets": {"t_test": 10000}}]


def loadOriginalFiles(datadict, destpath):
    firstDl = True
    for du in datadict:
        if not os.path.exists(destpath):
            print("Creating directory:", destpath)
            os.makedirs(destpath)
        destfile = os.path.join(destpath, du["name"])
        if os.path.isfile(destfile) is True:
            if os.stat(destfile).st_size == du["size"]:
                print(destfile, "is already in local storage.")
                continue
            else:
                print(destfile, "local version is of wrong size, trying to re-download.")
        if firstDl is True:
            firstDl = False
            print("Downloading original MNIST files from: http://yann.lecun.com/exdb/mnist/")
            print("Reference: [LeCun et al., 1998a]")
            print("   Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. 'Gradient-based learning")
            print("   applied to document recognition.' Proceedings of the IEEE, 86(11):2278-2324,")
            print("   November 1998.")
        print("Downloading", destfile, "from", du["url"])
        try:
            dataresp = urlopen(du["url"])
            data = dataresp.read()
        except URLError as e:
            print("Failed to download", destfile, e)
            return False
        f = open(destfile, "wb")
        f.write(data)
        f.close()
        if os.path.isfile(destfile) is False:
            print("Error: failed to write local file", destfile)
            return False
        if os.stat(destfile).st_size != du["size"]:
            print("Error:", destfile, " downloaded, unexpected size error!")
            return False
    return True


def encodeOriginalsToH5(datadict, zippath, h5path):
    destname = os.path.join(h5path, "mnist.h5")
    if os.path.exists(destname) is True:
        print(destname, "already exists, not overwriting.")
        return True
    hf = h5py.File(destname, 'w')
    for du in datadict:
        sourcefile = os.path.join(zippath, du["name"])
        f = gzip.open(sourcefile, 'rb')
        data = f.read()
        f.close()
        ex = du["offset"]
        for ds in du["datasets"]:
            ex += du["datasets"][ds] * du["entrysize"]
        if len(data) != ex:
            print(sourcefile, len(data), "expected: ", ex, "BAD data format!")
            return False
        offs = du["offset"]
        for ds in du["datasets"]:
            print("Creating HDF5 dataset", ds)
            n = du["datasets"][ds]
            w = du["entrysize"]
            shape = (n, w)
            if du["format"] == "int":
                di = np.zeros(shape, dtype='int32')
                dtype = 'i'
                for y in range(n):
                    for x in range(w):
                        di[y, x] = data[offs+y*w+x]
            if du["format"] == "float":
                di = np.zeros(shape, dtype='float')
                dtype = 'f'
                for y in range(n):
                    for x in range(w):
                        di[y, x] = float(data[offs+y*w+x]) / du["norm"]
            offs += du["datasets"][ds] * du["entrysize"]
            print(ds, shape, dtype)
            dset = hf.create_dataset(ds, shape, dtype=dtype,
                                     compression='gzip')
            dset[...] = di
    hf.close()
    return True


def checkFile():
    with h5py.File('mnist.hdf5', 'r') as hf:
        print('List of arrays in this file: \n', list(hf.keys()))
        for ds in hf.keys():
            data = hf.get(ds)
            np_data = np.array(data)
            print('Shape of the array ', ds, ': \n', np_data.shape)
            print(len(data))
            print("0:", data[0])


localpath = os.path.dirname(os.path.realpath(__file__))
destpath = os.path.join(localpath, "originals")
if loadOriginalFiles(mnist_urls, destpath) is False:
    print("Download failed.")
else:
    if encodeOriginalsToH5(mnist_urls, destpath, localpath) is True:
        print("database file is now in:", destpath)
    else:
        print("Error during HDF5 file creation")
