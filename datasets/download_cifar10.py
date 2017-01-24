from __future__ import print_function
import numpy as np
import h5py
import tarfile
import os
try:
    # Python 3.x
    from urllib.request import urlopen
    from urllib.error import URLError
except:
    # Python 2.x
    from urllib2 import urlopen
    from urllib2 import URLError

cifar_dict = {"url": 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
              "name": 'cifar-10-python.tar.gz',
              "size": 170498071}

def loadOriginalFile(cfdict, destpath):
    if not os.path.exists(destpath):
        print("Creating directory:", destpath)
        os.makedirs(destpath)
    destfile = os.path.join(destpath, cfdict["name"])
    cfdict["filename"] = destfile
    if os.path.isfile(destfile) is True:
        if os.stat(destfile).st_size == cfdict["size"]:
            print(destfile, "is already in local storage.")
            return True
        else:
            print(destfile, "local version is of wrong size, trying to re-download.")
    print("The CIFAR-10 dataset.")
    print("See: http://www.cs.toronto.edu/~kriz/cifar.html")
    print("Downloading", destfile, "from", cfdict["url"])
    try:
        dataresp = urlopen(cfdict["url"])
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
    if os.stat(destfile).st_size != cfdict["size"]:
        print("Error:", destfile, " downloaded, unexpected size error!")
        return False
    cfdict["filename"] = destfile
    return True


def uncompressOriginal(filename, destpath):
    if (filename.endswith("tar.gz")):
        try:
            tar = tarfile.open(filename, "r:gz")
            tar.extractall(path=destpath)
            tar.close()
            return True
        except:
            print("Extraction failed")
            return False
    else:
        print("Unexpected filename extension, need 'tar.gz'")
        return False


'''
def encodeOriginalsToH5(datadict, zippath, h5path):
    destname = os.path.join(h5path, "mnist.h5")
    if os.path.exists(destname) is True:
        print(destname, "already exists, not overwriting.")
        return True
    hf = h5py.File(destname, 'w')
    for du in datadict:
        sourcefile = os.path.join(zippath, du["name"])
        f = gzip.open(sourcefile, 'rb')
        data = bytearray(f.read())
        f.close()
        ex = du["offset"]
        for ds in du["datasets"]:
            ex += du["datasets"][ds] * du["entrysize"]
        if len(data) != ex:
            print(sourcefile, len(data), "expected: ", ex, "BAD data format!")
            return False
        offs = du["offset"]
        for ds in du["datasets"]:
            print("Creating HDF5 dataset", ds, "...")
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
            # print(ds, shape, dtype)
            dset = hf.create_dataset(ds, shape, dtype=dtype,
                                     compression='gzip')
            dset[...] = di
    hf.close()
    return True
'''


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
if loadOriginalFile(cifar_dict, destpath) is False:
    print("Download failed.")
else:
    print('Downloaded to:', cifar_dict["filename"])
    if (uncompressOriginal(cifar_dict["filename"], destpath)):
        print("Archive expansion completed")
    else:
        print("Archive expansion failed.")
'''
else:
    if encodeOriginalsToH5(mnist_urls, destpath, localpath) is True:
        print("database file is now in:", localpath)
    else:
        if
        print("Error during HDF5 file creation")
'''
