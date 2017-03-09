from __future__ import print_function
import numpy as np
import h5py
import tarfile
import os
import sys
import hashlib
try:
    # Python 3.x
    from urllib.request import urlopen
    from urllib.error import URLError
except:
    # Python 2.x
    from urllib2 import urlopen
    from urllib2 import URLError
try:
    # Python 3.x
    import pickle
except:
    # Python 2.x
    import cPickle as pickle


cifar_dict = {"url": 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
              "name": 'cifar-10-python.tar.gz',
              "size": 170498071,
              "md5": 'c58f30108f718f92721af3b95e74349a',
              "batchfolder": 'cifar-10-batches-py',
              "trainbatches": ['data_batch_1', 'data_batch_2', 'data_batch_3',
                               'data_batch_4', 'data_batch_5'],
              "testbatches": ['test_batch']}


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
            print(destfile, "local version is of wrong size, downloading...")
    print("The CIFAR-10 dataset.")
    print("See: http://www.cs.toronto.edu/~kriz/cifar.html")
    print("Downloading", destfile, "from", cfdict["url"])
    try:
        dataresp = urlopen(cfdict["url"])
        data = dataresp.read()
    except URLError as e:
        print("Failed to download", destfile, e)
        return False
    if hashlib.md5(data).hexdigest() != cfdict["md5"]:
        print("Bad MD5 hash!")
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


def uncompressOriginal(filename, destpath, batchpath):
    if os.path.exists(batchpath):
        print("Destination {} exists, not expanding again.".format(destpath))
        return True
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


def loadPickle(filename):
    if os.path.exists(filename) is not True:
        print("File {} does not exist, can't unpickle".format(filename))
        return False
    try:
        f = open(filename, 'rb')
        if sys.version_info < (3, 0, 0):
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='latin1')
        f.close()
        return (data['data'], data['labels'])
    except:
        print("Failed to unpickle {}".format(filename))
        return None


def loadBatches(cfdict, entry):
    data = []
    labels = []
    for ba in cfdict[entry]:
        filename = os.path.join(batchpath, ba)
        dc = loadPickle(filename)
        if dc is None:
            return None
        data.append(dc[0])
        labels.append(dc[1])
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    length = len(labels)
    X, y = data.reshape(length, 3, 32, 32), labels
    return (X, y)


def encodeCifar(cfdict, h5path, batchpath):
    destname = os.path.join(h5path, "cifar10.h5")
    if os.path.exists(destname) is True:
        print(destname, "already exists, not overwriting.")
        return True
    da = loadBatches(cfdict, "trainbatches")
    if da is None:
        return False
    X, y = da
    da = loadBatches(cfdict, "testbatches")
    if da is None:
        return False
    Xt, yt = da

    hf = h5py.File(destname, 'w')

    hf.create_dataset('train-data', data=X, compression='gzip')
    hf.create_dataset('train-labels', data=y.astype(np.int32),
                      compression='gzip')
    hf.create_dataset('test-data', data=Xt, compression='gzip')
    hf.create_dataset('test-labels', data=yt.astype(np.int32),
                      compression='gzip')
    hf.close()

    return True


localpath = os.path.dirname(os.path.realpath(__file__))
destpath = os.path.join(localpath, "originals")
batchpath = os.path.join(destpath, cifar_dict["batchfolder"])
if loadOriginalFile(cifar_dict,destpath) is False:
    print("Download failed.")
else:
    print('CIFAR file at:', cifar_dict["filename"])
    if (uncompressOriginal(cifar_dict["filename"], destpath, batchpath)):
        print("Archive expansion completed")
        if encodeCifar(cifar_dict, localpath, batchpath) is False:
            print("Failed to create H5 database.")
        else:
            print("Database file is now in:", localpath)
    else:
        print("Archive expansion failed.")
