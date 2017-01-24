# Check, if two hdf5 databases have same content.
from __future__ import print_function
import numpy as np
import h5py


def loadFile(filename):
    dd = {}
    with h5py.File(filename, 'r') as hf:
        print('List of arrays in this file: \n', list(hf.keys()))
        for ds in hf.keys():
            data = hf.get(ds)
            np_data = np.array(data)
            dd[ds] = np_data
            print('Shape of the array ', ds, ': ', np_data.shape)
            # print(len(data))
            # print("0:", data[0])
    return dd


def compareFiles(file1, file2):
    dd1 = loadFile(file1)
    dd2 = loadFile(file2)
    shapeok = True
    dataok = True
    if len(dd1) != len(dd2):
        print("Different number of datasets!")
        return False
    for ds in dd1:
        d1 = dd1[ds]
        if ds not in dd2:
            print("dataset {} is not in second database".format(ds))
            return False
        d2 = dd2[ds]
        if d1.shape != d2.shape:
            shapeok = False
        if np.array_equal(d1, d2):
            print("Arrays for {} equal.".format(ds))
        else:
            dataok = False
            pr = np.in1d(d1, d2)
            # print(pr)
            isperm = True
            dif = 0
            ndif = 0
            for p in pr:
                # print(p)
                if p != True:
                    isperm = False
                    dif += 1
                else:
                    ndif += 1
            if isperm:
                print("Arrays for {} are permutated, but otherwise equal.".format(ds))
            else:
                print("Arrays for {} NOT equal, {} differ, {} identical.".format(ds, dif, ndif))

            # return False
    if shapeok:
        print("Shapes are identical")
    return dataok


# if compareFiles("mnist.h5", "mnist.h5.old") is True:
if compareFiles("cifar10.h5", "cifar10.h5.old") is True:
    print("All ok.")
else:
    print("Files differ.")
