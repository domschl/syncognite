# Helper scripts to download the original test-data for CIFAR10 and MNIST

## Requirements

* [HDF5](https://support.hdfgroup.org/HDF5/) OS drivers (Arch linux: ```hdf5-cpp-fortran```, Mac homebrew: ```hdf5```)
* Python HDF5 support: [python-h5py](http://docs.h5py.org/en/latest/index.html), or ```pip install h5py```.

## MNIST: download_mnist.py

Downloads the original handwritten mnist digits from [Yan LeCun's website](http://yann.lecun.com/exdb/mnist/).
The data is transformed into a single HDF5 database file and partioned into 50000 train, 10000 validation and 10000 test entries. (The original data is 60000-train, 10000-test with no seperate validation data.)

### Download required data

```bash
python download_mnist.py
```

### Result

```mnist.h5``` HDF5 database with handwritten digits and labels in directory ```datasets```.

Then use [mnisttest](../cpmnist/):

```bash
mnisttest <path-to-database>/mnist.h5
```

## CIFAR10: download_cifar10.py

Downloads the CIFAR10 image database from a site of the [University of Toronto](http://www.cs.toronto.edu/~kriz/cifar.html). The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The data is transformed into a single HDF5 database file.

### Download required data for cifar

```bash
python download_cifar10.py
```

### Result

```cifar10.h5``` HDF5 database with images and labels in directory ```datasets```.

Then use [cifar10test](../cpcifar10/):

```bash
cifar10test <path-to-database>/cifar10.h5
```

## RNNReader: texts

The recurrent text generator `rnnreader` is trained with an UTF-8 text file.

### Included: tiny_shakespeare.txt

Part of the repository is a subset of Shakespeare's collected works, taken
from Justin Johnson's [torch-rnn](https://github.com/jcjohnson/torch-rnn/blob/master/data/tiny-shakespeare.txt) repository.

To test text generation via RNNs, use [rnnreader](../rnnreader):

```bash
rnnreader <path-to-text>/tiny-shakespeare.txt
```
### The Complete Collected Works of William Shakespeare

Use the python script `download_shakespeare.py` to get the unabrivated version, about 5x larger:

```bash
pip install -U ml-indie-tools
python download_shakespeare.py
```

Results in `shakespeare.txt` with the complete works.

### Women writers

Use the python script `download_women_writers` to download a collection of about 20 books by
authors Emilie Brontë, Jane Austen, and Virginia Woolf from Project Gutenberg:

```bash
pip install -U ml-indie-tools
python download_women_writers.py
```

Note: have a look at the download script, it can be easily modified for other authors, subjects or collections.

The resulting file `women_writers.txt`, which contains all book texts (about 12MB) concatenated.

## References

This uses [`ml-indie-tools`](https://github.com/domschl/ml-indie-tools) to download the Complete Works from Project Gutenberg.
The library can be used to download arbitrary book-collections from Project Gutenberg, see [Documentation](https://github.com/domschl/ml-indie-tools#gutenberg_dataset).
