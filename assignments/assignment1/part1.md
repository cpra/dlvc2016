
# Deep Learning VU - Assignment 1 - Part 1 #

The first assignment allows you to become familiar with basic dataset handling, image processing, and machine learning. Please read the [general information](https://owncloud.tuwien.ac.at/index.php/s/tejPFjV5uz39rBL) before you start.

-----

As this is a new course, this text might not be without errors. If you find a significant error (not just typos but errors that affect the assignment), please contact us via [email](mailto:dlvc@caa.tuwien.ac.at). Students who find and report such errors will get extra points.

-----

[Download](https://www.cs.toronto.edu/~kriz/cifar.html) the CIFAR10 dataset. Make sure to read the website carefully to understand the dataset structure. There are different versions available on the website, choose depending on the language you selected.

Alternatively, you can convert the dataset to HDF5, which is easy to use and there are language bindings to all languages allowed (for instance, you could write a Python script to convert the Python version of the dataset to HDF5 and then use it in Java). If you do so, you must generate two HDF5 files `train.h5` and `test.h5` that correspond to the training and test set, respectively. Other versions are not allowed.

-----

Write an abstract class that represents an image dataset for classification purposes:

```python
class ImageDataset:
    # A dataset, consisting of multiple samples/images
    # and corresponding class labels.

    def size(self):
        # Returns the size of the dataset (number of images).

    def nclasses(self):
        # Returns the number of different classes.
        # Class labels start with 0 and are consecutive.

    def classname(self, cid):
        # Returns the name of a class as a string.

    def sample(self, sid):
        # Returns the sid-th sample in the dataset, and the
        # corresponding class label. Depending of your language,
        # this can be a Matlab struct, Python tuple or dict, etc.
        # Sample IDs start with 0 and are consecutive.
        # The channel order of samples must be RGB.
        # Throws an error if the sample does not exist.
```

-----

Then wrap the CIFAR10 dataset. This dataset is already split into a training and a test set. We will use a part of the training set for evaluation purposes, namely the **last 20%** of samples of each class.

```python
class Cifar10Dataset(ImageDataset):
    # The CIFAR10 dataset.

    def __init__(self, fdir, split):
        # Ctor. fdir is a path to a directory in which the CIFAR10
        # files reside (e.g. data_batch_1 and test_batch).
        # split is a string that specifies which dataset split to load.
        # Can be 'train' (training set), 'val' (validation set) or 'test' (test set).

    # implement the other members ...
```

> Python users: If you use Python3, you will need to use `latin1` encoding when unpickling the files, e.g. `pickle.load(f, encoding='latin1')`.

Make sure you understand how to load and split the dataset: In order to preserve the sample order (as far as possible), the files must be read in order (with the Python version, this means loading `data_batch_1`, then `data_batch_2`, and so on). For the same purpose, the samples must be processed sequentially when creating the training and validation sets. The following toy example demonstrates how your code should behave. Assume there is a dataset with 10 samples of 2 classes:

    Sample  Class  ID
    A       0      0
    B       0      1
    C       1      2
    D       0      3
    E       1      4
    F       1      5
    G       1      6
    H       0      7
    I       0      8
    J       1      9

Then the training set (first 80% of samples per class) should be as follows:

    Sample  Class  ID
    A       0      0
    B       0      1
    C       1      2
    D       0      3
    E       1      4
    F       1      5
    G       1      6
    H       0      7

And the validation set (last 20% of samples per class) should be as follows:

    Sample  Class  ID
    I       0      0
    J       1      1

-----

For now we will use only a small subset of CIFAR10 in order to speed up training and testing. This subset should expose only the **first 10%** of samples for every class, i.e. the training set has 4k images, and the validation and test sets have 1k images each. The source dataset is the corresponding subset of `Cifar10Dataset`, i.e. the validation set of `TinyCifar10Dataset` must be the first 10% of samples per class of `Cifar10Dataset`s validation set. The same applies for the other subsets.

```python
class TinyCifar10Dataset(ImageDataset):
    # Subset of CIFAR10 that exposes only 10% of samples.

    def __init__(self, fdir, split):
        # See Cifar10Dataset
```

> Note that the "load the first n% of samples per class" functionality is required twice, for loading the training set of `Cifar10Dataset` and for loading 10% subsets for `TinyCifar10Dataset`. Encapsulating this functionality and reusing it, as well as utilizing `Cifar10Dataset` from `TinyCifar10Dataset` might save you some work, but this is not required.

-----

Create a script `test_cifar10_dataset.py` that tests the `Cifar10Dataset` class and produces an output similar to the following. Make sure that your script behaves exactly as follows -- if not, you have a bug somewhere. The formating does not have to be identical, but it should be similar.

    [train] 10 classes, name of class #1: automobile
    [val] 10 classes, name of class #1: automobile
    [test] 10 classes, name of class #1: automobile

    [train] 40000 samples
     Class #0: 4000 samples
     Class #1: 4000 samples
     Class #2: 4000 samples
     Class #3: 4000 samples
     Class #4: 4000 samples
     Class #5: 4000 samples
     Class #6: 4000 samples
     Class #7: 4000 samples
     Class #8: 4000 samples
     Class #9: 4000 samples

    [val] 10000 samples
     Class #0: 1000 samples
     Class #1: 1000 samples
     Class #2: 1000 samples
     Class #3: 1000 samples
     Class #4: 1000 samples
     Class #5: 1000 samples
     Class #6: 1000 samples
     Class #7: 1000 samples
     Class #8: 1000 samples
     Class #9: 1000 samples

    [test] 10000 samples
     Class #0: 1000 samples
     Class #1: 1000 samples
     Class #2: 1000 samples
     Class #3: 1000 samples
     Class #4: 1000 samples
     Class #5: 1000 samples
     Class #6: 1000 samples
     Class #7: 1000 samples
     Class #8: 1000 samples
     Class #9: 1000 samples

    [train] Sample #499: horse
    [val] Sample #499: deer
    [test] Sample #499: airplane

The script should also save the 500th sample (#499) of each subset to disk, to the directory in which the script resides. The 500th training image shows a brown horse, the 500th validation image shows a deer in front of a street (?), the 500th test image shows an airplane with a triangular shape with gray background. Again, if this is not the case, there is a problem in your code.

-----

Create another script `test_tinycifar10_dataset.py` that does the same for `TinyCifar10Dataset`. This is as simple as copying `test_cifar10_dataset.py` and replacing the class name during initialization. Make sure that your script produces the following output:

    [train] 10 classes, name of class #1: automobile
    [val] 10 classes, name of class #1: automobile
    [test] 10 classes, name of class #1: automobile

    [train] 4000 samples
     Class #0: 400 samples
     Class #1: 400 samples
     Class #2: 400 samples
     Class #3: 400 samples
     Class #4: 400 samples
     Class #5: 400 samples
     Class #6: 400 samples
     Class #7: 400 samples
     Class #8: 400 samples
     Class #9: 400 samples

    [val] 1000 samples
     Class #0: 100 samples
     Class #1: 100 samples
     Class #2: 100 samples
     Class #3: 100 samples
     Class #4: 100 samples
     Class #5: 100 samples
     Class #6: 100 samples
     Class #7: 100 samples
     Class #8: 100 samples
     Class #9: 100 samples

    [test] 1000 samples
     Class #0: 100 samples
     Class #1: 100 samples
     Class #2: 100 samples
     Class #3: 100 samples
     Class #4: 100 samples
     Class #5: 100 samples
     Class #6: 100 samples
     Class #7: 100 samples
     Class #8: 100 samples
     Class #9: 100 samples

    [train] Sample #499: horse
    [val] Sample #499: deer
    [test] Sample #499: airplane

The 500th samples per subset must be the same as before.
