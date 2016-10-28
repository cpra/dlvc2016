
# Deep Learning VU - Assignment 1 - Part 3 #

The first assignment allows you to become familiar with basic dataset handling, image processing, and machine learning. Please read the [general information](https://github.com/cpra/dlvc2016/blob/master/assignments/general.md) before you start.

This part builds upon [Assignment 1 - Part 2](https://github.com/cpra/dlvc2016/blob/master/assignments/assignment1/part2.md), which must be finished first.

-----

As this is a new course, this text might not be without errors. If you find a significant error (not just typos but errors that affect the assignment), please contact us via [email](mailto:dlvc@caa.tuwien.ac.at). Students who find and report such errors will get extra points.

-----

In Part 2 you've seen that the kNN classifier performs poorly on image data. This is mainly because it is unable to find discriminative features on its own. The first task is to confirm experimentally that this is the case by validating the classifier on features extracted from images instead of these images themselves.

For this purpose, I've extracted the popular [HOG features](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) from all images in the `TinyCifar10Dataset`. You can download these features here ([train](https://owncloud.tuwien.ac.at/index.php/s/rrgQdzce0Uoavso), [validation](https://owncloud.tuwien.ac.at/index.php/s/VqNqjQWGbIqKXpp), [test](https://owncloud.tuwien.ac.at/index.php/s/cueltRAaYSkr1MN)). The files are in HDF5 format, a database format that is supported by all languages ([Python](http://www.h5py.org/), [Matlab](https://de.mathworks.com/help/matlab/high-level-functions.html), [Java](https://support.hdfgroup.org/HDF5/examples/api-java.html), [Lua](https://github.com/deepmind/torch-hdf5)). Each file contains two databases, `features` and `labels`. The former is a matrix of size `n*144`, with `n` being the number of images in the original dataset. The `i`th row in this matrix corresponds to the extracted HOG features of the `i`th sample in the original `TinyCifar10Dataset`. `labels` is a `n`-vector of labels for every sample.

Write a class that loads such a file and exposes the contents as a `FeatureVectorDataset`:

```python
class HDF5FeatureVectorDataset(FeatureVectorDataset):
    # A dataset stored in a HDF5 file including the datasets
    # features (n*f matrix) and labels (n-vector).

    def __init__(self, fpath, class_names):
        # Ctor. fpath is a path to the HDF5 file.
        # class_names is a mapping from labels to
        # class names, for every label.

    # implement the members of FeatureVectorDataset
```

The HDF5 files provided store all information required to implement all members of `FeatureVectorDataset`, with the exception of `classname()`. For this reason, the constructor expects the client to provide this information via `class_names`. This can be any suitable datastructure, for instance a dictionary.

By using `HDF5FeatureVectorDataset` with the provided files, you are able to obtain a HOG-feature version of `TinyCifar10Dataset` like so:

```python
cifar10_classnames = { 0: 'airplane', 1: 'automobile', ... }  # must match TinyCifar10Dataset
train = HDF5FeatureVectorDataset('features_tinycifar10_train.h5', cifar10_classnames)
val = HDF5FeatureVectorDataset('features_tinycifar10_val.h5', cifar10_classnames)
```

-----

Copy the `knn_classify_tinycifar10.py` script from Part 2 and rename the copy to `knn_classify_hog_tinycifar10.py`. Adapt the code so that it performs hyperparameter optimization using the HOG-feature versions of `TinyCifar10Dataset` and finally tests the best hyperparameter combination on the test set, just like before. Example output:

    Performing hyperparameter search ...
     [train] 4000 samples
     [val] 1000 samples
     k=01, cmp=l2, accuracy: 30.8%
     k=07, cmp=l2, accuracy: 33.9%
     k=14, cmp=l2, accuracy: 33.9%
     k=24, cmp=l1, accuracy: 39.1%
     k=05, cmp=l1, accuracy: 38.1%
     k=23, cmp=l1, accuracy: 40.5%
     k=09, cmp=l1, accuracy: 38.5%
     k=12, cmp=l2, accuracy: 34.2%
     k=18, cmp=l1, accuracy: 40.1%
     k=21, cmp=l1, accuracy: 40.7%
     k=19, cmp=l1, accuracy: 40.1%
     k=25, cmp=l1, accuracy: 40.1%
     k=16, cmp=l1, accuracy: 40.2%
     k=21, cmp=l1, accuracy: 40.7%
     k=33, cmp=l1, accuracy: 38.8%
     k=36, cmp=l1, accuracy: 38.5%
     k=29, cmp=l2, accuracy: 32.3%
     k=29, cmp=l1, accuracy: 39.5%
     k=12, cmp=l1, accuracy: 39.5%
     k=31, cmp=l2, accuracy: 32.1%
     k=12, cmp=l1, accuracy: 39.5%
     k=03, cmp=l1, accuracy: 36.8%
     k=38, cmp=l2, accuracy: 32.9%
     k=08, cmp=l2, accuracy: 33.7%
     k=23, cmp=l1, accuracy: 40.5%
     k=08, cmp=l1, accuracy: 37.4%
     k=27, cmp=l1, accuracy: 39.5%
     k=16, cmp=l2, accuracy: 33.7%
     k=34, cmp=l2, accuracy: 33.2%
     k=14, cmp=l2, accuracy: 33.9%
     k=06, cmp=l1, accuracy: 37.1%
    Testing best combination (21, l1) on test set ...
     [test] 1000 samples
     Accuracy: 40.1%

The results show that adding a feature extraction step leads to significant performance improvements (about 10% in the example case, which is a relative improvement of about 30%).

-----

Write a report that summarizes assignment 1. See [general information](https://github.com/cpra/dlvc2016/blob/master/assignments/general.md) for requirements. The overall structure and contents of the report must be as follows:

1. **Image Classification.** Describe the image classification problem.
2. **The CIFAR10 Dataset.** Describe the CIFAR10 dataset.
3. **Training, Validation, and Test Sets.** Describe the purpose of these sets, why they are required, and how you obtained a validation set in case of CIFAR10.
4. **kNN Classifiers.** Describe how the kNN classifier works and how it can be used with images, which are not vectors. Explain what hyperparameters are in general and in case of kNN. How does parameter `k` generally influence the results?
5. **kNN Classification of CIFAR10.** Introduce the tiny version of CIFAR10. Describe what hyperparameter optimization is and why it is important. Explain your search strategy and visualize the results based on the output of `knn_classify_tinycifar10.py`.
6. **The Importance of Features.** Think about reasons why performing kNN classification directly on the raw images does not work well. Describe what a feature is and why operating on features instead of raw images is beneficial in case of kNN. Briefly explain what HOG features are. Compare the results when using these features (`knn_classify_hog_tinycifar10.py`) to those obtained using raw images and discuss them. Even with HOG features, the performance is still much lower than that of CNNs (90% accuracy and more on the whole dataset). Think of reasons for why this is the case.

The report should be 3 to 5 pages long.

-----

[Submit](https://github.com/cpra/dlvc2016/blob/master/assignments/general.md) assignment 1, including all three parts. The **deadline is Sun, 6.11., 23:00**.
