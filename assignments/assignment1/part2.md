
# Deep Learning VU - Assignment 1 - Part 2 #

The first assignment allows you to become familiar with basic dataset handling, image processing, and machine learning. Please read the [general information](https://owncloud.tuwien.ac.at/index.php/s/tejPFjV5uz39rBL) before you start.

This part builds upon [Assignment 1 - Part 1](https://owncloud.tuwien.ac.at/index.php/s/K6kta76H8hvIGoo), which must be finished first.

-----

As this is a new course, this text might not be without errors. If you find a significant error (not just typos but errors that affect the assignment), please contact us via [email](mailto:dlvc@caa.tuwien.ac.at). Students who find and report such errors will get extra points.

-----

Now that we have access to `TinyCifar10Dataset`, we can build and image classifier. We start out with the simplest example, a kNN classifier. Before we can do so, we must ensure that our datasets are in a format supported by the classifier, which expects that every sample is a feature vector of size `D`.

For this purpose, create a wrapper around an `ImageDataset` that reshapes each image to a `D`-vector. The wrapper should subclass the abstract class `FeatureVectorDataset`:

```python
class FeatureVectorDataset:
    # A dataset, consisting of multiple feature vectors
    # and corresponding class labels.

    def size(self):
        # Returns the size of the dataset (number of feature vectors).

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
        # Throws an error if the sample does not exist.
```

The `FeatureVectorDataset` behaves exactly like `ImageDataset` but the samples returned are feature vectors instead of images. Use a vector or matrix type provided by the chosen image library for representing feature vectors (see [general information](https://owncloud.tuwien.ac.at/index.php/s/tejPFjV5uz39rBL)).

The wrapper must have the following name and interface:

```python
class ImageVectorizer(FeatureVectorDataset):
    # Wraps an image dataset and exposes its contents.
    # Samples obtained using sample() are returned as 1D feature vectors.
    # Use devectorize() to convert a vector back to an image.

    def __init__(self, dataset):
        # Ctor. dataset is the dataset to wrap (type ImageDataset).

    def devectorize(self, fvec):
        # Convert a feature vector fvec obtained using sample()
        # back to an image and return the converted version.

    # implement the members of FeatureVectorDataset
```

`ImageVectorizer` is simple to implement as most methods can simply delegate work to the underlying `dataset`. Reshaping matrices and vectors is supported by all image libraries, you don't have to implement this functionality yourself.

-----

Create a script `test_image_vectorizer.py` that tests the `ImageVectorizer` class in conjunction with `TinyCifar10Dataset`. The script should initialize the vectorizer, obtain the 500th training sample (ID 499) as a vector, convert it back to an image, and save the result. Confirm that the saved image is correct, it must show a brown horse and a blue sky. The output of your script should be as below.

    4000 samples
    10 classes, name of class #1: automobile
    Sample #499: horse, shape: (3072,)
    Shape after devectorization: (32, 32, 3)

> The shape of feature vectors depends on the image library used and might be `3072` (1D, as above), `3072x1` or `1x3072`. All choices are fine.

-----

Implement a kNN classifier that supports the L1 and L2 distance measures, as discussed in the lecture. You are **not** allowed to use available implementations. The implementation should be simple, based on an exhaustive search for neighbors of a given sample (as covered in the lecture).

```python
class KnnClassifier:
    # k-nearest-neighbors classifier.

    def __init__(self, k, cmp):
        # Ctor. k is the number of nearest neighbors to search for,
        # and cmp is a string specifying the distance measure to
        # use, namely `l1` (L1 distance) or `l2` (L2 distance).

    def train(self, dataset):
        # Train on a dataset (type FeatureVectorDataset).

    def predict(self, fvec):
        # Return the predicted class label for a given feature vector fvec.
        # If the label is ambiguous, any of those in question is returned.
```

With `k > 1` the label might not be unique, e.g. if `k = 2` and the classes of the two nearest neighbors are 0 and 1, respectively. In this case, the classifier should return only a single label, i.e. 0 or 1.

-----

Create a script `knn_classify_tinycifar10.py` that finds optimal values in terms of accuracy for the hyperparameters `k` and `cmp` using the training and validation sets of `TinyCifar10Dataset`. You can choose between grid search and random search for finding good parameter combinations. You are **not** allowed to simply hard-code parameter combinations. You do not have to test all parameter combinations, but test at least 20. `k` should be varied within `[1,40]`.

Make sure that the script tests with `k=1` and `cmp='l2'` (you can hard-code this specific combination) in order to be able to verify that your code behaves correctly: the validation accuracy should be 25.2%.

The script should report every parameter combination tested and the corresponding validation accuracy. Finally, the script should report the test accuracy of the best parameter combination found. The output should be similar to the following:

    Performing random hyperparameter search ...
     [train] 4000 samples
     [val] 1000 samples
     k=01, cmp=l2, accuracy: 25.2%
     k=14, cmp=l2, accuracy: 25.7%
     k=37, cmp=l1, accuracy: 27.7%
     k=06, cmp=l2, accuracy: 26.0%
     k=29, cmp=l1, accuracy: 27.1%
     k=23, cmp=l1, accuracy: 27.3%
     k=13, cmp=l2, accuracy: 26.6%
     k=16, cmp=l1, accuracy: 26.9%
     k=22, cmp=l1, accuracy: 27.2%
     k=33, cmp=l2, accuracy: 25.4%
     k=24, cmp=l2, accuracy: 25.6%
     k=08, cmp=l2, accuracy: 25.5%
     k=35, cmp=l1, accuracy: 27.0%
     k=39, cmp=l1, accuracy: 27.5%
     k=28, cmp=l1, accuracy: 26.6%
     k=19, cmp=l1, accuracy: 28.1%
     k=18, cmp=l2, accuracy: 27.0%
     k=01, cmp=l1, accuracy: 27.4%
     k=04, cmp=l1, accuracy: 26.9%
     k=16, cmp=l1, accuracy: 26.9%
     k=22, cmp=l2, accuracy: 26.0%
     k=29, cmp=l1, accuracy: 27.1%
     k=34, cmp=l2, accuracy: 25.0%
     k=04, cmp=l2, accuracy: 24.7%
     k=02, cmp=l1, accuracy: 24.0%
     k=37, cmp=l1, accuracy: 27.7%
     k=06, cmp=l1, accuracy: 28.5%
     k=11, cmp=l2, accuracy: 26.7%
     k=39, cmp=l2, accuracy: 25.1%
     k=17, cmp=l2, accuracy: 26.2%
     k=20, cmp=l2, accuracy: 26.3%
    Testing best combination (6, l1) on test set ...
     [test] 1000 samples
     Accuracy: 30.4%
