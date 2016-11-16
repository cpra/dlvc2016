
# Deep Learning VU - Assignment 2 - Part 2 #

In the second assignment, you will learn how to use Convolutional Neural Networks (CNNs) for solving image classification tasks. Please read the [general information](https://github.com/cpra/dlvc2016/blob/master/assignments/general.md) before you start.

In this part, we will make use of a Deep Learning library for the first time. We will, however, not train deep networks yet but instead stick to linear classification for now. Once this works, the extension to deep networks will be straight-forward as the data handling, preprocessing, training, and validation steps are identical. The purpose of this part is to implement these steps.

Assignment 2 builds upon Assignment 1, which must be finished first. This means that you must stick to the programming language you used to implement Assignment 1. This part also builds upon [Assignement 2 - Part 1](https://github.com/cpra/dlvc2016/blob/master/assignments/assignment2/part1.md), which must be finished first.

-----

As this is a new course, this text might not be without errors. If you find a significant error (not just typos but errors that affect the assignment), please contact us via [email](mailto:dlvc@caa.tuwien.ac.at). Students who find and report such errors will get extra points.

-----

We will reuse the following classes from Assignment 1, so copy the corresponding source files to your Assignment 2 code directory: `TinyCifar10Dataset`, `ImageVectorizer`, and the corresponding base classes.

As we will use minibatch-based training, we need functionality for splitting datasets into minibatches. We'll do so using a generic `MiniBatchGenerator` class. In order for this class to be able to support both `ImageDataset`s and `FeatureVectorDataset`s, we'll first add a common abstract base class:

```python
class ClassificationDataset:
    # A dataset consisting of multiple samples and corresponding class labels.

    def size(self):
        # Return the size of the dataset (number of samples).

    def nclasses(self):
        # Return the number of different classes.
        # Class labels start with 0 and are consecutive.

    def classname(self, cid):
        # Return the name of a class as a string.

    def sample(self, sid):
        # Return the sid-th sample in the dataset, and the
        # corresponding class label. Depending of your language,
        # this can be a Matlab struct, Python tuple or dict, etc.
        # Sample IDs start with 0 and are consecutive.
        # Throws an error if the sample does not exist.
```

The interface of this class is identical to those of `ImageDataset` and `FeatureVectorDataset`, except for the fact that it does not specify any sample properties.

Modify `ImageDataset` and `FeatureVectorDataset` so that they derive from `ClassificationDataset`.

-----

Create functionality for sample preprocessing. For flexibility we will write a class for every preprocessing operation as well as functionality for chaining multiple such operations together to form compound operations. All such classes must derive from the following base class:

```python
class SampleTransformation:
    # A sample transformation.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
```

Implement the following transformation classes (we'll add more later):

```python
class IdentityTransformation(SampleTransformation):
    # A transformation that does not do anything.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
```

```python
class FloatCastTransformation(SampleTransformation):
    # Casts the sample datatype to single-precision float (e.g. numpy.float32).

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
```

```python
class SubtractionTransformation(SampleTransformation):
    # Subtract a scalar from all features.

    @staticmethod
    def from_dataset_mean(dataset, tform=None):
        # Return a transformation that will subtract by the global mean
        # over all samples and features in a dataset.
        # tform is an optional SampleTransformation to apply before computation.

    def __init__(self, value):
        # Constructor.
        # value is a scalar to subtract.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # The sample datatype must be single-precision float.

    def value(self):
        # Return the subtracted value.
```

```python
class DivisionTransformation(SampleTransformation):
    # Divide all features by a scalar.

    @staticmethod
    def from_dataset_stddev(dataset, tform=None):
        # Return a transformation that will divide by the global standard deviation
        # over all samples and features in a dataset.
        # tform is an optional SampleTransformation to apply before computation.

    def __init__(self, value):
        # Constructor.
        # value is a scalar divisor != 0.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # The sample datatype must be single-precision float.

    def value(self):
        # Return the divisor.
```

If your language does not support static methods, implement those via constructor overloading or as an ordinary function.

Furthermore, create a class that represents a compound transformation:

```python
class TransformationSequence(SampleTransformation):
    # Applies a sequence of transformations
    # in the order they were added via add_transformation().

    def add_transformation(self, transformation):
        # Add a transformation (type SampleTransformation) to the sequence.

    def get_transformation(self, tid):
        # Return the id-th transformation added via add_transformation.
        # The first transformation added has ID 0.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
```

-----

Write a script `test_sample_transformations.py` that tests each `SampleTransformation` type using the training set of `TinyCifar10Dataset`. The output of the script must be as follows. The implementation of the script should be clear from the output:

    Computing SubtractionTransformation from TinyCifar10Dataset [train] mean
     Value: 120.38
    Computing DivisionTransformation from TinyCifar10Dataset [train] stddev
     Value: 64.40
    First sample of TinyCifar10Dataset [train]: shape: (32, 32, 3), data type: uint8, mean: 103.4, min: 0.0, max: 255.0
    After applying IdentityTransformation: shape: (32, 32, 3), data type: uint8, mean: 103.4, min: 0.0, max: 255.0
    After applying FloatCastTransformation: shape: (32, 32, 3), data type: float32, mean: 103.4, min: 0.0, max: 255.0
    After applying sequence FloatCast -> SubtractionTransformation: shape: (32, 32, 3), data type: float32, mean: -16.9, min: -120.4, max: 134.6
    After applying sequence FloatCast -> SubtractionTransformation -> DivisionTransformation: shape: (32, 32, 3), data type: float32, mean: -0.3, min: -1.9, max: 2.1

-----

Create a class called `MiniBatchGenerator` that wraps a `ClassificationDataset` and exposes the samples as minibatches. The interface must be as follows:

```python
class MiniBatchGenerator:
    # Create minibatches of a given size from a dataset.
    # Preserves the original sample order unless shuffle() is used.

    def __init__(self, dataset, bs, tform=None):
        # Constructor.
        # dataset is a ClassificationDataset to wrap.
        # bs is an integer specifying the minibatch size.
        # tform is an optional SampleTransformation.
        # If given, tform is applied to all samples returned in minibatches.

    def batchsize(self):
        # Return the number of samples per minibatch.
        # The size of the last batch might be smaller.

    def nbatches(self):
        # Return the number of minibatches.

    def shuffle(self):
        # Shuffle the dataset samples so that each
        # ends up at a random location in a random minibatch.

    def batch(self, bid):
        # Return the bid-th minibatch.
        # Batch IDs start with 0 and are consecutive.
        # Throws an error if the minibatch does not exist.
```

The `batch()` method should return a suitable datatype that holds three chunks of information:

* A tensor of all samples that form the given minibatch.
* A vector of corresponding class labels.
* A vector of corresponding dataset sample IDs.

The batch generator should support samples of arbitrary shapes and prepend one dimension for the sample index. Use a tensor class provided by the chosen library, as usual (e.g. `numpy.ndarray`). Examples:

* Each sample of `TinyCifar10Dataset` is a 3D tensor with shape `32x32x3`, so sample tensors returned by `MiniBatchGenerator` should have the shape `bsx32x32x3`, with `bs` being the minibatch size. The corresponding label and sample ID vectors both have `bs` elements.
* Each sample of `FeatureVectorDataset` is an 1D feature vector with `D` elements, so sample tensors returned by `MiniBatchGenerator` should have the shape `bsxD`.

> The only somewhat tricky part in implementing `MiniBatchGenerator` is how to assign dataset samples (randomly) to minibatches. The easiest way is to use index vectors that store the sample IDs of every minibatch. These vectors should be computed once in `__init__()`  and `shuffle()`, respectively.

-----

Create a script `test_minibatchgenerator.py` that tests the `MiniBatchGenerator` class in conjunction with one `ImageDataset` and one `FeatureVectorDataset`. Use the `TinyCifar10Dataset` training set for the former and a vectorized version (using `ImageVectorizer`) for the latter. The output of the script must be as follows. The implementation of the script shoule be clear from the output:

    === Testing with TinyCifar10Dataset ===

    Dataset has 4000 samples
    Batch generator has 67 minibatches, minibatch size: 60

    Minibatch #0 has 60 samples
     Data shape: (60, 32, 32, 3)
     First 10 sample IDs: 0,1,2,3,4,5,6,7,8,9
    Minibatch #66 has 40 samples
     First 10 sample IDs: 3960,3961,3962,3963,3964,3965,3966,3967,3968,3969

    Shuffling samples

    Minibatch #0 has 60 samples
     First 10 sample IDs: 2939,3096,3214,1966,1047,3262,3817,1110,3621,133
    Minibatch #66 has 40 samples
     First 10 sample IDs: 872,2815,3673,2056,2830,3108,3895,22,1232,2127

    === Testing with ImageVectorizer ===

    Dataset has 4000 samples
    Batch generator has 67 minibatches, minibatch size: 60

    Minibatch #0 has 60 samples
     Data shape: (60, 3072)
     First 10 sample IDs: 0,1,2,3,4,5,6,7,8,9
    Minibatch #66 has 40 samples
     First 10 sample IDs: 3960,3961,3962,3963,3964,3965,3966,3967,3968,3969

    Shuffling samples

    Minibatch #0 has 60 samples
     First 10 sample IDs: 26,1299,106,1426,2008,3629,667,177,3564,1927
    Minibatch #66 has 40 samples
     First 10 sample IDs: 2427,2238,1718,721,2854,1835,2628,2135,2987,108

Note that the generator must preserve the order of samples initially, before `shuffle()` is called. As shuffling is a random operation, your shuffled sample IDs will be different from those above.

-----

Write a script `softmax_classify_tinycifar10.py` that uses the above functionality to train and test a softmax classifier with the help of your chosen Deep Learning library. Every library supports this because softmax classifiers are linear models, and linear models can be represented by simple neural networks consisting of one input layer and one output layer. In Keras, the corresponding network is:

```python
model = Sequential()
model.add(Dense(output_dim=10, input_dim=3072))
model.add(Activation('softmax'))
```

In Keras there is no specific input layer, so the above specification contains only the output layer. This layer contains of 10 neurons because `TinyCifar10Dataset` has 10 classes. As this is a `Dense` (fully-connected) layer, each neuron is connected to all inputs. In our case, there are 3072 such inputs because `32x32x3=3072`. The input is expected to be a feature vector, so `ImageVectorizer` is required. The softmax function is usually represented as a layer as well. Details will be explained in the next lectures.

The script should perform the following steps:

1. Load the `TinyCifar10Dataset` training and validation sets and wrap them using `ImageVectorizer`.
2. Setup preprocessing in the form of conversion of samples to `float` followed by normalization so that the training set has zero mean and unit variance. This can be implemented using `TransformationSequence`, as done in `test_sample_transformations.py`.
3. Initialize `MiniBatchGenerator`s for both datasets. The minibatch size for the validation set does not affect the results and can be set to 100, for example. That of the training set is also not critical, set to 64 or any other value that works well for you.
4. Setup a softmax classifier and optimization (SGD with standard or Nesterov momentum) using your chosen Deep Learning library. Find a learning rate that works well (momentum can be fixed to 0.9).
5. Train for 200 epochs. After every epoch, print the average loss, training accuracy, and validation accuracy over all minibatches.

You must implement minibatch training and validation yourself. Every library has functionality for training and prediction (and, in some cases, testing) on a minibatch for this purpose. For instance, Keras has `model.train_on_batch()` for training. Note that some libraries expect ground-truth labels in one-hot encoding. Keras has a `to_categorical()` function for obtaining one-hot vectors from scalar labels.

The overall structure of the training and validation step will look as follows:

    for each of 200 epochs:
        for each minibatch in training set:
            train classifier, store loss and training accuracy
        for each minibatch in validation set:
            test classifier, store validation accuracy
        compute and report means over loss and accuracies

Training for 200 epochs should not take long even on less powerful CPUs, so there is no need to use the GPU server for this purpose (you can do that though if you want). With a suitable minibatch size and learning rate, the training accuracy should exceed 95% after 200 epochs. The output of your script should be similar to the following:

    Setting up preprocessing ...
     Adding FloatCastTransformation
     Adding SubtractionTransformation [train] (value: 120.38)
     Adding DivisionTransformation [train] (value: 64.40)
    Initializing minibatch generators ...
     [train] 4000 samples, 63 minibatches of size 64
     [val]   1000 samples, 10 minibatches of size 100
    Initializing softmax classifier and optimizer ...
    Training for 200 epochs ...
     [Epoch 001] loss: 2.687, training accuracy: 0.262, validation accuracy: 0.198
     [Epoch 002] loss: 2.289, training accuracy: 0.357, validation accuracy: 0.208
     [Epoch 003] loss: 2.071, training accuracy: 0.395, validation accuracy: 0.221
     [Epoch 004] loss: 1.925, training accuracy: 0.438, validation accuracy: 0.236
     [Epoch 005] loss: 1.817, training accuracy: 0.469, validation accuracy: 0.246
     [Epoch 006] loss: 1.728, training accuracy: 0.489, validation accuracy: 0.248
     [Epoch 007] loss: 1.651, training accuracy: 0.514, validation accuracy: 0.247
     [Epoch 008] loss: 1.583, training accuracy: 0.534, validation accuracy: 0.247
     [Epoch 009] loss: 1.523, training accuracy: 0.552, validation accuracy: 0.247
     [Epoch 010] loss: 1.470, training accuracy: 0.564, validation accuracy: 0.251
     [Epoch 011] loss: 1.423, training accuracy: 0.579, validation accuracy: 0.255
     [Epoch 012] loss: 1.381, training accuracy: 0.597, validation accuracy: 0.255
     [Epoch 013] loss: 1.342, training accuracy: 0.608, validation accuracy: 0.257
     [Epoch 014] loss: 1.307, training accuracy: 0.616, validation accuracy: 0.247
     [Epoch 015] loss: 1.275, training accuracy: 0.624, validation accuracy: 0.249
     [Epoch 016] loss: 1.245, training accuracy: 0.632, validation accuracy: 0.253
     [Epoch 017] loss: 1.220, training accuracy: 0.641, validation accuracy: 0.253
     [Epoch 018] loss: 1.194, training accuracy: 0.647, validation accuracy: 0.251
     [Epoch 019] loss: 1.171, training accuracy: 0.655, validation accuracy: 0.251
     [Epoch 020] loss: 1.149, training accuracy: 0.664, validation accuracy: 0.254
     [Epoch 021] loss: 1.129, training accuracy: 0.670, validation accuracy: 0.253
     [Epoch 022] loss: 1.109, training accuracy: 0.675, validation accuracy: 0.254
     [Epoch 023] loss: 1.091, training accuracy: 0.679, validation accuracy: 0.254
     [Epoch 024] loss: 1.073, training accuracy: 0.687, validation accuracy: 0.258
     [Epoch 025] loss: 1.056, training accuracy: 0.694, validation accuracy: 0.256
     [Epoch 026] loss: 1.040, training accuracy: 0.699, validation accuracy: 0.258
     [Epoch 027] loss: 1.024, training accuracy: 0.704, validation accuracy: 0.254
     [Epoch 028] loss: 1.008, training accuracy: 0.709, validation accuracy: 0.255
     [Epoch 029] loss: 0.994, training accuracy: 0.716, validation accuracy: 0.257
     [Epoch 030] loss: 0.980, training accuracy: 0.719, validation accuracy: 0.260
     [Epoch 031] loss: 0.967, training accuracy: 0.725, validation accuracy: 0.257
     [Epoch 032] loss: 0.954, training accuracy: 0.729, validation accuracy: 0.258
     [Epoch 033] loss: 0.942, training accuracy: 0.733, validation accuracy: 0.257
     [Epoch 034] loss: 0.930, training accuracy: 0.738, validation accuracy: 0.254
     [Epoch 035] loss: 0.919, training accuracy: 0.742, validation accuracy: 0.252
     [Epoch 036] loss: 0.909, training accuracy: 0.745, validation accuracy: 0.252
     [Epoch 037] loss: 0.898, training accuracy: 0.753, validation accuracy: 0.254
     [Epoch 038] loss: 0.888, training accuracy: 0.752, validation accuracy: 0.253
     [Epoch 039] loss: 0.879, training accuracy: 0.757, validation accuracy: 0.251
     [Epoch 040] loss: 0.869, training accuracy: 0.761, validation accuracy: 0.253
     [Epoch 041] loss: 0.860, training accuracy: 0.763, validation accuracy: 0.253
     [Epoch 042] loss: 0.851, training accuracy: 0.767, validation accuracy: 0.253
     [Epoch 043] loss: 0.843, training accuracy: 0.772, validation accuracy: 0.251
     [Epoch 044] loss: 0.835, training accuracy: 0.776, validation accuracy: 0.255
     [Epoch 045] loss: 0.827, training accuracy: 0.779, validation accuracy: 0.258
     [Epoch 046] loss: 0.819, training accuracy: 0.780, validation accuracy: 0.257
     [Epoch 047] loss: 0.812, training accuracy: 0.781, validation accuracy: 0.257
     [Epoch 048] loss: 0.806, training accuracy: 0.783, validation accuracy: 0.257
     [Epoch 049] loss: 0.800, training accuracy: 0.783, validation accuracy: 0.258
     [Epoch 050] loss: 0.793, training accuracy: 0.787, validation accuracy: 0.259
     ...

Note that the validation accuracy increases only initially and then decreases again due to overfitting. We'll see how to combat this in the next lecture and assignment.
