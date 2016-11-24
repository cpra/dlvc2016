
# Deep Learning VU - Assignment 2 - Part 4 #

In the second assignment, you will learn how to use Convolutional Neural Networks (CNNs) for solving image classification tasks. In this part, we'll transition from the traditional feature extraction + classification approach to classification (as followed in Part 3) to a modern CNN-based approach. We'll also see how to use a trained CNN for classifying new images.

We will reuse the following classes from Assignment 1, so copy the corresponding source files to your Assignment 2 code directory: `Cifar10Dataset` and the corresponding base classes.

-----

As this is a new course, this text might not be without errors. If you find a significant error (not just typos but errors that affect the assignment), please contact us via [email](mailto:dlvc@caa.tuwien.ac.at). Students who find and report such errors will get extra points.

-----

In [Part 3](https://github.com/cpra/dlvc2016/blob/master/assignments/assignment2/part3.md) we've seen that the classification pipeline of HOG feature extraction followed by softmax classification did not perform particularly well, achieving a test accuracy of about 42%. The classifier also underperformed in terms of training accuracy even without regularization, indicating that it's capacity is too low. This seems reasonable as the softmax classifier is a linear model.

It can be shown easily that this is the case by replacing the softmax classifier with a more powerful classifier, such as a multilayer perceptron. Such a classifier, even a relatively "simple" one with a single hidden layer of 50 neurons or so, achieves a training accuracy close to 100%. Increasing the number of neurons while introducing weight decay and early stopping leads to a validation and test accuracy of about 47%. This is a substantial improvement over the softmax classifier, but still nowhere close to CNNs. We could maybe squeeze out a few percent more by tuning the model, but at some point the limitations of the HOG features hinder further progress. To this end, we'll now finally switch to deep learning and CNNs.

**Bonus task (5 points):** Reproduce the results above or try to beat them. If you choose to do this task, copy `opt_softmax_classify_hog_tinycifar10.py` from Part 3 to `opt_mlp_classify_hog_tinycifar10.py` and use the copy as a starting point.

> **Bonus tasks** are not required to reach all points possible for a given assignment, but they will count as a buffer in case you get point reductions. For instance, a bonus task that awards 5 points will "eat up" a reduction of 5 points. If you get no point reductions, bonus points have no effect.

-----

As CNNs operate on raw images instead of feature vectors, we'll first introduce additional preprocessing methods that are optimized for images. Specifically, we'll create versions of `SubtractionTransformation` and `DivisionTransformation` -- which we implemented in [Part 2](https://github.com/cpra/dlvc2016/blob/master/assignments/assignment2/part2.md) for feature normalization -- that operate on a per-channel basis. This is preferred because statistics used for normalization can differ significantly between the individual color channels. The interfaces of these classes are:

```python
class PerChannelSubtractionImageTransformation(SampleTransformation):
    # Perform per-channel subtraction of of image samples with a scalar.

    @staticmethod
    def from_dataset_mean(dataset, tform=None):
        # Return a transformation that will subtract by the global mean
        # over all samples and features in a dataset, independently for every color channel.
        # tform is an optional SampleTransformation to apply before computation.
        # samples must be 3D tensors with shape [rows,cols,channels].
        # rows, cols, channels can be arbitrary values > 0.

    def __init__(self, values):
        # Constructor.
        # values is a vector of c values to subtract, one per channel.
        # c can be any value > 0.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,c].
        # The sample datatype must be single-precision float.

    def values(self):
        # Return the subtracted values.
```

```python
class PerChannelDivisionImageTransformation(SampleTransformation):
    # Perform per-channel division of of image samples with a scalar.

    @staticmethod
    def from_dataset_stddev(dataset, tform=None):
        # Return a transformation that will divide by the global standard deviation
        # over all samples and features in a dataset, independently for every color channel.
        # tform is an optional SampleTransformation to apply before computation.
        # samples must be 3D tensors with shape [rows,cols,channels].
        # rows, cols, channels can be arbitrary values > 0.

    def __init__(self, values):
        # Constructor.
        # values is a vector of c divisors, one per channel.
        # c can be any value > 0.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,c].
        # The sample datatype must be single-precision float.

    def values(self):
        # Return the divisors.
```

-----

Update the `test_sample_transformations.py` from Part 2 to test the above transformations as well. The additional functionality and output of the script should be as follows:

    ...
    Computing PerChannelSubtractionImageTransformation from TinyCifar10Dataset [train] mean
     Values: 124.66, 122.35, 121.49
    Computing PerChannelDivisionImageTransformation from TinyCifar10Dataset [train] stddev
     Values: 67.48, 66.20, 65.64
    After applying sequence FloatCast -> PerChannelSubtractionImageTransformation -> PerChannelDivisionImageTransformation: shape: (32, 32, 3), data type: float32, mean: -0.3, min: -1.9, max: 2.0

We see that, in case of `TinyCifar10Dataset`, the per-channel statistics are quite similar.

-----

Time for deep learning. Copy `softmax_classify_hog_tinycifar10.py` from Part 3 to `cnn_classify_cifar10.py`. Change the datasets to `Cifar10Dataset` (**not** the tiny version). Replace `SubtractionTransformation` and `DivisionTransformation` with the per-channels versions from above. Finally, replace the softmax classifier with a CNN with the following architecture:

Layer Type | Parameters | Output Shape
---------- | ---------- | ------------
input | 32x32x3 | 32x32x3
conv | 3x3 @ 16 | 32x32x16
relu | |
pool | 2x2 max | 16x16x16
conv | 3x3 @ 32 | 16x16x32
relu | |
pool | 2x2 max | 8x8x32
conv | 3x3 @ 32 | 8x8x32
relu | |
pool | 2x2 max | 4x4x32
flatten | | 512
fc | 10 | 10
softmax | | 10

Layer descriptions:

* **input.** some libraries have an explicit input layer, others (like Keras) don't. The output shape, namely the index of the channel dimension, depends on the library. The above format is for TensorFlow, while others expect 3x32x32 (in Keras this can be [configured](https://keras.io/backend/)). In any case, if the chosen library expects samples to be in 3x32x32 format, create another `SampleTransformation` that performs the necessary transformation for use with the batch generators.
* **conv.** a 2D convolutional layer with padding such that the output resolution is unchanged (`border_mode='same'` in Keras). `3x3 @ 16` means 3x3 convolution and 16 feature maps.
* **relu.** ReLU nonlinearity.
* **pool.** a 2D pooling layer. `2x2 max` means max-pooling of 2x2 blocks. The stride should be 2 so that the output resolution is half the input resolution.
* **flatten.** a layer that flattens the input tensor to a vector. Required in most libraries before fc layers, which expect vector inputs.
* **fc.** a fully-connected (dense) layer. The only parameter is the number of hidden units.
* **softmax.** layer that applies the softmax function to the input.

Some libraries (like Keras) will by default take care of proper parameter initialization, using a method similar to what we've covered in Lecture 6. These defaults are fine. If you use TensorFlow, you have to do this yourself using e.g. `tf.get_variable(..., initializer=tf.contrib.layers.xavier_initializer())`.

Output the structure of your network to the console to be able to verify that your architecture matches the one above. Every library has functionality for this purpose, in Keras there is `model.summary()`.

> This small CNN has only about 20k parameters, significantly less than the much simpler softmax classifier we used previously for classifying images.

Train the CNN for at most 100 epochs using a global weight decay of 0.0001, a learning rate of 0.001, and a Nesterov momentum of 0.9. Use early stopping, aborting if the validation accuracy does not improve for 10 epochs. This might take a few hours unless you have a CUDA GPU, so you might want to use the [server](https://github.com/cpra/dlvc2016/blob/master/assignments/server.md). If you use the server, please don't test other hyperparameter combinations to conserve resources. The best model in terms of validation accuracy must be saved to disk **and submitted** (copied to the submission directory). The script output should be similar to the following:

    Loading Cifar10Dataset ...
    Setting up preprocessing ...
     Adding FloatCastTransformation
     Adding PerChannelSubtractionImageTransformation [train] (values: [ 124.7004776   122.28782654  121.54527283])
     Adding PerChannelDivisionImageTransformation [train] (values: [ 67.5210495   66.09591675  65.43208313])
    Initializing minibatch generators ...
     [train] 40000 samples, 625 minibatches of size 64
     [val]   10000 samples, 100 minibatches of size 100
    Initializing CNN and optimizer ...
    Training for 100 epochs ...
    ...
     [Epoch 074] loss: 0.513, training accuracy: 0.834, validation accuracy: 0.675
      New best validation accuracy, saving model to "model_best.h5"
     [Epoch 075] loss: 0.509, training accuracy: 0.836, validation accuracy: 0.674
     [Epoch 076] loss: 0.506, training accuracy: 0.837, validation accuracy: 0.674
     [Epoch 077] loss: 0.503, training accuracy: 0.837, validation accuracy: 0.674
     [Epoch 078] loss: 0.500, training accuracy: 0.838, validation accuracy: 0.672
     [Epoch 079] loss: 0.497, training accuracy: 0.839, validation accuracy: 0.673
     [Epoch 080] loss: 0.494, training accuracy: 0.841, validation accuracy: 0.672
     [Epoch 081] loss: 0.491, training accuracy: 0.842, validation accuracy: 0.674
     [Epoch 082] loss: 0.488, training accuracy: 0.843, validation accuracy: 0.673
     [Epoch 083] loss: 0.485, training accuracy: 0.844, validation accuracy: 0.671
     [Epoch 084] loss: 0.483, training accuracy: 0.845, validation accuracy: 0.671
     [Epoch 085] loss: 0.480, training accuracy: 0.846, validation accuracy: 0.670
      Validation accuracy did not improve for 10 epochs, stopping
    Testing best model on test set ...
     [test]   10000 samples, 100 minibatches of size 100
     Accuracy: 67.1%

You should reach a validation accuracy of about 67%, as shown above. The exact score will vary depending on the initial weights, which are chosen randomly. For comparison, a multilayer perceptron with tuned hyperparameters trained on HOG features of CIFAR10 achieves an accuracy of about 60%. The CNN clearly performs better, but still not as good as we'd like. We'll improve the results in Assignment 3.

-----

In practice, we'll want to apply our trained classifier to new data (images). In order to get familiar with the corresponding processing pipeline, create a script `cnn_classify_image.py`. This script should expect the following runtime arguments:

* `--model`: path to a classifier saved by `cnn_classify_cifar10.py`
* `--image`: path to a RGB input image
* `--means`: means for `PerChannelSubtractionImageTransformation`
* `--stds`: standard deviations for `PerChannelDivisionImageTransformation`

Use argument parsers provided by your programming language, e.g. `argparse` in Python. `--means` and `--stds` can expect any desired input format, such as `--means 1 2 3` or `--means 1,2,3`.

The script should load the classifier and image, preprocess the image using the same operations as carried out during classifier training (hence `--means` and `--stds`), classify the image, and report the class scores as well as the ID of the most likely class to the console. The script should be flexible enough to support arbitrary classifiers and image resolutions, so do not hard-code CIFAR10-specific things such as the input resolution expected by the classifier; this information can be obtained from the loaded classifier.

The image resolution must also be flexible, but images are expected to be square and in RGB format. The script thus requires functionality for resizing square images. Instead of implementing this functionality in the script itself, create another `SampleTransformation` subclass for this purpose:

```python
class ResizeImageTransformation(SampleTransformation):
    # Resize samples so that their smaller side length
    # (width or height) is as specified. For instance, if
    # the specified size is 32 and an input image is of
    # shape 50x60x3, the output size will be 32x38x3.

    def __init__(self, size):
        # Constructor.
        # size is the desired size of the smaller side of samples.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,channels].
        # Throws an error if min(rows,cols) < size.
```

Test the script with own samples as well as [this](https://github.com/cpra/dlvc2016/blob/master/assignments/assignment2/cat.jpg) example image, in which case the script output should be as follows (the per-class probabilities and predicted class might differ):

    $ python cnn_classify_image.py --model model_best.h5 --image ../specification/cat.jpg --means 124.7,122.29,121.55 --stds 67.52,66.1,65.43
    Parsing arguments ...
     Model: model_best.h5
     Image: ../specification/cat.jpg
     Means: [124.7, 122.29, 121.55]
     Stds: [67.52, 66.1, 65.43]
    Loading image ...
     Shape: (78, 78, 3)
    Loading classifier ...
     Input shape: (32, 32, 3), 10 classes
    Preprocessing image ...
     Transformations in order:
      ResizeImageTransformation
      FloatCastTransformation
      PerChannelSubtractionImageTransformation ([ 124.7   122.29  121.55])
      PerChannelDivisionImageTransformation ([ 67.52  66.1   65.43])
     Result: shape: (32, 32, 3), dtype: float32, mean: -1.844, std: 0.006
    Classifying image ...
     Class scores: [0.03, 0.00, 0.07, 0.49, 0.33, 0.05, 0.02, 0.01, 0.00, 0.00]
     ID of most likely class: 3 (score: 0.49)


In the example output, the image is correctly assigned ID 3 (cat), although with a rather low confidence.

-----

Write a report that summarizes assignment 2. See [general information](https://github.com/cpra/dlvc2016/blob/master/assignments/general.md) for requirements. The overall structure and contents of the report must be as follows:

1. **Deep Learning Libraries.** Briefly state which deep learning library you used and if you used the server for training. One or two sentences are sufficient.
2. **Linear Models.** Describe what parametric models are and how they differ from the k nearest neighbors classifier we used in Assignment 1. Describe what a linear model is and how these models are used for classification. What to the parameters (weights and biases) of linear classifiers specify?
3. **Minibatch Gradient Descent.** Explain the purpose and tasks involved in training parametric models. Give an overview of gradient descent and loss functions in general as well as cross-entropy loss. You don't have to use math for this, a summary that is easy to understand is sufficient. For instance, you could explain gradient descent using the hiking analogy I used in the lecture. What is minibatch gradient descent and why is it preferred to batch gradient descent?
4. **Preprocessing.** Explain what preprocessing is and why normalization (the main operation we used in this assignment) is important. How is preprocessing handled with respect to the different datasets (train,val,test)?
5. **Optimization vs. Machine Learning.** Summarize the differences between pure optimization and machine learning. Describe what regularization is and how weight decay and early stopping work (no need to use formulas).
6. **Convolutional Neural Networks.** Explain what a CNN is, as well as the general layer structure that all CNNs for classification share. Explain the purpose and operation of convolutional, pooling, and fully-connected layers.

The report should be 3 to 5 pages long.

-----

[Submit](https://github.com/cpra/dlvc2016/blob/master/assignments/general.md) Assignment 2, including all four parts. The **deadline is Thu, 8.12., 23:00**.
