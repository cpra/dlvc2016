
# Deep Learning VU - Assignment 2 - Part 3 #

In the second assignment, you will learn how to use Convolutional Neural Networks (CNNs) for solving image classification tasks. Part 3 is about extending `softmax_classify_tinycifar10.py` by focusing on machine learning instead of pure optimization. We'll also go back to feature-based classification, utilizing the HOG datasets from Assignment 1.

We will reuse the following classes from Assignment 1, so copy the corresponding source files to your Assignment 2 code directory: `HDF5FeatureVectorDataset`, and the corresponding base classes.

-----

As this is a new course, this text might not be without errors. If you find a significant error (not just typos but errors that affect the assignment), please contact us via [email](mailto:dlvc@caa.tuwien.ac.at). Students who find and report such errors will get extra points.

-----

In Part 2, we saw that the `TinyCifar10Dataset` training set is linearly separable in *image space*, i.e. when using the individual pixels as features. This is because this space is very high dimensional (3072D) but we only have 4000 samples. As a result, the classifier overfits the training set despite being a simple linear model.

To demonstrate this, copy `softmax_classify_tinycifar10.py` to `softmax_classify_hog_tinycifar10.py` and change the script to use the HOG-feature version of `TinyCifar10Dataset`, as done in [Assignment 1](https://github.com/cpra/dlvc2016/blob/master/assignments/assignment1/part3.md). Set the learning rate to `0.001` but leave the reset unchanged. After 200 epochs, you should achieve a training and validation accuracy of about 54.3% and 40.8%, respectively, While the training accuracy is much lower than before, which is bad from an optimization perspective, the validation accuracy is much improved. In machine (and deep) learning, we usually care only about the latter as it indicates how well our classifier will work on data unseen during training, and thus in practice.

-----

Extend `softmax_classify_hog_tinycifar10.py` by implementing early stopping as covered in the [lecture](https://github.com/cpra/dlvc2016/blob/master/lectures/lecture6.pdf). The script should keep track of the current best model/parameters in terms of validation accuracy by saving it to disk. Every library has functionality for this purpose (e.g. Keras has `model.save()`). The script should abort training if the validation accuracy does not improve for several epochs (can be chosen freely but should not be set too low). The script should report corresponding information to the console, like so:

    ...
    [Epoch 043] loss: 1.445, training accuracy: 0.505, validation accuracy: 0.417
     New best validation accuracy, saving model to "model_best.h5"
    [Epoch 044] loss: 1.443, training accuracy: 0.506, validation accuracy: 0.416
    [Epoch 045] loss: 1.440, training accuracy: 0.507, validation accuracy: 0.416
    [Epoch 046] loss: 1.437, training accuracy: 0.508, validation accuracy: 0.414
    [Epoch 047] loss: 1.435, training accuracy: 0.509, validation accuracy: 0.416
    [Epoch 048] loss: 1.432, training accuracy: 0.510, validation accuracy: 0.417
    [Epoch 049] loss: 1.430, training accuracy: 0.511, validation accuracy: 0.416
    [Epoch 050] loss: 1.428, training accuracy: 0.512, validation accuracy: 0.416
    [Epoch 051] loss: 1.426, training accuracy: 0.512, validation accuracy: 0.416
    [Epoch 052] loss: 1.423, training accuracy: 0.513, validation accuracy: 0.414
    [Epoch 053] loss: 1.421, training accuracy: 0.514, validation accuracy: 0.414
    [Epoch 054] loss: 1.419, training accuracy: 0.514, validation accuracy: 0.414
    [Epoch 055] loss: 1.418, training accuracy: 0.516, validation accuracy: 0.414
    [Epoch 056] loss: 1.416, training accuracy: 0.517, validation accuracy: 0.413
    [Epoch 057] loss: 1.414, training accuracy: 0.518, validation accuracy: 0.412
    [Epoch 058] loss: 1.412, training accuracy: 0.518, validation accuracy: 0.412
    [Epoch 059] loss: 1.411, training accuracy: 0.519, validation accuracy: 0.413
    [Epoch 060] loss: 1.409, training accuracy: 0.521, validation accuracy: 0.413
    [Epoch 061] loss: 1.408, training accuracy: 0.522, validation accuracy: 0.413
    [Epoch 062] loss: 1.406, training accuracy: 0.522, validation accuracy: 0.413
    [Epoch 063] loss: 1.405, training accuracy: 0.523, validation accuracy: 0.413
     Validation accuracy did not improve for 20 epochs, stopping
    Best validation accuracy: 0.42 (epoch 43)

-----

Copy `softmax_classify_hog_tinycifar10.py` (after implementing early stopping) to `opt_softmax_classify_hog_tinycifar10.py` and implement hyperparamter optimization as in [Assignment 1](https://github.com/cpra/dlvc2016/blob/master/assignments/assignment1/part2.md). The script should search for a good combination of two hyperparameters, learning rate and [weight decay](https://github.com/cpra/dlvc2016/blob/master/lectures/lecture6.pdf), using either grid or random search. You don't have to implement weight decay yourself, all libraries support it. Choose a search range based on initial testing. Once a suitable search range was found (can be hard-coded), test at least 20 combinations. Training will be fast even on less powerful CPUs, so there is no need to use the GPU server (unless you want to).

The script should report every combination tested as well as the corresponding validation accuracy (using early stopping). Finally, the script should load the best model from disk and report its test accuracy. The output should be similar to the following:

    Loading HDF5FeatureVectorDataset ...
    Setting up preprocessing ...
     Adding FloatCastTransformation
     Adding SubtractionTransformation [train] (value: 0.01)
     Adding DivisionTransformation [train] (value: 0.01)
    Initializing minibatch generators ...
     [train] 4000 samples, 63 minibatches of size 64
     [val]   1000 samples, 10 minibatches of size 100
    Performing random hyperparameter search ...
     learning rate=0.0007, weight decay=0.0363, accuracy: 0.434 (epoch 62)
     learning rate=0.0001, weight decay=0.0739, accuracy: 0.427 (epoch 185)
     learning rate=0.0004, weight decay=0.1160, accuracy: 0.428 (epoch 60)
     learning rate=0.0006, weight decay=0.0347, accuracy: 0.428 (epoch 55)
     learning rate=0.0001, weight decay=0.2453, accuracy: 0.409 (epoch 99)
     learning rate=0.0001, weight decay=0.1652, accuracy: 0.381 (epoch 128)
     learning rate=0.0005, weight decay=0.1097, accuracy: 0.429 (epoch 47)
     learning rate=0.0008, weight decay=0.2163, accuracy: 0.419 (epoch 13)
     learning rate=0.0009, weight decay=0.2334, accuracy: 0.416 (epoch 36)
     learning rate=0.0003, weight decay=0.1726, accuracy: 0.416 (epoch 63)
     learning rate=0.0009, weight decay=0.1122, accuracy: 0.431 (epoch 27)
     learning rate=0.0001, weight decay=0.0336, accuracy: 0.386 (epoch 122)
     learning rate=0.0005, weight decay=0.1418, accuracy: 0.424 (epoch 39)
     learning rate=0.0004, weight decay=0.0996, accuracy: 0.431 (epoch 55)
     learning rate=0.0008, weight decay=0.0141, accuracy: 0.421 (epoch 47)
     learning rate=0.0002, weight decay=0.1983, accuracy: 0.414 (epoch 70)
     learning rate=0.0009, weight decay=0.2808, accuracy: 0.408 (epoch 13)
     learning rate=0.0009, weight decay=0.0611, accuracy: 0.435 (epoch 34)
     learning rate=0.0009, weight decay=0.1387, accuracy: 0.427 (epoch 18)
     learning rate=0.0004, weight decay=0.1485, accuracy: 0.420 (epoch 66)
     learning rate=0.0010, weight decay=0.2284, accuracy: 0.415 (epoch 39)
     learning rate=0.0001, weight decay=0.0257, accuracy: 0.414 (epoch 194)
     learning rate=0.0001, weight decay=0.0558, accuracy: 0.404 (epoch 133)
     learning rate=0.0002, weight decay=0.0628, accuracy: 0.420 (epoch 80)
     learning rate=0.0009, weight decay=0.2745, accuracy: 0.410 (epoch 45)
     learning rate=0.0001, weight decay=0.1388, accuracy: 0.405 (epoch 200)
     learning rate=0.0002, weight decay=0.1801, accuracy: 0.418 (epoch 63)
     learning rate=0.0003, weight decay=0.0642, accuracy: 0.424 (epoch 85)
     learning rate=0.0009, weight decay=0.2508, accuracy: 0.410 (epoch 54)
     learning rate=0.0009, weight decay=0.0362, accuracy: 0.428 (epoch 57)
     learning rate=0.0003, weight decay=0.1359, accuracy: 0.425 (epoch 56)
    Testing best model (learning rate=0.0009, weight decay=0.0611) on test set ...
     [test]   1000 samples, 10 minibatches of size 100
     Accuracy: 41.7%

> There is not a huge improvement in this case because (i) the loss function of softmax classifiers is convex and thus easy to optimize, and (ii) the model is not prone to overfitting in the first place. This will change when we (finally!) switch from linear models to CNNs in the next part. Note that the test accuracy of the softmax classifier is only slightly better than that of the k nearest neighbors classifer tested in Assignment 1.
