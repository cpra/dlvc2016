
# Deep Learning VU - Assignment 3 #

In this assignment, you will improve the CNN-based classification pipeline developed in Assignment 2 based on what's been covered in lectures 8 and 9 in order to increase CIFAR10 performance. This assignment consists of a single part and is less strict than the others -- the goal is to encourage you to experiment with different network architectures, regularization strategies, data augmentation methods, and so on.

-----

As this is a new course, this text might not be without errors. If you find a significant error (not just typos but errors that affect the assignment), please contact us via [email](mailto:dlvc@caa.tuwien.ac.at). Students who find and report such errors will get extra points.

-----

You'll be awarded points based on the following factors:

* **Effort.** Experimenting with different data augmentation strategies, network architectures, making use of oversampling at test time, etc. will get you extra points. Note that this is not about quantity; don't just perform many trial-and-error experiments but put to use what you've learned in the lecture.
* **Report Quality.** You must submit a report that documents the experiments you carried out, *why you carried them out* (e.g. "the results of experiment X show that the model overfits, so we next attempt to account for this by doing Y"), the experimental results (e.g. in the form of a visualization of the training and test accuracy) and a discussion (e.g. "The results in Figure X show that the capacity of this model is not high enough because ..."). Again, this is not about quantity but about quality. The most important thing is that you document what you did and why in order for me to be able to judge whether you understand and are able to apply the concepts presented in the lecture.
* **Performance.** I'll compare and rank all submissions using a **private** test set. The higher your rank, the more points you will get. I'll maintain a leaderboard in order for you to be able to assess the quality of your submission before the deadline. See below for more information.

As a guideline, you can expect to get about 75% of the maximum number of points if you carry out a few sensible experiments, document them well, and exceed a CIFAR10 validation performance of 90%. The remaining points will be awarded based on additional effort and private test set performance.

-----

You should reuse as much code as possible from the previous assignments, including `Cifar10Dataset`, `MiniBatchGenerator` and different `SampleTransformation`s.

-----

You must use *at least* two forms of data augmentation: horizontal mirroring and random cropping. The interfaces of these classes must be as follows:

```python
class HorizontalMirroringTransformation(SampleTransformation):
    # Perform horizontal mirroring of samples with a given probability.

    def __init__(self, proba):
        # Constructor.
        # proba is a value in [0,1] that determines how likely it is
        # that a sample is mirrored (as opposed to left unchanged).

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,channels].
```

```python
class RandomCropTransformation(SampleTransformation):
    # Randomly crop samples to a given size.

    def __init__(self, width, height):
        # Constructor.
        # Images are cropped randomly to the specified width and height.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,channels].
        # If rows < height or cols < width, an error is raised.
```

Both transformations are easy to implement with the help of Python's `random` module and numpy. You are free to implement any additional transformations. Note that the resolution of training samples will be smaller than the default 32 by 32 pixels due to random cropping (unless you implement some form of padding transformation), so you need some transformation to obtain validation samples of the same size. Which transformation you use for this purpose is up to you.

-----

Create a script called `test_data_augmentation.py` that tests all transformations you implemented. The script should load [this image](https://github.com/cpra/dlvc2016/blob/master/assignments/assignment2/cat.jpg), initialize all transformations using sensible values, and combine them using the already existing `TransformationSequence` from Assignment 2. Apply the combination ten times to the image and save all results (ten in total) to folder `augmented_samples/`. Confirm that about half of the images are mirrored, that all images have the same and correct size, and that the cropped regions vary.

-----

Create a script called `train_best_model.py` that trains "your best model" for this assignment and saves it to `best_model.h5`. By "best model" I mean the best model you obtained through your experiments *in terms of CIFAR10 validation set performance*. As such, the script should not contain code for all experiments you carried out, but only the final one with the best architecture and hyperparameters you obtained experimentally. The script will thus be very similar to `cnn_classify_cifar10.py` from Assignment 2 and should produce similar output. The script should, however, not perform a final validation on the CIFAR10 test set.

You must obtain a validation accuracy (single model, no oversampling) of at least 90% in order to get all possible points. If you follow the guidelines presented in the lectures, this is not challenging. Visualize the training progress (training and test accuracy), add the visualization to your report, and discuss the results. Explain the network architecture your selected and why you selected it. Briefly explain every means for improving performance you use. For instance, if you use dropout, you should explain briefly what dropout is and why you use it. Also explain how you set the individual hyperparameters (e.g. dropout probability).

If you want to train a model ensemble in a way that requires multiple training scripts, call these scripts `train_best_model_X.py`, with `X` being some short description, and describe how the scripts differ in the report.

In any case, you must submit "all best models" you obtained. If you trained a model ensemble, submit all models in the ensemble as `best_model_1.h5`, `best_model_2.h5`, and so on. Otherwise you must submit exactly one file `best_model.h5`. I'll use this model (or the ensemble) to compute your final score on the private test set.

-----

Create a script `test_best_model.py` that tests "your best model" (`best_model.h5` or your model ensemble) on the CIFAR10 test set and reports the accuracy. I'll use this script to assess your group's performance on my private test dataset (by replacing `Cifar10Dataset`), so make sure to implement any strategy for improving test performance in this script.

Remember that the purpose of the test set is for a final performance evaluation. Do not use `test_best_model.py` multiple times with different models in an attempt to optimize the test performance. My private test set is different anyways, so a better score on CIFAR10 test does not necessarily imply a better score on my private test set.

-----

The current leaderboard is [here](https://github.com/cpra/dlvc2016/tree/master/assignments/assignment3/leaderboard.md). If you want to know how well you are doing before the deadline, write me a mail with your group ID and the path to the model you want tested. I'll use this model and your `test_best_model.py` (both must be in your group's submission directory) to compute the accuracy on the private test set and update the leaderboard. As I have to do this manually, you can request this evaluation only three times before the deadline. Note that it might take a day or two until you have the result.

-----

[Submit](https://github.com/cpra/dlvc2016/blob/master/assignments/general.md) Assignment 3 including the report. The deadline is Thu, 26.01., 23:00.
