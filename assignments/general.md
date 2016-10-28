
# General Information on Assignments #

The following applies to all DLVC assignments unless stated otherwise in the corresponding specifications. Make sure you understand and adhere to all the requirements, as failing to do so will cost you points.

## Groups ##

You can do the assignments alone or in groups of two. If you want to form a group, send a mail with the Matriklenummer and names of both members to [dlvc@caa.tuwien.ac.at](mailto:dlvc@caa.tuwien.ac.at) until October 20, 11pm. No group changes are allowed afterwards, and it is not possible to change groups later on.

## Programming Language ##

You can implement the assignments in Matlab, Java, Python (2 or 3), or Lua, but we suggest you use Python. We will reuse code in later assignments, so choose a language you want to use for the rest of the semester. Some assignments might require to use another language for a certain subtask.

The following image / Deep Learning libraries must be used.

* Matlab: [MatConvNet](http://www.vlfeat.org/matconvnet/)
* Java: [nd4j](http://nd4j.org/) / [dl4j](http://deeplearning4j.org/index.html)
* Python: [numpy](http://www.numpy.org/) / [Keras](https://keras.io/) or [TensorFlow](https://www.tensorflow.org/)
* Lua: [torch7](http://torch.ch/) with the standard packages (`nn`, `image`, `optim` etc.)

No other libraries are allowed, except for the following:

* Python: [pillow](https://python-pillow.org/) or [scikit-image](http://scikit-image.org/) for loading and saving images
* Java: does anybody even use Java? If so, please let us know

Your code must be cross-platform, so make sure that you use only parts of your language's standard library that are available on all platforms.

## Code Templates ##

All assignments come with code templates that define the overall structure. They are in Python but easy to translate to other languages. As their main purpose is to be easy to read, they might not run without modifications.

Regardless of the language you choose, you must implement all code templates exactly as provided. For instance, if there is a template

```python
class Foo:
    # some description
    def bar(arg1, arg2):
      pass
```

then your code must have a class `Foo` that exactly matches this type in terms of interface (member names, argument names and types, and return types) and behavior (as stated in the code comments). This implies that you must use object-oriented programming, even if you use Matlab.

## Filenames ##

If there are filenames specified, you must use them as well (replace the endings depending on your language, e.g. `.m` instead of `.py`).

## Code Style ##

Your code must be easily readable and commented (similar to the provided templates).

## Indices ##

Python and Java start to count at 0, while Matlab and Lua start at 1. The assignments assume that the former is the case, e.g. the 500th image in an (ordered) dataset has ID 499. When using Matlab or Lua, you have to add 1 to all IDs given in the assignments.

## Reports ##

Every assignment includes a short report that summarizes the assignment. The report can be in German or English and must be in **PDF format**. It must be written in a narrative form, in a style that allows readers unfamiliar with the topic to get an overview of the tasks involved and how you solved them.

For example, if a task was to implement a kNN classifier, the corresponding part in the report should explain how the kNN classifier works, before going over implementation details. These implementation details should be general, without going into language specifics. For instance, in case of kNN you might write something like:

> During training, kNN classifiers store all training samples as vectors together with the corresponding class labels. In our implementation, the samples are stored as a matrix of size n*f, with n being the number of training samples and f being the vector dimensionality. The labels are stored as a n-vector. kNN training is implemented in the `KnnClassifier` class in file `knn_classifier.py`.

This writing style is easy to follow and detailed enough to understand how kNN classifiers perform training, and how this can be implemented. Please do not write something like this:

> We use Python's `pickle` module to load the CIFAR10 files. There are several such files, which are loaded in the `for` loop that starts at line 15 in file `knn_classifier.py`. Here is a copy of that code: ...

Again, the purpose of the report is to summarize a given assignment in a way that is easy to read, detailed enough to understand the topics covered and how you solved them, and concise. Do not paste code into your report.

## Submission ##

Submissions must be handed in fully, not in parts. Every submission must include the code for all parts of that submission, as well as a written report in PDF format.

Assignments are submitted via our Deep Learning server. See [this text](https://owncloud.tuwien.ac.at/index.php/s/3Cvwex1V3rPSVn4) for instructions on how to connect to this server. Once connected, you will find a `groupX/submissions/assignmentY` folder in your home directory, with `X` being your [group ID](https://owncloud.tuwien.ac.at/index.php/s/fu7q9EMmwMu3DMg) and `Y` being the assignment ID. To submit assignment `Y`, simply copy your submission into this folder. The folder is shared between all members of group `X`, so submission is required only once per group.

The submission deadlines are stated in the individual assignments. These deadlines are hard: you will no longer be able to change a submission folder after the deadline has passed. So please don't start uploading your submission five minutes before the deadline.

## Plagiarism ##

Do not copy code from other students or the web. If we find out that you do, you will get 0 points.
