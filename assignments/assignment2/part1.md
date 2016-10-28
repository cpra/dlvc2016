
# Deep Learning VU - Assignment 2 - Part 1 #

In the second assignment, you will learn how to use Convolutional Neural Networks (CNNs) for solving image classification tasks. Please read the [general information](https://owncloud.tuwien.ac.at/index.php/s/tejPFjV5uz39rBL) before you start.

Assignment 2 builds upon Assignment 1, which must be finished first. This means that you must stick to the programming language you used to implement Assignment 1.

-----

As this is a new course, this text might not be without errors. If you find a significant error (not just typos but errors that affect the assignment), please contact us via [email](mailto:dlvc@caa.tuwien.ac.at). Students who find and report such errors will get extra points.

-----

> All supported deep learning libraries can use both the CPU and the GPU for calculations, although the later requires a somewhat recent Nvidia GPU for CUDA support. Unfortunately, solving any practical task using CNNs virtually requires such a GPU because training is generally too slow on CPUs. Still, you must install a library even if you don't have such a GPU. This way you can write and test your code on your local machine, and then use our server to perform a full training run.

Select, install, and test a deep learning library of choice, depending on the programing language you selected. See [general information](https://owncloud.tuwien.ac.at/index.php/s/tejPFjV5uz39rBL) for a list of libraries to choose from. You must select a library from this list, other libraries are not allowed. If you have a Nvidia GPU, you should install [CUDA](https://developer.nvidia.com/cuda-toolkit) first in order to enable GPU support. In this case, you might also want to install [cuDNN](https://developer.nvidia.com/cudnn) for optimal speed. We suggest you use library versions that are similar (ideally identical) to those on the server (see below). Otherwise, you might run into problems when trying to run your code remotely.

For testing, simply try one of the examples that come with all libraries.

> Some deep learning libraries can be a pain to install on Windows. This applies particularly to tensorflow and theano, one of which is required if you use Python (Keras can use both as the backend). In fact, tensorflow is not fully supported under Windows right now. It seems that the easiest way of installing either library is using Docker images ([tensorflow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#docker-installation), [theano](http://deeplearning.net/software/theano/install.html)). I'm not a Windows user, so my experience in this area is very limited though.

-----

> Our server is currently not yet reachable from outside, and the information in this section is not final yet. We'll notify you once the server is available.

Training CNNs is a very complex task that usually takes too long even on modern desktop CPUs. For this reason, CNNs are usually trained on powerful GPUs. In order to support students who do not have such a GPU (any decent Nvidia GPU will do for this assignment), we have set up a powerful GPU server that can be used for training. Apart from running code, this server is also used for handing in assignments.

Information on how to use this server is available [here](https://owncloud.tuwien.ac.at/index.php/s/3Cvwex1V3rPSVn4). This part of the assignment is simple: connect to the server using the credentials you have received via mail. First change the default password using the `passwd` command. Then take a look at the directory structure in your home folder and familiarize yourself with with the scheduler by submitting some dummy job.

All supported deep learning libraries are already installed on the server, so you don't have to install anything. Write a simple script to ensure that the library you chose is indeed available. For instance, if you use tensorflow locally, your script might be:

```python
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))
```

The script does not have to do anything meaningful and you don't have to hand it in.
