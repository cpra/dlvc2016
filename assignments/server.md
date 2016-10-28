
# Deep Learning Server #

We set up a dedicated server for students of this lecture. This server serves two purposes: handing in assignments and training deep networks.

## Access ##

The server is reachable from within the TU Wien university network only. If you want to reach the server from home, you can use [VPN](https://www.zid.tuwien.ac.at/tunet/vpn/client/). You can connect to the server via SSH:

    ssh eMATRIKELNUMMER@dlvc.caa.tuwien.ac.at

If you are on Windows, you can use a SSH client such as [PuTTY](http://www.putty.org/).

The password for logging in was sent to you via email. If you have not received such a mail, please contact us. The first thing you should do after connecting for the first time is changing your password using the `passwd` command.

## Your Home Directory ##

After logging in, you are inside your home directory. This is your private space, only you can access this directory. Inside your directory there is a `groupX/` folder, with `X` being your [group ID](https://owncloud.tuwien.ac.at/index.php/s/fu7q9EMmwMu3DMg). If you are in a group of two, you can use this folder to share data with your colleague. The main purpose of this directory is for handing in assignments. See the [general information](https://owncloud.tuwien.ac.at/index.php/s/tejPFjV5uz39rBL) for details. Do not delete the `groupX/` directory or the `submissions/` subdirectory.

You can upload and download data using `scp` or an SFTP GUI like [filezilla](https://filezilla-project.org/). Please do not upload datasets. All datasets required are available inside the `datasets/` folder. See the Dataset section.

## Deep Learning Libraries ##

All deep learning libraries allowed are already installed on the server. Please don't install your own versions of these libraries locally, always use the preinstalled versions. The following libraries and software versions are installed:

* **tensorflow.** version r0.11 for Python2 and Python3. To use, `import tensorflow`.
* **keras.** version git commit `028aae19bf5ae6efe0b32d25d1c700224eebfcf9` for Python2 and Python3, with tensorflow backend. To use, `import keras`.
* **torch.** version git commit `1037772e952e910ece55820683852cf369022a96`. To use, `require('torch')`. All standard packages for CNN training are installed as well, see `luarocks list`.
* **matconvnet.** version git commit `4ce2871ec55f0d7deed1683eb5bd77a8a19a50cd`. The library is at `/opt/matconvnet`, so you have to call `addpath /opt/matconvnet/matlab` in Matlab before you can use the library.
* **dl4j.** The compiled and packaged examples repository, version git commit `711c08313d8e45ba4b0d571ca5cbd3314350385e`, is at `/opt/dl4j-examples/`. To be honest, I don't know if that's all you need as I have little experience with Java. If you need anything else, please contact me (Christopher).

We strongly suggest that you install the same versions on your computer as most of these libraries are in heavy development, leading to frequent API changes.

## Datasets ##

All datasets you need in the assignments are available inside the `datasets/` directory. (If not, please contact us.) Please don't upload datasets yourself in order to conserve disk space.

## Task Scheduling ##

The server is quite powerful, it has 40 logical CPU cores, 256 GB RAM, and 4 powerful GPUs (two [GeForce TITAN X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) and two [GeForce GTX 1080](http://www.geforce.com/hardware/10series/geforce-gtx-1080)). Despite this, task scheduling is required because training a CNN typically results in full utilization of a single GPU. This means that at most four processes that require GPU resources should run in parallel and on different GPUs. The task scheduler ensures this.

### Submitting Jobs ###

> The `getFreeGPU` does currently not work with the scheduler, causing an error. This will be fixed in the future. Until then, you can select a free GPU yourself by executing `nvidia-smi` and using the GPU (IDs 0 to 3) with the lowest utilization. Then specify the GPU manually, e.g. `export CUDA_VISIBLE_DEVICES=1`. This obviously does not work if your job is queued for some time (which is the very reason why `getFreeGPU` exists) but will be sufficient for Assignment 2 - Part 1. Please do not run jobs that take long until this problem is fixed.

To schedule a job that requires access to the GPU, create a bash script that includes the corresponding commands. The first lines should always be as follows:

```bash
#!/bin/bash
#PBS -m bea
#PBS -M YOUR_EMAIL_ADDRESS

# Request free GPU
export CUDA_VISIBLE_DEVICES=$(getFreeGPU)
```

Change `YOUR_EMAIL_ADDRESS` in line 3 to an email address to which notifications should be sent. The scheduler will notify you if your script starts and when it has finished. The `export ...` line is always required in this exact form. It ensures that your scripts will run on the GPU that is currently utilized the least.

Afterwards, simply add the command(s) to execute, one per line, e.g.:

```bash
python my_python_script_that_requires_gpu.py
```

Once the script is ready, submit it to the scheduler like so:

```bash
qsub my_script.sh
```

### Job Administration ###

Use `qstat` to view your job queue. It will return something like this:

    Job id                    Name             User            Time Use S Queue
    ------------------------- ---------------- --------------- -------- - -----
    69.localhost              test.sh          stefan          00:00:00 C batch
    70.localhost              test.sh          stefan                 0 R batch
    71.localhost              test.sh          stefan                 0 Q batch
    72.localhost              test.sh          stefan                 0 Q batch

The `S` column shows the job status:

    C     job is completed
    R     job is running
    Q     job is waiting the queue

If you want to cancel a job, use `qdel JOB`, with `JOB` being the Job ID shown in `qstat`.

### Accessing Job Output ###

If your scripts write output to stdout or stderr (i.e. if they log to the console), you might want to retrieve this output once the job has finished. The scheduler writes all output to stdout and stderr to a file called `JOBNAME.oJOBID` and `JOBNAME.eJOBID`, respectively. You can find both files in your working directory once the job starts to run.

### Don't Cheat ###

Do not attempt to bypass the scheduler by executing your scripts directly. The server will detect this and kill the corresponding process. Also don't chain several long-running commands inside a single job script. There is a time limit after which every job will be killed. Basically, be fair and share the limited resources we have with your colleagues.

### Limited Resources and Queues ###

There are 40 groups that have to share access to 4 GPUs. We expect that this will lead to long queues as an assignment deadline approaches. In this case, you'll probably have to wait for hours before your script even starts. In order to minimize wait times, please do the following:

* Write and test your code locally on your system. If you have a decent Nvidia GPU, please train locally and don't use the server. If you don't have such a GPU, perform training for a few epochs on the CPU to ensure that your code works. If this is the case, upload your code to our server and do a full training run there.
* Don't schedule multiple training runs in a single job, and don't submit multiple long jobs. Be fair.
* If you want to train on the server, do so as early as possible. If everyone starts two days before the deadline, there will be long queues and your job might not finish soon enough.
