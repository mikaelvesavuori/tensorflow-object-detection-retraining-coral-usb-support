# Object detection and retraining with support for Coral USB Accelerator

This notebook helps you retrain an existing COCO (Common Objects in Context) model, and output it with Tensorflow Lite to be a quantized model that can do object detection on your custom labels. Quantized models will be compatible with an Edge TPU device like the Coral USB Accelerator, given that the original model is supported by one.

This entire notebook is to a large extent built on fantastic previous work that has been done by contributors globally. The section on preparation is based on [https://blog.ml6.eu/our-edge-tpu-demo-project-cbc9bea5a355](https://blog.ml6.eu/our-edge-tpu-demo-project-cbc9bea5a355), and the training section is adapted from [https://github.com/dctian/DeepPiCar/blob/master/models/object_detection/code/tensorflow_traffic_sign_detection.ipynb](https://github.com/dctian/DeepPiCar/blob/master/models/object_detection/code/tensorflow_traffic_sign_detection.ipynb).

At the time of writing (late April 2020), coming in as a newbie in hardware-accelerated machine learning can certainly be a confusing experience. Some of the factors delaying taking those baby steps are:

- Tensorflow version 2 is out since some time back, but seems unequivocally rejected for actual object detection use. Not sure why that is.
- Overall learning curve is high, further complicated by a mix of Python (which I've only encountered in bits before), tons of different writing styles and libraries, as well as the TF v1 and TF v2 conflict mentioned above.
- Using the Coral USB Accelerator, you will find a number of examples online (including the very good documentation) but for me who learns best from articles and examples there are not very many "working" examples that I myself can explain. Some are hidden behind large scripts and until quite some time later, I never understood that there were "built-in" scripts that come with the TF models.
- And of course: All of the specificities around what can and cannot compile to an Edge TPU, including the range of opaque and mystical errors that the ML world can produce when trying to find one's way.
- Finally, I wanted an example in notebook format, ideally Google Colab, since it's a more stable ML platform than I am managing to set up my Macbook Pro to be.

I hope that this notebook will serve as a complete, well-documented example for how to prepare data and then train it, and finally to compile it for your Coral USB Accelerator. You should be able to find good information here, even if you are not interested in all of the steps.

My intent is not to take any credit (since again, most material here is simply repurposed and adjuster), beyond hopefully whatever gratefulness that you may feel if I manage to make learning this stuff easier and faster.

With that said, I might not have covered all the most minute details. Expect to have to fill in some blanks as you go.

PS: Obviously, as time passes and TF v2 settles as the standard, we will have to see whether this notebook remains relevant in 2021 and beyond.

## Requirements for compiling to Edge TPU devices

The two basic requirements are:

1.  To train a model, you **should ideally** use a pre-trained model (suggestion: use the [Tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md))
2.  The model **must** be quantized

## Note on using Coral USB Accelerator with Raspberry Pi Zero W

The above list should give you a sense that the ML landscape is shifting quite fast, and because we are dealing with both software and hardware, things change on a frequent basis.

One of my failed attempts while working on this was getting the Coral stick to work with Raspberry Pi Zero W. It's technically possible, but you would need to compile for the version 10 runtime. This is because the Zero uses an ARMv6 architecture, while TF (if I remember correctly) is compiled for ARMv7. Yes, you can rebuild/recompile, but that's out of my league. Ultimately I put that side-project on ice. So:

- Yes, Coral works on the Pi Zero W given the official demos
- No, it's not very easy (for me at least) to compile custom models and get them working on the Zero W

## References

- [https://blog.ml6.eu/our-edge-tpu-demo-project-cbc9bea5a355](https://blog.ml6.eu/our-edge-tpu-demo-project-cbc9bea5a355)
- [https://gist.github.com/StanCallewaert](https://gist.github.com/StanCallewaert)
- [https://github.com/dctian/DeepPiCar/blob/master/models/object_detection/code/tensorflow_traffic_sign_detection.ipynb](https://github.com/dctian/DeepPiCar/blob/master/models/object_detection/code/tensorflow_traffic_sign_detection.ipynb)
- [https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi)
- [https://github.com/toborobot/train_own_tensorflow_colab/blob/master/train_detect_own_dataset_tf_colab.ipynb](https://github.com/toborobot/train_own_tensorflow_colab/blob/master/train_detect_own_dataset_tf_colab.ipynb)
- [https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/](https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/)
- [https://github.com/datitran/raccoon_dataset](https://github.com/datitran/raccoon_dataset)
- [https://pythonprogramming.net/creating-tfrecord-files-tensorflow-object-detection-api-tutorial/](https://pythonprogramming.net/creating-tfrecord-files-tensorflow-object-detection-api-tutorial/)

## Resources

- Retrain an object detection model
- Train object detection model on Google Colab with TPU

## Tensorflow: GitHub Models Repository Commits

We are going to use Tensorflow 1.15, so it's important that models and the library itself is not "latest".

From [https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10):

"If you are using an older version of TensorFlow, here is a table showing which GitHub commit of the repository you should use. I generated this by going to the release branches for the models repository and getting the commit before the last commit for the branch. (They remove the research folder as the last commit before they create the official version release.)"

- TF v1.7: [https://github.com/tensorflow/models/tree/adfd5a3aca41638aa9fb297c5095f33d64446d8f](https://github.com/tensorflow/models/tree/adfd5a3aca41638aa9fb297c5095f33d64446d8f)
- TF v1.8: [https://github.com/tensorflow/models/tree/abd504235f3c2eed891571d62f0a424e54a2dabc](https://github.com/tensorflow/models/tree/abd504235f3c2eed891571d62f0a424e54a2dabc)
- TF v1.9: [https://github.com/tensorflow/models/tree/d530ac540b0103caa194b4824af353f1b073553b](https://github.com/tensorflow/models/tree/d530ac540b0103caa194b4824af353f1b073553b)
- TF v1.10: [https://github.com/tensorflow/models/tree/b07b494e3514553633b132178b4c448f994d59df](https://github.com/tensorflow/models/tree/b07b494e3514553633b132178b4c448f994d59df)
- TF v1.11: [https://github.com/tensorflow/models/tree/23b5b4227dfa1b23d7c21f0dfaf0951b16671f43](https://github.com/tensorflow/models/tree/23b5b4227dfa1b23d7c21f0dfaf0951b16671f43)
- TF v1.12: [https://github.com/tensorflow/models/tree/r1.12.0](https://github.com/tensorflow/models/tree/r1.12.0)
- TF v1.13: [https://github.com/tensorflow/models/tree/r1.13.0](https://github.com/tensorflow/models/tree/r1.13.0)
- Latest version: [https://github.com/tensorflow/models](https://github.com/tensorflow/models)
