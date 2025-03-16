# Evaluating Spatial Robustness of Detectron2 Object Detection and Instance Segmentation

Extension on [NERO paper](https://arxiv.org/abs/2305.19889) to evaluate robustness of computer vision models.

# Background

# Method
I use [Detectron2](https://github.com/facebookresearch/detectron2) from Facebook AI Research in object detection and instance segmentation tasks across 10 settings, each setting consisting of several cropped versions of the same image. For each image, one singular object is identified as the target object, and the model's prediction of its position and confidence rating are the outcome variables used to create the NERO plots and evaluate model robustness. To ensure that images used for evaluation are not present in the training data, the 10 settings are chosen from my own camera roll.

Cropping:

Crucially, the images are selected such that the object of interest is far enough away from the edges of the image so that all croppings keep the whole image in the frame.
