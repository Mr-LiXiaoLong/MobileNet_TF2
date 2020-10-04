# MobileNet_tf2

## Introduction to the Modification 

MobileNet supports any input size larger than 32 x 32, with larger image 
sizes offering better performance. The number of parameters and number of 
multiply-adds can be modified by the `alpha` parameter, which increases/
decreases the number of filters in each layer. 

By altering the image size and `alpha` parameter, all 16 models from the 
paper can be built, with ImageNet weights provided. It greatly increase
the model's scalability and performance. 

Make the necessary changes to adapt to the environment of TensorFlow 2.3, 
Keras 2.4.3, CUDA Toolkit 11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, 
write the new lines of code to replace the deprecated code.  

## Environment: 

Ubuntu 18.04 

TensorFlow 2.3

Keras 2.4.3

CUDA Toolkit 11.0 

cuDNN 8.0.1

CUDA 450.57
