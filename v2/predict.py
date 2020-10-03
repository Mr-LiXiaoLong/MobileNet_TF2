#!/usr/bin/env python
# coding: utf-8

# predict.py

"""
MobileNet v2 models for Keras.

MobileNetV2 is similar to the original MobileNet, except that it uses inverted residual 
blocks with bottlenecking features. It has a drastically lower parameter count than the 
original MobileNet. MobileNets support any input size greater than 32 x 32, with larger 
image sizes offering better performance.

$ python predict.py

The number of parameters and number of multiply-adds can be modified by using the `alpha` 
parameter, which increases/decreases the number of filters in each layer. By altering the 
image size and `alpha` parameter, all 22 models from the paper can be built, with ImageNet 
weights provided.

The weights for all 16 models are obtained and translated from the Tensorflow checkpoints
from TensorFlow checkpoints found [here]
(https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md).

# Reference
This file contains building code for MobileNetV2, based on
[MobileNetV2: Inverted Residuals and Linear Bottlenecks]
(https://arxiv.org/abs/1801.04381) (CVPR 2018)

Comparing this model to the existing Tensorflow model can be found at [mobilenet_v2_keras]
(https://github.com/JonathanCMitchell/mobilenet_v2_keras)
"""

import numpy as np
import tensorflow as tf 
from keras.preprocessing import image
from mobilenet_v2_func import MobileNetV2
from keras.applications.imagenet_utils import decode_predictions


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def preprocess_input(x):
    # Process any given image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    output = np.multiply(x, 2.0)

    return output


if __name__ == '__main__':

    for r in [96, 128, 160, 192, 224]: # r means rows 
        for a in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]: # a means alpha 
            if r == 224:
                model = MobileNetV2(include_top=True, weights='imagenet', 
                                    input_shape=(r,r,3), alpha=a)
                img_path = '/home/mike/Documents/keras_mobilenet/v2/elephant.jpg'
                img = image.load_img(img_path, target_size=(r,r))
                output = preprocess_input(img)
                print('Input image shape:', output.shape)

                preds = model.predict(output)
                print(np.argmax(preds))
                print('Predicted:', decode_predictions(preds,1))

            model = MobileNetV2(include_top=False, weights='imagenet')

            model.summary()