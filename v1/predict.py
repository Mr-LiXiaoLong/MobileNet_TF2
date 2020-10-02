#!/usr/bin/env python
# coding: utf-8

# predict.py

"""
MobileNet v1 models for Keras.

MobileNets support any input size greater than 32 x 32, with larger image 
sizes offering better performance.The number of parameters and number of 
multiply-adds can be modified by the `alpha` parameter, which increases/
decreases the number of filters in each layer. By altering the image size 
and `alpha` parameter, all 16 models from the paper can be built, with 
ImageNet weights provided.

$ python predict.py

Predicted: [[('n02690373', 'airliner', 0.979478)]]
Model: "mobilenet_1.00_224"

The paper demonstrates the performance of MobileNets using `alpha` values 
of 1.0(called 100 % MobileNet), 0.75, 0.5 and 0.25. For each of the `alpha`
values, weights for 4 different input image sizes are provided (224, 192, 
160, 128). Modify some lines of code to comply with TensorFlow 2.2 and Keras
2.4.3 based on the code contributed by Francois Cholett/Somshubra Majumdar

https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md

# Reference
- MobileNets: Efficient Convolutional Neural Networks for Mobile Vision 
  Applications
- https://arxiv.org/pdf/1704.04861.pdf
"""

import numpy as np
import tensorflow as tf 
from keras.preprocessing import image
from mobilenet_v1_obj import MobileNet
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
    for r in [128, 160, 192, 224]:
        for a in [0.25, 0.50, 0.75, 1.0]:
            if r == 224:
                model = MobileNet(include_top=True, weights='imagenet', 
                                  input_shape=(r,r,3), alpha=a)
                img_path = '/home/mike/Documents/keras_mobilenet/v1/plane.jpg'
                img = image.load_img(img_path, target_size=(r, r))
                output = preprocess_input(img)
                print('Input image shape:', output.shape)

                preds = model.predict(output)
                print(np.argmax(preds))
                print('Predicted:', decode_predictions(preds, 1))

            model = MobileNet(include_top=False, weights='imagenet')
            model.summary()
