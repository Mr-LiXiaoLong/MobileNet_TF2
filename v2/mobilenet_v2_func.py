#!/usr/bin/env python
# coding: utf-8

# mobilenet_v2_func.py

"""
MobileNet v2 models for Keras.

MobileNetV2 is a general architecture for multiple use cases.Depending on the use case, 
it can use different input layer size and different width factors. This allows different 
width models to reduce the number of multiply-adds and thereby reduce inference cost on 
mobile devices.

MobileNetV2 is similar to the original MobileNet, except that it uses inverted residual 
blocks with bottlenecking features. It has a drastically lower parameter count than the 
original MobileNet. MobileNets support any input size greater than 32 x 32, with larger 
image sizes offering better performance.

The number of parameters and number of multiply-adds can be modified by using the `alpha` 
parameter, which increases/decreases the number of filters in each layer. By altering the 
image size and `alpha` parameter, all 22 models from the paper can be built, with ImageNet 
weights provided.

The paper demonstrates the performance of MobileNets using `alpha` values of 1.0 (called 
100 % MobileNet), 0.35, 0.5, 0.75, 1.0, 1.3, and 1.4 For each of these `alpha` values, 
weights for 5 different input image sizes are provided (224, 192, 160, 128, and 96).

Modify some lines of code to comply with TensorFlow 2.2 and Keras 2.4.3 based on the script 
publised by Francois Cholett. 

The following table describes the performance of MobileNet on various input sizes:
------------------------------------------------------------------------
MACs stands for Multiply Adds

 Classification Checkpoint| MACs (M) | Parameters (M)| Top 1 Accuracy| Top 5 Accuracy
--------------------------|------------|---------------|---------|----|-------------
| [mobilenet_v2_1.4_224]  | 582 | 6.06 |          75.0 | 92.5 |
| [mobilenet_v2_1.3_224]  | 509 | 5.34 |          74.4 | 92.1 |
| [mobilenet_v2_1.0_224]  | 300 | 3.47 |          71.8 | 91.0 |
| [mobilenet_v2_1.0_192]  | 221 | 3.47 |          70.7 | 90.1 |
| [mobilenet_v2_1.0_160]  | 154 | 3.47 |          68.8 | 89.0 |
| [mobilenet_v2_1.0_128]  | 99  | 3.47 |          65.3 | 86.9 |
| [mobilenet_v2_1.0_96]   | 56  | 3.47 |          60.3 | 83.2 |
| [mobilenet_v2_0.75_224] | 209 | 2.61 |          69.8 | 89.6 |
| [mobilenet_v2_0.75_192] | 153 | 2.61 |          68.7 | 88.9 |
| [mobilenet_v2_0.75_160] | 107 | 2.61 |          66.4 | 87.3 |
| [mobilenet_v2_0.75_128] | 69  | 2.61 |          63.2 | 85.3 |
| [mobilenet_v2_0.75_96]  | 39  | 2.61 |          58.8 | 81.6 |
| [mobilenet_v2_0.5_224]  | 97  | 1.95 |          65.4 | 86.4 |
| [mobilenet_v2_0.5_192]  | 71  | 1.95 |          63.9 | 85.4 |
| [mobilenet_v2_0.5_160]  | 50  | 1.95 |          61.0 | 83.2 |
| [mobilenet_v2_0.5_128]  | 32  | 1.95 |          57.7 | 80.8 |
| [mobilenet_v2_0.5_96]   | 18  | 1.95 |          51.2 | 75.8 |
| [mobilenet_v2_0.35_224] | 59  | 1.66 |          60.3 | 82.9 |
| [mobilenet_v2_0.35_192] | 43  | 1.66 |          58.2 | 81.2 |
| [mobilenet_v2_0.35_160] | 30  | 1.66 |          55.7 | 79.1 |
| [mobilenet_v2_0.35_128] | 20  | 1.66 |          50.8 | 75.0 |
| [mobilenet_v2_0.35_96]  | 11  | 1.66 |          45.5 | 70.4 |

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


import warnings
import tensorflow as tf 
from keras.models import Model

from keras import backend as K
from keras.layers import Add, Input, ReLU, Dropout, Dense, Conv2D, DepthwiseConv2D, \
   BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D


from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from imagenet_utils import _obtain_input_shape


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Give the base_weights_path to v1.1
BASE_WEIGHT_PATH = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/'
                    'releases/download/v1.1/')


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor/2) // divisor * divisor)
    # Make sure the round-down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(backend, inputs, kernel_size):
    # Return a tuple for zero-padding for 2D convolution with downsampling.
    """
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 1 if K.image_data_format() == 'channels_last' else 2
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0]%2, 1 - input_size[1]%2)

    correct = (kernel_size[0]//2, kernel_size[1]//2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    channel_axis = -1 if K.image_data_format() == 'channels_last' else 1
    in_channels = K.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters*alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand the in_channels 
        x = Conv2D(expansion*in_channels, kernel_size=1, padding='same', use_bias=False, 
                   activation=None, name=prefix+'expand')(x)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, 
                               name=prefix+'expand_BN')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Apply the Depthwise Conv. 
    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(K,x,3), name=prefix+'pad')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, 
                        padding='same' if stride == 1 else 'valid', 
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, 
                           name=prefix+'depthwise_BN')(x)

    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, 
               activation=None, name=prefix+'project')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, 
                           name=prefix+'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix+'add')([inputs,x])

    return x


def MobileNetV2(input_shape=None, alpha=1.0, include_top=True, weights='imagenet',
                input_tensor=None, pooling=None, num_classes=1000, **kwargs):
    # Instantiate the MobileNetV2 architecture.
    """
    # Arguments
        input_shape: optional tuple such as (224, 224, 3) or infer input_shape from 
            an input_tensor; If selecting include both input_tensor and input_shape,
            input_shape will be used if matching or throwing  an error.
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases filters # in each layer.
            - If `alpha` > 1.0, proportionally increases filters # in each layer.
            - If `alpha` = 1, default filters # from the paper used at each layer.
        include_top: whether to include the FC layer at the top of the network.
        weights: `None` (random initialization)or 'imagenet'. 
        input_tensor: optional Keras tensor (output of `layers.Input()`)
        pooling: Optional mode for feature extraction when `include_top` is `False`.
            - `None`: the output of model is the 4D tensor of the last conv layer 
            - `avg` means global average pooling and the output as a 2D tensor.
            - `max` means global max pooling will be applied.
        num_classes: specified if `include_top` is True
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights` or invalid input shape.
        RuntimeError: run the model with a backend without support separable conv.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` '
                         '(random initialization) or `imagenet` or the'
                         'path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and num_classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `num_classes` should be 1000')

    # Determine the proper input shape/default size if both input_shape and input_tensor 
    # are used while being matched. 
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = K.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = K.is_keras_tensor(
                    get_source_inputs(input_tensor))
            except ValueError:
                raise ValueError('input_tensor: ', input_tensor,
                                 'is not type input_tensor')

        if is_input_t_tensor:
            if K.image_data_format == 'channels_last':
                if K.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape,
                                     'and input_tensor: ', input_tensor,
                                     'do not meet the same shape requirements')
            else: 
                if K.int_shape(input_tensor)[1] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape,
                                     'and input_tensor: ', input_tensor,
                                     'do not meet the same shape requirements')
        else:
            raise ValueError('input_tensor specified: ', input_tensor,
                             'is not a keras tensor')

    # Infer the shape from input_tensor if input_shape as None
    if input_shape is None and input_tensor is not None:

        try:
            K.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError('input_tensor: ', input_tensor,
                             'is type: ', type(input_tensor),
                             'which is not a valid type')

        if input_shape is None and not K.is_keras_tensor(input_tensor):
            default_size = 224
        elif input_shape is None and K.is_keras_tensor(input_tensor):
            if K.image_data_format() == 'channels_last':
                rows = K.int_shape(input_tensor)[1]
                cols = K.int_shape(input_tensor)[2]
            else: 
                rows = K.int_shape(input_tensor)[2]
                cols = K.int_shape(input_tensor)[3]
            if rows == cols and rows in [96, 128, 160, 192, 224]:
                default_size = rows
            else:
                default_size = 224

    # If input_shape is None and no input_tensor
    elif input_shape is None:
        default_size = 224

    # Assume the default size if input_shape as not None
    else:
        if K.image_data_format() == 'channels_last':
            rows = input_shape[0]
            cols = input_shape[1]
        else: 
            rows = input_shape[1]
            cols = input_shape[2]
        if rows == cols and rows in [96, 128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if K.image_data_format() == 'channels_last':    
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of `0.35`, `0.50`, `0.75`, '
                             '`1.0`, `1.3` or `1.4` only.')

        if rows != cols or rows not in [96, 128, 160, 192, 224]:
            rows = 224
            warnings.warn('`input_shape` is undefined or non-square, '
                          'or `rows` is not in [96, 128, 160, 192, 224].'
                          ' Weights for input shape (224, 224) will be'
                          ' loaded as the default.')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = -1 if K.image_data_format() == 'channels_last' else 1

    first_block_filters = _make_divisible(32 * alpha, 8)

    # Call the function of correct_pad()
    x = ZeroPadding2D(padding=correct_pad(K,img_input,3), name='Conv1_pad')(img_input)
    x = Conv2D(first_block_filters, kernel_size=3, strides=(2,2), padding='valid', 
               use_bias=False, name='Conv1')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

    # Increase the output channels # if the width multiplier > 1.0
    if alpha > 1.0: # No alpha applied to last conv as stated in the paper:
        last_block_filters = _make_divisible(1280*alpha, 8)
    else:
        last_block_filters = 1280

    x = Conv2D(last_block_filters, kernel_size=1, use_bias=False, name='Conv_1')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = ReLU(6., name='out_relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure the model considers any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Build the model.
    model = Model(inputs, x, name='mobilenetv2_%0.2f_%s' % (alpha,rows))

    # Load the weights.
    if weights == 'imagenet':
        if include_top:
            model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + 
                          str(alpha) + '_' + str(rows) + '.h5')
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name, weight_path, cache_subdir='models')
        else:
            model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
                          str(alpha) + '_' + str(rows) + '_no_top' + '.h5')
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model