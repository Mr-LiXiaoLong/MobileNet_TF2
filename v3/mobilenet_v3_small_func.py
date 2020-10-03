#!/usr/bin/env python
# coding: utf-8

# mobile_v3_small_func.py

"""
MobileNet v3 models for Keras.

The following table describes the performance of MobileNets:
------------------------------------------------------------------------
MACs stands for Multiply Adds

| Classification Checkpoint| MACs(M)| Parameters(M)| Top1 Accuracy| Pixel1 CPU(ms)|
| [mobilenet_v3_large_1.0_224]              | 217 | 5.4 |   75.6   |   51.2   |
| [mobilenet_v3_large_0.75_224]             | 155 | 4.0 |   73.3   |   39.8   |
| [mobilenet_v3_large_minimalistic_1.0_224] | 209 | 3.9 |   72.3   |   44.1   |
| [mobilenet_v3_small_1.0_224]              | 66  | 2.9 |   68.1   |   15.8   |
| [mobilenet_v3_small_0.75_224]             | 44  | 2.4 |   65.4   |   12.8   |
| [mobilenet_v3_small_minimalistic_1.0_224] | 65  | 2.0 |   61.9   |   12.2   |


Predicted: [[('n02504458', 'African_elephant', 0.86706144)]]
Model: "mobilenetv2_1.00_224"

Modify some lines of code to comply with TensorFlow 2.2 and Keras 2.4.3 based on the 
script publised by Francois Cholett. 

The Mobilenet weights for all 6 models are obtained and translated from the Tensorflow 
checkpoints from TensorFlow checkpoints found [here]
(https://github.com/tensorflow/models/tree/master/research/
slim/nets/mobilenet/README.md).

# Reference
This file contains building code for MobileNetV3, based on [Searching for MobileNetV3]
(https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)
"""


import os
import warnings
import numpy as np
import tensorflow as tf 

from keras.preprocessing import image
from keras.layers import Add, Input, ReLU, Dropout, Dense, Flatten, Reshape, Multiply, Activation, Conv2D, \
    Softmax, DepthwiseConv2D, BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.models import Model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs

from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from imagenet_utils import _obtain_input_shape


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


BASE_WEIGHT_PATH = ('https://github.com/DrSlink/mobilenet_v3_keras/'
                    'releases/download/v1.0/')
WEIGHTS_HASHES = {
    'large_224_0.75_float': (
        '765b44a33ad4005b3ac83185abf1d0eb',
        'c256439950195a46c97ede7c294261c6'),
    'large_224_1.0_float': (
        '59e551e166be033d707958cf9e29a6a7',
        '12c0a8442d84beebe8552addf0dcb950'),
    'large_minimalistic_224_1.0_float': (
        '675e7b876c45c57e9e63e6d90a36599c',
        'c1cddbcde6e26b60bdce8e6e2c7cae54'),
    'small_224_0.75_float': (
        'cb65d4e5be93758266aa0a7f2c6708b7',
        'c944bb457ad52d1594392200b48b4ddb'),
    'small_224_1.0_float': (
        '8768d4c2e7dee89b9d02b2d03d65d862',
        '5bec671f47565ab30e540c257bba8591'),
    'small_minimalistic_224_1.0_float': (
        '99cd97fb2fcdad2bf028eb838de69e37',
        '1efbf7e822e03f250f45faa3c6bbe156'),
}


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


def relu(x):
    return ReLU()(x)


def hard_sigmoid(x):
    return ReLU(6.)(x+3.) * (1./6.)


def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor/2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = GlobalAveragePooling2D(name=prefix+'squeeze_excite/AvgPool')(inputs)
    if K.image_data_format() == 'channels_first':
        x = Reshape((filters,1,1))(x)
    else:
        x = Reshape((1,1,filters))(x)
    x = Conv2D(_depth(filters*se_ratio), kernel_size=1, padding='same', 
               name=prefix + 'squeeze_excite/Conv')(x)
    x = ReLU(name=prefix+'squeeze_excite/Relu')(x)
    x = Conv2D(filters, kernel_size=1, padding='same', 
               name=prefix + 'squeeze_excite/Conv_1')(x)
    x = Activation(hard_sigmoid)(x)
    if K.backend() == 'theano':
        # For the Theano backend, make the excitation weights broadcastable explicitly.
        x = Lambda(lambda br: K.pattern_broadcast(br, [True,True,True,False]),
                   output_shape=lambda input_shape: input_shape, 
                   name=prefix + 'squeeze_excite/broadcast')(x)
    x = Multiply(name=prefix+'squeeze_excite/Mul')([inputs,x])

    return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride, 
                        se_ratio, activation, block_id):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = K.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = Conv2D(_depth(infilters*expansion), kernel_size=1, padding='same', 
                          use_bias=False, name=prefix + 'expand')(x)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, 
                               name=prefix+'expand/BatchNorm')(x)
        x = Activation(activation)(x)

    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(K, x, kernel_size), 
                          name=prefix+'depthwise/pad')(x)
    x = DepthwiseConv2D(kernel_size, strides=stride, padding='same' if stride == 1 else 'valid', 
                        use_bias=False, name=prefix+'depthwise')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, 
                           name=prefix+'depthwise/BatchNorm')(x)
    x = Activation(activation)(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters*expansion), se_ratio, prefix)

    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False, 
               name=prefix+'project')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, 
                           name=prefix+'project/BatchNorm')(x)

    if stride == 1 and infilters == filters:
        x = Add(name=prefix+'Add')([shortcut,x])

    return x


def MobileNetV3(stack_fn, last_point_ch, input_shape=None, alpha=1.0, model_type='large',
                minimalistic=False, include_top=True, weights='imagenet', input_tensor=None,
                num_classes=1000, pooling=None, dropout_rate=0.2, **kwargs):
    # Instantiate the MobileNetV3 architecture.
    """
    # Arguments
        stack_fn: a function that returns output tensor for the stacked residual blocks.
        last_point_ch: number channels at the last layer (before top)
        input_shape: optional tuple such as (224,224,3) or infer input_shape from 
            an input_tensor; If selecting include both input_tensor and input_shape,
            input_shape will be used if matching or throwing  an error.
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases filters # in each layer.
            - If `alpha` > 1.0, proportionally increases filters # in each layer.
            - If `alpha` = 1, default filters # from the paper used at each layer.
        model_type: MobileNetV3 is defined as two models: large and small. .
        minimalistic: Also contains the minimalistic models. 
        include_top: whether to include the FC layer at the top of the network.
        weights: `None` (random initialization), 'imagenet' or the path to any weights. 
        input_tensor: optional Keras tensor (output of `layers.Input()`)
        pooling: Optional mode for feature extraction when `include_top` is `False`.
            - `None`: the output of model is the 4D tensor of the last conv layer 
            - `avg` means global average pooling and the output as a 2D tensor.
            - `max` means global max pooling will be applied.
        num_classes: specified if `include_top` is True
        num_classes: specified if `include_top` is True
        dropout_rate: fraction of the input units to drop on the last layer
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights` or invalid input shape.
        RuntimeError: run the model with a backend without support separable conv
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

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

        if K.is_keras_tensor(input_tensor):
            if K.image_data_format() == 'channels_last':
                rows = K.int_shape(input_tensor)[1]
                cols = K.int_shape(input_tensor)[2]
                input_shape = (cols, rows, 3)
            else: 
                rows = K.int_shape(input_tensor)[2]
                cols = K.int_shape(input_tensor)[3]
                input_shape = (3, cols, rows)

    # If input_shape is None and input_tensor is None using standart shape
    if input_shape is None and input_tensor is None:
        input_shape = (None, None, 3)

    if K.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if rows and cols and (rows<32 or cols<32):
        raise ValueError('Input size must be at least 32x32; got `input_shape=' +
                         str(input_shape) + '`')
    if weights == 'imagenet':
        if minimalistic is False and alpha not in [0.75, 1.0] \
                or minimalistic is True and alpha != 1.0:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of `0.75`, `1.0` for non minimalistic'
                             ' or `1.0` for minimalistic only.')

        if rows != cols or rows != 224:
            warnings.warn('`input_shape` is undefined or non-square, '
                          'or `rows` is not 224.'
                          ' Weights for input shape (224,224) will be'
                          ' loaded as the default.')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = -1 if K.image_data_format() == 'channels_last' else 1

    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25

    x = ZeroPadding2D(padding=correct_pad(K,img_input,3), name='Conv_pad')(img_input)
    x = Conv2D(16, kernel_size=3, strides=(2,2), padding='valid', use_bias=False, name='Conv')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm')(x)
    x = Activation(activation)(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = _depth(K.int_shape(x)[channel_axis]*6)

    # Increase the output channels # if the width multiplier > 1.0
    if alpha > 1.0: # No alpha applied to last conv as stated in the paper
        last_point_ch = _depth(last_point_ch * alpha)

    x = Conv2D(last_conv_ch, kernel_size=1, padding='same', use_bias=False, name='Conv_1')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_1/BatchNorm')(x)
    x = Activation(activation)(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        if channel_axis == 1:
            x = Reshape((last_conv_ch,1,1))(x)
        else:
            x = Reshape((1,1,last_conv_ch))(x)

        x = Conv2D(last_point_ch, kernel_size=1, padding='same', name='Conv_2')(x)
        x = Activation(activation)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        x = Conv2D(num_classes, kernel_size=1, padding='same', name='Logits')(x)
        x = Flatten()(x)
        x = Softmax(name='Predictions/Softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure the model considers any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Build the model.
    model = Model(inputs, x, name='MobilenetV3'+model_type)

    # Load the weights.
    if weights == 'imagenet':
        model_name = "{}{}_224_{}_float".format(
            model_type, '_minimalistic' if minimalistic else '', str(alpha))
        if include_top:
            file_name = 'weights_mobilenet_v3_' + model_name + '.h5'
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = 'weights_mobilenet_v3_' + model_name + '_no_top.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = get_file(file_name, BASE_WEIGHT_PATH+file_name,
                                cache_subdir='models', file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def MobileNetV3Small(input_shape=None, alpha=1.0, minimalistic=False, include_top=True,
                     weights='imagenet', input_tensor=None, num_classes=1000,
                     pooling=None, dropout_rate=0.2, **kwargs):

    def stack_fn(x, kernel, activation, se_ratio):

        def depth(d):

            return _depth(d*alpha)

        x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
        x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
        x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
        x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
        x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10)

        return x

    return MobileNetV3(stack_fn, 1024, input_shape, alpha, 'small', minimalistic, include_top,
                       weights, input_tensor, num_classes, pooling, dropout_rate, **kwargs)


def MobileNetV3Large(input_shape=None, alpha=1.0, minimalistic=False, include_top=True,
                     weights='imagenet', input_tensor=None, num_classes=1000, 
                     pooling=None, dropout_rate=0.2, **kwargs):

    def stack_fn(x, kernel, activation, se_ratio):

        def depth(d):

            return _depth(d * alpha)

        x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
        x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5)
        x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
        x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11)
        x = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio,
                                activation, 12)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio,
                                activation, 13)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio,
                                activation, 14)
        
        return x

    return MobileNetV3(stack_fn, 1280, input_shape, alpha, 'large', minimalistic, include_top,
                       weights, input_tensor, num_classes, pooling, dropout_rate, **kwargs)


# -setattr(MobileNetV3Small, '__doc__', MobileNetV3.__doc__)
# -setattr(MobileNetV3Large, '__doc__', MobileNetV3.__doc__)


if __name__ == '__main__':

    model = MobileNetV3Small(include_top=True, weights='imagenet')
    # -model = MobileNetV3Small(include_top=False, weights='imagenet')
 
    model.summary()
