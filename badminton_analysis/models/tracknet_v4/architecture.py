"""TrackNetV4 Type-A/Type-B architecture using Keras 3 portable ops.

Adapted from the upstream TensorFlow implementation so the saved ``.keras``
weights can run on Good-Badminton's existing PyTorch backend.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import ops
from keras.layers import (
    Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, UpSampling2D,
    concatenate, Reshape, Layer
)
from keras.models import Model

# Utility functions for tensor rearrangement
def rearrange_tensor(input_tensor, order):
    """
    Rearranges the dimensions of a tensor according to the specified order.

    Args:
        input_tensor (tf.Tensor): The input tensor to rearrange.
        order (str): A string specifying the desired dimension order (e.g., 'BCHWT').

    Returns:
        tf.Tensor: The rearranged tensor.
    """
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all([dim in order for dim in "BCHWT"]), "Order must contain all of BCHWT"
    return ops.transpose(input_tensor, [order.index(dim) for dim in "BTCHW"])

def reverse_rearrange_tensor(input_tensor, order):
    """
    Reverses the rearrangement of a tensor to its original order.

    Args:
        input_tensor (tf.Tensor): The tensor to reverse rearrange.
        order (str): A string specifying the original dimension order (e.g., 'BCHWT').

    Returns:
        tf.Tensor: The tensor in the original order.
    """
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    return ops.transpose(input_tensor, ["BTCHW".index(dim) for dim in order])

class MotionPromptLayer(Layer):
    """
    A custom Keras layer that generates attention maps from video sequences using attention maps.
    This TensorFlow implementation has been customized based on https://github.com/q1xiangchen/VMPs/.
    """
    def __init__(self, penalty_weight=0.0, **kwargs):
        super(MotionPromptLayer, self).__init__(**kwargs)
        # Default configs
        self.input_permutation = "BTCHW"
        self.input_color_order = "RGB"
        self.color_map = {'R': 0, 'G': 1, 'B': 2}
        self.gray_scale = ops.convert_to_tensor([0.299, 0.587, 0.114], dtype="float32")

        # Power normalization parameters
        self.pn = power_normalization
        self.a = self.add_weight(shape=(), initializer=keras.initializers.Constant(0.1), trainable=True, name='a')
        self.b = self.add_weight(shape=(), initializer=keras.initializers.Constant(0.0), trainable=True, name='b')
        
        # Temporal attention variation regularization parameter
        self.lambda1 = penalty_weight

    def call(self, video_seq):
        loss = ops.convert_to_tensor(0.0, dtype="float32")

        # Rearrange the input tensor to BTCHW (if different)
        video_seq = rearrange_tensor(video_seq, self.input_permutation)

        # Normalize the input tensor back to [0, 1]
        norm_seq = video_seq * 0.225 + 0.45

        # Transfer the input tensor to grayscale 
        idx_list = [self.color_map[idx] for idx in self.input_color_order]
        gray_scale_tensor = ops.take(self.gray_scale, idx_list)
        weights = ops.cast(gray_scale_tensor, dtype=norm_seq.dtype)
        grayscale_video_seq = ops.einsum("btcwh,c->btwh", norm_seq, weights)

        # Frame difference
        B, T, H, W = grayscale_video_seq.shape
        frame_diff = grayscale_video_seq[:, 1:] - grayscale_video_seq[:, :-1]

        # Power normalization
        attention_map = self.pn(frame_diff, self.a, self.b)
        norm_attention = ops.expand_dims(attention_map, axis=2)

        if B is None:
            B = 1

        # Temporal attention variation regularization
        if self.trainable:
            temp_diff = norm_attention[:, 1:] - norm_attention[:, :-1]
            temporal_loss = ops.sum(ops.square(temp_diff)) / (H * W * (T - 2) * B)
            loss = self.lambda1 * temporal_loss
            self.add_loss(loss)

        return attention_map, loss

# Power normalization function
def power_normalization(input, a, b):
    return 1 / (1 + ops.exp(-(5 / (0.45 * ops.abs(ops.tanh(a)) + 1e-1)) * (ops.abs(input) - 0.6 * ops.tanh(b))))

# Motion fusion layers
class FusionLayerTypeA(Layer):
    """
    A Keras layer that incorporates motion using attention maps - version 1.
    """
    def call(self, inputs):
        feature_map, attention_map = inputs
        output_1 = feature_map[:, 0, :, :]
        output_2 = feature_map[:, 1, :, :] * attention_map[:, 0, :, :]
        output_3 = feature_map[:, 2, :, :] * attention_map[:, 1, :, :]

        return ops.stack([output_1, output_2, output_3], axis=1)

class FusionLayerTypeB(Layer):
    """
    A Keras layer that incorporates motion using attention maps - version 2.
    """
    def call(self, inputs):
        feature_map, attention_map = inputs
        output_1 = feature_map[:, 0, :, :] * attention_map[:, 0, :, :]
        output_2 = feature_map[:, 1, :, :] * ((attention_map[:, 0, :, :] + attention_map[:, 1, :, :])/2)
        output_3 = feature_map[:, 2, :, :] * attention_map[:, 1, :, :]

        return ops.stack([output_1, output_2, output_3], axis=1)

def TrackNetV4(input_height, input_width, fusion_layer_type="TypeA"):
    """
    Builds the TrackNetV4 model that incorporates motion prompts using a multi-layer CNN.

    Args:
        input_height (int): The height of the input.
        input_width (int): The width of the input.

    Returns:
        Model: A Keras model instance.
    """
    if fusion_layer_type == "TypeA":
        fusion_layer = FusionLayerTypeA()
    elif fusion_layer_type == "TypeB":
        fusion_layer = FusionLayerTypeB()
    else:
        raise ValueError("Unknown Motion Fusion Type")

    imgs_input = Input(shape=(9, input_height, input_width))
    motion_input = Reshape((3, 3, input_height, input_width))(imgs_input)
    
    # Motion prompt layer integration
    residual_maps, _ = MotionPromptLayer()(motion_input)

    # Layer 1
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(imgs_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 2
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x1 = BatchNormalization()(x)

    # Layer 3
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x1)

    # Layer 4
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 5
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x2 = BatchNormalization()(x)

    # Layer 6
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x2)

    # Layer 7
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 8
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 9
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x3 = BatchNormalization()(x)

    # Layer 10
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x3)

    # Layer 11
    x = Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 12
    x = Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 13
    x = Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 14
    x = concatenate([UpSampling2D((2, 2), data_format='channels_first')(x), x3], axis=1)

    # Layer 15
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 16
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 17
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 18
    x = concatenate([UpSampling2D((2, 2), data_format='channels_first')(x), x2], axis=1)

    # Layer 19
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 20
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 21
    x = concatenate([UpSampling2D((2, 2), data_format='channels_first')(x), x1], axis=1)

    # Layer 22
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 23
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 24
    x = Conv2D(3, (1, 1), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = fusion_layer([x, residual_maps])
    x = Activation('sigmoid')(x)

    # Model creation
    model = Model(inputs=imgs_input, outputs=x)
    return model
