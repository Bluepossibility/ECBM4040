from .SpectralPool import Common_Spectral_Pool
from .FrequencyDropout import freq_dropout_mask
import tensorflow as tf
import numpy as np
"""Leveraging Keras, custom all the layers we might need in Jupyter Notebook here"""


class Spectral_Conv_Layer(tf.keras.layers.Layer):
    """
    Leveraging Keras, custom a Spectral Convolution Layer
    """
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(Spectral_Conv_Layer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel = None

    # Use build method to add trainable parameters to the layer
    def build(self, input_shape):
        self.sample_weight = self.add_weight(shape=(self.kernel_size, self.kernel_size),
                                 initializer=tf.keras.initializers.GlorotUniform(),
                                 trainable=True)
        self.bias = self.add_weight(shape=(self.kernel_size, self.kernel_size),
                                 initializer=tf.keras.initializers.GlorotUniform(),
                                 trainable=True)

    def call(self, input_tensor):
        #print("tf.shape(input_tensor)",input_tensor.shape), used for debug
        complex_sample_weight = tf.cast(self.sample_weight, dtype=tf.complex64)
        # Calculate spectral weight based on sample weight
        fft2d_sample_weight = tf.signal.fft2d(complex_sample_weight)
        real_init = tf.math.real(fft2d_sample_weight)
        imag_init = tf.math.imag(fft2d_sample_weight)
        spectral_weight = tf.complex(real_init, imag_init)
        # Convert spectral weight back to spatial and take the real part of it
        complex_spatial_weight = tf.signal.ifft2d(spectral_weight)
        spatial_weight = tf.math.real(complex_spatial_weight)
        self.kernel = spatial_weight + self.bias
        # -----------------------------------------------------------
        self.kernel = tf.expand_dims(self.kernel, axis=-1, name=None)
        self.kernel = tf.keras.layers.Concatenate(axis=-1)([self.kernel for _ in range(input_tensor.shape[-1])])
        self.kernel = tf.expand_dims(self.kernel, axis=-1, name=None)
        self.kernel = tf.keras.layers.Concatenate(axis=-1)([self.kernel for _ in range(self.filters)])
        #print("self.kernel",self.kernel)
        # Perform the convolution based on the calculated spatial weight
        return tf.nn.conv2d(input=input_tensor, filters=self.kernel, strides=[1,1,1,1], padding='SAME')

class Spectral_Pool_Layer(tf.keras.layers.Layer):
    """
    Leveraging Keras, custom a Spectral Pooling Layer via subclassing
    """
    def __init__(self, out_channels, kernel_size=3, freq_dropout_lower_bound=None, freq_dropout_upper_bound=None, **kwargs):
        super(Spectral_Pool_Layer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.training = True
        self.freq_dropout_lower_bound = freq_dropout_lower_bound
        self.freq_dropout_upper_bound = freq_dropout_upper_bound

    def call(self, input_tensor, activation=tf.nn.relu):
        # Fast Fourier Transform 2d
        img_fft = tf.signal.fft2d(tf.cast(input_tensor, tf.complex64))
        # Truncate the spectrum
        img_truncated = Common_Spectral_Pool(img_fft, self.kernel_size)
        # Calculate output regarding the frequency drop out bound
        if (self.freq_dropout_lower_bound is not None) and (self.freq_dropout_upper_bound is not None):
            # If we are in the training phase, we need to drop all frequencies above a certain randomly determined level.
            if self.training:
                tf_random_cutoff = tf.random.uniform([], self.freq_dropout_lower_bound, self.freq_dropout_upper_bound)
                dropout_mask = freq_dropout_mask(self.kernel_size, tf_random_cutoff)
                dropout_mask = tf.expand_dims(dropout_mask, axis=-1, name=None)
                dropout_mask = tf.expand_dims(dropout_mask, axis=0, name=None)
                #print(img_truncated.shape, dropout_mask.shape), used for debug
                img_down_sampled = img_truncated[:,:,:,:] * dropout_mask

            # In the testing phase, return the truncated frequency matrix unchanged.
            else:
                img_down_sampled = img_truncated
            output = tf.math.real(tf.signal.ifft2d(img_down_sampled))
        else:
            output = tf.math.real(tf.signal.ifft2d(img_truncated))

        if activation is not None:
            return activation(output)
        else:
            #print('output type of Spectral_Pool_Layer',type(output)), used for debug
            return output

def conv_2d_layer(filters, kernel_size):
    """
    Leveraging Keras, custom a Convolution2d Layer via subclassing
    """
    conv_2d = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid")
    return conv_2d(filters, kernel_size)

def Dense_layer(units):
    """
    Leveraging Keras, custom a Dense Layer
    """
    Dense = tf.keras.layers.Dense(
        units,
        activation='relu',
        se_bias=tf.constant(True, dtype=tf.bool))
    return Dense(units)

def global_average_layer():
    """
    Leveraging Keras, custom a Global Average Pooling 2D Layer
    """
    global_average = tf.keras.layers.GlobalAveragePooling2D()
    return global_average()

