from .CustomLayers import Spectral_Conv_Layer, Spectral_Pool_Layer
import tensorflow as tf

class CNN_Spectral_Pool(tf.keras.Model):
    """CNN with spectral pooling layers and options for convolution layers."""
    def __init__(self,
                 M,
                 l2_norm,
                 num_classes=10,
                 alpha=0.3,
                 beta=0.15,
                 max_num_filters=288,
                 use_parameterization = True
                 ):
        super(CNN_Spectral_Pool, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        self.num_classes = num_classes
        self.M = M  # M is the total number of convolution and spectral-pool layer-pairs
        self.alpha = alpha
        self.beta = beta
        self.max_num_filters = max_num_filters
        self.parameterization_or_not = use_parameterization
        # Define Layers to be used in this custom Keras Model-----------------------------------------------------------
        self.Spectral_Conv_Layer = Spectral_Conv_Layer
        self.Spectral_Pool_Layer = Spectral_Pool_Layer
        self.conv2d = tf.keras.layers.Conv2D(filters=64, padding='same', kernel_size=3, activation="relu", trainable=True)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1000 = tf.keras.layers.Dense(1000, activation='relu',activity_regularizer=tf.keras.regularizers.l2(l=l2_norm))
        self.dense100 = tf.keras.layers.Dense(100, activation='relu',activity_regularizer=tf.keras.regularizers.l2(l=l2_norm))
        self.dense10 = tf.keras.layers.Dense(10, activation='relu',activity_regularizer=tf.keras.regularizers.l2(l=l2_norm))
        self.global_avg = tf.keras.layers.GlobalAveragePooling2D()
        self.softmax = tf.keras.layers.Softmax()
        self.Spectral_Conv_Layer_list = []
        self.Spectral_Pool_Layer_list = []
        self.conv2d_list = []
        self.filter_size = 3
        for m in range(1, self.M + 1):  # For m from 1 to M, spectral convolution layer and spectral pool layer differs
            freq_dropout_lower_bound, freq_dropout_upper_bound = self.Freq_Dropout_Bounds(self.filter_size, m)
            num_of_filters = self.Num_of_Filters(m)
            self.conv2d_list.append(
                tf.keras.layers.Conv2D(
                    filters=num_of_filters,
                    padding='same',
                    kernel_size=3,
                    activation="relu",
                    trainable=True))
            self.Spectral_Conv_Layer_list.append(
                self.Spectral_Conv_Layer(filters=10, name='Spectral_Conv_Layer{0}'.format(m), trainable=True))
            self.Spectral_Pool_Layer_list.append(
                self.Spectral_Pool_Layer(
                out_channels=10,
                freq_dropout_lower_bound=freq_dropout_lower_bound,
                freq_dropout_upper_bound=freq_dropout_upper_bound,
                name='Spectral_Pool_Layer{0}'.format(m)))


    def Num_of_Filters(self, m):
        """
        :param m: Current layer number, no more than 6
        :return: Number of filters for CNN, no more than 288
        """
        return min(self.max_num_filters, 96 + 32 * m)

    def Freq_Dropout_Bounds(self, size, idx):
        """
        Get the bounds for frequency dropout.
        This function implements the linear parameterization of the
        probability distribution for frequency dropout in the orginal paper
        :param size: size of image in layer
        :param idx: current layer index
        :return: freq_dropout_lower_bound: The lower bound for the frequency drop off
                freq_dropout_upper_bound: The upper bound for the frequency drop off
        """
        c = self.alpha + (idx / self.M) * (self.beta - self.alpha)
        freq_dropout_lower_bound = c * (1. + size // 2)
        freq_dropout_upper_bound = (1. + size // 2)
        return freq_dropout_lower_bound, freq_dropout_upper_bound

    def call(self, input_tensor):
        # The first part of the CNN model is pairs of convolutional and spectral pooling layers.
        for m in range(self.M):  # For m from 1 to M
            if m == 0:  # It's the first layer, should have input_shape as an argument
                # x = self.conv2d_list[0](input_tensor) if not use_spectral_parameterization
                # x = self.Spectral_Conv_Layer_list[0](input_tensor) if use_spectral_parameterization
                if self.parameterization_or_not:
                    x = self.conv2d_list[0](input_tensor)
                    #print('after {0} spectral_conv'.format(m),x), used for debug
                else:
                    x = self.Spectral_Conv_Layer_list[0](input_tensor)  # spectral_conv_layer
                    #print('after {0} conv'.format(m),x), used for debug
            else:  # It's not the first layer
                # x = self.conv2d_list[m](input_tensor) if not use_spectral_parameterization
                # x = self.Spectral_Conv_Layer_list[m](input_tensor) if use_spectral_parameterization
                if self.parameterization_or_not:
                    x = self.conv2d_list[m](x)
                    #print('after {0} spectral_conv'.format(m),x),used for debug
                else:
                    x = self.Spectral_Conv_Layer_list[m](x) # spectral_conv_layer
                    #print('after {0} conv'.format(m),x), used for debug
            x = self.Spectral_Pool_Layer_list[m](x)  # spectral_pool_layer is added whatever
        ###########################################################################################
        x = self.flatten(input_tensor)
        x = self.dense1000(x)
        x = self.dense100(x)
        if self.num_classes == 10:
            x = self.dense10(x)
        x = self.softmax(x)

        return x

