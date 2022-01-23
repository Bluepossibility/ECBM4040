import numpy as np
import tensorflow as tf
from .CustomLayers import Spectral_Conv_Layer
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam


class CNN_Spectral_Param():
	"""
	This class builds and trains the generic and deep CNN architectures
	with and without spectral pooling to make comparison and derive the conclusion
	"""
	def __init__(self, num_out = 10, architect = 'generic', use_spectral_params = True, kernel_size = 3,
				 l2_norm = 0.001, filters = 128, learning_rate = 1e-6):
		"""
		The initialization of parameters in class CNN_Spectral_Param()
		:param num_output: Number of classes to predict for output_
		:param arcchitecture: Defines which architecture to build (either deep or generic)
		:param use_spectral_params: Flag to turn spectral parameterization on and off
		:param kernel_size: size of convolutional kernel
		:param l2_norm: Scale factor for l2 norm of CNN weights when calculating l2 loss
		:param learning_rate: Learning rate for Adam AdamOptimizer
		:param data_format: Format of input images, either 'NHWC' or 'NCHW'
		:param random_seed: Seed for initializers to create reproducable results
		"""
		self.num_out = num_out
		self.architect = architect
		self.use_spectral_params = use_spectral_params
		self.kernel_size = kernel_size
		self.l2_norm = l2_norm
		self.filters = filters
		self.learning_rate = learning_rate


	def _build_generic_model(self, use_spectral_params):
		"""
		Builds the generic architecture of CNN with and withou spectal pooling
		This architecture is a pair of convolution and maxpooling layers,
		or the spectral convolution and maxpooling layer followed by three
		fully-connected layers and a softmax or golobal averaging layers.
		:param use_spectral_params: define if use spectral convolution layers
		:return: the tensorflow keras model
		"""
		if use_spectral_params == True:

			model = Sequential()
			# spectral_conv1
			model.add(Spectral_Conv_Layer(self.filters,self.kernel_size, input_shape=(32, 32, 3)))
			# maxpool1
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# spectral_conv2
			model.add(tf.keras.layers.Conv2D(filters=192, padding='same', kernel_size=self.kernel_size,
												   activation="relu", trainable=True))
			# maxpool2
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# flatten
			model.add(tf.keras.layers.Flatten())
			# dense1
			model.add(tf.keras.layers.Dense(1024, activation='relu',
										   activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			# dense2
			model.add(tf.keras.layers.Dense(512, activation='relu',
										   activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			# dense3
			model.add(tf.keras.layers.Dense(self.num_out, activation='relu',
										   activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			model.add(tf.keras.layers.Softmax())
			for layers in model.layers:
					layers.trainable = True
			optimizer = Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
# 			optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.0, nesterov=False)
			model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
			return model

		elif use_spectral_params == False:

			model = Sequential()
			# generic_conv1
			model.add(tf.keras.layers.Conv2D(filters=96, padding='same', kernel_size=self.kernel_size,
												   activation="relu", trainable=True,input_shape=(32, 32, 3)))
			# maxpool1
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# generic_conv2
			model.add(tf.keras.layers.Conv2D(filters=192, padding='same', kernel_size=self.kernel_size,
												   activation="relu", trainable=True))
			# maxpool2
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# flatten
			model.add(tf.keras.layers.Flatten())
			# dense1
			model.add(tf.keras.layers.Dense(1024, activation='relu',
											activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			# dense2
			model.add(tf.keras.layers.Dense(512, activation='relu',
											activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			# dense3
			model.add(tf.keras.layers.Dense(256, activation='relu',
											activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			model.add(tf.keras.layers.Dense(self.num_out, activation='softmax'))

			for layers in model.layers:
					layers.trainable = True
			optimizer = Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
			model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
			return model


	def _build_deep_model(self, use_spectral_params):
		"""
		Builds the deep architecture of CNN with and without spectral convolution layer.
		This architecture is defined as follows:
			back-to-back spectral convolutions, back-to-back spectral convolutions, max-pool,
			back-to-back convolutions, max-pool back-to-back 10-filter convolutions,
			and a global averaging layer(for both scenario)
		:param use_spectral_params: define if use spectral convolution layers
		:return: The tensorflow keras model
		"""
		if use_spectral_params == True:

			model = Sequential()
			# spectral_conv1
			model.add(Spectral_Conv_Layer(self.filters-36, self.kernel_size, input_shape=(32, 32, 3)))
			# spectral_conv2
			model.add(tf.keras.layers.Conv2D(filters=96, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# spectral_conv3
			model.add(tf.keras.layers.Conv2D(filters=96, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# spectral_conv4
			model.add(Spectral_Conv_Layer(self.filters - 36, self.kernel_size))
			# maxpool1
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# spectral_conv5
			model.add(tf.keras.layers.Conv2D(filters=192, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# spectral_conv6
			model.add(tf.keras.layers.Conv2D(filters=192, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# maxpool2
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# spectral_conv7
			model.add(tf.keras.layers.Conv2D(filters= self.num_out, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# global average pooling
			model.add(tf.keras.layers.GlobalAveragePooling2D())
			# print(type(model.layers[-2].output))
			# flatten
			# model.add(tf.keras.layers.Flatten())
			# # dense1
			# model.add(tf.keras.layers.Dense(1024, activation='relu',
			# 								activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			# # dense2
			# model.add(tf.keras.layers.Dense(256, activation='relu',
			# 								activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			# model.add(tf.keras.layers.Dense(self.num_out, activation='softmax'))
			for layers in model.layers:
					layers.trainable = True
			optimizer = Adam(learning_rate=self.learning_rate, decay=0.01)
			model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
			return model

		elif use_spectral_params == False:

			model = Sequential()
			# deep_conv1
			model.add(tf.keras.layers.Conv2D(filters=96, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True, input_shape=(32, 32, 3)))
			# deep_conv2
			model.add(tf.keras.layers.Conv2D(filters=96, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# # Batch normalization
			# model.add(BatchNormalization())
			# maxpool1
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# deep_conv3
			model.add(tf.keras.layers.Conv2D(filters=192, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# deep_conv4
			model.add(tf.keras.layers.Conv2D(filters=192, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# # Batch normalization
			# model.add(BatchNormalization())
			# maxpool2
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
# 			# deep_conv5
# 			model.add(tf.keras.layers.Conv2D(filters=512, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
# 											 activation="relu", trainable=True))
# 			# deep_conv6
# 			model.add(tf.keras.layers.Conv2D(filters=512, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
# 											 activation="relu", trainable=True))
# 			# # Batch normalization
# 			# model.add(BatchNormalization())
# 			# maxpool3
# 			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

			# # Batch normalization
			# model.add(BatchNormalization())
			# maxpool4
			# model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

			# # Batch normalization
			# model.add(BatchNormalization())
			# # maxpool5
			# model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# deep_conv7
			model.add(tf.keras.layers.Conv2D(filters=10, padding='same', kernel_size=self.kernel_size,strides=(1, 1),
											 activation="relu", trainable=True))
			# maxpool6
			# model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# global average pooling
			model.add(tf.keras.layers.GlobalAveragePooling2D())
			# # flatten
			# model.add(tf.keras.layers.Flatten())
			# # dense1
			# model.add(tf.keras.layers.Dense(128, activation='relu',
			# 								activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			# # dense2
			# model.add(tf.keras.layers.Dense(self.num_out, activation='softmax'))
			for layers in model.layers:
					layers.trainable = True
			optimizer = Adam(learning_rate=self.learning_rate, decay=0.01)
			model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
			return model