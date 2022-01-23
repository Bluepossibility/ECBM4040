import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

def create_model(lr=1e-3):
    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/EfficientNetB0
    base_model=EfficientNetB0(weights='imagenet',include_top=False, input_shape=(128,128,3)) 
    x=base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    # Dense layer 1
    x=Dense(512,activation='relu')(x) 
    x = BatchNormalization()(x)
    # Dense layer 2
    x=Dense(512,activation='relu')(x) 
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    # Dense layer 3
    x=Dense(64,activation='relu')(x) 
    # Final layer with softmax activation
    preds=Dense(5,activation='softmax')(x) 
    model=Model(inputs=base_model.input,outputs=preds)

    
    for layers in model.layers:
        layers.trainable = True   
        
    optimizer = Adam(learning_rate=lr, decay=0.01)        
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    return model