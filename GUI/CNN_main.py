import numpy as np
import os
import h5py

with open(os.path.expanduser('~')+'\\.keras\\keras.json','w') as f:
    new_settings = """{\r\n
    "epsilon": 1e-07,\r\n
    "image_data_format": "channels_last",\n
    "backend": "theano",\r\n
    "floatx": "float32"\r\n
    }"""
    f.write(new_settings)

import keras

# Load data
sPathIn = 'C:/Users/Sebastian Milde/Pictures/Universitaet/Masterarbeit/Data_train_test/'
sPathOut = 'C:\Users\Sebastian Milde\Pictures\Universitaet\Masterarbeit\Results'
with h5py.File(sPathIn, 'r') as hf:
    X_train = hf['X_train'][:]
    X_test = hf['X_test'][:]
    y_train = hf['y_train'][:]
    y_test = hf['y_test'][:]
    patchSize = hf['patchSize'][:]

num_classes = 8

# convert training data in 4D - Tensor --> shape(batch, channels, height, width)
X_train = np.expand_dims(X_train, axis = 1)
X_test = np.expand_dims(X_test, axis = 1)
#convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)