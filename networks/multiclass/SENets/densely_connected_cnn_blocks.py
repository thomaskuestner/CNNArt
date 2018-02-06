'''
@author: Yannick Wilhelm
@email: yannick.wilhelm@gmx.de
@date: February 2017

Keras implementation of DenseNet Blocks in accordance with the original paper
(Huang 2017, Densely Connected CNNs)

'''


import keras.backend as K
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Activation
from keras.layers import concatenate

from networks.multiclass.SENets import squeeze_excitation_block


def transition_layer(input_tensor, numFilters, compressionFactor=1.0):

    numOutPutFilters = int(numFilters*compressionFactor)

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(input_tensor)
    x = Activation('relu')(x)

    x = Conv2D(numOutPutFilters, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', name='conv1')(x)

    # downsampling
    x = AveragePooling2D((2, 2), strides=(2, 2), padding='valid', data_format='channels_last', name='')(x)

    return x, numOutPutFilters


def dense_block(input_tensor, numInputFilters, numLayers=1, growthRate_k=12, bottleneck_enabled=False):

    if K.image_data_format() == 'channels_last':
        concat_axis = -1
        bn_axis = -1
    else:
        concat_axis = 1
        bn_axis = 1

    concat_features = [input_tensor]

    for i in range(numLayers):
        x = BatchNormalization(axis=bn_axis, name='')(concat_features)
        x = Activation('relu')(x)

        if bottleneck_enabled == True:
            x = Conv2D(4*growthRate_k, (1,1), strides=(1,1), kernel_initializer='he_normal', padding='same')(x)    # "in our experiments, we let each 1x1 conv produce 4k feature maps
            x = BatchNormalization(axis=bn_axis)(x)
            x = Activation('relu')(x)

        x = Conv2D(growthRate_k, (3,3), strides=(1,1), kernel_initializer='he_normal', padding='same')(x)
        concat_features = concatenate([x, concat_features], axis=concat_axis)

        numInputFilters += growthRate_k

    return concat_features, numInputFilters


def transition_SE_layer(input_tensor, numFilters, compressionFactor=1.0, se_ratio=16):

    numOutPutFilters = int(numFilters*compressionFactor)

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(input_tensor)
    x = Activation('relu')(x)

    x = Conv2D(numOutPutFilters, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', name='conv1')(x)

    # SE Block
    x = squeeze_excitation_block(x, ratio=se_ratio)
    #x = BatchNormalization(axis=bn_axis)(x)

    # downsampling
    x = AveragePooling2D((2, 2), strides=(2, 2), padding='valid', data_format='channels_last', name='')(x)

    return x, numOutPutFilters