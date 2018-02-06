'''
@author: Yannick Wilhelm
@email: yannick.wilhelm@gmx.de
@date: Febraury 2017

Deep residual learning blocks
(He 2015, Deep Residual Learning)

'''

from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Add
from keras.layers import Lambda
import keras.backend as K
from networks.multiclass.SENets.squeeze_excitation_block import *

def identity_bottleneck_block(input_tensor, filters, stage, block, se_enabled=False, se_ratio=16):


    numFilters1, numFilters2, numFilters3 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(numFilters1, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(numFilters2, (3,3), padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(numFilters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
    # squeeze and excitation block
    if se_enabled:
        x = squeeze_excitation_block(x, ratio=se_ratio)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])

    x = Activation('relu')(x)
    return x



def identity_block(input_tensor, filters, stage, block, se_enabled=False, se_ratio=16):

    numFilters1, numFilters2 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + '_' + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_branch'

    x = Conv2D(numFilters1, (3, 3), padding='same', kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(numFilters2, (3, 3), padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
    # squeeze and excitation block
    if se_enabled:
        x = squeeze_excitation_block(x, ratio=se_ratio)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    x = Add()([x, input_tensor])

    x = Activation('relu')(x)
    return x



def projection_block(input_tensor, filters, stage, block, se_enabled=False, se_ratio=16):

    numFilters1, numFilters2 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + '_' + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_branch'

    # downsampling directly by convolution with stride 2
    x = Conv2D(numFilters1, (3, 3), padding='same', strides=(2,2), kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(numFilters2, (3, 3), padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
    # squeeze and excitation block
    if se_enabled:
        x = squeeze_excitation_block(x, ratio=se_ratio)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    # projection shortcut convolution
    x_shortcut = Conv2D(numFilters2, (1,1), strides=(2, 2), kernel_initializer='he_normal', name=conv_name_base + '1')(input_tensor)
    x_shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(x_shortcut)

    # addition of shortcut
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x



def projection_bottleneck_block(input_tensor, filters, stage, block, se_enabled=False, se_ratio=16):
    numFilters1, numFilters2, numFilters3 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + '_' + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_branch'

    x = Conv2D(numFilters1, (1, 1), strides=(2,2), kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(numFilters2, (3, 3), padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(numFilters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
    # squeeze and excitation block
    if se_enabled:
        x = squeeze_excitation_block(x, ratio=se_ratio)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # projection shortcut
    x_shortcut = Conv2D(numFilters3, (1,1), strides=(2,2), kernel_initializer='he_normal', name=conv_name_base+'1')(input_tensor)
    x_shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(x_shortcut)

    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x



def zero_padding_block(input_tensor, filters, stage, block, se_enabled=False, se_ratio=16):
    numFilters1, numFilters2 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + '_' + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_branch'

    # downsampling directly by convolution with stride 2
    x = Conv2D(numFilters1, (3, 3), strides=(2, 2), kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(numFilters2, (3, 3), kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
    # squeeze and excitation block
    if se_enabled:
        x = squeeze_excitation_block(x, ratio=se_ratio)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    # zero padding and downsampling with 1x1 conv shortcut connection
    x_shortcut = Conv2D(1, (1, 1), strides=(2, 2), kernel_initializer='he_normal', name=conv_name_base + '1')(input_tensor)
    x_shortcut2 = MaxPooling2D(pool_size=(1, 1), strides=(2, 2), border_mode='same')(input_tensor)
    x_shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(x_shortcut)

    x_shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(x_shortcut)

    # addition of shortcut
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x



def zeropad(x):
    y = K.zeros_like(x)
    return K.concatenate([x, y], axis=1)



def zeropad_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[1] *= 2
    return tuple(shape)