'''
@author: Yannick Wilhelm
@email: yannick.wilhelm@gmx.de
@date: Febraury 2017

Deep residual learning blocks
(He 2015, Deep Residual Learning)

'''

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K
from networks.FullyConvolutionalNetworks.motion.squeeze_excitation_block_tf import *



def identity_bottleneck_block(input_tensor, filters, stage, block, se_enabled=False, se_ratio=16):


    numFilters1, numFilters2, numFilters3 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + '_' + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_branch'

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


########################################################################################################################
### 3D Residual Blocks #################################################################################################
########################################################################################################################

def identity_block_3D(input_tensor, filters, kernel_size=(3, 3, 3), stage=0, block=0, se_enabled=False, se_ratio=16):

    numFilters1, numFilters2 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + '_' + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_branch'

    x = Conv3D(filters=numFilters1,
               kernel_size=kernel_size,
               strides=(1, 1, 1),
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2a')(input_tensor)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = Conv3D(filters=numFilters2,
               kernel_size=kernel_size,
               strides=(1, 1, 1),
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2b')(x)

    # squeeze and excitation block
    if se_enabled:
        x = squeeze_excitation_block_3D(x, ratio=se_ratio)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    x = Add()([x, input_tensor])

    x = LeakyReLU(alpha=0.01)(x)


    return x



def projection_block_3D(input_tensor, filters, kernel_size=(3,3,3) , stage=0, block=0, se_enabled=False, se_ratio=16):

    numFilters1, numFilters2 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + '_' + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_branch'

    # downsampling directly by convolution with stride 2
    x = Conv3D(filters=numFilters1,
               kernel_size=kernel_size,
               strides=(2,2,2),
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2a')(input_tensor)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = Conv3D(filters=numFilters2,
               kernel_size=kernel_size,
               strides=(1, 1, 1),
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2b')(x)

    # squeeze and excitation block
    if se_enabled:
        x = squeeze_excitation_block_3D(x, ratio=se_ratio)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    # projection shortcut convolution
    x_shortcut = Conv3D(filters=numFilters2,
                        kernel_size=(2,2,2),
                        strides=(2,2,2),
                        padding='same',
                        kernel_initializer='he_normal',
                        name=conv_name_base + '1')(input_tensor)
    x_shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(x_shortcut)

    # addition of shortcut
    x = Add()([x, x_shortcut])

    x = LeakyReLU(alpha=0.01)(x)

    return x


########################################################################################################################


########################################################################################################################
### Transposed 3D projection block
########################################################################################################################

def transposed_projection_block_3D(input_tensor, filters, kernel_size=(3, 3, 3), stage=0, block=0, se_enabled=False, se_ratio=16):

    numFilters1, numFilters2 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    transposed_conv_name_base = 'transposed_res' + str(stage) + '_' + str(block) + '_branch'
    conv_name_base = 'res' + str(stage) + '_' + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_branch'

    # upsampling directly by transposed convolution with stride 2
    x = Conv3DTranspose(filters=numFilters1,
                        kernel_size=kernel_size,
                        strides=(2, 2, 2),
                        padding='same',
                        data_format=K.image_data_format(),
                        kernel_initializer='he_normal',
                        name=transposed_conv_name_base + '2a')(input_tensor)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = Conv3D(filters=numFilters2,
               kernel_size=kernel_size,
               strides=(1, 1, 1),
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2b')(x)

    # squeeze and excitation block
    if se_enabled:
        x = squeeze_excitation_block_3D(x, ratio=se_ratio)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    # projection shortcut transposed convolution
    x_shortcut = Conv3DTranspose(filters=numFilters2,
                        kernel_size=(2,2,2),
                        strides=(2, 2, 2),
                        padding='same',
                        data_format=K.image_data_format(),
                        kernel_initializer='he_normal',
                        name=transposed_conv_name_base + '2a')(input_tensor)
    x_shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(x_shortcut)

    # addition of shortcut
    x = Add()([x, x_shortcut])

    x = LeakyReLU(alpha=0.01)(x)

    return x

########################################################################################################################
