'''
@author: Yannick Wilhelm
@email: yannick.wilhelm@gmx.de
@date: February 2017

Keras implementation of a Squeeze-and-Excitation-Block in accordance with the original paper
(Hu 2017, Squeeze and Excitation Networks)
'''

from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute, GlobalAveragePooling3D
from keras import backend

def squeeze_excitation_block(inputSE, ratio=16):
    '''
    Creates a squeeze and excitation block
    :param input: input tensor
    :param ratio: reduction ratio r for bottleneck given by the two FC layers
    :return: keras tensor
    '''

    if backend.image_data_format() == 'channels_first':
        channels = 1
    else:
        channels = -1

    # number of input filters/channels
    inputSE_shape = backend.int_shape(inputSE)
    numChannels = inputSE_shape[channels]

    #squeeze operation
    output = GlobalAveragePooling2D(data_format=backend.image_data_format())(inputSE)

    #excitation operation
    output = Dense(numChannels//ratio, activation='relu', use_bias=True, kernel_initializer='he_normal')(output)
    output = Dense(numChannels, activation='sigmoid', use_bias=True, kernel_initializer='he_normal')(output)

    #scale operation
    output = multiply([inputSE, output])

    return output



def squeeze_excitation_block_3D(inputSE, ratio=16):
    '''
    Creates a squeeze and excitation block
    :param input: input tensor
    :param ratio: reduction ratio r for bottleneck given by the two FC layers
    :return: keras tensor
    '''

    if backend.image_data_format() == 'channels_first':
        channels = 1
    else:
        channels = -1

    # number of input filters/channels
    inputSE_shape = backend.int_shape(inputSE)
    numChannels = inputSE_shape[channels]

    #squeeze operation
    output = GlobalAveragePooling3D(data_format=backend.image_data_format())(inputSE)

    #excitation operation
    output = Dense(numChannels//ratio, activation='relu', use_bias=True, kernel_initializer='he_normal')(output)
    output = Dense(numChannels, activation='sigmoid', use_bias=True, kernel_initializer='he_normal')(output)

    #scale operation
    output = multiply([inputSE, output])

    return output