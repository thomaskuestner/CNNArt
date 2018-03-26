from keras.layers import Dense, Reshape, Flatten, Dropout
from utils.MotionCorrection.network import *


def encode(input, patchSize):
    if list(patchSize) == [80, 80]:
        # output shape: (32, 80, 80)
        conv_1 = fCreateLeakyReluConv2D(filters=32, strides=1, kernel_size=5)(input)
        # output shape: (64, 40, 40)
        down_1 = fCreateLeakyReluConv2D(filters=64, strides=2, kernel_size=5)(conv_1)
        # output shape: (128, 40, 40)
        output = fCreateLeakyReluConv2D(filters=128, strides=1, kernel_size=3)(down_1)

    elif list(patchSize) == [48, 48]:
        # output shape: (32, 48, 48)
        conv_1 = fCreateLeakyReluConv2D(filters=32, strides=1, kernel_size=7)(input)
        # output shape: (64, 24, 24)
        down_1 = fCreateLeakyReluConv2D(filters=64, strides=2, kernel_size=7)(conv_1)
        # output shape: (128, 24, 24)
        output = fCreateLeakyReluConv2D(filters=128, strides=1, kernel_size=3)(down_1)

    elif list(patchSize) == [80, 80, 16]:
        # output shape: (32, 80, 80, 16)
        conv_1 = fCreateLeakyReluConv3D(filters=32, strides=1, kernel_size=5)(input)
        # output shape: (64, 40, 40, 8)
        down_1 = fCreateLeakyReluConv3D(filters=64, strides=2, kernel_size=5)(conv_1)
        # output shape: (128, 40, 40, 8)
        output = fCreateLeakyReluConv3D(filters=128, strides=1, kernel_size=3)(down_1)

    return output


def encode_shared(input, patchSize):
    if list(patchSize) == [80, 80]:
        # output shape: (256, 40, 40)
        conv_shared_1 = fCreateLeakyReluConv2D(filters=256, strides=1, kernel_size=3)(input)
        # output shape: (256, 20, 20)
        down_shared_1 = fCreateLeakyReluConv2D(filters=256, strides=2, kernel_size=3)(conv_shared_1)
        # output shape: (256, 20, 20)
        conv_shared_2 = fCreateLeakyReluConv2D(filters=256, strides=1, kernel_size=3)(down_shared_1)
        # output shape: (256, 10, 10)
        down_shared_2 = fCreateLeakyReluConv2D(filters=256, strides=2, kernel_size=3)(conv_shared_2)

        flat = Flatten()(down_shared_2)

        z_mean = Dense(512)(flat)
        z_log_var = Dense(512)(flat)

        z = Lambda(samplingDense, output_shape=(512,))([z_mean, z_log_var])

    elif list(patchSize) == [48, 48]:
        # output shape: (256, 24, 24)
        conv_shared_1 = fCreateLeakyReluConv2D(filters=256, strides=1, kernel_size=3)(input)
        # output shape: (256, 12, 12)
        conv_shared_2 = fCreateLeakyReluConv2D(filters=256, strides=2, kernel_size=3)(conv_shared_1)
        # output shape: (256, 12, 12)
        conv_shared_3 = fCreateLeakyReluConv2D(filters=256, strides=1, kernel_size=3)(conv_shared_2)
        # output shape: (256, 6, 6)
        conv_shared_4 = fCreateLeakyReluConv2D(filters=256, strides=2, kernel_size=3)(conv_shared_3)

        flat = Flatten()(conv_shared_4)

        z_mean = Dense(512)(flat)

        z_log_var = Dense(512)(flat)

        z = Lambda(samplingDense, output_shape=(512,))([z_mean, z_log_var])

    elif list(patchSize) == [80, 80, 16]:
        # output shape: (256, 20, 20, 4)
        conv_shared_1 = fCreateLeakyReluConv3D(filters=256, strides=2, kernel_size=3)(input)
        # output shape: (256, 20, 20, 4)
        conv_shared_2 = fCreateLeakyReluConv3D(filters=256, strides=1, kernel_size=3)(conv_shared_1)
        # output shape: (256, 10, 10, 2)
        conv_shared_3 = fCreateLeakyReluConv3D(filters=256, strides=2, kernel_size=3)(conv_shared_2)
        # output shape: (256, 10, 10, 2)
        conv_shared_4 = fCreateLeakyReluConv3D(filters=256, strides=1, kernel_size=3)(conv_shared_3)
        # output shape: (256, 5, 5, 1)
        conv_shared_5 = fCreateLeakyReluConv3D(filters=256, strides=2, kernel_size=3)(conv_shared_4)

        flat = Flatten()(conv_shared_5)

        z_mean = Dense(256)(flat)

        z_log_var = Dense(256)(flat)

        z = Lambda(samplingDense, output_shape=(256,))([z_mean, z_log_var])

    return z, z_mean, z_log_var


def decode(input, patchSize, dropout):
    if list(patchSize) == [80, 80]:
        dense = Dense(256 * 10 * 10)(input)
        dropout = Dropout(dropout)(dense)
        # output shape: (256, 10, 10)
        reshape = Reshape((256, 10, 10))(dropout)
        # output shape: (256, 20, 20)
        output = fCreateConv2DTranspose(filters=256, kernel_size=3, strides=2)(reshape)
        # output shape: (256, 20, 20)
        output = fCreateConv2DTranspose(filters=256, kernel_size=3, strides=1)(output)
        # output shape: (128, 40, 40)
        output = fCreateConv2DTranspose(filters=128, kernel_size=3, strides=2)(output)
        # output shape: (64, 40, 40)
        output = fCreateConv2DTranspose(filters=64, kernel_size=3, strides=1)(output)
        # output shape: (32, 80, 80)
        output = fCreateConv2DTranspose(filters=32, kernel_size=3, strides=2)(output)
        output = Conv2DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)

    elif list(patchSize) == [48, 48]:
        dense = Dense(256 * 6 * 6)(input)
        dropout = Dropout(dropout)(dense)
        # output shape: (256, 6, 6)
        reshape = Reshape((256, 6, 6))(dropout)
        # output shape: (256, 12, 12)
        output = fCreateConv2DTranspose(filters=256, kernel_size=3, strides=2)(reshape)
        # output shape: (256, 12, 12)
        output = fCreateConv2DTranspose(filters=256, kernel_size=3, strides=1)(output)
        # output shape: (128, 24, 24)
        output = fCreateConv2DTranspose(filters=128, kernel_size=3, strides=2)(output)
        # output shape: (64, 24, 24)
        output = fCreateConv2DTranspose(filters=64, kernel_size=3, strides=1)(output)
        # output shape: (32, 48, 48)
        output = fCreateConv2DTranspose(filters=32, kernel_size=3, strides=2)(output)
        output = Conv2DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)

    elif list(patchSize) == [80, 80, 16]:
        dense = Dense(256 * 5 * 5)(input)
        dropout = Dropout(dropout)(dense)
        # output shape: (256, 5, 5, 1)
        reshape = Reshape((256, 5, 5, 1))(dropout)
        # output shape: (256, 10, 10, 2)
        output = fCreateConv3DTranspose(filters=256, strides=2, kernel_size=3)(reshape)
        # output shape: (256, 10, 10, 2)
        output = fCreateConv3DTranspose(filters=256, strides=1, kernel_size=3)(output)
        # output shape: (256, 20, 20, 4)
        output = fCreateConv3DTranspose(filters=256, strides=2, kernel_size=3)(output)
        # output shape: (256, 20, 20, 4)
        output = fCreateConv3DTranspose(filters=256, strides=1, kernel_size=3)(output)
        # output shape: (256, 40, 40, 8)
        output = fCreateConv3DTranspose(filters=128, strides=2, kernel_size=3)(output)
        # output shape: (64, 40, 40, 8)
        output = fCreateConv3DTranspose(filters=64, strides=1, kernel_size=3)(output)
        # output shape: (64, 80, 80, 16)
        output = fCreateConv3DTranspose(filters=32, strides=2, kernel_size=3)(output)
        output = Conv3DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)

    return output