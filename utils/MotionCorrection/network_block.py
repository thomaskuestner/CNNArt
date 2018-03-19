from keras.layers import Dense, Reshape, Flatten, Dropout
from utils.MotionCorrection.network import *


def encode(input, patchSize):
    if list(patchSize) == [80, 80]:
        conv_1 = fCreateLeakyReluConv2D(filters=32)(input)
        down_1 = fCreateLeakyReluConv2D(filters=64, strides=2, kernel_size=2)(conv_1)
        output = fCreateLeakyReluConv2D(filters=128)(down_1)

    elif list(patchSize) == [48, 48]:
        conv_1 = fCreateLeakyReluBNConv2D(filters=32, strides=1, kernel_size=7)(input)
        down_1 = fCreateLeakyReluBNConv2D(filters=64, strides=2, kernel_size=7)(conv_1)
        output = fCreateLeakyReluBNConv2D(filters=128, strides=1, kernel_size=3)(down_1)

    elif list(patchSize) == [48, 48, 4]:
        conv_1 = fCreateLeakyReluBNConv3D(filters=32, strides=1, kernel_size=7)(input)
        down_1 = fCreateLeakyReluBNConv3D(filters=64, strides=2, kernel_size=7)(conv_1)
        output = fCreateLeakyReluBNConv3D(filters=128, strides=1, kernel_size=3)(down_1)

    return output


def encode_shared(input, patchSize, isIncep):
    if isIncep:
        if list(patchSize) == [80, 80]:
            res_1 = fCreateConv2D_InceptionBlock(filters=[32, 64, 128])(input)
            down_1 = fCreateLeakyReluBNConv2D(filters=256, strides=2, kernel_size=2)(res_1)
            res_2 = fCreateConv2D_InceptionBlock(filters=[32, 64, 128])(down_1)
            down_2 = fCreateLeakyReluBNConv2D(filters=256, strides=2, kernel_size=2)(res_2)
            flat = Flatten()(down_2)

            z_mean = Dense(256)(flat)
            z_log_var = Dense(256)(flat)

            z = Lambda(samplingDense, output_shape=(256,))([z_mean, z_log_var])

        elif list(patchSize) == [48, 48]:
            res_1 = fCreateConv2D_InceptionBlock(filters=[32, 64, 128])(input)
            down_1 = fCreateLeakyReluConv2D(filters=256, strides=2, kernel_size=2)(res_1)
            flat = Flatten()(down_1)

            z_mean = Dense(128)(flat)
            z_log_var = Dense(128)(flat)

            z = Lambda(samplingDense, output_shape=(128,))([z_mean, z_log_var])

    else:
        if list(patchSize) == [80, 80]:
            conv_shared_1 = fCreateLeakyReluConv2D(filters=256)(input)
            down_shared_1 = fCreateLeakyReluConv2D(filters=256, strides=2, kernel_size=2)(conv_shared_1)
            conv_shared_2 = fCreateLeakyReluConv2D(filters=256)(down_shared_1)
            down_shared_2 = fCreateLeakyReluConv2D(filters=256, strides=2, kernel_size=2)(conv_shared_2)

            flat = Flatten()(down_shared_2)

            z_mean = Dense(512)(flat)
            z_log_var = Dense(512)(flat)

            z = Lambda(samplingDense, output_shape=(512,))([z_mean, z_log_var])

        elif list(patchSize) == [48, 48]:
            conv_shared_1 = fCreateLeakyReluBNConv2D(filters=256, strides=1, kernel_size=3)(input)
            down_shared_1 = fCreateLeakyReluBNConv2D(filters=256, strides=2, kernel_size=3)(conv_shared_1)
            conv_shared_2 = fCreateLeakyReluBNConv2D(filters=256, strides=1, kernel_size=3)(down_shared_1)
            down_shared_2 = fCreateLeakyReluBNConv2D(filters=256, strides=2, kernel_size=3)(conv_shared_2)

            flat = Flatten()(down_shared_2)

            z_mean = Dense(512)(flat)

            z_log_var = Dense(512)(flat)

            z = Lambda(samplingDense, output_shape=(512,))([z_mean, z_log_var])

        elif list(patchSize) == [48, 48, 4]:
            conv_shared_1 = fCreateLeakyReluBNConv3D(filters=256, strides=1, kernel_size=3)(input)
            down_shared_1 = fCreateLeakyReluBNConv3D(filters=256, strides=(2, 2, 1), kernel_size=3)(conv_shared_1)
            conv_shared_2 = fCreateLeakyReluBNConv3D(filters=256, strides=1, kernel_size=3)(down_shared_1)
            down_shared_2 = fCreateLeakyReluBNConv3D(filters=256, strides=2, kernel_size=3)(conv_shared_2)

            flat = Flatten()(down_shared_2)

            z_mean = Dense(512)(flat)

            z_log_var = Dense(512)(flat)

            z = Lambda(samplingDense, output_shape=(512,))([z_mean, z_log_var])

    return z, z_mean, z_log_var


def decode(input, patchSize, dropout):
    if list(patchSize) == [80, 80]:
        dense = Dense(256 * 10 * 10)(input)
        dropout = Dropout(dropout)(dense)
        reshape = Reshape((256, 10, 10))(dropout)
        output = fCreateConv2DTranspose(filters=256, kernel_size=3, strides=2)(reshape)
        output = fCreateConv2DTranspose(filters=256, kernel_size=3, strides=1)(output)
        output = fCreateConv2DTranspose(filters=256, kernel_size=3, strides=2)(output)
        output = fCreateConv2DTranspose(filters=128, kernel_size=3, strides=1)(output)
        output = fCreateConv2DTranspose(filters=64, kernel_size=3, strides=2)(output)
        output = Conv2DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)

    elif list(patchSize) == [48, 48]:
        dense = Dense(256 * 6 * 6)(input)
        dropout = Dropout(dropout)(dense)
        reshape = Reshape((256, 6, 6))(dropout)
        output = fCreateConv2DTranspose(filters=256, kernel_size=3, strides=2)(reshape)
        output = fCreateConv2DTranspose(filters=256, kernel_size=3, strides=1)(output)
        output = fCreateConv2DTranspose(filters=128, kernel_size=3, strides=2)(output)
        output = fCreateConv2DTranspose(filters=64, kernel_size=3, strides=1)(output)
        output = fCreateConv2DTranspose(filters=32, kernel_size=3, strides=2)(output)
        output = Conv2DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)

    elif list(patchSize) == [48, 48, 4]:
        dense = Dense(256 * 6 * 6)(input)
        dropout = Dropout(dropout)(dense)
        reshape = Reshape((256, 6, 6, 1))(dropout)
        output = fCreateConv3DTranspose(filters=128, strides=2, kernel_size=3)(reshape)
        output = fCreateConv3DTranspose(filters=128, strides=1, kernel_size=3)(output)
        output = fCreateConv3DTranspose(filters=64, strides=(2, 2, 1), kernel_size=3)(output)
        output = fCreateConv3DTranspose(filters=64, strides=1, kernel_size=3)(output)
        output = fCreateConv3DTranspose(filters=64, strides=2, kernel_size=3)(output)
        output = Conv3DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)

    return output