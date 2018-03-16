from keras.layers import Dense, Reshape, Flatten, Lambda, Dropout
from utils.MotionCorrection.network import *


def encode(input, patchSize):
    if list(patchSize) == [80, 80]:
        input = fCreateLeakyReluConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1))(input)
        res_1 = fCreateEncoder2D_Block(filters=32)(input)
        down_1 = fCreateDownConv2D_Block(filters=64)(res_1)
        res_2 = fCreateEncoder2D_Block(filters=64)(down_1)
        output = fCreateDownConv2D_Block(filters=128)(res_2)

    elif list(patchSize) == [48, 48]:
        input = fCreateLeakyReluConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1))(input)
        res_1 = fCreateEncoder2D_Block(filters=32)(input)
        down_1 = fCreateDownConv2D_Block(filters=64)(res_1)
        res_2 = fCreateEncoder2D_Block(filters=64)(down_1)
        output = fCreateDownConv2D_Block(filters=128)(res_2)

    elif list(patchSize) == [80, 80, 10]:
        res_1 = fCreateEncoder3D_Block(filters=32)(input)
        down_1 = fCreateDownConv3D_Block(filters=64, strides=(2, 2, 2), kernel_size=(2, 2, 2))(res_1)
        res_2 = fCreateEncoder3D_Block(filters=64)(down_1)
        output = fCreateDownConv3D_Block(filters=128, strides=(2, 2, 1), kernel_size=(2, 2, 1))(res_2)

    return output


def encode_shared(input, patchSize, isIncep):
    if isIncep:
        if list(patchSize) == [80, 80]:
            res_1 = fCreateInception2D_Block(filters=[32, 64, 128])(input)
            down_1 = fCreateDownConv2D_Block(filters=256)(res_1)
            res_2 = fCreateInception2D_Block(filters=[32, 64, 128])(down_1)
            down_2 = fCreateDownConv2D_Block(filters=256)(res_2)
            flat = Flatten()(down_2)

            z_mean = Dense(128)(flat)
            z_log_var = Dense(128)(flat)

            z = Lambda(sampling, output_shape=(128,))([z_mean, z_log_var])

        elif list(patchSize) == [48, 48]:
            res_1 = fCreateInception2D_Block(filters=[32, 64, 128])(input)
            down_1 = fCreateDownConv2D_Block(filters=256)(res_1)
            flat = Flatten()(down_1)

            z_mean = Dense(128)(flat)
            z_log_var = Dense(128)(flat)

            z = Lambda(sampling, output_shape=(128,))([z_mean, z_log_var])

        elif list(patchSize) == [80, 80, 10]:
            res_1 = fCreateInception3D_Block(filters=[32, 64, 128])(input)
            down_1 = fCreateDownConv3D_Block(filters=256, strides=(2, 2, 1), kernel_size=(2, 2, 1))(res_1)
            flat = Flatten()(down_1)

            z_mean = Dense(128)(flat)
            z_log_var = Dense(128)(flat)

            z = Lambda(sampling, output_shape=(128,))([z_mean, z_log_var])
    else:
        if list(patchSize) == [80, 80]:
            res_1 = fCreateEncoder2D_Block(filters=128)(input)
            down_1 = fCreateDownConv2D_Block(filters=256)(res_1)
            res_2 = fCreateEncoder2D_Block(filters=256)(down_1)
            down_2 = fCreateDownConv2D_Block(filters=512)(res_2)
            flat = Flatten()(down_2)

            z_mean = Dense(128)(flat)
            z_log_var = Dense(128)(flat)

            z = Lambda(sampling, output_shape=(128,))([z_mean, z_log_var])

        elif list(patchSize) == [48, 48]:
            res_1 = fCreateEncoder2D_Block(filters=128)(input)
            down_1 = fCreateDownConv2D_Block(filters=256)(res_1)
            flat = Flatten()(down_1)

            dropout = Dropout(0.8)(flat)

            z_mean = Dense(128)(dropout)
            z_log_var = Dense(128)(dropout)

            z = Lambda(sampling, output_shape=(128,))([z_mean, z_log_var])

        elif list(patchSize) == [80, 80, 10]:
            res_1 = fCreateEncoder3D_Block(filters=128)(input)
            down_1 = fCreateDownConv3D_Block(filters=256, strides=(2, 2, 1), kernel_size=(2, 2, 1))(res_1)
            flat = Flatten()(down_1)

            z_mean = Dense(128)(flat)
            z_log_var = Dense(128)(flat)

            z = Lambda(sampling, output_shape=(128,))([z_mean, z_log_var])

    return z, z_mean, z_log_var


def decode(input, patchSize, dropout):
    if list(patchSize) == [80, 80]:
        dense = Dense(256 * 5 * 5)(input)
        dropout = Dropout(dropout)(dense)
        reshape = Reshape((256, 5, 5))(dropout)
        output = fCreateConv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same')(reshape)
        output = fCreateConv2DTranspose(filters=256, kernel_size=3, strides=1, padding='same')(output)
        output = fCreateConv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')(output)
        output = fCreateConv2DTranspose(filters=128, kernel_size=3, strides=1, padding='same')(output)
        output = fCreateConv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(output)
        output = fCreateConv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same')(output)
        output = fCreateConv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')(output)
        output = Conv2DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)

    elif list(patchSize) == [48, 48]:
        dense = Dense(256 * 6 * 6)(input)
        dropout = Dropout(dropout)(dense)
        reshape = Reshape((256, 6, 6))(dropout)
        output = fCreateConv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')(reshape)
        output = fCreateConv2DTranspose(filters=128, kernel_size=3, strides=1, padding='same')(output)
        output = fCreateConv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(output)
        output = fCreateConv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same')(output)
        output = fCreateConv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')(output)
        output = Conv2DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)

    elif list(patchSize) == [80, 80, 10]:
        dense = Dense(25600 * 5)(input)
        dropout = Dropout(dropout)(dense)
        reshape = Reshape((256, 10, 10, 5))(dropout)
        output = fCreateConv3DTranspose(filters=256, strides=(2, 2, 1))(reshape)
        output = fCreateConv3DTranspose(filters=256, strides=(1, 1, 1))(output)
        output = fCreateConv3DTranspose(filters=128, strides=(2, 2, 1))(output)
        output = fCreateConv3DTranspose(filters=128, strides=(1, 1, 1))(output)
        output = fCreateConv3DTranspose(filters=64, strides=(2, 2, 2))(output)
        output = Conv3DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)

    return output