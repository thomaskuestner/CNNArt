from keras.layers import Dense, Reshape, Flatten, Dropout
from utils.MotionCorrection.network import *

def encode(input, patchSize):
    if list(patchSize) == [48, 48]:
        # output shape: (64, 48, 48)
        conv_1 = fCreateConv2D_ResBlock(filters=64, kernel_size=(7, 7), strides=(1, 1))(input)

        # output shape: (128, 24, 24)
        output = fCreateConv2D_ResBlock(filters=128)(conv_1)

    elif list(patchSize) == [80, 80]:
        # output shape: (64, 80, 80)
        conv_1 = fCreateConv2D_ResBlock(filters=64, kernel_size=(7, 7), strides=(1, 1))(input)

        # output shape: (128, 40, 40)
        output = fCreateConv2D_ResBlock(filters=128)(conv_1)

    return output, conv_1


def encode_shared(input, patchSize):
    if list(patchSize) == [48, 48]:
        # output shape: (256, 12, 12)
        conv_shared_1 = fCreateConv2D_ResBlock(filters=256)(input)
        # output shape: (512, 6, 6)
        conv_shared_2 = fCreateConv2D_ResBlock(filters=512)(conv_shared_1)

        flat = Flatten()(conv_shared_2)
        z_mean = Dense(512)(flat)
        z_log_var = Dense(512)(flat)
        z = Lambda(samplingDense, output_shape=(512,))([z_mean, z_log_var])

        conv_shared_2 = None

    elif list(patchSize) == [80, 80]:
        # output shape: (256, 20, 20)
        conv_shared_1 = fCreateConv2D_ResBlock(filters=256)(input)
        # output shape: (512, 10, 10)
        conv_shared_2 = fCreateConv2D_ResBlock(filters=512)(conv_shared_1)
        # output shape: (512, 5, 5)
        conv_shared_3 = fCreateConv2D_ResBlock(filters=512)(conv_shared_2)

        flat = Flatten()(conv_shared_3)
        z_mean = Dense(512)(flat)
        z_log_var = Dense(512)(flat)
        z = Lambda(samplingDense, output_shape=(512,))([z_mean, z_log_var])

    return z, z_mean, z_log_var, conv_shared_1, conv_shared_2


def decode(input, patchSize, conv_1, conv_2, conv_3, conv_4, arch):
    if arch == "vae-mocounet":
        if list(patchSize) == [48, 48]:
            dense = Dense(512 * 6 * 6)(input)
            # output shape: (512, 6, 6)
            reshape = Reshape((512, 6, 6))(dense)
            # output shape: (256, 12, 12)
            output = fCreateConv2DTranspose_ResBlock(filters=256)(reshape)
            output = add([output, conv_3])
            # output shape: (128, 24, 24)
            output = fCreateConv2DTranspose_ResBlock(filters=128)(output)
            output = add([output, conv_2])
            # output shape: (64, 48, 48)
            output = fCreateConv2DTranspose_ResBlock(filters=64)(output)
            output = add([output, conv_1])
            # output shape: (64, 48, 48)
            output = fCreateConv2DTranspose_ResBlock(filters=64, kernel_size=(7, 7), strides=(1, 1))(output)
            # output shape: (1, 48, 48)
            output = Conv2DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)

        elif list(patchSize) == [80, 80]:
            dense = Dense(512 * 5 * 5)(input)
            # output shape: (512, 5, 5)
            reshape = Reshape((512, 5, 5))(dense)
            # output shape: (512, 10, 10)
            output = fCreateConv2DTranspose_ResBlock(filters=512)(reshape)
            output = add([output, conv_4])
            # output shape: (256, 20, 20)
            output = fCreateConv2DTranspose_ResBlock(filters=256)(output)
            output = add([output, conv_3])
            # output shape: (128, 40, 40)
            output = fCreateConv2DTranspose_ResBlock(filters=128)(output)
            output = add([output, conv_2])
            # output shape: (64, 80, 80)
            output = fCreateConv2DTranspose_ResBlock(filters=64)(output)
            output = add([output, conv_1])
            # output shape: (64, 80, 80)
            output = fCreateConv2DTranspose_ResBlock(filters=64, kernel_size=(7, 7), strides=(1, 1))(output)
            # output shape: (1, 80, 80)
            output = Conv2DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)

    elif arch == "vae-mononet":
         if list(patchSize) == [48, 48]:
            dense = Dense(512 * 6 * 6)(input)
            # output shape: (512, 6, 6)
            reshape = Reshape((512, 6, 6))(dense)
            # output shape: (256, 12, 12)
            output = fCreateConv2DTranspose_ResBlock(filters=256)(reshape)
            # output shape: (128, 24, 24)
            output = fCreateConv2DTranspose_ResBlock(filters=128)(output)
            # output shape: (64, 48, 48)
            output = fCreateConv2DTranspose_ResBlock(filters=64)(output)
            # output shape: (64, 48, 48)
            output = fCreateConv2DTranspose_ResBlock(filters=64, kernel_size=(7, 7), strides=(1, 1))(output)
            # output shape: (1, 48, 48)
            output = Conv2DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)

        elif list(patchSize) == [80, 80]:
            dense = Dense(512 * 5 * 5)(input)
            # output shape: (512, 5, 5)
            reshape = Reshape((512, 5, 5))(dense)
            # output shape: (512, 10, 10)
            output = fCreateConv2DTranspose_ResBlock(filters=512)(reshape)
            # output shape: (256, 20, 20)
            output = fCreateConv2DTranspose_ResBlock(filters=256)(output)
            # output shape: (128, 40, 40)
            output = fCreateConv2DTranspose_ResBlock(filters=128)(output)
            # output shape: (64, 80, 80)
            output = fCreateConv2DTranspose_ResBlock(filters=64)(output)
            # output shape: (64, 80, 80)
            output = fCreateConv2DTranspose_ResBlock(filters=64, kernel_size=(7, 7), strides=(1, 1))(output)
            # output shape: (1, 80, 80)
            output = Conv2DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)       

    return output
