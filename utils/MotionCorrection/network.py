from keras.layers import Conv2D, LeakyReLU, Conv2DTranspose, concatenate
from keras.regularizers import l1_l2


def LeakyReluConv2D(filters, kernel_size, strides, padding):
    l1_reg = 0
    l2_reg = 1e-6

    def f(inputs):
        conv2d = Conv2D(filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        kernel_regularizer=l1_l2(l1_reg, l2_reg))(inputs)
        return LeakyReLU()(conv2d)
    return f


def LeakyReluConv2DTranspose(filters, kernel_size, strides, padding):
    l1_reg = 0
    l2_reg = 1e-6

    def f(inputs):
        conv2d = Conv2DTranspose(filters=filters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding,
                                 kernel_regularizer=l1_l2(l1_reg, l2_reg))(inputs)
        return LeakyReLU()(conv2d)
    return f


def fCreateEncoder_Block(filters, strides=(1, 1), kernel_size=(3, 3), padding='same'):
    l1_reg = 0
    l2_reg = 1e-6

    def f(inputs):
        conv_1 = Conv2D(filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        kernel_regularizer=l1_l2(l1_reg, l2_reg))(inputs)
        active_1 =  LeakyReLU()(conv_1)

        conv_2 = Conv2D(filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        kernel_regularizer=l1_l2(l1_reg, l2_reg))(active_1)
        active_2 =  LeakyReLU()(conv_2)

        output = concatenate([inputs, active_2], axis=1)
        return output
    return f


def fCreateEncoder_DownConv_Block(filters, strides=(2, 2), kernel_size=(2, 2), padding='valid'):
    l1_reg = 0
    l2_reg = 1e-6

    def f(inputs):
        conv_1 = Conv2D(filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        kernel_regularizer=l1_l2(l1_reg, l2_reg))(inputs)
        output = LeakyReLU()(conv_1)

        return output
    return f



def fCreateConv2DTranspose(filters, strides, kernel_size=(3, 3), padding='same'):
    l1_reg = 0
    l2_reg = 1e-6

    def f(inputs):
        conv2d = Conv2DTranspose(filters=filters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding,
                                 kernel_regularizer=l1_l2(l1_reg, l2_reg))(inputs)
        return LeakyReLU()(conv2d)
    return f

