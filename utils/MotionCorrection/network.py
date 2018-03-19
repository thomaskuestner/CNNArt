from keras.layers import Conv2D, LeakyReLU, Conv2DTranspose
from keras.regularizers import l1_l2,l2

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

