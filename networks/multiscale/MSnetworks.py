# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 15:53:28 2018

@author: cyf

All the network models about multiscale
"""

import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda, Reshape
from keras.activations import relu, elu, softmax
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.initializers import Constant
from keras.layers import  concatenate, add, average, GlobalAveragePooling3D
from keras.layers.convolutional import Conv3D, UpSampling3D, MaxPooling3D, Conv3DTranspose, Cropping3D
from keras.regularizers import l1_l2,l2
import theano.tensor as T
from networks.motion.VNetArt.motion_VNetArt import fGetActivation, fCreateVNet_Block, fCreateVNet_DownConv_Block


def fgetKernelNumber():
    # KN[0] for conv with kernel 1X1X1, KN[1] for conv with kernel 3X3X3 or 5X5X5
    KernelNumber = [32, 64, 128]
    return KernelNumber
def fgetLayerNumConv():
    LayerNum = 2
    return LayerNum
def fgetLayerNumIncep():
    LayerNum = 2
    return LayerNum

# Parameters for fCreateVNet_DownConv_Block, which influence the input size of the 2nd pathway
def fgetKernels():
    Kernels = fgetStrides()
    return Kernels
def fgetStrides():
    Strides = [[2,2,2], [2,2,1], [2,2,1]]
    return Strides
def fgetLayerNum():
    LayerNum = 3
    return LayerNum

# Parameters for fCreateVNet_Block, the output size is as same as input size
def fgetKernelsUnscaled():
    Kernels = [[3,3,3], [3,3,3]]
    return Kernels
def fgetStridesUnscaled():
    Strides = [[1,1,1], [1,1,1]]
    return Strides
def fgetLayerNumUnscaled():
    LayerNum = 2
    return LayerNum


# Models of dual-pathway VNet
def fCreateModel_VNet_MultiPath(patchSize, patchSize_down=None, ScaleFactor=1, learningRate=1e-3, optimizer='SGD',
                     dr_rate=0.0, input_dr_rate=0.0, max_norm=5, iPReLU=0, l2_reg=1e-6):
    # Total params: 2,503,010

    input_orig = Input(shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2])))
    path_orig_output = fpathway(input_orig)
    input_down = Input(shape=(1, int(patchSize_down[0]), int(patchSize_down[1]), int(patchSize_down[2])))
    path_down = fpathway(input_down)
    path_down_output = fUpSample(path_down, ScaleFactor)
    multi_scale_connect = fconcatenate(path_orig_output, path_down_output)

    # fully connect layer as dense
    flat_out = Flatten()(multi_scale_connect)
    dropout_out = Dropout(dr_rate)(flat_out)
    dense_out = Dense(units=2,
                          kernel_initializer='normal',
                          kernel_regularizer=l2(l2_reg))(dropout_out)

    output_fc = Activation('softmax')(dense_out)
    cnn_ms = Model(inputs=[input_orig, input_down], outputs=output_fc)
    return cnn_ms

def fconcatenate(path_orig, path_down):
    if path_orig._keras_shape == path_down._keras_shape:
        path_down_cropped = path_down
    else:
        crop_x_1 = int(np.ceil((path_down._keras_shape[2] - path_orig._keras_shape[2]) / 2))
        crop_x_0 = path_down._keras_shape[2] - path_orig._keras_shape[2] - crop_x_1
        crop_y_1 = int(np.ceil((path_down._keras_shape[3] - path_orig._keras_shape[3]) / 2))
        crop_y_0 = path_down._keras_shape[3] - path_orig._keras_shape[3] - crop_y_1
        crop_z_1 = int(np.ceil((path_down._keras_shape[4] - path_orig._keras_shape[4]) / 2))
        crop_z_0 = path_down._keras_shape[4] - path_orig._keras_shape[4] - crop_z_1
        path_down_cropped = Cropping3D(cropping=((crop_x_0, crop_x_1), (crop_y_0, crop_y_1), (crop_z_0, crop_z_1)))(path_down)
    connected = average([path_orig, path_down_cropped])
    return connected

def fUpSample(up_in, factor, method='repeat'):
    factor = int(np.round(1 / factor))
    if method == 'repeat':
        up_out = UpSampling3D(size=(factor, factor, factor), data_format='channels_first')(up_in)
        #else:  use inteporlation
        #up_out = scaling.fscalingLayer3D(up_in, factor, [up_in._keras_shape[2],up_in._keras_shape[3],up_in._keras_shape[4]])
    return up_out

def fpathway(input_t, dr_rate=0.0, iPReLU=0, l2_reg=1e-6):
    Strides = fgetStrides()
    KernelNumber = fgetKernelNumber()
    after_res1_t = fCreateVNet_Block(input_t, KernelNumber[0], type=fgetLayerNumUnscaled(), iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    after_DownConv1_t = fCreateVNet_DownConv_Block(after_res1_t, after_res1_t._keras_shape[1], Strides[0],
                                                     iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_res2_t = fCreateVNet_Block(after_DownConv1_t, KernelNumber[1], type=fgetLayerNumUnscaled(), iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    after_DownConv2_t = fCreateVNet_DownConv_Block(after_res2_t, after_res2_t._keras_shape[1], Strides[1],
                                                     iPReLU=iPReLU, l2_reg=l2_reg, dr_rate=dr_rate)

    after_res3_t = fCreateVNet_Block(after_DownConv2_t, KernelNumber[2], type=fgetLayerNumUnscaled(), iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    after_DownConv3_t = fCreateVNet_DownConv_Block(after_res3_t, after_res3_t._keras_shape[1], Strides[2],
                                                     iPReLU=iPReLU, l2_reg=l2_reg, dr_rate=dr_rate)
    return after_DownConv3_t
 
    
# Models of SPP    
def fCreateModel_SPP(patchSize,dr_rate=0.0, iPReLU=0, l2_reg=1e-6):
    # Total params: 1,036,856
    # The third down sampling convolutional layer is replaced by the SPP module
    Strides = fgetStrides()
    kernelnumber = fgetKernelNumber()
    inp = Input(shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2])))

    after_Conv_1 = fCreateVNet_Block(inp, kernelnumber[0], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_1 = fCreateVNet_DownConv_Block(after_Conv_1, after_Conv_1._keras_shape[1], Strides[0],
                                                     iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Conv_2 = fCreateVNet_Block(after_DownConv_1, kernelnumber[1], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_2 = fCreateVNet_DownConv_Block(after_Conv_2, after_Conv_2._keras_shape[1], Strides[1],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Conv_3 = fCreateVNet_Block(after_DownConv_2, kernelnumber[2], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_3 = fSPP(after_Conv_3, level=3)

    dropout_out = Dropout(dr_rate)(after_DownConv_3)
    dense_out = Dense(units=2,
                          kernel_initializer='normal',
                          kernel_regularizer=l2(l2_reg))(dropout_out)

    outp = Activation('softmax')(dense_out)
    cnn_spp = Model(inputs=inp, outputs=outp)
    return cnn_spp

def fCreateModel_SPP_MultiPath(patchSize, patchSize2, dr_rate=0.0, iPReLU=0, l2_reg=1e-6):
    # Total params: 2,073,710
    # There are 2 pathway, whose receptive fields are in multiple relation.
    # Their outputs are averaged as the final prediction
    # The third down sampling convolutional layer in each pathway is replaced by the SPP module
    Strides = fgetStrides()
    kernelnumber = fgetKernelNumber()
    
    sharedConv1 = fCreateVNet_Block
    sharedDown1 = fCreateVNet_DownConv_Block
    sharedConv2 = fCreateVNet_Block
    sharedDown2 = fCreateVNet_DownConv_Block
    sharedConv3 = fCreateVNet_Block
    sharedSPP = fSPP
    
    inp1 = Input(shape=(1, patchSize[0], patchSize[1], patchSize[2]))
    inp1_Conv_1 = sharedConv1(inp1, kernelnumber[0], type=fgetLayerNumConv(), l2_reg=l2_reg)
    inp1_DownConv_1 = sharedDown1(inp1_Conv_1, inp1_Conv_1._keras_shape[1], Strides[0],
                                                     iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    inp1_Conv_2 = sharedConv2(inp1_DownConv_1, kernelnumber[1], type=fgetLayerNumConv(), l2_reg=l2_reg)
    inp1_DownConv_2 = sharedDown2(inp1_Conv_2, inp1_Conv_2._keras_shape[1], Strides[1],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    inp1_Conv_3 = sharedConv3(inp1_DownConv_2, kernelnumber[2], type=fgetLayerNumConv(), l2_reg=l2_reg)
    inp1_SPP = sharedSPP(inp1_Conv_3, level=3)
    
    inp2 = Input(shape=(1, patchSize2[0], patchSize2[1], patchSize2[2]))
    inp2_Conv_1 = sharedConv1(inp2, kernelnumber[0], type=fgetLayerNumConv(), l2_reg=l2_reg)
    inp2_DownConv_1 = sharedDown1(inp2_Conv_1, inp2_Conv_1._keras_shape[1], Strides[0],
                                                     iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    inp2_Conv_2 = sharedConv2(inp2_DownConv_1, kernelnumber[1], type=fgetLayerNumConv(), l2_reg=l2_reg)
    inp2_DownConv_2 = sharedDown2(inp2_Conv_2, inp2_Conv_2._keras_shape[1], Strides[1],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    inp2_Conv_3 = sharedConv3(inp2_DownConv_2, kernelnumber[2], type=fgetLayerNumConv(), l2_reg=l2_reg)
    inp2_SPP = sharedSPP(inp2_Conv_3, level=3)    
    SPP_aver = average([inp1_SPP, inp2_SPP])
    
    dropout_out = Dropout(dr_rate)(SPP_aver)
    dense_out = Dense(units=2,
                          kernel_initializer='normal',
                          kernel_regularizer=l2(l2_reg))(dropout_out)
    output_fc = Activation('softmax')(dense_out)
    model_shared = Model(inputs=[inp1, inp2], outputs = output_fc)    
    return model_shared

def fSPP(inp, level=3):
    inshape = inp._keras_shape[2:]
    Kernel = [[0] * 3 for i in range(level)]
    Stride = [[0] * 3 for i in range(level)]
    SPPout = T.tensor5()
    for iLevel in range(level):
        Kernel[iLevel] = np.ceil(np.divide(inshape, iLevel+1, dtype = float)).astype(int)
        Stride[iLevel] = np.floor(np.divide(inshape, iLevel+1, dtype = float)).astype(int)
        if inshape[2]%3==2:
            Kernel[2][2] = Kernel[2][2] + 1
        poolLevel = MaxPooling3D(pool_size=Kernel[iLevel], strides=Stride[iLevel])(inp)
        if iLevel == 0:
            SPPout = Flatten()(poolLevel)
        else:
            poolFlat = Flatten()(poolLevel)
            SPPout = concatenate([SPPout,poolFlat], axis=1)
    return SPPout


# Models of FCN
def fCreateModel_FCN_simple(patchSize,dr_rate=0.0, iPReLU=0, l1_reg=0.0, l2_reg=1e-6):
    # Total params: 1,223,831
    # Replace the dense layer with a convolutional layer with filters=2 for the two classes
    Strides = fgetStrides()
    kernelnumber = fgetKernelNumber()
    inp = Input(shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2])))

    after_Conv_1 = fCreateVNet_Block(inp, kernelnumber[0], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_1 = fCreateVNet_DownConv_Block(after_Conv_1, after_Conv_1._keras_shape[1], Strides[0],
                                                     iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Conv_2 = fCreateVNet_Block(after_DownConv_1, kernelnumber[1], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_2 = fCreateVNet_DownConv_Block(after_Conv_2, after_Conv_2._keras_shape[1], Strides[1],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Conv_3 = fCreateVNet_Block(after_DownConv_2, kernelnumber[2], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_3 = fCreateVNet_DownConv_Block(after_Conv_3, after_Conv_3._keras_shape[1], Strides[2],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    dropout_out = Dropout(dr_rate)(after_DownConv_3)
    fclayer = Conv3D(2,
                       kernel_size=(1,1,1),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l1_l2(l1_reg, l2_reg),
                       )(dropout_out)
    fclayer = GlobalAveragePooling3D()(fclayer)
    outp = Activation('softmax')(fclayer)
    cnn_spp = Model(inputs=inp, outputs=outp)
    return cnn_spp

def fCreateModel_FCN_MultiFM(patchSize, dr_rate=0.0, iPReLU=0,l1_reg=0, l2_reg=1e-6):
    # Total params: 1,420,549
    # The dense layer is repleced by a convolutional layer with filters=2 for the two classes
    # The FM from the third down scaled convolutional layer is upsempled by deconvolution and
    # added with the FM from the second down scaled convolutional layer.
    # The combined FM goes through a convolutional layer with filters=2 for the two classes
    # The two predictions are averages as the final result.
    Strides = fgetStrides()
    kernelnumber = fgetKernelNumber()
    inp = Input(shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2])))

    after_Conv_1 = fCreateVNet_Block(inp, kernelnumber[0], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_1 = fCreateVNet_DownConv_Block(after_Conv_1, after_Conv_1._keras_shape[1], Strides[0],
                                                     iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Conv_2 = fCreateVNet_Block(after_DownConv_1, kernelnumber[1], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_2 = fCreateVNet_DownConv_Block(after_Conv_2, after_Conv_2._keras_shape[1], Strides[1],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Conv_3 = fCreateVNet_Block(after_DownConv_2, kernelnumber[2], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_3 = fCreateVNet_DownConv_Block(after_Conv_3, after_Conv_3._keras_shape[1], Strides[2],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    # fully convolution over the FM from the deepest level
    dropout_out1 = Dropout(dr_rate)(after_DownConv_3)
    fclayer1 = Conv3D(2,
                       kernel_size=(1,1,1),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l1_l2(l1_reg, l2_reg),
                       )(dropout_out1)
    fclayer1 = GlobalAveragePooling3D()(fclayer1)
    
    # Upsample FM from the deepest level, add with FM from level 2, 
    UpedFM_Level3 = Conv3DTranspose(filters=97, kernel_size=(3,3,1), strides=(2,2,1), padding='same')(after_DownConv_3)
    conbined_FM_Level23 = add([UpedFM_Level3, after_DownConv_2])    
    fclayer2 = Conv3D(2,
                       kernel_size=(1,1,1),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l1_l2(l1_reg, l2_reg),
                       )(conbined_FM_Level23)
    fclayer2 = GlobalAveragePooling3D()(fclayer2)

    # combine the two predictions using average
    fcl_aver = average([fclayer1, fclayer2])
    predict = Activation('softmax')(fcl_aver)
    cnn_fcl_msfm = Model(inputs=inp, outputs=predict)
    return cnn_fcl_msfm

def fCreateModel_FCN_MultiFM_MultiPath(patchSize, patchSize_down, dr_rate=0.0, iPReLU=0, l1_reg=0, l2_reg=1e-6):
    # Total params: 2,841,098
    # The dense layer is repleced by a convolutional layer with filters=2 for the two classes
    # The FM from the third down scaled convolutional layer is upsempled by deconvolution and
    # added with the FM from the second down scaled convolutional layer.
    # The combined FM goes through a convolutional layer with filters=2 for the two classes
    # The four predictions from the two pathways are averages as the final result.
    Strides = fgetStrides()
    kernelnumber = fgetKernelNumber()
    sharedConv1 = fCreateVNet_Block
    sharedDown1 = fCreateVNet_DownConv_Block
    sharedConv2 = fCreateVNet_Block
    sharedDown2 = fCreateVNet_DownConv_Block
    sharedConv3 = fCreateVNet_Block
    sharedDown3 = fCreateVNet_DownConv_Block

    inp1 = Input(shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2])))
    after_1Conv_1 = sharedConv1(inp1, kernelnumber[0], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_1DownConv_1 = sharedDown1(after_1Conv_1, after_1Conv_1._keras_shape[1], Strides[0],
                                                  iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_1Conv_2 = sharedConv2(after_1DownConv_1, kernelnumber[1], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_1DownConv_2 = sharedDown2(after_1Conv_2, after_1Conv_2._keras_shape[1], Strides[1],
                                                  iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_1Conv_3 = sharedConv3(after_1DownConv_2, kernelnumber[2], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_1DownConv_3 = sharedDown3(after_1Conv_3, after_1Conv_3._keras_shape[1], Strides[2],
                                                  iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    inp2 = Input(shape=(1, int(patchSize_down[0]), int(patchSize_down[1]), int(patchSize_down[2])))
    after_2Conv_1 = sharedConv1(inp2, kernelnumber[0], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_2DownConv_1 = sharedDown1(after_2Conv_1, after_2Conv_1._keras_shape[1], Strides[0],
                                                  iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_2Conv_2 = sharedConv2(after_2DownConv_1, kernelnumber[1], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_2DownConv_2 = sharedDown2(after_2Conv_2, after_2Conv_2._keras_shape[1], Strides[1],
                                                  iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_2Conv_3 = sharedConv3(after_2DownConv_2, kernelnumber[2], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_2DownConv_3 = sharedDown3(after_2Conv_3, after_2Conv_3._keras_shape[1], Strides[2],
                                                  iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    # fully convolution over the FM from the deepest level
    dropout_out1 = Dropout(dr_rate)(after_1DownConv_3)
    fclayer1 = Conv3D(2,
                      kernel_size=(1, 1, 1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='valid',
                      strides=(1, 1, 1),
                      kernel_regularizer=l1_l2(l1_reg, l2_reg),
                      )(dropout_out1)
    fclayer1 = GlobalAveragePooling3D()(fclayer1)

    # Upsample FM from the deepest level, add with FM from level 2,
    UpedFM_1Level3 = Conv3DTranspose(filters=97, kernel_size=(3, 3, 1), strides=(2, 2, 1), padding='same')(
        after_1DownConv_3)
    conbined_FM_1Level23 = add([UpedFM_1Level3, after_1DownConv_2])
    fclayer2 = Conv3D(2,
                      kernel_size=(1, 1, 1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='valid',
                      strides=(1, 1, 1),
                      kernel_regularizer=l1_l2(l1_reg, l2_reg),
                      )(conbined_FM_1Level23)
    fclayer2 = GlobalAveragePooling3D()(fclayer2)

    dropout_out2 = Dropout(dr_rate)(after_2DownConv_3)
    fclayer3 = Conv3D(2,
                      kernel_size=(1, 1, 1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='valid',
                      strides=(1, 1, 1),
                      kernel_regularizer=l1_l2(l1_reg, l2_reg),
                      )(dropout_out2)
    fclayer3 = GlobalAveragePooling3D()(fclayer3)

    # Upsample FM from the deepest level, add with FM from level 2,
    UpedFM_2Level3 = Conv3DTranspose(filters=97, kernel_size=(3, 3, 1), strides=(2, 2, 1), padding='same')(
        after_2DownConv_3)
    conbined_FM_2Level23 = add([UpedFM_2Level3, after_2DownConv_2])
    fclayer4 = Conv3D(2,
                      kernel_size=(1, 1, 1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='valid',
                      strides=(1, 1, 1),
                      kernel_regularizer=l1_l2(l1_reg, l2_reg),
                      )(conbined_FM_2Level23)
    fclayer4 = GlobalAveragePooling3D()(fclayer4)
    # combine the two predictions using average
    fcl_aver = average([fclayer1, fclayer2, fclayer3, fclayer4])
    predict = Activation('softmax')(fcl_aver)
    cnn_fcl_2p = Model(inputs=[inp1, inp2], outputs=predict)
    return cnn_fcl_2p


# models of inception module
def fCreateModel_Inception_Archi1(patchSize,dr_rate=0.0, iPReLU=0, l2_reg=1e-6):
    # Total params: 5,483,161
    # Each convolution layer before down sampling is replaced by the inception block,
    # all other convolution layers are reserved.
    Strides = fgetStrides()
    kernelnumber = fgetKernelNumber()
    inp = Input(shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2])))

    after_Incep_1 =  fConvIncep(inp, KB=kernelnumber[0], layernum=fgetLayerNumIncep(), l2_reg=l2_reg)
    after_DownConv_1 = fCreateVNet_DownConv_Block(after_Incep_1, after_Incep_1._keras_shape[1], Strides[0],
                                                     iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Incep_2 = fConvIncep(after_DownConv_1, KB=kernelnumber[1], layernum=fgetLayerNumIncep(), l2_reg=l2_reg)
    after_DownConv_2 = fCreateVNet_DownConv_Block(after_Incep_2, after_Incep_2._keras_shape[1], Strides[1],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Incep_3 = fConvIncep(after_DownConv_2, KB=kernelnumber[2], layernum=fgetLayerNumIncep(), l2_reg=l2_reg)
    after_DownConv_2 = fCreateVNet_DownConv_Block(after_Incep_3, after_Incep_3._keras_shape[1], Strides[2],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    # fully connect layer as dense
    flat_out = Flatten()(after_DownConv_2)
    dropout_out = Dropout(dr_rate)(flat_out)
    dense_out = Dense(units=2,
                          kernel_initializer='normal',
                          kernel_regularizer=l2(l2_reg))(dropout_out)

    outp = Activation('softmax')(dense_out)
    cnn_incep = Model(inputs=inp, outputs=outp)
    return cnn_incep

def fCreateModel_Inception_Archi2(patchSize,dr_rate=0.0, iPReLU=0, l2_reg=1e-6):
    # Total params: 2,802,041
    # The three higher convolution layers except down sampling are replaced by the inception block,
    # the three lower convolution layers are reserved.
    # In work of GoogLeNet, it's beneficial for memory efficiency during training
    # to start using inception modules at higher layers and to keep lower layers in traditional convolutional fashion.
    Strides = fgetStrides()
    kernelnumber = fgetKernelNumber()
    inp = Input(shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2])))

    after_Incep_1 =  fCreateVNet_Block(inp, kernelnumber[0], type=fgetLayerNumIncep(), iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    after_DownConv_1 = fCreateVNet_DownConv_Block(after_Incep_1, after_Incep_1._keras_shape[1], Strides[0],
                                                     iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Incep_2 = fConvIncep(after_DownConv_1, KB=kernelnumber[1], layernum=fgetLayerNumIncep(), l2_reg=l2_reg)
    after_DownConv_2 = fCreateVNet_DownConv_Block(after_Incep_2, after_Incep_2._keras_shape[1], Strides[1],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Incep_3 = fIncepChain(after_DownConv_2, layernum=fgetLayerNumIncep(), l2_reg=l2_reg, iPReLU=iPReLU)
    after_DownConv_2 = fCreateVNet_DownConv_Block(after_Incep_3, after_Incep_3._keras_shape[1], Strides[2],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    # fully connect layer as dense
    flat_out = Flatten()(after_DownConv_2)
    dropout_out = Dropout(dr_rate)(flat_out)
    dense_out = Dense(units=2,
                          kernel_initializer='normal',
                          kernel_regularizer=l2(l2_reg))(dropout_out)

    outp = Activation('softmax')(dense_out)
    cnn_incep = Model(inputs=inp, outputs=outp)
    return cnn_incep

def fConvIncep(input_t, KB=64, layernum=2, l1_reg=0.0, l2_reg=1e-6, iPReLU=0):
    tower_t = Conv3D(filters=KB,
                     kernel_size=[2,2,1],
                     kernel_initializer='he_normal',
                     weights=None,
                     padding='same',
                     strides=(1, 1, 1),
                     kernel_regularizer=l1_l2(l1_reg, l2_reg),
                     )(input_t)
    incep = fGetActivation(tower_t, iPReLU=iPReLU)

    for counter in range(1,layernum):
        incep = InceptionBlock(incep, l1_reg=l1_reg, l2_reg=l2_reg)

    incepblock_out = concatenate([incep, input_t], axis=1)
    return incepblock_out

def fIncepChain(input_t, layernum=2, l1_reg=0.0, l2_reg=1e-6, iPReLU=0):
    incep = InceptionBlock(input_t, l1_reg=l1_reg, l2_reg=l2_reg)

    for counter in range(1,layernum):
        incep = InceptionBlock(incep, l1_reg=l1_reg, l2_reg=l2_reg)

    incepblock_out = concatenate([incep, input_t], axis=1)
    return incepblock_out

def InceptionBlock(inp, l1_reg=0.0, l2_reg=1e-6):
    KN = fgetKernelNumber()
    branch1 = Conv3D(filters=KN[0], kernel_size=(1,1,1), kernel_initializer='he_normal', weights=None,padding='same',
                     strides=(1,1,1),kernel_regularizer=l1_l2(l1_reg, l2_reg),activation='relu')(inp)

    branch3 = Conv3D(filters=KN[0], kernel_size=(1, 1, 1), kernel_initializer='he_normal', weights=None, padding='same',
                     strides=(1, 1, 1), kernel_regularizer=l1_l2(l1_reg, l2_reg), activation='relu')(inp)
    branch3 = Conv3D(filters=KN[2], kernel_size=(3, 3, 3), kernel_initializer='he_normal', weights=None, padding='same',
                     strides=(1, 1, 1), kernel_regularizer=l1_l2(l1_reg, l2_reg), activation='relu')(branch3)

    branch5 = Conv3D(filters=KN[0], kernel_size=(1, 1, 1), kernel_initializer='he_normal', weights=None, padding='same',
                     strides=(1, 1, 1), kernel_regularizer=l1_l2(l1_reg, l2_reg), activation='relu')(inp)
    branch5 = Conv3D(filters=KN[1], kernel_size=(5, 5, 5), kernel_initializer='he_normal', weights=None, padding='same',
                     strides=(1, 1, 1), kernel_regularizer=l1_l2(l1_reg, l2_reg), activation='relu')(branch5)

    branchpool = MaxPooling3D(pool_size=(3,3,3),strides=(1,1,1),padding='same',data_format='channels_first')(inp)
    branchpool = Conv3D(filters=KN[0], kernel_size=(1, 1, 1), kernel_initializer='he_normal', weights=None, padding='same',
                     strides=(1, 1, 1), kernel_regularizer=l1_l2(l1_reg, l2_reg), activation='relu')(branchpool)
    out = concatenate([branch1, branch3, branch5, branchpool], axis=1)
    return out