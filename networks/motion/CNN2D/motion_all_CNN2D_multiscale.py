# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 2018

@author: Thomas Kuestner
"""
import os.path
import scipy.io as sio
import numpy as np  # for algebraic operations, matrices
import keras
import keras.optimizers
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda, Reshape
from keras.activations import relu, elu, softmax
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.initializers import Constant
from keras.layers.merge import concatenate, add
from keras.layers.convolutional import Conv3D, Conv2D, MaxPooling3D, MaxPooling2D, ZeroPadding3D, UpSampling2D, Cropping2D
from keras.regularizers import l1_l2, l2
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def fgetKernelNumber():
    KernelNumber = [32, 64, 128]
    return KernelNumber

def fgetKernels():
    Kernels = [[14,14], [7,7], [3,3]]
    return Kernels

def fgetStrides():
    Strides = [[1,1], [1,1], [1,1]]
    return Strides

def fgetLayerNum():
    LayerNum = 3
    return LayerNum

def createModel(patchSize, patchSize_down=None, ScaleFactor=1, learningRate=1e-3, optimizer='SGD',
                     dr_rate=0.0, input_dr_rate=0.0, max_norm=5, iPReLU=0, l2_reg=1e-6):
    input_orig = Input(shape=(1, int(patchSize[0]), int(patchSize[1])))
    input_down = Input(shape=(1, int(patchSize_down[0]), int(patchSize_down[1])))

    path_down = fConveBlock(input_down)
    path_down_output = fUpSample(path_down, ScaleFactor)
    path_orig_output = fConveBlock(input_orig)
    path_orig_output_shape=np.shape(path_orig_output)
    path_down_output_shape=np.shape(path_down_output)
    crop_x_0, crop_x_1, crop_y_0, crop_y_1 = fcalculateToCropSize(path_orig_output_shape, path_down_output_shape)
    path_down_crop = Cropping2D(cropping=((crop_x_0, crop_x_1), (crop_y_0, crop_y_1)))(path_down_output)
    multi_scale_connect = concatenate([path_orig_output, path_down_crop], axis=0)

    # fully connect layer as dense
    flat_out = Flatten()(multi_scale_connect)
    dropout_out = Dropout(dr_rate)(flat_out)
    dense_out = Dense(units=2,
                          kernel_initializer='normal',
                          kernel_regularizer=l2(l2_reg))(dropout_out)
    # Fully connected layer as convo with 1X1 ?

    output_fc = Activation('softmax')(dense_out)
    cnn_ms = Model(inputs=[input_orig, input_down], outputs=[output_fc])
    return cnn_ms

def fcalculateToCropSize(path_orig, path_down):
    if path_orig==path_down:
        crop_x_0=crop_x_1=crop_y_0=crop_y_1=0
    else:
        crop_x_1 = int(np.ceil(path_orig[1]-path_down[1]/2))
        crop_x_0 = path_orig[1]-path_down[1] - crop_x_1
        crop_y_1 = int(np.ceil(path_orig[2]-path_down[2]/2))
        crop_y_0 = path_orig[2]-path_down[2] - crop_y_1
    return crop_x_0, crop_x_1, crop_y_0, crop_y_1
     
def fUpSample(up_in, factor, method = 'repeat'):
    if method == 'repeat':
        up_out = UpSampling2D(size=(factor,factor), data_format='channels_first')
    # else: use inteporlation
    return up_out

def fConveBlock(conv_input,l1_reg=0.0, l2_reg=1e-6, dr_rate=0):
    Kernels = fgetKernels()
    Strides = fgetStrides()
    KernelNumber = fgetKernelNumber()
    # All parameters about kernels and so on are identical with original 2DCNN
    drop_out_1 = Dropout(dr_rate)(conv_input)
    conve_out_1 = Conv2D(KernelNumber[0],
                   kernel_size=Kernels[0],
                   kernel_initializer='he_normal',
                   weights=None,
                   padding='valid',
                   strides=Strides[0],
                   kernel_regularizer=l1_l2(l1_reg, l2_reg)
                   )(drop_out_1)
    # input shape : 1 means grayscale... richtig uebergeben...
    active_out_1 = Activation('relu')(conve_out_1)

    drop_out_2 = Dropout(dr_rate)(active_out_1)
    conve_out_2 = Conv2D(KernelNumber[1],  # learning rate: 0.1 -> 76%
                   kernel_size=Kernels[1],
                   kernel_initializer='he_normal',
                   weights=None,
                   padding='valid',
                   strides=Strides[1],
                   kernel_regularizer=l1_l2(l1_reg, l2_reg)
                   )(drop_out_2)
    active_out_2 = Activation('relu')(conve_out_2)

    drop_out_3 = Dropout(dr_rate)(active_out_2)
    conve_out_3 = Conv2D(KernelNumber[2],  # learning rate: 0.1 -> 76%
                   kernel_size=Kernels[2],
                   kernel_initializer='he_normal',
                   weights=None,
                   padding='valid',
                   strides=Strides[2],
                   kernel_regularizer=l1_l2(l1_reg, l2_reg)
                         )(drop_out_3)
    active_out_3 = Activation('relu')(conve_out_3)
    return active_out_3


def fTrain(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSizes=None, learningRates=None, iEpochs=None,
           CV_Patient=0, X_train_p2=None, y_train_p2=None, X_test_p2=None, y_test_p2=None, patchSize_down=None, ScaleFactor=None):
    # grid search on batch_sizes and learning rates
    # parse inputs
    batchSizes = [64] if batchSizes is None else batchSizes
    learningRates = [0.01] if learningRates is None else learningRates
    iEpochs = 300 if iEpochs is None else iEpochs

    # change the shape of the dataset
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    y_train = np.asarray([y_train[:], np.abs(np.asarray(y_train[:], dtype=np.float32) - 1)]).T
    y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32) - 1)]).T

    for iBatch in batchSizes:
        for iLearn in learningRates:
            fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, iBatch, iLearn, iEpochs,
                        CV_Patient=CV_Patient, X_train_p2=X_train_p2, y_train_p2=y_train_p2, X_test_p2=X_test_p2, y_test_p2=y_test_p2, patchSize_down=patchSize_down,ScaleFactor=ScaleFactor)


def fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSize=None, learningRate=None, iEpochs=None,
                CV_Patient=0, X_train_p2=None, y_train_p2=None, X_test_p2=None, y_test_p2=None, patchSize_down=None, ScaleFactor=None):
    # parse inputs
    batchSize = [64] if batchSize is None else batchSize
    learningRate = [0.01] if learningRate is None else learningRate
    iEpochs = 300 if iEpochs is None else iEpochs

    print('Training 2D CNN')
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)
    model_name = sPath + '/' + sFilename  + '/' + sFilename + '_MS' + '_lr_' + str(learningRate) + '_bs_' + str(batchSize)
    if CV_Patient != 0: model_name = model_name + '_' + 'CV' + str(
        CV_Patient)  # determine if crossValPatient is used...
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '_json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        return

    # create model
    cnn = createModel(patchSize, patchSize_down=patchSize_down, ScaleFactor=ScaleFactor)

    # opti = SGD(lr=learningRate, momentum=1e-8, decay=0.1, nesterov=True);#Adag(lr=0.01, epsilon=1e-06)
    opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
    # callbacks.append(
    #    ModelCheckpoint('/home/sXXXX/no_backup/sXXXX/checkpoints/checker.hdf5', monitor='val_acc', verbose=0,
    #                    period=5, save_best_only=True))  # overrides the last checkpoint, its just for security
    # callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-4, verbose=1))

    cnn.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
    print(cnn.summary)

    result = cnn.fit(x = [X_train, X_train_p2],
                     y = [y_train, y_train_p2],
                     validation_data=([X_test,X_test_p2], [y_test,y_test_p2]),
                     epochs=iEpochs,
                     batch_size=batchSize,
                     callbacks=callbacks,
                     verbose=1)

    loss_test, acc_test = cnn.evaluate([X_test, X_test_p2], [y_test, y_test_p2], batch_size=batchSize)

    prob_test = cnn.predict([X_test,X_test_p2], batch_size=batchSize, verbose=0)

    # save model
    json_string = cnn.to_json()
    open(model_json, 'w').write(json_string)
    # wei = cnn.get_weights()
    cnn.save_weights(weight_name, overwrite=True)
    # cnn.save(model_all) # keras > v0.7

    # matlab
    acc = result.history['acc']
    loss = result.history['loss']
    val_acc = result.history['val_acc']
    val_loss = result.history['val_loss']

    print('Saving results: ' + model_name)
    sio.savemat(model_name, {'model_settings': model_json,
                             'model': model_all,
                             'weights': weight_name,
                             'acc': acc,
                             'loss': loss,
                             'val_acc': val_acc,
                             'val_loss': val_loss,
                             'loss_test': loss_test,
                             'acc_test': acc_test,
                             'prob_test': prob_test})


def fPredict(X_test, y_test, model_name, sOutPath, patchSize, batchSize):
    weight_name = sOutPath + '/' + model_name + '_weights.h5'
    model_json = sOutPath + model_name + '_json'
    model_all = sOutPath + model_name + '_model.h5'

    #    # load weights and model (OLD WAY)
    #    conten = sio.loadmat(model_name)
    #    weig = content['wei']
    #    nSize = weig.shape
    #    weigh = []
    #
    #    for i in drange(0,nSize[1],2):
    #    	w0 = weig[0,i]
    #    	w1 = weig[0,i+1]
    #    	w1=w1.T
    #    	w1 = np.concatenate(w1,axis=0)
    #
    #    	weigh= weigh.extend([w0, w1])
    #
    #    model = model_from_json(model_json)
    #    model.set_weights(weigh)

    # load weights and model (new way)
    # model = model_from_json(model_json)
    model = createModel(patchSize)
    opti = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]

    model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
    model.load_weights(weight_name)

    # load complete model (including weights); keras > 0.7
    # model = load_model(model_all)

    # assume artifact affected shall be tested!
    # y_test = np.ones((len(X_test),1))

    X_test = np.expand_dims(X_test, axis=1)
    y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32) - 1)]).T

    score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batchSize)
    prob_pre = model.predict(X_test, batchSize, 1)

    # modelSave = model_name[:-5] + '_pred.mat'
    modelSave = sOutPath + '/' + model_name + '_pred.mat'
    sio.savemat(modelSave, {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test})
    model.save(model_all)


###############################################################################
## OPTIMIZATIONS ##
###############################################################################
def fHyperasTrain(X_train, Y_train, X_test, Y_test, patchSize):
    # explicitly stated here instead of cnn = createModel() to allow optimization
    cnn = Sequential()
    #    cnn.add(Convolution2D(32,
    #                            14,
    #                            14,
    #                            init='normal',
    #                           # activation='sigmoid',
    #                            weights=None,
    #                            border_mode='valid',
    #                            subsample=(1, 1),
    #                            W_regularizer=l2(1e-6),
    #                            input_shape=(1, patchSize[0,0], patchSize[0,1])))
    #    cnn.add(Activation('relu'))

    cnn.add(Convolution2D(32,  # 64
                          7,
                          7,
                          init='normal',
                          # activation='sigmoid',
                          weights=None,
                          border_mode='valid',
                          subsample=(1, 1),
                          W_regularizer=l2(1e-6)))
    cnn.add(Activation('relu'))
    cnn.add(Convolution2D(64,  # learning rate: 0.1 -> 76%
                          3,
                          3,
                          init='normal',
                          # activation='sigmoid',
                          weights=None,
                          border_mode='valid',
                          subsample=(1, 1),
                          W_regularizer=l2(1e-6)))
    cnn.add(Activation('relu'))

    cnn.add(Convolution2D(128,  # learning rate: 0.1 -> 76%
                          3,
                          3,
                          init='normal',
                          # activation='sigmoid',
                          weights=None,
                          border_mode='valid',
                          subsample=(1, 1),
                          W_regularizer=l2(1e-6)))
    cnn.add(Activation('relu'))

    # cnn.add(pool2(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'))

    cnn.add(Flatten())
    # cnn.add(Dense(input_dim= 100,
    #              output_dim= 100,
    #              init = 'normal',
    #              #activation = 'sigmoid',
    #              W_regularizer='l2'))
    # cnn.add(Activation('sigmoid'))
    cnn.add(Dense(input_dim=100,
                  output_dim=2,
                  init='normal',
                  # activation = 'sigmoid',
                  W_regularizer='l2'))
    cnn.add(Activation('softmax'))

    opti = SGD(lr={{choice([0.1, 0.01, 0.05, 0.005, 0.001])}}, momentum=1e-8, decay=0.1, nesterov=True)
    cnn.compile(loss='categorical_crossentropy',
                optimizer=opti)

    epochs = 300

    result = cnn.fit(X_train, Y_train,
                     batch_size=128,  # {{choice([64, 128])}}
                     nb_epoch=epochs,
                     show_accuracy=True,
                     verbose=2,
                     validation_data=(X_test, Y_test))
    score_test, acc_test = cnn.evaluate(X_test, Y_test, verbose=0)

    return {'loss': -acc_test, 'status': STATUS_OK, 'model': cnn, 'trainresult': result, 'score_test': score_test}


## helper functions
def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
    r += step