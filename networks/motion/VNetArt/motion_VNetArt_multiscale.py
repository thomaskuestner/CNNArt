import os.path
import scipy.io as sio
import numpy as np
import keras
import keras.optimizers
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten,   Dropout, Lambda, Reshape
from keras.activations import relu, elu, softmax
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.initializers import Constant
from keras.layers import  concatenate, add, average
from keras.layers.convolutional import Conv3D, UpSampling3D, MaxPooling3D, Cropping3D, ZeroPadding3D
from keras.regularizers import l1_l2,l2
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

def fgetKernelNumber(): # identical for fCreateVNet_Block and fCreateVNet_DownConv_Block
    KernelNumber = [32, 64, 128]
    return KernelNumber

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

def fTrain(X_train, Y_train, X_test, Y_test, sOutPath, patchSize, batchSizes=None, learningRates=None, iEpochs=None,sInPaths=None,sInPaths_valid=None, CV_Patient=0,
           X_train_p2=None, y_train_p2=None, X_test_p2=None, y_test_p2=None, patchSize_down=None, ScaleFactor=None):

    batchSizes = [64] if batchSizes is None else batchSizes
    learningRates = [0.01] if learningRates is None else learningRates
    iEpochs = 300 if iEpochs is None else iEpochs

    # change the shape of the dataset
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    Y_train = np.asarray([Y_train[:], np.abs(np.asarray(Y_train[:], dtype=np.float32) - 1)]).T
    Y_test = np.asarray([Y_test[:], np.abs(np.asarray(Y_test[:], dtype=np.float32) - 1)]).T
    if len(X_train_p2) != 0:
        X_train_p2 = np.expand_dims(X_train_p2, axis=1)
        X_test_p2 = np.expand_dims(X_test_p2, axis=1)
        y_train_p2 = np.asarray([y_train_p2[:], np.abs(np.asarray(y_train_p2[:], dtype=np.float32) - 1)]).T
        y_test_p2 = np.asarray([y_test_p2[:], np.abs(np.asarray(y_test_p2[:], dtype=np.float32) - 1)]).T

    for iBatch in batchSizes:
        for iLearn in learningRates:
            fTrainInner(sOutPath, learningRate=iLearn, X_train=X_train, y_train=Y_train, X_test=X_test, y_test=Y_test, patchSize=patchSize,batchSize=iBatch, iEpochs=iEpochs, CV_Patient=CV_Patient,
                        X_train_p2=X_train_p2, y_train_p2=y_train_p2, X_test_p2=X_test_p2, y_test_p2=y_test_p2, patchSize_down=patchSize_down,ScaleFactor=ScaleFactor)

def fTrainInner(sOutPath, learningRate=0.001, patchSize=None, sInPaths=None, sInPaths_valid=None, X_train=None, y_train=None, X_test=None, y_test=None,  batchSize=64, iEpochs=299, CV_Patient=0
                , X_train_p2=None, y_train_p2=None, X_test_p2=None, y_test_p2=None, patchSize_down=None, ScaleFactor=None):
    '''train a model with training data X_train with labels Y_train. Validation Data should get the keywords Y_test and X_test'''

    print('Training VNet multiscale')
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath,sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)
    model_name = sPath + '/' + sFilename  + '/' + sFilename + '_VNet_MS' +'_lr_' + str(learningRate) + '_bs_' + str(batchSize)
    if CV_Patient != 0: model_name = model_name +'_'+ 'CV' + str(CV_Patient)# determine if crossValPatient is used...
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '_json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        print('----------already trained->go to next----------')
        return

    model = fCreateModel(patchSize, patchSize_down=patchSize_down, ScaleFactor=ScaleFactor, learningRate=learningRate, optimizer='Adam')
    opti, loss = fGetOptimizerAndLoss(optimizer='Adam', learningRate=learningRate)  # loss cat_crosent default
    model.compile(optimizer=opti, loss=loss, metrics=['accuracy'])
    model.summary()

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
    callbacks.append(ModelCheckpoint('/no_backup/s1241/checkpoints/checker.hdf5', monitor='val_acc', verbose=0,
       period=5, save_best_only=True))# overrides the last checkpoint, its just for security
    callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-4, verbose=1))

    result =model.fit(x = [X_train, X_train_p2],
                      y = y_train,
                      validation_data=([X_test, X_test_p2], y_test),
                      epochs=iEpochs,
                      batch_size=batchSize,
                      callbacks=callbacks,
                      verbose=1)

    print('\nscore and acc on test set:')
    score_test, acc_test = model.evaluate([X_test, X_test_p2], y_test, batch_size=batchSize, verbose=1)
    print('\npredict class probabillities:')
    prob_test = model.predict([X_test,X_test_p2], batch_size=batchSize, verbose=1)

    # save model
    json_string = model.to_json()
    open(model_json +'.txt', 'w').write(json_string)
    model.save_weights(weight_name, overwrite=True)

    # matlab
    print('Saving results: ' + model_name)
    sio.savemat(model_name, {'model_settings': model_json,
                             'model': model_all,
                             'weights': weight_name,
                             'training_result': result.history,
                             'val_loss': score_test,
                             'val_acc': acc_test,
                             'prob_test': prob_test})

def fPredict(X_test,y_test, model_name, sOutPath, batchSize=64, X_test_p2=[], y_test_p2=[], patchSize=[]):
    # Takes an already trained model and computes the loss and Accuracy over the samples X with their Labels y

    X = np.expand_dims(X_test, axis=1)
    y = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32) - 1)]).T
    X_p2 = np.expand_dims(X_test_p2, axis=1)
    y_p2 = np.asarray([y_test_p2[:], np.abs(np.asarray(y_test_p2[:], dtype=np.float32) - 1)]).T

    weight_name = sOutPath + '/' + model_name + '_weights.h5'
    model_json = sOutPath + '/' + model_name + '_json.txt'
    model_all = sOutPath + '/' + model_name + '_model.h5'

    # load weights and model (new way)
    model_json= open(model_json, 'r')
    model_string=model_json.read()
    model_json.close()
    model = model_from_json(model_string)

    model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.load_weights(weight_name)

    test_loss, p1_loss, p2_loss, p1_acc, p2_acc = model.evaluate([X, X_p2], [y, y_p2],batch_size=batchSize, verbose=1)
    print('p1_loss:'+str(p1_loss)+ '   p1_acc:'+ str(p1_acc)+'   p2_loss:'+str(p2_loss)+ '   p2_acc:'+ str(p2_acc))
    prob_pre = model.predict([X, X_p2], batch_size=batchSize, verbose=1)

    modelSave = sOutPath + '/' + model_name + '_pred.mat'
    print('saving Model:{}'.format(modelSave))
    sio.savemat(modelSave, {'prob_pre': prob_pre, 'p1_loss': p1_loss, 'p1_acc': p1_acc, 'p2_loss': p2_loss, 'p2_acc': p2_acc})
    #model.save(model_all)

def fCreateModel(patchSize, patchSize_down=None, ScaleFactor=1, learningRate=1e-3, optimizer='SGD',
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

def sliceP1(input):
    return input[:input.shape[0] // 2, :]

def sliceP2(input):
    return input[input.shape[0] // 2:, :]

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

def fGetActivation(input_t,  iPReLU=0):
    init=0.25
    if iPReLU == 1:  # one alpha for each channel
        output_t = PReLU(alpha_initializer=Constant(value=init), shared_axes=[2, 3, 4])(input_t)
    elif iPReLU == 2:  # just one alpha for each layer
        output_t = PReLU(alpha_initializer=Constant(value=init), shared_axes=[2, 3, 4, 1])(input_t)
    else:
        output_t = Activation('relu')(input_t)
    return output_t

def fpathway(input_t, dr_rate=0.0, iPReLU=0, l2_reg=1e-6):
    Kernels = fgetKernels()
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

def fCreateVNet_DownConv_Block(input_t, channels, stride, l1_reg=0.0, l2_reg=1e-6, iPReLU=0, dr_rate=0):
    output_t=Dropout(dr_rate)(input_t)
    output_t=Conv3D(channels,
                    kernel_size=stride,
                    strides=stride,
                    weights=None,
                    padding='valid',
                    kernel_regularizer=l1_l2(l1_reg, l2_reg),
                    kernel_initializer='he_normal'
                    )(output_t)
    output_t=fGetActivation(output_t,iPReLU=iPReLU)
    return output_t

def fCreateVNet_Block(input_t, channels, type=1, kernel_size=(3,3,3),l1_reg=0.0, l2_reg=1e-6, iPReLU=0, dr_rate=0):
    tower_t= Dropout(dr_rate)(input_t)
    tower_t = Conv3D(channels,
                           kernel_size=kernel_size,
                           kernel_initializer='he_normal',
                           weights=None,
                           padding='same',
                           strides=(1, 1, 1),
                           kernel_regularizer=l1_l2(l1_reg, l2_reg),
                           )(tower_t)

    tower_t = fGetActivation(tower_t, iPReLU=iPReLU)
    for counter in range(1, type):
        tower_t = Dropout(dr_rate)(tower_t)
        tower_t = Conv3D(channels,
                           kernel_size=kernel_size,
                           kernel_initializer='he_normal',
                           weights=None,
                           padding='same',
                           strides=(1, 1, 1),
                           kernel_regularizer=l1_l2(l1_reg, l2_reg),
                           )(tower_t)
        tower_t = fGetActivation(tower_t, iPReLU=iPReLU)
    tower_t = concatenate([tower_t, input_t], axis=1)
    return tower_t


def fGetOptimizerAndLoss(optimizer,learningRate=0.001, loss='categorical_crossentropy'):
    if optimizer not in ['Adam', 'SGD', 'Adamax', 'Adagrad', 'Adadelta', 'Nadam', 'RMSprop']:
        print('this optimizer does not exist!!!')
        return None
    loss='categorical_crossentropy'

    if optimizer == 'Adamax':  # leave the rest as default values
        opti = keras.optimizers.Adamax(lr=learningRate)
        loss = 'categorical_crossentropy'
    elif optimizer == 'SGD':
        opti = keras.optimizers.SGD(lr=learningRate, momentum=0.9, decay=5e-5)
        loss = 'categorical_crossentropy'
    elif optimizer == 'Adagrad':
        opti = keras.optimizers.Adagrad(lr=learningRate)
    elif optimizer == 'Adadelta':
        opti = keras.optimizers.Adadelta(lr=learningRate)
    elif optimizer == 'Adam':
        opti = keras.optimizers.Adam(lr=learningRate, decay=5e-5)
        loss = 'categorical_crossentropy'
    elif optimizer == 'Nadam':
        opti = keras.optimizers.Nadam(lr=learningRate)
        loss = 'categorical_crossentropy'
    elif optimizer == 'RMSprop':
        opti = keras.optimizers.RMSprop(lr=learningRate)
    return opti, loss