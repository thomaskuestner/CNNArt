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
from keras.layers import  concatenate, add
from keras.layers.convolutional import Conv3D,Conv2D, MaxPooling3D, MaxPooling2D, ZeroPadding3D
from keras.regularizers import l1_l2,l2
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras import backend as K
K.set_image_dim_ordering('th')  # set channel first reordering

def fTrain(X_train, Y_train, X_test, Y_test, sOutPath, patchSize, batchSizes=None, learningRates=None, iEpochs=None,sInPaths=None,sInPaths_valid=None, CV_Patient=0, model='motion_head'):#rigid for loops for simplicity
    #add for loops here
    batchSizes = [64] if batchSizes is None else batchSizes
    learningRates = [0.01] if learningRates is None else learningRates
    iEpochs = 300 if iEpochs is None else iEpochs

    # change the shape of the dataset
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    Y_train = np.asarray([Y_train[:], np.abs(np.asarray(Y_train[:], dtype=np.float32) - 1)]).T
    Y_test = np.asarray([Y_test[:], np.abs(np.asarray(Y_test[:], dtype=np.float32) - 1)]).T

    for iBatch in batchSizes:
        for iLearn in learningRates:
            cnn = fCreateModel(patchSize, learningRate=iLearn, optimizer='Adam')
            fTrainInner(sOutPath, cnn, learningRate=iLearn, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,batchSize=iBatch, iEpochs=iEpochs, CV_Patient=CV_Patient)

def fTrainInner(sOutPath, model, learningRate=0.001, patchSize=None, sInPaths=None, sInPaths_valid=None, X_train=None, Y_train=None, X_test=None, Y_test=None,  batchSize=64, iEpochs=299, CV_Patient=0):
    '''train a model with training data X_train with labels Y_train. Validation Data should get the keywords Y_test and X_test'''

    print('Training VNet')
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath,sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)
    model_name = sPath + '/' + sFilename  + '/' + sFilename + '_VNet' +'_lr_' + str(learningRate) + '_bs_' + str(batchSize)
    if CV_Patient != 0: model_name = model_name +'_'+ 'CV' + str(CV_Patient)# determine if crossValPatient is used...
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '_json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        print('----------already trained->go to next----------')
        return
    model.summary()

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
    callbacks.append(ModelCheckpoint('checkpoint/checker.hdf5', monitor='val_acc', verbose=0,
        period=5, save_best_only=True))# overrides the last checkpoint, its just for security
    callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-4, verbose=1))

    result =model.fit(X_train,
                         Y_train,
                         validation_data=[X_test, Y_test],
                         epochs=iEpochs,
                         batch_size=batchSize,
                         callbacks=callbacks,
                         verbose=1)

    print('\nscore and acc on test set:')
    score_test, acc_test = model.evaluate(X_test, Y_test, batch_size=batchSize, verbose=1)
    print('\npredict class probabillities:')
    prob_test = model.predict(X_test, batchSize, verbose=1)

    # save model
    json_string = model.to_json()
    open(model_json +'.txt', 'w').write(json_string)

    model.save_weights(weight_name, overwrite=True)


    # matlab
    acc = result.history['acc']
    loss = result.history['loss']
    val_acc = result.history['val_acc']
    val_loss = result.history['val_loss']


    print('\nSaving results: ' + model_name)
    sio.savemat(model_name, {'model_settings': model_json,
                             'model': model_all,
                             'weights': weight_name,
                             'acc_history': acc,
                             'loss_history': loss,
                             'val_acc_history': val_acc,
                             'val_loss_history': val_loss,
                             'loss_test': score_test,
                             'acc_test': acc_test,
                             'prob_test': prob_test})

def fPredict(X_test,y_test,  model_name, sOutPath, batchSize=64,patchSize=[40,40,5]):
    """Takes an already trained model and computes the loss and Accuracy over the samples X with their Labels y
    Input:
        X: Samples to predict on. The shape of X should fit to the input shape of the model
        y: Labels for the Samples. Number of Samples should be equal to the number of samples in X
        sModelPath: (String) full path to a trained keras model. It should be *_json.txt file. there has to be a corresponding *_weights.h5 file in the same directory!
        sOutPath: (String) full path for the Output. It is a *.mat file with the computed loss and accuracy stored.
                    The Output file has the Path 'sOutPath'+ the filename of sModelPath without the '_json.txt' added the suffix '_pred.mat'
        batchSize: Batchsize, number of samples that are processed at once"""
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

    X_test = np.expand_dims(X_test, axis=1)
    y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32) - 1)]).T

    score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batchSize)
    print('loss'+str(score_test)+ '   acc:'+ str(acc_test))
    prob_pre = model.predict(X_test, batch_size=batchSize, verbose=1)

    modelSave = sOutPath + '/' + model_name + '_pred.mat'
    print('saving Model:{}'.format(modelSave))
    sio.savemat(modelSave, {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test})


def fCreateModel(patchSize, learningRate=1e-3, optimizer='SGD',
                     dr_rate=0.0, input_dr_rate=0.0, max_norm=5, iPReLU=0, l2_reg=1e-6):
    l2_reg = 1e-4
    # using SGD lr 0.001
    # motion_head:unkorrigierte Version 3steps with only type(1,1,1)(149K params)--> val_loss: 0.2157 - val_acc: 0.9230
    # motion_head:korrigierte Version type(1,2,2)(266K params) --> val_loss: 0.2336 - val_acc: 0.9149 nach abbruch...
    # double_#channels(type 122) (870,882 params)>
    # functional api...
    input_t = Input(shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2])))

    after_res1_t = fCreateVNet_Block(input_t, 32, type=2, iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    after_DownConv1_t = fCreateVNet_DownConv_Block(after_res1_t, after_res1_t._keras_shape[1], (2, 2, 2),
                                                     iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_res2_t = fCreateVNet_Block(after_DownConv1_t, 64, type=2, iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    after_DownConv2_t = fCreateVNet_DownConv_Block(after_res2_t, after_res2_t._keras_shape[1], (2, 2, 1),
                                                     iPReLU=iPReLU, l2_reg=l2_reg, dr_rate=dr_rate)

    after_res3_t = fCreateVNet_Block(after_DownConv2_t, 128, type=2, iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    after_DownConv3_t = fCreateVNet_DownConv_Block(after_res3_t, after_res3_t._keras_shape[1], (2, 2, 1),
                                                     iPReLU=iPReLU, l2_reg=l2_reg, dr_rate=dr_rate)

    after_flat_t = Flatten()(after_DownConv3_t)
    after_dense_t = Dropout(dr_rate)(after_flat_t)
    after_dense_t = Dense(units=2,
                          kernel_initializer='normal',
                          kernel_regularizer=l2(l2_reg))(after_dense_t)
    output_t = Activation('softmax')(after_dense_t)

    cnn = Model(inputs=[input_t], outputs=[output_t])

    opti, loss = fGetOptimizerAndLoss(optimizer, learningRate=learningRate)  # loss cat_crosent default
    cnn.compile(optimizer=opti, loss=loss, metrics=['accuracy'])
    sArchiSpecs = '_t222_l2{}_dr{}'.format(l2_reg, dr_rate)
    return cnn

def fGetActivation(input_t,  iPReLU=0):
    init=0.25
    if iPReLU == 1:  # one alpha for each channel
        output_t = PReLU(alpha_initializer=Constant(value=init), shared_axes=[2, 3, 4])(input_t)
    elif iPReLU == 2:  # just one alpha for each layer
        output_t = PReLU(alpha_initializer=Constant(value=init), shared_axes=[2, 3, 4, 1])(input_t)
    else:
        output_t = Activation('relu')(input_t)
    return output_t

def fCreateVNet_DownConv_Block(input_t,channels, stride, l1_reg=0.0, l2_reg=1e-6, iPReLU=0, dr_rate=0):
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


def fCreateVNet_Block( input_t, channels, type=1, kernel_size=(3,3,3),l1_reg=0.0, l2_reg=1e-6, iPReLU=0, dr_rate=0):
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
