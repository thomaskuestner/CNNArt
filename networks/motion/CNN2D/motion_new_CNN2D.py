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


def fTrain(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSizes, learningRates, iEpochs, CV_Patient=0, model='motion_head'):#rigid for loops for simplicity
    #add for loops here
    batchSizes = [64] if batchSizes is None else batchSizes
    learningRates = [0.001] if learningRates is None else learningRates
    iEpochs = 300 if iEpochs is None else iEpochs

    # change the shape of the dataset
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    y_train = np.asarray([y_train[:], np.abs(np.asarray(y_train[:], dtype=np.float32)-1)]).T
    y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32)-1)]).T

    for iBatch in batchSizes:
        for iLearn in learningRates:
            print "Model: 2D_CNN"
            print "learning rate: " + str(iLearn)
            print "batch size: " + str(iBatch)
            cnn= fCreateModel(patchSize, learningRate=iLearn, optimizer='Adam')
            fTrainInner(cnn, X_train, y_train, X_test, y_test, sOutPath, patchSize, iBatch, iLearn, iEpochs, CV_Patient=CV_Patient)


def fTrainInner(cnn, X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSize=64, learningRate=0.001, iEpochs=299, CV_Patient=0):

    print 'Training CNN'
    print 'with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize )+ ', maxEpochs=' +str(iEpochs)

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)
    model_name = sPath + '/' + sFilename + '/'
    if CV_Patient != 0: model_name = model_name + 'CV' + str(
        CV_Patient) + '_'  # determine if crossValPatient is used...
    model_name = model_name + str(int(patchSize[0])) + str(int(patchSize[0])) \
                 + '_lr_' + str(learningRate) + '_bs_' + str(batchSize)
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '_json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        return

    # create model

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
    callbacks.append(
        ModelCheckpoint('/home/s1241/no_backup/s1241/checkpoints/checker.hdf5', monitor='val_acc', verbose=0,
                        period=5, save_best_only=True))  # overrides the last checkpoint, its just for security
    callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-4, verbose=1))

    result = cnn.fit(X_train,
                     y_train,
                     validation_data=[X_test, y_test],
                     epochs=iEpochs,
                     batch_size=batchSize,
                     callbacks=callbacks,
                     verbose=1)

    loss_test, acc_test = cnn.evaluate(X_test, y_test, batch_size=batchSize)

    prob_test = cnn.predict(X_test, batchSize, 0)

    # save model
    json_string = cnn.to_json()
    open(model_json, 'w').write(json_string)
    cnn.save_weights(weight_name, overwrite=True)


    # matlab
    acc = result.history['acc']
    loss = result.history['loss']
    val_acc = result.history['val_acc']
    val_loss = result.history['val_loss']

    print 'Saving results: ' + model_name
    sio.savemat(model_name, {'model_settings': model_json,
                             'model': model_all,
                             'weights': weight_name,
                             'acc_history': acc,
                             'loss_history': loss,
                             'val_acc_history': val_acc,
                             'val_loss_history': val_loss,
                             'loss_test': loss_test,
                             'acc_test': acc_test,
                             'prob_test': prob_test})


def fPredict(X_test, y_test, sOutPath, patchSize, batchSizes=64, learningRates=0.001, CV_Patient=0):
    # takes the .mat file as a string

    # change the shape of the dataset

    X_test = np.expand_dims(X_test, axis=1)
    y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32)-1)]).T

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)


    for iBatch in batchSizes:
        for iLearn in learningRates:
            model_name = sPath + '/' + sFilename + '/'
            if CV_Patient != 0: model_name = model_name + 'CV' + str(
                CV_Patient) + '_'  # determine if crossValPatient is used...
            sModelName = model_name + str(int(patchSize[0])) + str(int(patchSize[0])) \
                         + '_lr_' + str(iLearn) + '_bs_' + str(iBatch)
            print "Predict in Model: 2D_CNN"
            print "learning rate: " + str(iLearn)
            print "batch size: " + str(iBatch)
            # sModelPath = sModelName.replace(".mat", "")
            # sModelPath = sModelPath.replace("_json", "")
            weight_name = sModelName + '_weights.h5'
            model_json = sModelName + '_json'
            model_all = sModelName + '_model.h5'

            model_json = open(model_json, 'r')
            model_string = model_json.read()
            model_json.close()
            model = model_from_json(model_string)

            model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
            model.load_weights(weight_name)

            score_test, acc_test = model.evaluate(X_test, y_test, batch_size=iBatch)
            print 'score:' + str(score_test) + 'acc:' + str(acc_test)
            prob_pre = model.predict(X_test, batch_size=iBatch, verbose=1)

            #_, sModelFileSave = os.path.split(sOutPath)

            modelSave = sModelName + '_pred.mat'
            print modelSave
            sio.savemat(modelSave, {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test})


def fCreateModel(patchSize, optimizer='Adam', learningRate=0.001):# only on linse 3!!!!!!!!!!!!
    l1_reg=0
    l2_reg=1e-6
    cnn = Sequential()
    #Total params: 272,994
    cnn.add(Conv2D(32,
                            kernel_size=(14,14),
                            kernel_initializer='he_normal',
                            weights=None,
                            padding='valid',
                            strides=(1, 1),
                            kernel_regularizer=l1_l2(l1_reg, l2_reg),
                            input_shape=(1, int(patchSize[0]), int(patchSize[0]))))
     #input shape : 1 means grayscale... richtig uebergeben...
    cnn.add(Activation('relu'))

    cnn.add(Conv2D(64,                    #learning rate: 0.1 -> 76%
                            kernel_size=(7,7),
                            kernel_initializer='he_normal',
                            weights=None,
                            padding='valid',
                            strides=(1, 1),
                            kernel_regularizer=l1_l2(l1_reg, l2_reg),
                            #data_format='channels_first'
                    ))
    cnn.add(Activation('relu'))
    cnn.add(Conv2D(128,                    #learning rate: 0.1 -> 76%
                            kernel_size=(3,3),
                            kernel_initializer='he_normal',
                            weights=None,
                            padding='valid',
                            strides=(1, 1),
                            kernel_regularizer=l1_l2(l1_reg, l2_reg)))
    cnn.add( Activation('relu'))
    cnn.add(Flatten())
    cnn.add(Dense(units=2,
                  kernel_initializer= 'he_normal',
                  kernel_regularizer= 'l2'))
    cnn.add(Activation('softmax'))
    loss='categorical_crossentropy'
    opti=keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #opti,loss=archi.fGetOptimizerAndLoss(optimizer=optimizer,learningRate=learningRate)
    cnn.compile(loss=loss, optimizer=opti, metrics=['accuracy'])
    print cnn.summary()
    return cnn