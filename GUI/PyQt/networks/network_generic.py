# # generate a network
# # create model in createModel
# # change callbacks settings in fTrainInner
# # change loss function in fTrainInner
# # change network compile settings in fTrainInner

import os
from DLart.Constants_DLart import *
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices)
import os.path
import scipy.io as sio
import numpy as np
import keras
from keras.layers.core import Flatten
from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.regularizers import l2  # , activity_l2

from utils.image_preprocessing import ImageDataGenerator
from utils.LivePlotCallback import LivePlotCallback
from matplotlib import pyplot as plt

from networks.multiclass.SENets.densely_connected_cnn_blocks import *


def createModel(patchSize, numClasses):
    """create network model
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    growthRate_k = 12
    compressionFactor = 1.0

    # input layer
    input_tensor = None

    # hidden layers

    # output layer
    output = None

    # create model
    sModelName = ''  # model name
    cnn = Model(input_tensor, output, name=sModelName)

    return cnn, sModelName


def fTrain(X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None, y_test=None, sOutPath=None, patchSize=0,
           batchSizes=None, learningRates=None, iEpochs=None, dlart_handle=None):
    # grid search on batch_sizes and learning rates
    # parse inputs
    batchSize = batchSizes[0]
    learningRate = learningRates[0]

    if len(X_train.shape) == 3:
        # change the shape of the dataset -> at color channel -> here one for grey scale
        dimExpandEnable = False
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

        if X_valid is not None and y_valid is not None:
            X_valid = np.expand_dims(X_valid, axis=-1)
    elif len(X_train.shape) == 4:
        dimExpandEnable = True

    # number of classes
    numClasses = np.shape(y_train)[-1]

    # create cnn model
    cnn, sModelName = createModel(patchSize=patchSize, numClasses=numClasses)

    fTrainInner(cnn,
                sModelName,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
                sOutPath=sOutPath,
                patchSize=patchSize,
                batchSize=batchSize,
                learningRate=learningRate,
                iEpochs=iEpochs,
                dlart_handle=dlart_handle,
                dimExpandEnable=dimExpandEnable)

    K.clear_session()


def fTrainInner(cnn, modelName, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None, y_test=None,
                sOutPath=None, patchSize=0, batchSize=None, learningRate=None, iEpochs=None, dlart_handle=None, dimExpandEnable=False):
    print('Training CNN')
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)

    model_name = sOutPath + os.sep + sFilename
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '.json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        print('------- already trained -> go to next')
        return

    # create optimizer
    if dlart_handle != None:
        if dlart_handle.getOptimizer() == SGD_OPTIMIZER:
            opti = keras.optimizers.SGD(lr=learningRate,
                                        momentum=dlart_handle.getMomentum(),
                                        decay=dlart_handle.getWeightDecay(),
                                        nesterov=dlart_handle.getNesterovEnabled())
        elif dlart_handle.getOptimizer() == RMS_PROP_OPTIMIZER:
            opti = keras.optimizers.RMSprop(lr=learningRate, decay=dlart_handle.getWeightDecay())
        elif dlart_handle.getOptimizer() == ADAGRAD_OPTIMIZER:
            opti = keras.optimizers.Adagrad(lr=learningRate, epsilon=None, decay=dlart_handle.getWeightDecay())
        elif dlart_handle.getOptimizer() == ADADELTA_OPTIMIZER:
            opti = keras.optimizers.Adadelta(lr=learningRate, rho=0.95, epsilon=None,
                                             decay=dlart_handle.getWeightDecay())
        elif dlart_handle.getOptimizer() == ADAM_OPTIMIZER:
            opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None,
                                         decay=dlart_handle.getWeightDecay())
        else:
            raise ValueError("Unknown Optimizer!")
    else:
        opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    cnn.summary()

    # compile model
    loss_func = 'categorical_crossentropy'  # change loss function
    cnn.compile(loss=loss_func, optimizer=opti, metrics=['accuracy'])

    # callbacks
    callback_earlyStopping = EarlyStopping(monitor='val_loss', patience=25, verbose=1)
    # callback_tensorBoard = keras.callbacks.TensorBoard(log_dir=dlart_handle.getLearningOutputPath() + '/logs',
    # histogram_freq=2,
    # batch_size=batchSize,
    # write_graph=True,
    # write_grads=True,
    # write_images=True,
    # embeddings_freq=0,
    # embeddings_layer_names=None,
    #  embeddings_metadata=None)

    callbacks = [callback_earlyStopping,
                 ModelCheckpoint(sOutPath + os.sep + 'checkpoints/checker.hdf5', monitor='val_acc', verbose=0, period=5,
                                 save_best_only=True), LearningRateScheduler(schedule=step_decay, verbose=1),
                 LivePlotCallback(dlart_handle)]
    # callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-4, verbose=1))

    # data augmentation
    if dlart_handle.getDataAugmentationEnabled():
        # Initialize Image Generator
        # all shifted and rotated images are filled with zero padded pixels
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=dlart_handle.getZCA_Whitening(),
            zca_epsilon=1e-6,
            rotation_range=dlart_handle.getRotation(),
            width_shift_range=dlart_handle.getWidthShift(),
            height_shift_range=dlart_handle.getHeightShift(),
            shear_range=0.,
            zoom_range=dlart_handle.getZoom(),
            channel_shift_range=0.,
            fill_mode='constant',
            cval=0.,
            horizontal_flip=dlart_handle.getHorizontalFlip(),
            vertical_flip=dlart_handle.getVerticalFlip(),
            rescale=None,
            histogram_equalization=dlart_handle.getHistogramEqualization(),
            contrast_stretching=dlart_handle.getContrastStretching(),
            adaptive_equalization=dlart_handle.getAdaptiveEqualization(),
            preprocessing_function=None,
            data_format=K.image_data_format()
        )

        datagen_val = ImageDataGenerator(featurewise_center=False,
                                         samplewise_center=False,
                                         featurewise_std_normalization=False,
                                         samplewise_std_normalization=False,
                                         zca_whitening=dlart_handle.getZCA_Whitening(),
                                         zca_epsilon=1e-6,
                                         rotation_range=0.,
                                         width_shift_range=0.,
                                         height_shift_range=0.,
                                         shear_range=0.,
                                         zoom_range=0.,
                                         channel_shift_range=0.,
                                         fill_mode='constant',
                                         cval=0.,
                                         horizontal_flip=False,
                                         vertical_flip=False,
                                         rescale=None,
                                         histogram_equalization=dlart_handle.getHistogramEqualization(),
                                         contrast_stretching=dlart_handle.getContrastStretching(),
                                         adaptive_equalization=dlart_handle.getAdaptiveEqualization(),
                                         preprocessing_function=None,
                                         data_format=K.image_data_format())

        datagen_test = ImageDataGenerator(featurewise_center=False,
                                          samplewise_center=False,
                                          featurewise_std_normalization=False,
                                          samplewise_std_normalization=False,
                                          zca_whitening=dlart_handle.getZCA_Whitening(),
                                          zca_epsilon=1e-6,
                                          rotation_range=0.,
                                          width_shift_range=0.,
                                          height_shift_range=0.,
                                          shear_range=0.,
                                          zoom_range=0.,
                                          channel_shift_range=0.,
                                          fill_mode='constant',
                                          cval=0.,
                                          horizontal_flip=False,
                                          vertical_flip=False,
                                          rescale=None,
                                          histogram_equalization=dlart_handle.getHistogramEqualization(),
                                          contrast_stretching=dlart_handle.getContrastStretching(),
                                          adaptive_equalization=dlart_handle.getAdaptiveEqualization(),
                                          preprocessing_function=None,
                                          data_format=K.image_data_format())

        # fit parameters from dataset
        datagen.fit(X_train)
        datagen_test.fit(X_test)

        # configure batch size and get one batch of images
        for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
            # display first 9 images
            for i in range(0, 9):
                plt.subplot(330 + 1 + i)
                plt.imshow(x_batch[i].reshape(x_batch.shape[1], x_batch.shape[2]), cmap='gray')
            plt.show()
            break

        if X_valid is not None and y_valid is not None:
            # fit model on data
            # use validation/test split
            datagen_val.fit(X_valid)
            result = cnn.fit_generator(datagen.flow(X_train, y_train, batch_size=batchSize),
                                       steps_per_epoch=X_train.shape[0] // batchSize,
                                       epochs=iEpochs,
                                       validation_data=datagen_val.flow(X_valid, y_valid, batch_size=batchSize),
                                       callbacks=callbacks,
                                       workers=1,
                                       use_multiprocessing=False)

        else:
            # fit model on data
            # use test data for validation and test

            result = cnn.fit_generator(datagen.flow(X_train, y_train, batch_size=batchSize),
                                       steps_per_epoch=X_train.shape[0] // batchSize,
                                       epochs=iEpochs,
                                       validation_data=datagen_test.flow(X_test, y_test, batch_size=batchSize),
                                       callbacks=callbacks,
                                       workers=1,
                                       use_multiprocessing=False)

        # return the loss value and metrics values for the model in test mode
        score_test, acc_test = cnn.evaluate_generator(datagen_test.flow(X_test, y_test, batch_size=batchSize),
                                                      steps=None,
                                                      max_queue_size=10,
                                                      workers=1,
                                                      use_multiprocessing=False)

        prob_test = cnn.predict_generator(datagen_test.flow(X_test, y_test, batch_size=batchSize),
                                          steps=None,
                                          max_queue_size=10,
                                          workers=1,
                                          use_multiprocessing=False,
                                          verbose=1)

    else:
        if X_valid is not None and y_valid is not None:
            if dimExpandEnable:
                X_valid = np.expand_dims(X_valid, axis=-1)
            # use validation/test split
            result = cnn.fit(X_train,
                             y_train,
                             validation_data=(X_valid, y_valid),
                             epochs=iEpochs,
                             batch_size=batchSize,
                             callbacks=callbacks,
                             verbose=1)
        else:
            if dimExpandEnable:
                X_train = np.expand_dims(X_train, axis=-1)
                X_test = np.expand_dims(X_test, axis=-1)

            # use test set for validation and test
            result = cnn.fit(X_train,
                             y_train,
                             validation_data=(X_test, y_test),
                             epochs=iEpochs,
                             batch_size=batchSize,
                             callbacks=callbacks,
                             verbose=1)

        # return the loss value and metrics values for the model in test mode
        score_test, acc_test = cnn.evaluate(X_test, y_test, batch_size=batchSize, verbose=1)

        prob_test = cnn.predict(X_test, batchSize, 0)

    # save model
    json_string = cnn.to_json()
    with open(model_json, 'w') as jsonFile:
        jsonFile.write(json_string)

    # wei = cnn.get_weights()
    cnn.save_weights(weight_name, overwrite=True)
    cnn.save(model_all)  # keras > v0.7
    model_png_dir = sOutPath + os.sep + "model.png"
    from keras.utils import plot_model
    plot_model(cnn, to_file=model_png_dir, show_shapes=True, show_layer_names=True)

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
                             'score_test': score_test,
                             'acc_test': acc_test,
                             'prob_test': prob_test})


def step_decay(epoch, lr):
    drop = 0.1
    epochs_drop = 10.0
    print("Current Learning Rate: " + str(lr))
    if epoch == epochs_drop or epoch == 2 * epochs_drop or epoch == 3 * epochs_drop or epoch == 4 * epochs_drop:
        lr = drop * lr
        print("Reduce Learningrate by 0.1 to " + str(lr))

    return lr


def fPredict(X, y, sModelPath, sOutPath, batchSize=64):
    """Takes an already trained model and computes the loss and Accuracy over the samples X with their Labels y
        Input:
            X: Samples to predict on. The shape of X should fit to the input shape of the model
            y: Labels for the Samples. Number of Samples should be equal to the number of samples in X
            sModelPath: (String) full path to a trained keras model. It should be *_json.txt file. there has to be a corresponding *_weights.h5 file in the same directory!
            sOutPath: (String) full path for the Output. It is a *.mat file with the computed loss and accuracy stored.
                        The Output file has the Path 'sOutPath'+ the filename of sModelPath without the '_json.txt' added the suffix '_pred.mat'
            batchSize: Batchsize, number of samples that are processed at once"""
    sModelPath = sModelPath.replace("_json.txt", "")
    weight_name = sModelPath + '_weights.h5'
    model_json = sModelPath + '_json.txt'
    model_all = sModelPath + '_model.h5'

    # load weights and model (new way)
    model_json = open(model_json, 'r')
    model_string = model_json.read()
    model_json.close()
    model = model_from_json(model_string)

    loss_func = 'categorical_crossentropy'  # change loss function
    model.compile(loss=loss_func, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.load_weights(weight_name)

    score_test, acc_test = model.evaluate(X, y, batch_size=batchSize)
    print('loss' + str(score_test) + '   acc:' + str(acc_test))
    prob_pre = model.predict(X, batch_size=batchSize, verbose=1)
    print(prob_pre[0:14, :])
    _, sModelFileSave = os.path.split(sModelPath)

    modelSave = sOutPath + sModelFileSave + '_pred.mat'
    print('saving Model:{}'.format(modelSave))
    sio.savemat(modelSave, {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test})


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

    # opti = SGD(lr={{choice([0.1, 0.01, 0.05, 0.005, 0.001])}}, momentum=1e-8, decay=0.1, nesterov=True)
    # cnn.compile(loss='categorical_crossentropy', optimizer=opti)

    epochs = 300

    result = cnn.fit(X_train, Y_train,
                     batch_size=128,  # {{choice([64, 128])}}
                     nb_epoch=epochs,
                     show_accuracy=True,
                     verbose=2,
                     validation_data=(X_test, Y_test))
    score_test, acc_test = cnn.evaluate(X_test, Y_test, verbose=0)

    # return {'loss': -acc_test, 'status': STATUS_OK, 'model': cnn, 'trainresult': result, 'score_test': score_test}


## helper functions
def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
    r += step
