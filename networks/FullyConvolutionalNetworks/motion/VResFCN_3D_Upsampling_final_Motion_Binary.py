'''
Copyright: 2016-2019 Thomas Kuestner (thomas.kuestner@med.uni-tuebingen.de) under Apache2 license
@author: Thomas Kuestner
'''

import os
#os.environ["CUDA_DEVICE_ORDER"]="0000:02:00.0"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices)

import tensorflow as tf
import os.path
import numpy as np
import scipy.io as sio
import keras
from keras.layers import Input
import keras.backend as K
#from keras.layers import Conv2D
from keras.layers import BatchNormalization
#from keras.layers import GlobalAveragePooling2D
#from keras.activations import softmax
from keras.layers import concatenate
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Model
#from keras.models import Sequential
from keras.layers import UpSampling3D
#from keras.layers.convolutional import Convolution2D
from keras.layers import LeakyReLU
from keras.layers import Softmax

from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
#from keras.callbacks import ReduceLROnPlateau
#from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
#from keras.regularizers import l2  # , activity_l2
from sklearn.metrics import classification_report, confusion_matrix

#from utils.LivePlotCallback import LivePlotCallback

from networks.FullyConvolutionalNetworks.motion.deep_residual_learning_blocks import *
from utils.dlnetwork import *
from utils.image_preprocessing import *
from utils.DataGenerator import *
from utils.Training_Test_Split_FCN import fSplitSegmentationDataset_generator


def createModel(patchSize, numClasses, usingClassification=False):

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    input_tensor = Input(shape=(patchSize[0], patchSize[1], patchSize[2], 1))

    # first stage
    x = Conv3D(filters=16,
               kernel_size=(5, 5, 5),
               strides=(1, 1, 1),
               padding='same',
               kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x_after_stage_1 = LeakyReLU(alpha=0.01)(x)

    #x_after_stage_1 = Add()([input_tensor, x])

    # first down convolution
    x_down_conv_1 = projection_block_3D(x_after_stage_1,
                            filters=(32, 32),
                            kernel_size=(2, 2, 2),
                            stage=1,
                            block=1,
                            se_enabled=True,
                            se_ratio=4)

    # second stage
    x = identity_block_3D(x_down_conv_1, filters=(32, 32), kernel_size=(3, 3, 3), stage=2, block=1, se_enabled=True, se_ratio=4)
    x_after_stage_2 = identity_block_3D(x, filters=(32, 32), kernel_size=(3,3,3), stage=2, block=2, se_enabled=True, se_ratio=4)



    # second down convolution
    x_down_conv_2 = projection_block_3D(x_after_stage_2,
                                        filters=(64, 64),
                                        kernel_size=(2, 2, 2),
                                        stage=2,
                                        block=3,
                                        se_enabled=True,
                                        se_ratio=8)

    # third stage
    x = identity_block_3D(x_down_conv_2, filters=(64, 64), kernel_size=(3, 3, 3), stage=3, block=1, se_enabled=True, se_ratio=8)
    x_after_stage_3 = identity_block_3D(x, filters=(64, 64), kernel_size=(3, 3, 3), stage=3, block=2, se_enabled=True, se_ratio=8)
    #x = identity_block_3D(x, filters=(64, 64), kernel_size=(3, 3, 3), stage=3, block=3, se_enabled=False, se_ratio=16)

    # third down convolution
    x_down_conv_3 = projection_block_3D(x_after_stage_3,
                                        filters=(128, 128),
                                        kernel_size=(2, 2, 2),
                                        stage=3,
                                        block=4,
                                        se_enabled=True,
                                        se_ratio=16)

    # fourth stage
    x = identity_block_3D(x_down_conv_3, filters=(128, 128), kernel_size=(3, 3, 3), stage=4, block=1, se_enabled=True, se_ratio=16)
    x_after_stage_4 = identity_block_3D(x, filters=(128, 128), kernel_size=(3, 3, 3), stage=4, block=2, se_enabled=True, se_ratio=16)
    #x = identity_block_3D(x, filters=(128, 128), kernel_size=(3, 3, 3), stage=4, block=3, se_enabled=False, se_ratio=16)



    ### end of encoder path

    if usingClassification:
        # use x_after_stage_4 as quantification output
        # global average pooling
        x_class = GlobalAveragePooling3D(data_format=K.image_data_format())(x_after_stage_4)

        # fully-connected layer
        classification_output = Dense(units=numClasses,
                       activation='softmax',
                       kernel_initializer='he_normal',
                       name='classification_output')(x_class)

    ### decoder path


    # first 3D upsampling
    x = UpSampling3D(size=(2, 2, 2), data_format=K.image_data_format())(x_after_stage_4)
    x = Conv3D(filters=64,
               kernel_size=(3, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = concatenate([x, x_after_stage_3], axis=bn_axis)

    # first decoder stage
    x = identity_block_3D(x, filters=(128, 128), kernel_size=(3, 3, 3), stage=6, block=1, se_enabled=True, se_ratio=16)
    x = identity_block_3D(x, filters=(128, 128), kernel_size=(3, 3, 3), stage=6, block=2, se_enabled=True, se_ratio=16)

    # second 3D upsampling
    x = UpSampling3D(size=(2, 2, 2), data_format=K.image_data_format())(x)
    x = Conv3D(filters=32,
               kernel_size=(3, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = concatenate([x, x_after_stage_2], axis=bn_axis)

    # second decoder stage
    x = identity_block_3D(x, filters=(64, 64), kernel_size=(3, 3, 3), stage=7, block=1, se_enabled=True, se_ratio=8)
    x = identity_block_3D(x, filters=(64, 64), kernel_size=(3, 3, 3), stage=7, block=2, se_enabled=True, se_ratio=8)

    # third 3D upsampling
    x = UpSampling3D(size=(2, 2, 2), data_format=K.image_data_format())(x)
    x = Conv3D(filters=16,
               kernel_size=(3, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = concatenate([x, x_after_stage_1], axis=bn_axis)

    # third decoder stage
    x = identity_block_3D(x, filters=(32, 32), kernel_size=(3, 3, 3), stage=9, block=1, se_enabled=True, se_ratio=4)
    #x = identity_block_3D(x, filters=(32, 32), kernel_size=(3, 3, 3), stage=9, block=2, se_enabled=True, se_ratio=4)

    ### End of decoder

    ### last segmentation segments
    # 1x1x1-Conv3 produces 2 featuremaps for probabilistic  segmentations of the foreground and background
    x = Conv3D(filters=2,
               kernel_size=(1, 1, 1),
               strides=(1, 1, 1),
               padding='same',
               kernel_initializer='he_normal',
               name='conv_veryEnd')(x)
    #x = BatchNormalization(axis=bn_axis)(x) # warum leakyrelu vor softmax?
    #x = LeakyReLU(alpha=0.01)(x)

    segmentation_output = Softmax(axis=bn_axis, name='segmentation_output')(x)
    #segmentation_output = keras.layers.activations.sigmoid(x)

    # create model
    if usingClassification:
        cnn = Model(inputs =[input_tensor], outputs=[segmentation_output, classification_output], name='3D-VResFCN-Classification')
        sModelName = cnn.name
    else:
        cnn = Model(inputs=[input_tensor], outputs=[segmentation_output], name='3D-VResFCN')
        sModelName = cnn.name

    return cnn, sModelName





def fTrain(X_train=None, y_train=None, Y_segMasks_train=None, X_valid=None, y_valid=None, Y_segMasks_valid=None, X_test=None,
           y_test=None, Y_segMasks_test=None, sOutPath=None, patchSize=0, batchSize=None, learningRate=None, iEpochs=None,
           dlnetwork=None, data=None, numClasses=2):

    usingClassification = dlnetwork.usingClassification

    # grid search on batch_sizes and learning rates
    # parse inputs
    #batchSize = batchSizes[0]
    #learningRate = learningRates[0]



    #if y_valid is None or y_valid == 0:
    #    print("No Validation Dataset.")

    # sio.savemat('D:med_data/voxel_and_masks.mat',
    #                           {'voxel_train': X_train, 'Y_segMasks_train': Y_segMasks_train,
    #                            'y_train': y_train})

    #y_train = np.asarray([y_train[:], np.abs(np.asarray(y_train[:], dtype=np.float32) - 1)]).T
    #y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32) - 1)]).T

    # number of classes
    if y_train is not None:
        numClasses = np.shape(y_train)[1]

    #create cnn model
    print('Create model')
    cnn, sModelName = createModel(patchSize=patchSize, numClasses=numClasses, usingClassification=usingClassification)

    fTrainInner(cnn,
                sModelName,
                X_train=X_train,
                y_train=y_train,
                Y_segMasks_train=Y_segMasks_train,
                X_valid=X_valid,
                y_valid=y_valid,
                Y_segMasks_valid=Y_segMasks_valid,
                X_test=X_test,
                y_test=y_test,
                Y_segMasks_test=Y_segMasks_test,
                sOutPath=sOutPath,
                patchSize=patchSize,
                batchSize=batchSize,
                learningRate=learningRate,
                iEpochs=iEpochs,
                usingClassification=usingClassification,
                dlnetwork=dlnetwork,
                data=data)

    K.clear_session()

    # for iBatch in batchSizes:
    #     for iLearn in learningRates:
    #         fTrainInner(cnn,
    #                     sModelName,
    #                     X_train=X_train,
    #                     y_train=y_train,
    #                     Y_segMasks_train=Y_segMasks_train,
    #                     X_valid=X_valid,
    #                     y_valid=y_valid,
    #                     Y_segMasks_valid=Y_segMasks_valid,
    #                     X_test=X_test,
    #                     y_test=y_test,
    #                     Y_segMasks_test=Y_segMasks_test,
    #                     sOutPath=sOutPath,
    #                     patchSize=patchSize,
    #                     batchSize=iBatch,
    #                     learningRate=iLearn,
    #                     iEpochs=iEpochs,
    #                     usingClassification=usingClassification,
    #                     dlnetwork=dlnetwork)


def fTrainInner(cnn, modelName, X_train=None, y_train=None, Y_segMasks_train=None, X_valid=None, y_valid=None,
                Y_segMasks_valid=None, X_test=None, y_test=None, Y_segMasks_test=None, sOutPath=None, patchSize=0,
                batchSize=None, learningRate=None, iEpochs=None, usingClassification=False, dlnetwork=None, data=None):
    print('Training CNN')
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    # sio.savemat('D:med_data/' + 'checkdata_voxel_and_mask.mat',
    #             {'mask_train': Y_segMasks_train,
    #              'voxel_train': X_train,
    #              'mask_test': Y_segMasks_test,
    #              'voxel_test': X_test})

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
    if dlnetwork != None:
        if dlnetwork.optimizer == 'SGD':
            opti = keras.optimizers.SGD(lr=learningRate,
                                        momentum=dlnetwork.momentum,
                                        decay=dlnetwork.weightdecay,
                                        nesterov=dlnetwork.nesterov)

        elif dlnetwork.optimizer == 'RMSPROP':
            opti = keras.optimizers.RMSprop(lr=learningRate, decay=dlnetwork.weightdecay)

        elif dlnetwork.optimizer == 'ADAGRAD':
            opti = keras.optimizers.Adagrad(lr=learningRate, epsilon=None, decay=dlnetwork.weightdecay)

        elif dlnetwork.optimizer == 'ADADELTA':
            opti = keras.optimizers.Adadelta(lr=learningRate, rho=0.95, epsilon=None,
                                             decay=dlnetwork.weightdecay)

        elif dlnetwork.optimizer == 'ADAM':
            opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None,
                                         decay=dlnetwork.weightdecay)
        else:
            raise ValueError("Unknown Optimizer!")
    else:
        # opti = SGD(lr=learningRate, momentum=1e-8, decay=0.1, nesterov=True);#Adag(lr=0.01, epsilon=1e-06)
        opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    cnn.summary()

    # compile model
    if usingClassification:
        cnn.compile(
            loss={'segmentation_output': dice_coef_loss, 'classification_output': 'categorical_crossentropy'},
            optimizer=opti,
            metrics={'segmentation_output': dice_coef, 'classification_output': 'accuracy'})
    else:
        cnn.compile(loss=dice_coef_loss,
                    optimizer=opti,
                    metrics=[dice_coef]
                    )

    # callbacks
    #callback_earlyStopping = EarlyStopping(monitor='val_loss', patience=12, verbose=1)

    # callback_tensorBoard = keras.callbacks.TensorBoard(log_dir=dlart_handle.getLearningOutputPath() + '/logs',
    # histogram_freq=2,
    # batch_size=batchSize,
    # write_graph=True,
    # write_grads=True,
    # write_images=True,
    # embeddings_freq=0,
    # embeddings_layer_names=None,
    #  embeddings_metadata=None)

    #callbacks = [callback_earlyStopping]
    callbacks = []
    #callbacks.append(
     #   ModelCheckpoint(sOutPath + os.sep + 'checkpoints' + os.sep + 'checker.hdf5', monitor='val_acc', verbose=0,
      #                  period=1, save_best_only=True))  # overrides the last checkpoint, its just for security
    # callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-4, verbose=1))
    callbacks.append(LearningRateScheduler(schedule=step_decay, verbose=1))
    #callbacks.append(LivePlotCallback(dlart_handle))

    print('Start training')

    # TODO: add here data augmentation via ImageDataGenerator from utils/image_preprocessing
    if dlnetwork.trainMode == 'GENERATOR':
        # prepare data generators
        if os.path.exists(X_train):  # splitting was already done
            train_gen = DataGenerator(X_train, batch_size=batchSize, dim=patchSize, usingClassification=usingClassification)
            val_gen = DataGenerator(X_valid, batch_size=batchSize, dim=patchSize, usingClassification=usingClassification)
            test_gen = DataGenerator(X_test, batch_size=batchSize, dim=patchSize, usingClassification=usingClassification)
        else:  # splitting needs to be done
            datapath = os.path.dirname(X_train)
            datafiles = [f for f in os.listdir(datapath) if (os.path.isfile(os.path.join(datapath, f)) and f.endswith('.hdf5'))]
            train_files, val_files, test_files = fSplitSegmentationDataset_generator(datafiles, data.allPats, data.allTestPats, data.splittingMode, testTrainingDatasetRatio=data.trainTestDatasetRatio, validationTrainRatio=data.trainValidationRatio, nfolds=data.nfolds, isRandomShuffle=data.isRandomShuffle)
            train_gen = DataGenerator(datapath, batch_size=batchSize, dim=patchSize, usingClassification=usingClassification, list_IDs=train_files)
            val_gen = DataGenerator(datapath, batch_size=batchSize, dim=patchSize, usingClassification=usingClassification, list_IDs=val_files)
            test_gen = DataGenerator(datapath, batch_size=batchSize, dim=patchSize, usingClassification=usingClassification, list_IDs=test_files)
        existing_validation = True if len(val_gen.list_IDs) > 0 else False
    else:  # ARRAY
        existing_validation = (X_valid != 0 and X_valid is not None)

    if existing_validation:
        # using test set for validation
        if usingClassification:
            if dlnetwork.trainMode == 'ARRAY':
                result = cnn.fit(X_train,
                                 {'segmentation_output': Y_segMasks_train, 'classification_output': y_train},
                                 validation_data=(X_valid, {'segmentation_output': Y_segMasks_valid, 'classification_output': y_valid}),
                                 epochs=iEpochs,
                                 batch_size=batchSize,
                                 callbacks=callbacks,
                                 verbose=1)
            else:
                result = cnn.fit_generator(train_gen,
                                           validation_data=val_gen,
                                           epochs=iEpochs,
                                           batch_size=batchSize,
                                           callbacks=callbacks,
                                           use_multiprocessing=True,
                                           workers=8,
                                           max_queue_size=32,
                                           verbose=1)
        else:
            if dlnetwork.trainMode == 'ARRAY':
                result = cnn.fit(X_train,
                                 Y_segMasks_train,
                                 validation_data=(X_valid, Y_segMasks_valid),
                                 epochs=iEpochs,
                                 batch_size=batchSize,
                                 callbacks=callbacks,
                                 verbose=1)
            else:
                result = cnn.fit_generator(train_gen,
                                           validation_data=val_gen,
                                           epochs=iEpochs,
                                           batch_size=batchSize,
                                           callbacks=callbacks,
                                           use_multiprocessing=True,
                                           workers=8,
                                           max_queue_size=32,
                                           verbose=1)
    else:
        # using validation set for validation
        if usingClassification:
            if dlnetwork.trainMode == 'ARRAY':
                result = cnn.fit(X_train,
                                 {'segmentation_output': Y_segMasks_train, 'classification_output': y_train},
                                 validation_data=(X_test, {'segmentation_output': Y_segMasks_test, 'classification_output': y_test}),
                                 epochs=iEpochs,
                                 batch_size=batchSize,
                                 callbacks=callbacks,
                                 verbose=1)
            else:
                result = cnn.fit_generator(train_gen,
                                           validation_data=test_gen,
                                           epochs=iEpochs,
                                           batch_size=batchSize,
                                           callbacks=callbacks,
                                           use_multiprocessing=True,
                                           workers=8,
                                           max_queue_size=32,
                                           verbose=1)
        else:
            if dlnetwork.trainMode == 'ARRAY':
                result = cnn.fit(X_train,
                                 Y_segMasks_train,
                                 validation_data=(X_test, Y_segMasks_test),
                                 epochs=iEpochs,
                                 batch_size=batchSize,
                                 callbacks=callbacks,
                                 verbose=1)
            else:
                result = cnn.fit_generator(train_gen,
                                           validation_data=test_gen,
                                           epochs=iEpochs,
                                           batch_size=batchSize,
                                           callbacks=callbacks,
                                           use_multiprocessing=True,
                                           workers=8,
                                           max_queue_size=32,
                                           verbose=1)

    # return the loss value and metrics values for the model in test mode
    if dlnetwork.trainMode == 'ARRAY':
        if usingClassification:
            model_metrics = cnn.metrics_names
            loss_test, segmentation_output_loss_test, classification_output_loss_test, segmentation_output_dice_coef_test, classification_output_acc_test \
                = cnn.evaluate(X_test, {'segmentation_output': Y_segMasks_test, 'classification_output': y_test}, batch_size=batchSize, verbose=1)
        else:
            score_test, dice_coef_test = cnn.evaluate(X_test, Y_segMasks_test, batch_size=batchSize, verbose=1)

        prob_test = cnn.predict(X_test, batchSize, 0)
    else:
        if usingClassification:
            model_metrics = cnn.metrics_names
            loss_test, segmentation_output_loss_test, classification_output_loss_test, segmentation_output_dice_coef_test, classification_output_acc_test \
                = cnn.evaluate_generator(test_gen, batch_size=batchSize, verbose=1)
        else:
            score_test, dice_coef_test = cnn.evaluate_generator(test_gen, batch_size=batchSize, verbose=1)

        prob_test = cnn.predict_generator(test_gen, batchSize, 0)

    # save model
    json_string = cnn.to_json()
    with open(model_json, 'w') as jsonFile:
        jsonFile.write(json_string)

    # wei = cnn.get_weights()
    cnn.save_weights(weight_name, overwrite=True)
    # cnn.save(model_all) # keras > v0.7

    if not usingClassification:
        # matlab
        dice_coef_training = result.history['dice_coef']
        training_loss = result.history['loss']
        if X_valid != 0:
            val_dice_coef = result.history['val_dice_coef']
            val_loss = result.history['val_loss']
        else:
            val_dice_coef = 0
            val_loss = 0

        print('Saving results: ' + model_name)

        sio.savemat(model_name, {'model_settings': model_json,
                                 'model': model_all,
                                 'weights': weight_name,
                                 'dice_coef': dice_coef_training,
                                 'training_loss': training_loss,
                                 'val_dice_coef': val_dice_coef,
                                 'val_loss': val_loss,
                                 'score_test': score_test,
                                 'dice_coef_test': dice_coef_test,
                                 'prob_test': prob_test})
    else:
        # matlab
        segmentation_output_loss_training = result.history['segmentation_output_loss']
        classification_output_loss_training = result.history['classification_output_loss']
        segmentation_output_dice_coef_training = result.history['segmentation_output_dice_coef']
        classification_output_acc_training = result.history['classification_output_acc']

        if X_valid != 0:
            val_segmentation_output_loss = result.history['val_segmentation_output_loss']
            val_classification_output_loss = result.history['val_classification_output_loss']
            val_segmentation_output_dice_coef = result.history['val_segmentation_output_dice_coef']
            val_classification_output_acc = result.history['val_classification_output_acc']
        else:
            val_segmentation_output_loss = 0
            val_classification_output_loss = 0
            val_segmentation_output_dice_coef = 0
            val_classification_output_acc = 0

        print('Saving results: ' + model_name)

        sio.savemat(model_name, {'model_settings': model_json,
                                 'model': model_all,
                                 'weights': weight_name,
                                 'segmentation_output_loss_training': segmentation_output_loss_training,
                                 'classification_output_loss_training': classification_output_loss_training,
                                 'segmentation_output_dice_coef_training': segmentation_output_dice_coef_training,
                                 'classification_output_acc_training': classification_output_acc_training,
                                 'segmentation_output_loss_val': val_segmentation_output_loss,
                                 'classification_output_loss_val': val_classification_output_loss,
                                 'segmentation_output_dice_coef_val': val_segmentation_output_dice_coef,
                                 'classification_output_acc_val': val_classification_output_acc,
                                 'loss_test': loss_test,
                                 'segmentation_output_loss_test': segmentation_output_loss_test,
                                 'classification_output_loss_test': classification_output_loss_test,
                                 'segmentation_output_dice_coef_test': segmentation_output_dice_coef_test,
                                 'classification_output_acc_test': classification_output_acc_test,
                                 'segmentation_predictions': prob_test[0],
                                 'classification_predictions': prob_test[1]})


def step_decay(epoch, lr):
   drop = 0.1
   epochs_drop = 20.0
   print("Current Learning Rate: " + str(lr))
   if epoch == epochs_drop or epoch == 2*epochs_drop or epoch == 3*epochs_drop or epoch == 4*epochs_drop:
       lr = drop*lr
       print("Reduce Learningrate by 0.1 to " + str(lr))

   return lr


def fPredict(X_test, Y_test=None, Y_segMasks_test=None, sModelPath=None, batch_size=64, usingClassification=False, usingSegmentationMasks=True, dlnetwork=None):
    """Takes an already trained model and computes the loss and Accuracy over the samples X with their Labels y
        Input:
            X: Samples to predict on. The shape of X should fit to the input shape of the model
            y: Labels for the Samples. Number of Samples should be equal to the number of samples in X
            sModelPath: (String) full path to a trained keras model. It should be *_json.txt file. there has to be a corresponding *_weights.h5 file in the same directory!
            sOutPath: (String) full path for the Output. It is a *.mat file with the computed loss and accuracy stored.
                        The Output file has the Path 'sOutPath'+ the filename of sModelPath without the '_json.txt' added the suffix '_pred.mat'
            batchSize: Batchsize, number of samples that are processed at once"""

    if usingSegmentationMasks:
        X_test = np.expand_dims(X_test, axis=-1)
        Y_segMasks_test_foreground = np.expand_dims(Y_segMasks_test, axis=-1)
        Y_segMasks_test_background = np.ones(Y_segMasks_test_foreground.shape) - Y_segMasks_test_foreground
        Y_segMasks_test = np.concatenate((Y_segMasks_test_background, Y_segMasks_test_foreground), axis=-1)

    else:
        X_test = np.expand_dims(X_test, axis=-1)

    _, sPath = os.path.splitdrive(sModelPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)

    #listdir = os.listdir(sModelPath)

    #sModelPath = sModelPath.replace("_json.txt", "")
    #weight_name = sModelPath + '_weights.h5'
    #model_json = sModelPath + '_json.txt'
    #model_all = sModelPath + '_model.h5'

    # load weights and model (new way)
    with open(sPath + os.sep + sFilename + '.json', 'r') as fp:
        model_string = fp.read()

    model = model_from_json(model_string)
    # create optimizer
    if dlnetwork != None:
        if dlnetwork.optimizer == 'SGD':
            opti = keras.optimizers.SGD(lr=dlnetwork.learningRate,
                                        momentum=dlnetwork.momentum,
                                        decay=dlnetwork.weightdecay,
                                        nesterov=dlnetwork.nesterov)

        elif dlnetwork.optimizer == 'RMSPROP':
            opti = keras.optimizers.RMSprop(lr=dlnetwork.learningRate, decay=dlnetwork.weightdecay)

        elif dlnetwork.optimizer == 'ADAGRAD':
            opti = keras.optimizers.Adagrad(lr=dlnetwork.learningRate, epsilon=None, decay=dlnetwork.weightdecay)

        elif dlnetwork.optimizer == 'ADADELTA':
            opti = keras.optimizers.Adadelta(lr=dlnetwork.learningRate, rho=0.95, epsilon=None,
                                             decay=dlnetwork.weightdecay)

        elif dlnetwork.optimizer == 'ADAM':
            opti = keras.optimizers.Adam(lr=dlnetwork.learningRate, beta_1=0.9, beta_2=0.999, epsilon=None,
                                         decay=dlnetwork.weightdecay)
        else:
            raise ValueError("Unknown Optimizer!")
    else:
        # opti = SGD(lr=learningRate, momentum=1e-8, decay=0.1, nesterov=True);#Adag(lr=0.01, epsilon=1e-06)
        opti = keras.optimizers.Adam(lr=dlnetwork.learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.summary()

    if usingSegmentationMasks:
        if usingClassification:
            model.compile(
                loss={'segmentation_output': dice_coef_loss, 'classification_output': 'categorical_crossentropy'},
                optimizer=opti,
                metrics={'segmentation_output': dice_coef, 'classification_output': 'accuracy'})

            model.load_weights(sPath + os.sep + sFilename + '_weights.h5')

            loss_test, segmentation_output_loss_test, classification_output_loss_test, segmentation_output_dice_coef_test, classification_output_acc_test \
                = model.evaluate(X_test,
                                 {'segmentation_output': Y_segMasks_test, 'classification_output': Y_test},
                                 batch_size=batch_size, verbose=1)

            print('loss' + str(loss_test) + ' segmentation loss:' + str(
                segmentation_output_loss_test) + ' classification loss: ' + str(classification_output_loss_test) + \
                  ' segmentation dice coef: ' + str(
                segmentation_output_dice_coef_test) + ' classification accuracy: ' + str(
                classification_output_acc_test))

            prob_test = model.predict(X_test, batch_size=batch_size, verbose=1)

            predictions = {'prob_pre': prob_test[0],
                           'classification_predictions': prob_test[1],
                           'loss_test': loss_test,
                           'segmentation_output_loss_test': segmentation_output_loss_test,
                           'classification_output_loss_test': classification_output_loss_test,
                           'segmentation_output_dice_coef_test': segmentation_output_dice_coef_test,
                           'classification_output_acc_test': classification_output_acc_test}

        else:
            model.compile(loss=dice_coef_loss, optimizer=opti, metrics=[dice_coef])
            model.load_weights(sPath + os.sep + sFilename + '_weights.h5')

            score_test, acc_test = model.evaluate(X_test, Y_segMasks_test, batch_size=batch_size)
            print('loss: ' + str(score_test) + '   dice coef:' + str(acc_test))

            prob_test = model.predict(X_test, batch_size=batch_size, verbose=1)

            predictions = {'prob_pre': prob_test, 'score_test': score_test, 'acc_test': acc_test}

    else:
        model.compile(loss=dice_coef_loss, optimizer=opti, metrics=[dice_coef])

        model.load_weights(sPath + os.sep + sFilename+'_weights.h5')

        score_test, acc_test = model.evaluate(X_test, Y_segMasks_test, batch_size=2)
        print('loss' + str(score_test) + '   acc:' + str(acc_test))

        prob_pre = model.predict(X_test, batch_size=batch_size, verbose=1)

        probability_predictions = {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test}

        classification_summary = classification_report(np.argmax(Y_test, axis=1),
                                                       np.argmax(probability_predictions, axis=1),
                                                       target_names=None, digits=4)

        # confusion matrix
        confusionMatrix = confusion_matrix(y_true=np.argmax(Y_test, axis=1),
                                           y_pred=np.argmax(probability_predictions, axis=1),
                                           labels=range(int(probability_predictions.shape[1])))

        predictions = {
            'predictions': probability_predictions,
            'score_test': score_test,
            'acc_test': acc_test,
            'classification_report': classification_summary,
            'confusion_matrix': confusionMatrix
        }

    return predictions


def dice_coef(y_true, y_pred, epsilon=1e-5):
    dice_numerator = 2.0 * K.sum(y_true*y_pred, axis=[1,2,3,4])
    dice_denominator = K.sum(K.square(y_true), axis=[1,2,3,4]) + K.sum(K.square(y_pred), axis=[1,2,3,4])

    dice_score = dice_numerator / (dice_denominator + epsilon)
    return K.mean(dice_score, axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def dice_coef_2(ground_truth, prediction, weight_map=None):
    """
    Function to calculate the dice loss with the definition given in

        Milletari, F., Navab, N., & Ahmadi, S. A. (2016)
        V-net: Fully convolutional neural
        networks for volumetric medical image segmentation. 3DV 2016

    using a square in the denominator

    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    ground_truth = tf.to_int64(ground_truth)
    prediction = tf.cast(prediction, tf.float32)
    ids = tf.range(tf.to_int64(tf.shape(ground_truth)[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(tf.shape(prediction)))
    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(weight_map_nclasses * tf.square(prediction),
                          reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot * weight_map_nclasses,
                                 reduction_axes=[0])
    else:
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(tf.square(prediction), reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    # dice_score.set_shape([n_classes])
    # minimising (1 - dice_coefficients)

    #return 1.0 - tf.reduce_mean(dice_score)
    return tf.reduce_mean(dice_score)



def jaccard_distance_loss(y_true, y_pred):
    return -jaccard_distance(y_true, y_pred)


def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (1 - jac) * smooth