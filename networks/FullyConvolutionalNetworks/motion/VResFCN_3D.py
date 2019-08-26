import os
#os.environ["CUDA_DEVICE_ORDER"]="0000:02:00.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices)

import os.path
import scipy.io as sio
import numpy as np
import math
import keras
from keras.layers import Input
import keras.backend as K
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.activations import softmax
from keras.layers import concatenate
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import LeakyReLU
from keras.layers import Softmax
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.regularizers import l2  # , activity_l2

from keras.optimizers import SGD
from networks.multiclass.SENets.deep_residual_learning_blocks import *
from DeepLearningArt.DLArt_GUI.dlart import DeepLearningArtApp
from utils.image_preprocessing import ImageDataGenerator
from matplotlib import pyplot as plt




def createModel(patchSize, numClasses):

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
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = LeakyReLU(alpha=0.01)(x)

    x_after_stage_1 = Add()([input_tensor, x])

    # first down convolution
    x_down_conv_1 = projection_block_3D(x_after_stage_1,
                            filters=(32, 32),
                            kernel_size=(2, 2, 2),
                            stage=1,
                            block=1,
                            se_enabled=False,
                            se_ratio=16)

    # second stage
    x = identity_block_3D(x_down_conv_1, filters=(32, 32), kernel_size=(3, 3, 3), stage=2, block=1, se_enabled=False, se_ratio=16)
    #x = identity_block_3D(x, filters=(32, 32), kernel_size=(3,3,3), stage=2, block=2, se_enabled=False, se_ratio=16)
    x_after_stage_2 = x

    # second down convolution
    x_down_conv_2 = projection_block_3D(x_after_stage_2,
                                        filters=(64, 64),
                                        kernel_size=(2, 2, 2),
                                        stage=2,
                                        block=3,
                                        se_enabled=False,
                                        se_ratio=16)

    # third stage
    x = identity_block_3D(x_down_conv_2, filters=(64, 64), kernel_size=(3, 3, 3), stage=3, block=1, se_enabled=False, se_ratio=16)
    x = identity_block_3D(x, filters=(64, 64), kernel_size=(3, 3, 3), stage=3, block=2, se_enabled=False, se_ratio=16)
    #x = identity_block_3D(x, filters=(64, 64), kernel_size=(3, 3, 3), stage=3, block=3, se_enabled=False, se_ratio=16)
    x_after_stage_3 = x

    # third down convolution
    x_down_conv_3 = projection_block_3D(x_after_stage_3,
                                        filters=(128, 128),
                                        kernel_size=(2, 2, 2),
                                        stage=3,
                                        block=4,
                                        se_enabled=False,
                                        se_ratio=16)

    # fourth stage
    x = identity_block_3D(x_down_conv_3, filters=(128, 128), kernel_size=(3, 3, 3), stage=4, block=1, se_enabled=False, se_ratio=16)
    x = identity_block_3D(x, filters=(128, 128), kernel_size=(3, 3, 3), stage=4, block=2, se_enabled=False, se_ratio=16)
    #x = identity_block_3D(x, filters=(128, 128), kernel_size=(3, 3, 3), stage=4, block=3, se_enabled=False, se_ratio=16)
    x_after_stage_4 = x

    # fourth down convolution
    x_down_conv_4 = projection_block_3D(x_after_stage_4,
                                        filters=(256, 256),
                                        kernel_size=(2, 2, 2),
                                        stage=4,
                                        block=4,
                                        se_enabled=False,
                                        se_ratio=16)

    # fifth stage
    x = identity_block_3D(x_down_conv_4, filters=(256, 256), kernel_size=(3, 3, 3), stage=5, block=1, se_enabled=False, se_ratio=16)
    x = identity_block_3D(x, filters=(256, 256), kernel_size=(3, 3, 3), stage=5, block=2, se_enabled=False, se_ratio=16)
    # x = identity_block_3D(x, filters=(256, 256), kernel_size=(3, 3, 3), stage=5, block=3, se_enabled=False, se_ratio=16)
    x_after_stage_5 = x

    ### end of encoder path

    ### decoder path

    x = Conv3DTranspose(filters=128,
                        kernel_size=(2, 2, 2),
                        strides=(2, 2, 2),
                        padding='valid',
                        data_format=K.image_data_format(),
                        kernel_initializer='he_normal'
                        )(x_after_stage_5)

    #input_shape = (2, 2, 2, 1)

    # first transposed 3D Convolution
    x = transposed_projection_block_3D(x_after_stage_5,
                                       filters=(128, 128),
                                       kernel_size=(2, 2, 2),
                                       stage=5,
                                       block=3,
                                       se_enabled=False,
                                       se_ratio=16)
    x = concatenate([x, x_after_stage_4], axis=K.image_data_format())

    # first decoder stage
    x = projection_block_3D(x, filters=(128, 128), kernel_size=(3, 3, 3), stage=6, block=1, se_enabled=False, se_ratio=16)
    x = identity_block_3D(x, filters=(128, 128), kernel_size=(3, 3, 3), stage=6, block=2, se_enabled=False, se_ratio=16)

    # second transposed 3D Convolution
    x = transposed_projection_block_3D(x,
                                       filters=(64, 64),
                                       kernel_size=(2, 2, 2),
                                       stage=6,
                                       block=3,
                                       se_enabled=False,
                                       se_ratio=16)
    x = concatenate([x, x_after_stage_3], axis=K.image_data_format())


    # second decoder stage
    x = projection_block_3D(x, filters=(64, 64), kernel_size=(3, 3, 3), stage=7, block=1, se_enabled=False, se_ratio=16)
    x = identity_block_3D(x, filters=(64, 64), kernel_size=(3, 3, 3), stage=7, block=2, se_enabled=False, se_ratio=16)

    # third transposed 3D Convolution
    x = transposed_projection_block_3D(x,
                                       filters=(32, 32),
                                       kernel_size=(2, 2, 2),
                                       stage=7,
                                       block=3,
                                       se_enabled=False,
                                       se_ratio=16)
    x = concatenate([x, x_after_stage_2])

    # third decoder stage
    x = projection_block_3D(x, filters=(32, 32), kernel_size=(3, 3, 3), stage=8, block=1, se_enabled=False, se_ratio=16)
    x = identity_block_3D(x, filters=(32, 32), kernel_size=(3, 3, 3), stage=8, block=2, se_enabled=False, se_ratio=16)

    #fourth transposed 3D convolution
    x = transposed_projection_block_3D(x,
                                       filters=(16, 16),
                                       kernel_size=(2, 2, 2),
                                       stage=8,
                                       block=3,
                                       se_enabled=False,
                                       se_ratio=16)
    x = concatenate([x, x_after_stage_1], axis=K.image_data_format())

    # fourth decoder stage
    x = projection_block_3D(x, filters=(16, 16), kernel_size=(3, 3, 3), stage=9, block=1, se_enabled=False, se_ratio=16)

    ### End of decoder

    ### last segmentation segments
    # 1x1x1-Conv3 produces 2 featuremaps for probabilistic  segmentations of the foreground and background
    x = Conv3D(filters=2,
               kernel_size=(1, 1, 1),
               strides=(1, 1, 1),
               padding='same',
               kernel_initializer='he_normal',
               name='conv_veryEnd')(x)

    output = Softmax(axis=bn_axis)(x)

    # create model
    cnn = Model(input_tensor, output, name='3D-VResFCN')
    sModelName = cnn.name

    return cnn, sModelName


def fTrain(X_train=None, y_train=None, Y_segMasks_train=None, X_valid=None, y_valid=None, Y_segMasks_valid=None, X_test=None, y_test=None, Y_segMasks_test=None, sOutPath=None, patchSize=0, batchSizes=None, learningRates=None, iEpochs=None, dlart_handle=None):

    usingClassification = True

    # grid search on batch_sizes and learning rates
    # parse inputs
    batchSize = batchSizes[0]
    learningRate = learningRates[0]

    # change the shape of the dataset -> at color channel -> here one for grey scale
    X_train = np.expand_dims(X_train, axis=-1)
    Y_segMasks_train_foreground = np.expand_dims(Y_segMasks_train, axis=-1)
    Y_segMasks_train_background = np.ones(Y_segMasks_train_foreground.shape)-Y_segMasks_train_foreground
    Y_segMasks_train = np.concatenate((Y_segMasks_train_background, Y_segMasks_train_foreground), axis=-1)

    X_test = np.expand_dims(X_test, axis=-1)
    Y_segMasks_test_foreground = np.expand_dims(Y_segMasks_test, axis=-1)
    Y_segMasks_test_background = np.ones(Y_segMasks_test_foreground.shape)-Y_segMasks_test_foreground
    Y_segMasks_test = np.concatenate((Y_segMasks_test_background, Y_segMasks_test_foreground), axis=-1)

    if X_valid.size == 0 and y_valid.size == 0:
        print("No Validation Dataset.")
    else:
        X_valid = np.expand_dims(X_valid, axis=-1)
        Y_segMasks_valid_foreground = np.expand_dims(Y_segMasks_valid, axis=-1)
        Y_segMasks_valid_background = np.ones(Y_segMasks_valid_foreground.shape) - Y_segMasks_valid_foreground
        Y_segMasks_valid = np.concatenate((Y_segMasks_valid_background, Y_segMasks_valid_foreground), axis=-1)

    # sio.savemat('D:med_data/voxel_and_masks.mat',
    #                           {'voxel_train': X_train, 'Y_segMasks_train': Y_segMasks_train,
    #                            'y_train': y_train})

    #y_train = np.asarray([y_train[:], np.abs(np.asarray(y_train[:], dtype=np.float32) - 1)]).T
    #y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32) - 1)]).T

    # number of classes
    numClasses = np.shape(y_train)[1]

    #create cnn model
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
                dlart_handle=dlart_handle, usingClassification=usingClassification)

    K.clear_session()

    # for iBatch in batchSizes:
    #     for iLearn in learningRates:
    #         fTrainInner(cnn,
    #                     sModelName,
    #                     X_train=X_train,
    #                     y_train=y_train,
    #                     X_valid=X_valid,
    #                     y_valid=y_valid,
    #                     X_test=X_test,
    #                     y_test=y_test,
    #                     sOutPath=sOutPath,
    #                     patchSize=patchSize,
    #                     batchSize=iBatch,
    #                     learningRate=iLearn,
    #                     iEpochs=iEpochs,
    #                     dlart_handle=dlart_handle)


def fTrainInner(cnn, modelName, X_train=None, y_train=None, Y_segMasks_train=None, X_valid=None, y_valid=None,
                Y_segMasks_valid=None, X_test=None, y_test=None, Y_segMasks_test=None, sOutPath=None, patchSize=0,
                batchSize=None, learningRate=None, iEpochs=None, dlart_handle=None, usingClassification=False):
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
    if dlart_handle != None:
        if dlart_handle.getOptimizer() == DeepLearningArtApp.SGD_OPTIMIZER:
            opti = keras.optimizers.SGD(lr=learningRate,
                                        momentum=dlart_handle.getMomentum(),
                                        decay=dlart_handle.getWeightDecay(),
                                        nesterov=dlart_handle.getNesterovEnabled())

        elif dlart_handle.getOptimizer() == DeepLearningArtApp.RMS_PROP_OPTIMIZER:
            opti = keras.optimizers.RMSprop(lr=learningRate, decay=dlart_handle.getWeightDecay())

        elif dlart_handle.getOptimizer() == DeepLearningArtApp.ADAGRAD_OPTIMIZER:
            opti = keras.optimizers.Adagrad(lr=learningRate, epsilon=None, decay=dlart_handle.getWeightDecay())

        elif dlart_handle.getOptimizer() == DeepLearningArtApp.ADADELTA_OPTIMIZER:
            opti = keras.optimizers.Adadelta(lr=learningRate, rho=0.95, epsilon=None,
                                             decay=dlart_handle.getWeightDecay())

        elif dlart_handle.getOptimizer() == DeepLearningArtApp.ADAM_OPTIMIZER:
            opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None,
                                         decay=dlart_handle.getWeightDecay())
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

    if X_valid.size == 0 and y_valid.size == 0:
        # using test set for validation
        if usingClassification:
            result = cnn.fit(X_train,
                             {'segmentation_output': Y_segMasks_train, 'classification_output': y_train},
                             validation_split=0.15,
                             epochs=iEpochs,
                             batch_size=batchSize,
                             callbacks=callbacks,
                             verbose=1)
        else:
            result = cnn.fit(X_train,
                             Y_segMasks_train,
                             validation_split=0.15,
                             epochs=iEpochs,
                             batch_size=batchSize,
                             callbacks=callbacks,
                             verbose=1)
    else:
        # using validation set for validation
        if usingClassification:
            result = cnn.fit(X_train,
                             {'segmentation_output': Y_segMasks_train, 'classification_output': y_train},
                             validation_split=0.15,
                             epochs=iEpochs,
                             batch_size=batchSize,
                             callbacks=callbacks,
                             verbose=1)
        else:
            result = cnn.fit(X_train,
                             Y_segMasks_train,
                             validation_split=0.15,
                             epochs=iEpochs,
                             batch_size=batchSize,
                             callbacks=callbacks,
                             verbose=1)

    # return the loss value and metrics values for the model in test mode
    if usingClassification:
        model_metrics = cnn.metrics_names
        loss_test, segmentation_output_loss_test, classification_output_loss_test, segmentation_output_dice_coef_test, classification_output_acc_test \
            = cnn.evaluate(X_test, {'segmentation_output': Y_segMasks_test, 'classification_output': y_test}, batch_size=batchSize, verbose=1)
    else:
        score_test, dice_coef_test = cnn.evaluate(X_test, Y_segMasks_test, batch_size=batchSize, verbose=1)

    prob_test = cnn.predict(X_test, batchSize, 0)

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
        val_dice_coef = result.history['val_dice_coef']
        val_loss = result.history['val_loss']

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

        val_segmentation_output_loss = result.history['val_segmentation_output_loss']
        val_classification_output_loss = result.history['val_classification_output_loss']
        val_segmentation_output_dice_coef = result.history['val_segmentation_output_dice_coef']
        val_classification_output_acc = result.history['val_classification_output_acc']

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
   epochs_drop = 10.0
   print("Current Learning Rate: " + str(lr))
   if epoch == epochs_drop or epoch == 2*epochs_drop or epoch == 3*epochs_drop or epoch == 4*epochs_drop:
       lr = drop*lr
       print("Reduce Learningrate by 0.1 to " + str(lr))

   return lr




def fPredict(X_test, y=None, Y_segMasks_test=None, sModelPath=None, sOutPath=None, batch_size=64):
    """Takes an already trained model and computes the loss and Accuracy over the samples X with their Labels y
        Input:
            X: Samples to predict on. The shape of X should fit to the input shape of the model
            y: Labels for the Samples. Number of Samples should be equal to the number of samples in X
            sModelPath: (String) full path to a trained keras model. It should be *_json.txt file. there has to be a corresponding *_weights.h5 file in the same directory!
            sOutPath: (String) full path for the Output. It is a *.mat file with the computed loss and accuracy stored.
                        The Output file has the Path 'sOutPath'+ the filename of sModelPath without the '_json.txt' added the suffix '_pred.mat'
            batchSize: Batchsize, number of samples that are processed at once"""

    X_test = np.expand_dims(X_test, axis=-1)
    Y_segMasks_test_foreground = np.expand_dims(Y_segMasks_test, axis=-1)
    Y_segMasks_test_background = np.ones(Y_segMasks_test_foreground.shape) - Y_segMasks_test_foreground
    Y_segMasks_test = np.concatenate((Y_segMasks_test_background, Y_segMasks_test_foreground), axis=-1)

    _, sPath = os.path.splitdrive(sModelPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)

    listdir = os.listdir(sModelPath)

    #sModelPath = sModelPath.replace("_json.txt", "")
    #weight_name = sModelPath + '_weights.h5'
    #model_json = sModelPath + '_json.txt'
    #model_all = sModelPath + '_model.h5'

    # load weights and model (new way)
    with open(sModelPath + os.sep + sFilename + '.json', 'r') as fp:
        model_string = fp.read()

    model = model_from_json(model_string)

    model.summary()

    model.compile(loss=dice_coef_loss, optimizer=keras.optimizers.Adam(), metrics=[dice_coef])
    model.load_weights(sModelPath+ os.sep + sFilename+'_weights.h5')

    score_test, acc_test = model.evaluate(X_test, Y_segMasks_test, batch_size=2)
    print('loss' + str(score_test) + '   acc:' + str(acc_test))

    prob_pre = model.predict(X_test, batch_size=batch_size, verbose=1)

    predictions = {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test}

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

    #opti = SGD(lr={{choice([0.1, 0.01, 0.05, 0.005, 0.001])}}, momentum=1e-8, decay=0.1, nesterov=True)
    #cnn.compile(loss='categorical_crossentropy', optimizer=opti)

    epochs = 300

    result = cnn.fit(X_train, Y_train,
                     batch_size=128,  # {{choice([64, 128])}}
                     nb_epoch=epochs,
                     show_accuracy=True,
                     verbose=2,
                     validation_data=(X_test, Y_test))
    score_test, acc_test = cnn.evaluate(X_test, Y_test, verbose=0)

    #return {'loss': -acc_test, 'status': STATUS_OK, 'model': cnn, 'trainresult': result, 'score_test': score_test}


## helper functions
def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
    r += step
