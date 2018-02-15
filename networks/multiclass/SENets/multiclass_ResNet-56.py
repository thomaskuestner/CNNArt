import os.path
import scipy.io as sio
import numpy as np
import keras
from keras.layers import Input
import keras.backend as K
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Activation, Flatten

from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping
from keras.regularizers import l2  # , activity_l2

from keras.optimizers import SGD

from networks.multiclass.SENets.deep_residual_learning_blocks import *
from DeepLearningArt.DLArt_GUI.dlart import DeepLearningArtApp
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

def createModel(patchSize, numClasses):
    # ResNet-56 based on CIFAR-10, for 32x32 Images
    print(K.image_data_format())

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    input_tensor = Input(shape=(patchSize[0], patchSize[1], 1))

    # first conv layer
    x = Conv2D(16, (3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', name='conv1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)

    # first stage of 2n=2*9=18 Convs (3x3, 16)
    x = identity_block(x, [16, 16], stage=1, block=1)
    x = identity_block(x, [16, 16], stage=1, block=2)
    x = identity_block(x, [16, 16], stage=1, block=3)
    x = identity_block(x, [16, 16], stage=1, block=4)
    x = identity_block(x, [16, 16], stage=1, block=5)
    x = identity_block(x, [16, 16], stage=1, block=6)
    x = identity_block(x, [16, 16], stage=1, block=7)
    x = identity_block(x, [16, 16], stage=1, block=8)
    x = identity_block(x, [16, 16], stage=1, block=9)

    # second stage of 2n=2*9=18 convs (3x3, 32)
    x = projection_block(x, [32, 32], stage=2, block=1)
    x = identity_block(x, [32, 32], stage=2, block=2)
    x = identity_block(x, [32, 32], stage=2, block=3)
    x = identity_block(x, [32, 32], stage=2, block=4)
    x = identity_block(x, [32, 32], stage=2, block=5)
    x = identity_block(x, [32, 32], stage=2, block=6)
    x = identity_block(x, [32, 32], stage=2, block=7)
    x = identity_block(x, [32, 32], stage=2, block=8)
    x = identity_block(x, [32, 32], stage=2, block=9)

    # third stage of 3n=3*9=18 convs (3x3, 64)
    x = projection_block(x, [64, 64], stage=3, block=1)
    x = identity_block(x, [64, 64], stage=3, block=2)
    x = identity_block(x, [64, 64], stage=3, block=3)
    x = identity_block(x, [64, 64], stage=3, block=4)
    x = identity_block(x, [64, 64], stage=3, block=5)
    x = identity_block(x, [64, 64], stage=3, block=6)
    x = identity_block(x, [64, 64], stage=3, block=7)
    x = identity_block(x, [64, 64], stage=3, block=8)
    x = identity_block(x, [64, 64], stage=3, block=9)

    # global average pooling
    x = GlobalAveragePooling2D(data_format='channels_last')(x)

    # fully-connected layer
    output = Dense(units=numClasses,
                   activation='softmax',
                   kernel_initializer='he_normal',
                   name='fully-connected')(x)

    # create model
    cnn = Model(input_tensor, output, name='ResNet-56')

    return cnn


def fTrain(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSizes=None, learningRates=None, iEpochs=None, dlart_handle=None):
    # grid search on batch_sizes and learning rates
    # parse inputs
    batchSizes = [64] if batchSizes is None else batchSizes
    learningRates = [0.01] if learningRates is None else learningRates
    iEpochs = 300 if iEpochs is None else iEpochs

    # change the shape of the dataset -> at color channel -> here one for grey scale
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    #y_train = np.asarray([y_train[:], np.abs(np.asarray(y_train[:], dtype=np.float32) - 1)]).T
    #y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32) - 1)]).T

    for iBatch in batchSizes:
        for iLearn in learningRates:
            fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, iBatch, iLearn, iEpochs, dlart_handle)


def fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSize=None, learningRate=None, iEpochs=None, dlart_handle=None):
    # parse inputs
    batchSize = 64 if batchSize is None else batchSize
    learningRate = 0.01 if learningRate is None else learningRate
    iEpochs = 300 if iEpochs is None else iEpochs

    print('Training CNN')
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)

    model_name = sOutPath + os.sep + sFilename + '_lr_' + str(learningRate) + '_bs_' + str(batchSize)
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '_json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        return

    #number of classes
    numClasses = np.shape(y_train)[1]

    # create model
    cnn = createModel(patchSize, numClasses=numClasses)

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
    cnn.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])

    # callbacks
    callback_earlyStopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
    callback_tensorBoard = keras.callbacks.TensorBoard(log_dir=dlart_handle.getLearningOutputPath() + '/logs',
                                                       histogram_freq=2,
                                                       batch_size=batchSize,
                                                       write_graph=True,
                                                       write_grads=True,
                                                       write_images=True,
                                                       embeddings_freq=0,
                                                       embeddings_layer_names=None,
                                                       embeddings_metadata=None)

    callbacks = [callback_earlyStopping, callback_tensorBoard]


    # data augmentation
    if dlart_handle.getDataAugmentationEnabled() == True:
        # Initialize Image Generator
        # all shifted and rotated images are filled with zero padded pixels
        if dlart_handle.getRotation() == True and dlart_handle.getHeightShift() == False and dlart_handle.getWidthShift() == False:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=dlart_handle.getZCA_Whitening(),
                zca_epsilon=1e-6,
                rotation_range=30,
                width_shift_range=0.,
                height_shift_range=0.,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='constant',
                cval=0.,
                horizontal_flip=dlart_handle.getHorizontalFlip(),
                vertical_flip=dlart_handle.getVerticalFlip(),
                rescale=None,
                preprocessing_function=None,
                data_format=K.image_data_format()
            )
        elif dlart_handle.getRotation() == False and dlart_handle.getHeightShift() == True and dlart_handle.getWidthShift() == False:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=dlart_handle.getZCA_Whitening(),
                zca_epsilon=1e-6,
                rotation_range=0,
                width_shift_range=0.,
                height_shift_range=0.2,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='constant',
                cval=0.,
                horizontal_flip=dlart_handle.getHorizontalFlip(),
                vertical_flip=dlart_handle.getVerticalFlip(),
                rescale=None,
                preprocessing_function=None,
                data_format=K.image_data_format()
            )
        elif dlart_handle.getRotation() == False and dlart_handle.getHeightShift() == False and dlart_handle.getWidthShift() == True:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=dlart_handle.getZCA_Whitening(),
                zca_epsilon=1e-6,
                rotation_range=0,
                width_shift_range=0.2,
                height_shift_range=0,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='constant',
                cval=0.,
                horizontal_flip=dlart_handle.getHorizontalFlip(),
                vertical_flip=dlart_handle.getVerticalFlip(),
                rescale=None,
                preprocessing_function=None,
                data_format=K.image_data_format()
            )
        elif dlart_handle.getRotation() == True and dlart_handle.getHeightShift() == True and dlart_handle.getWidthShift() == False:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=dlart_handle.getZCA_Whitening(),
                zca_epsilon=1e-6,
                rotation_range=30,
                width_shift_range=0.,
                height_shift_range=0.2,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='constant',
                cval=0.,
                horizontal_flip=dlart_handle.getHorizontalFlip(),
                vertical_flip=dlart_handle.getVerticalFlip(),
                rescale=None,
                preprocessing_function=None,
                data_format=K.image_data_format()
            )
        elif dlart_handle.getRotation() == True and dlart_handle.getHeightShift() == False and dlart_handle.getWidthShift() == True:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=dlart_handle.getZCA_Whitening(),
                zca_epsilon=1e-6,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='constant',
                cval=0.,
                horizontal_flip=dlart_handle.getHorizontalFlip(),
                vertical_flip=dlart_handle.getVerticalFlip(),
                rescale=None,
                preprocessing_function=None,
                data_format=K.image_data_format()
            )
        elif dlart_handle.getRotation() == False and dlart_handle.getHeightShift() == True and dlart_handle.getWidthShift() == True:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=dlart_handle.getZCA_Whitening(),
                zca_epsilon=1e-6,
                rotation_range=0,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='constant',
                cval=0.,
                horizontal_flip=dlart_handle.getHorizontalFlip(),
                vertical_flip=dlart_handle.getVerticalFlip(),
                rescale=None,
                preprocessing_function=None,
                data_format=K.image_data_format()
            )
        elif dlart_handle.getRotation() == True and dlart_handle.getHeightShift() == True and dlart_handle.getWidthShift() == True:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=dlart_handle.getZCA_Whitening(),
                zca_epsilon=1e-6,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='constant',
                cval=0.,
                horizontal_flip=dlart_handle.getHorizontalFlip(),
                vertical_flip=dlart_handle.getVerticalFlip(),
                rescale=None,
                preprocessing_function=None,
                data_format=K.image_data_format()
            )
        elif dlart_handle.getRotation() == False and dlart_handle.getHeightShift() == False and dlart_handle.getWidthShift() == False:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=dlart_handle.getZCA_Whitening(),
                zca_epsilon=1e-6,
                rotation_range=0,
                width_shift_range=0.,
                height_shift_range=0.,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='constant',
                cval=0.,
                horizontal_flip=dlart_handle.getHorizontalFlip(),
                vertical_flip=dlart_handle.getVerticalFlip(),
                rescale=None,
                preprocessing_function=None,
                data_format=K.image_data_format()
            )
        else:
            raise ValueError("No valid data augmentation option configuration!")

        # fit parameters from dataset
        datagen.fit(X_train)

        # configure batch size and get one batch of images
        for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
            # display first 9 images
            for i in range(0, 9):
                plt.subplot(330+1+i)
                plt.imshow(x_batch[i].reshape(x_batch.shape[1], x_batch.shape[2]), cmap='gray')
            plt.show()
            break

        # fit model on data
        result = cnn.fit_generator(datagen.flow(X_train, y_train, batch_size=batchSize),
                                   steps_per_epoch=X_train.shape[0]//batchSize,
                                   epochs=iEpochs,
                                   validation_data=(X_test, y_test),
                                   callbacks=callbacks,
                                   workers=1,
                                   use_multiprocessing=False
        )
    else:
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
                             'score_test': score_test,
                             'acc_test': acc_test,
                             'prob_test': prob_test})


def fPredict(X_test, y_test, model_name, sOutPath, patchSize, batchSize):
    weight_name = model_name[0] + '_weights.h5'
    model_json = model_name[0] + '_json'
    model_all = model_name[0] + '_model.h5'

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

    model.compile(loss='categorical_crossentropy', optimizer=opti)
    model.load_weights(weight_name)

    # load complete model (including weights); keras > 0.7
    # model = load_model(model_all)

    # assume artifact affected shall be tested!
    # y_test = np.ones((len(X_test),1))

    score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batchSize, show_accuracy=True)
    prob_pre = model.predict(X_test, batchSize, 0)

    # modelSave = model_name[:-5] + '_pred.mat'
    modelSave = model_name[0] + '_pred.mat'
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
