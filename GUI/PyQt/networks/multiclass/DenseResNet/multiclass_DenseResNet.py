import h5py
import keras.models
import numpy as np  # for algebraic operations, matrices
import os.path
import scipy.io as sio
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK
# from theano import functionfrom keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, add
from keras.layers.advanced_activations import PReLU, ELU
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D as pool2
from keras.layers.core import Dense, Activation, Flatten, Dropout  # , Layer, Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Sequential
# from keras.layers import containers
from keras.models import model_from_json, Model, load_model
from keras.optimizers import SGD
from keras.preprocessing import image
# from keras.layers.convolutional import ZeroPadding2D as zero2d
from keras.regularizers import l2  # , activity_l2
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from networks.multiclass.DenseNet.Densenet import DenseNet


# elementary denseResBlock
def Block(input, num_filters, with_shortcut):
    out1 = Conv2D(filters=int(num_filters / 2), kernel_size=(1, 1), kernel_initializer='he_normal', weights=None,
                  padding='same',
                  strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(input)
    out2 = Conv2D(filters=int(num_filters), kernel_size=(3, 3), kernel_initializer='he_normal', weights=None,
                  padding='same',
                  strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out1)
    out3 = Conv2D(filters=int(num_filters), kernel_size=(1, 1), kernel_initializer='he_normal', weights=None,
                  padding='same',
                  strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out2)
    # out4 = pool2(pool_size=(3, 3), strides=(2, 2), data_format="channel_first")(out3)

    if with_shortcut:
        input = Conv2D(filters=num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None,
                       padding='same', strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(input)
        return add([input, out3])
    else:
        input = Conv2D(filters=num_filters, kernel_size=(1, 1), kernel_initializer='he_normal', weights=None,
                       padding='same', strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(input)
        return add([input, out3])


# DenseResNet 4040
def create4040Model(patchSize, outputDimension=11, iVersion=1):
    seed = 5
    np.random.seed(seed)
    input = Input(shape=(patchSize[0], patchSize[1], 1))

    # DenseResNet 4040 (selected)
    if iVersion == 1:
        out = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='same',
                     data_format='channels_last', strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(input)
        out1 = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='same',
                      data_format='channels_last', strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out)
        # out1=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

        out = Block(out1, 128, with_shortcut=True)
        out = Block(out, 128, with_shortcut=False)
        out = concatenate(inputs=[out1, out], axis=3)
        out2 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        out = Block(out2, 256, with_shortcut=True)
        out = Block(out, 256, with_shortcut=False)
        out1 = pool2(pool_size=(3, 3), strides=(2, 2), data_format="channels_first")(out1)
        out = concatenate(inputs=[out1, out2, out], axis=3)
        out3 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        # out5=GlobalAveragePooling2D(data_format='channels_first')(out3)

        out5 = Flatten()(out3)

        out6 = Dense(units=outputDimension,
                     kernel_initializer='normal',
                     kernel_regularizer='l2',
                     activation='softmax')(out5)

        cnn = Model(inputs=input, outputs=out6)

    # temp/resdensenet1 for 4040
    elif iVersion == 2:
        out = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='same',
                     data_format='channels_last', strides=(2, 2), kernel_regularizer=l2(1e-6), activation='relu')(input)
        out = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='same',
                     data_format='channels_last', strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out)
        out1 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        out = Block(out1, 128, with_shortcut=True)
        out = Block(out, 128, with_shortcut=False)
        out = concatenate(inputs=[out1, out], axis=3)
        out2 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        out = Block(out2, 256, with_shortcut=True)
        out = Block(out, 256, with_shortcut=False)
        out1 = pool2(pool_size=(2, 2), data_format="channels_first")(out1)
        out = concatenate(inputs=[out1, out2, out], axis=1)
        out3 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        # out5=GlobalAveragePooling2D(data_format='channels_first')(out4)

        out5 = Flatten()(out3)

        out6 = Dense(units=11,
                     kernel_initializer='normal',
                     kernel_regularizer='l2',
                     activation='softmax')(out5)

        cnn = Model(inputs=input, outputs=out6)

    # temp/resdensenet3 for 4040
    elif iVersion == 3:
        out = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='same',
                     strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(input)
        out1 = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='same',
                      strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out)
        # out1=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

        sout1 = Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='same',
                       strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out1)
        out = Block(out1, 128, with_shortcut=True)
        out = Block(out, 128, with_shortcut=False)
        out = add([sout1, out])
        out = concatenate(inputs=[out1, out], axis=1)
        out2 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        sout2 = Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='same',
                       strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out2)
        out = Block(out2, 256, with_shortcut=True)
        out = Block(out, 256, with_shortcut=False)
        out = add([sout2, out])
        out1 = pool2(pool_size=(3, 3), strides=(2, 2), data_format="channels_first")(out1)
        out = concatenate(inputs=[out1, out2, out], axis=1)
        out3 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        # out5=GlobalAveragePooling2D(data_format='channels_first')(out3)

        out5 = Flatten()(out3)

        out6 = Dense(units=11,
                     kernel_initializer='normal',
                     kernel_regularizer='l2',
                     activation='softmax')(out5)

        cnn = Model(inputs=input, outputs=out6)

    return cnn


# DenseResNet 180180
def create180180Model(patchSize, outputDimension=11, iVersion=1):
    seed = 5
    np.random.seed(seed)
    # define input shape=(xSize, ySize, colorChannel=1)
    input = Input(shape=(patchSize[0], patchSize[1], 1))

    # DenseResNet 180180 (selected)
    if (iVersion == 1):
        out = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='valid',
                     strides=(2, 2), kernel_regularizer=l2(1e-6), activation='relu')(input)
        out = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='valid',
                     strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out)
        out1 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        out = Block(out1, 128, with_shortcut=True)
        out = Block(out, 128, with_shortcut=False)
        out = concatenate(inputs=[out1, out], axis=1)
        out2 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        out = Block(out2, 128, with_shortcut=True)
        out = Block(out, 128, with_shortcut=False)
        out1 = pool2(pool_size=(2, 2), data_format="channels_first")(out1)
        out = concatenate(inputs=[out1, out2, out], axis=1)
        out3 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        out = Block(out3, 256, with_shortcut=True)
        out = Block(out, 256, with_shortcut=False)
        out2 = pool2(pool_size=(2, 2), data_format="channels_first")(out2)
        out = concatenate(inputs=[out2, out3, out], axis=1)
        out4 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        # out5=GlobalAveragePooling2D(data_format='channels_first')(out4)

        out5 = Flatten()(out4)

        out6 = Dense(units=outputDimension,
                     kernel_initializer='normal',
                     kernel_regularizer='l2',
                     activation='softmax')(out5)

        cnn = Model(inputs=input, outputs=out6)

    # temp/resdensenet2 for 180180
    elif iVersion == 2:
        out = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='valid',
                     strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(input)
        out = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='valid',
                     strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out)
        out1 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        out = Block(out1, 128, with_shortcut=True)
        out = Block(out, 128, with_shortcut=False)
        out = Block(out, 128, with_shortcut=False)
        out = concatenate(inputs=[out1, out], axis=1)
        out2 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        out = Block(out2, 128, with_shortcut=True)
        out = Block(out, 128, with_shortcut=False)
        out = Block(out, 128, with_shortcut=False)
        out1 = pool2(pool_size=(2, 2), data_format="channels_first")(out1)
        out = concatenate(inputs=[out1, out2, out], axis=1)
        out3 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        out = Block(out3, 256, with_shortcut=True)
        out = Block(out, 256, with_shortcut=False)
        out = Block(out, 256, with_shortcut=False)
        out2 = pool2(pool_size=(2, 2), data_format="channels_first")(out2)
        out = concatenate(inputs=[out2, out3, out], axis=1)
        out4 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        # out5=GlobalAveragePooling2D(data_format='channels_first')(out4)

        out5 = Flatten()(out4)

        out6 = Dense(units=11,
                     kernel_initializer='normal',
                     kernel_regularizer='l2',
                     activation='softmax')(out5)

        cnn = Model(inputs=input, outputs=out6)

    elif iVersion == 3:
        out = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='valid',
                     strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(input)
        out = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='valid',
                     strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out)
        out1 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        out = Block(out1, 128, with_shortcut=True)
        out = Block(out, 128, with_shortcut=False)
        out = Block(out, 128, with_shortcut=False)
        out = concatenate(inputs=[out1, out], axis=1)
        out2 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        out = Block(out2, 128, with_shortcut=True)
        out = Block(out, 128, with_shortcut=False)
        out = Block(out, 128, with_shortcut=False)
        out1 = pool2(pool_size=(3, 3), strides=(2, 2), data_format="channels_first")(out1)
        out = concatenate(inputs=[out1, out2, out], axis=1)
        out3 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        out = Block(out3, 256, with_shortcut=True)
        out = Block(out, 256, with_shortcut=False)
        out2 = pool2(pool_size=(3, 3), strides=(2, 2), data_format="channels_first")(out2)
        out = concatenate(inputs=[out2, out3, out], axis=1)
        out4 = pool2(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(out)

        # out5=GlobalAveragePooling2D(data_format='channels_first')(out4)

        out5 = Flatten()(out4)

        out6 = Dense(units=11,
                     kernel_initializer='normal',
                     kernel_regularizer='l2',
                     activation='softmax')(out5)

        cnn = Model(inputs=input, outputs=out6)

    return cnn


def fTrain(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSizes=None, learningRates=None, iEpochs=None):
    # grid search on batch_sizes and learning rates
    # parse inputs
    batchSizes = [64] if batchSizes is None else [batchSizes]
    learningRates = [0.01] if learningRates is None else learningRates
    iEpochs = 300 if iEpochs is None else iEpochs

    for iBatch in batchSizes:
        for iLearn in learningRates:
            fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, iBatch, iLearn, iEpochs)


def fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSize=None, learningRate=None, iEpochs=None):
    # parse inputs
    batchSize = [64] if batchSize is None else batchSize
    learningRate = [0.01] if learningRate is None else learningRate
    iEpochs = 300 if iEpochs is None else iEpochs

    outputDimension = y_train.shape[1]

    # reshpae input tensors
    X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], X_train.shape[2],
                                   1])  # to shape [numSamples, xSize, ySize, channels=1]
    X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1], X_test.shape[2], 1])

    print('Training CNN DenseResNet')
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)
    model_name = sPath + os.sep + sFilename + str(patchSize[0]) + str(patchSize[1]) + '_lr_' + str(
        learningRate) + '_bs_' + str(batchSize)
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '.json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        return

    # create model
    if (patchSize[0] == 40 & patchSize[1] == 40):
        cnn = create4040Model(patchSize, outputDimension=outputDimension)
    elif (patchSize[0] == 180 & patchSize[1] == 180):
        cnn = create180180Model(patchSize)
    else:
        print('NO models for patch size ' + patchSize[0] + patchSize[1])

    # opti = SGD(lr=learningRate, momentum=1e-8, decay=0.1, nesterov=True);#Adag(lr=0.01, epsilon=1e-06)
    optimizer = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                 ModelCheckpoint(filepath=model_name + 'bestweights.hdf5', monitor='val_acc', verbose=0,
                                 save_best_only=True, save_weights_only=False)]
    # callbacks = [ModelCheckpoint(filepath=model_name+'bestweights.hdf5',monitor='val_acc',verbose=0,save_best_only=True,save_weights_only=False)]

    cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    cnn.summary()

    # in keras fit() validation_data there is the test set used. -> no validation set used!
    # compare function definition in doc: fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    # two options: validation_split(splits validation set from training set) or validation_data (overrides validation_split)
    # BUT: consider if 'normal splitting' or 'cross validation splitting' is used
    result = cnn.fit(X_train,
                     y_train,
                     validation_data=[X_test, y_test],
                     epochs=iEpochs,
                     batch_size=batchSize,
                     callbacks=callbacks,
                     verbose=1)

    score_test, acc_test = cnn.evaluate(X_test, y_test, batch_size=batchSize)

    prob_test = cnn.predict(X_test, batchSize, 0)
    y_pred = np.argmax(prob_test, axis=1)
    y_test = np.argmax(y_test, axis=1)
    confusion_mat = confusion_matrix(y_test, y_pred)

    # save model
    json_string = cnn.to_json()
    open(model_json, 'w').write(json_string)
    # wei = cnn.get_weights()
    cnn.save_weights(weight_name, overwrite=True)
    cnn.save(model_all)  # keras > v0.7
    model_png_dir = sOutPath + os.sep +  "model.png"
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
                             'prob_test': prob_test,
                             'confusion_mat': confusion_mat})


def fPredict(X_test, y_test, model_name, sOutPath, patchSize, batchSize):
    weight_name = model_name[0]
    # model_json = model_name[1] + '.json'
    # model_all = model_name[0] + '.hdf5'
    _, sPath = os.path.splitdrive(sOutPath)
    sPath, sFilename = os.path.split(sOutPath)
    # sFilename, sExt = os.path.splitext(sFilename)

    # f = h5py.File(weight_name, 'r+')
    # del f['optimizer_weights']
    # f.close()
    model = load_model(weight_name)
    opti = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]

    # model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
    # model.load_weights(weight_name)
    model.summary()

    score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batchSize)
    prob_pre = model.predict(X_test, batchSize, 0)

    y_pred = np.argmax(prob_pre, axis=1)
    y_test = np.argmax(y_test, axis=1)
    confusion_mat = confusion_matrix(y_test, y_pred)
    # modelSave = model_name[:-5] + '_pred.mat'
    modelSave = sOutPath + '/' + sFilename + '_result.mat'
    sio.savemat(modelSave,
                {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test, 'confusion_mat': confusion_mat})


###############################################################################
## OPTIMIZATIONS ##
###############################################################################
def fStatistic(confusion_mat):  # each column represents a predicted label, each row represents a truth label
    cm = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, decimals=3)
    dim = cm.shape[0]
    BER = np.sum(np.diag(np.identity(dim) - cm), axis=0) / dim
    # Recall = np.sum(np.diag(confusion_mat)/np.sum(confusion_mat,axis=1),axis=1) / col
    # Precision = np.sum(np.diag(normal_cm),axis=1) / col
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    return BER


## helper functions
def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step
