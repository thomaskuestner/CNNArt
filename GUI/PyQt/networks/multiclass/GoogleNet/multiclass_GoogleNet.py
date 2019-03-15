import os.path
import scipy.io as sio
import numpy as np  # for algebraic operations, matrices
import keras.models
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout  # , Layer, Flatten
# from keras.layers import containers
from keras.models import model_from_json,Model
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK
from sklearn.metrics import confusion_matrix
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D as pool2
from keras.callbacks import EarlyStopping,ModelCheckpoint
# from keras.layers.convolutional import ZeroPadding2D as zero2d
from keras.regularizers import l2  # , activity_l2
# from theano import functionfrom keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from keras.layers.merge import concatenate
from keras.layers import Input,add
from keras.layers.advanced_activations import PReLU,ELU
from keras.layers.pooling import GlobalAveragePooling2D


def createModel(patchSize, iVersion = 1):
    seed=5
    np.random.seed(seed)
    input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))

    # googlenet7 for 120120 180180
    if iVersion == 1:
        out1=Conv2D(filters=32,
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(input)

        out2=Conv2D(filters=64,
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out1)
        out2=pool2(pool_size=(2,2),data_format='channels_first')(out2)

        out3=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out2)


        out4=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out3)
        out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

        out5_1=Conv2D(filters=32,
                      kernel_size=(1,1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='same',
                      strides=(1, 1),
                      kernel_regularizer=l2(1e-6),
                      activation='relu')(out4)

        out5_2=Conv2D(filters=32,  # learning rate: 0.1 -> 76%
                      kernel_size=(1,1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='same',
                      strides=(1, 1),
                      kernel_regularizer=l2(1e-6),
                      activation='relu')(out4)
        out5_2=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
                      kernel_size=(3,3),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='same',
                      strides=(1, 1),
                      kernel_regularizer=l2(1e-6),
                      activation='relu')(out5_2)

        out5_3=Conv2D(filters=32,  # learning rate: 0.1 -> 76%
                      kernel_size=(1,1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='same',
                      strides=(1, 1),
                      kernel_regularizer=l2(1e-6),
                      activation='relu')(out4)
        out5_3=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
                      kernel_size=(5,5),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='same',
                      strides=(1, 1),
                      kernel_regularizer=l2(1e-6),
                      activation='relu')(out5_3)

        out5_4=pool2(pool_size=(3,3),strides=(1,1),padding='same',data_format='channels_first')(out4)
        out5_4=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
                      kernel_size=(1,1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='same',
                      strides=(1, 1),
                      kernel_regularizer=l2(1e-6),
                      activation='relu')(out5_4)

        out5=concatenate(inputs=[out5_1,out5_2,out5_3],axis=1)

        out7=Conv2D(filters=256,  # learning rate: 0.1 -> 76%
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out5)
        #out7=pool2(pool_size=(2,2),data_format='channels_first')(out7)

        out8=Conv2D(filters=256,  # learning rate: 0.1 -> 76%
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out7)
        out8=pool2(pool_size=(2,2),data_format='channels_first')(out8)

        out9=Flatten()(out8)


        out10=Dense(units=11,
                   kernel_initializer='normal',
                   kernel_regularizer='l2',
                   activation='softmax')(out9)

        cnn = Model(inputs=input,outputs=out10)

    #336C2P for 4040
    elif iVersion == 2:
        out1=Conv2D(filters=32,
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(input)

        out2=Conv2D(filters=64,
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out1)

        out3=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out2)

        out4=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out3)
        out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

        out5=Conv2D(filters=256,
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out4)

        out6=Conv2D(filters=256,  # learning rate: 0.1 -> 76%
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out5)
        out6=pool2(pool_size=(2,2),data_format='channels_first')(out6)

        out7=Flatten()(out6)

        out10=Dense(units=11,
                   kernel_initializer='normal',
                   kernel_regularizer='l2',
                   activation='softmax')(out7)

        cnn = Model(inputs=input,outputs=out10)

    #temp/googlenet764
    elif iVersion == 3:
        out1=Conv2D(filters=64,
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(input)

        out2=Conv2D(filters=64,
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out1)
        out2=pool2(pool_size=(2,2),data_format='channels_first')(out2)

        out3=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out2)


        out4=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out3)
        out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

        out5_1=Conv2D(filters=32,
                      kernel_size=(1,1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='same',
                      strides=(1, 1),
                      kernel_regularizer=l2(1e-6),
                      activation='relu')(out4)

        out5_2=Conv2D(filters=32,  # learning rate: 0.1 -> 76%
                      kernel_size=(1,1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='same',
                      strides=(1, 1),
                      kernel_regularizer=l2(1e-6),
                      activation='relu')(out4)
        out5_2=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
                      kernel_size=(3,3),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='same',
                      strides=(1, 1),
                      kernel_regularizer=l2(1e-6),
                      activation='relu')(out5_2)

        out5_3=Conv2D(filters=32,  # learning rate: 0.1 -> 76%
                      kernel_size=(1,1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='same',
                      strides=(1, 1),
                      kernel_regularizer=l2(1e-6),
                      activation='relu')(out4)
        out5_3=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
                      kernel_size=(5,5),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='same',
                      strides=(1, 1),
                      kernel_regularizer=l2(1e-6),
                      activation='relu')(out5_3)

        out5_4=pool2(pool_size=(3,3),strides=(1,1),padding='same',data_format='channels_first')(out4)
        out5_4=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
                      kernel_size=(1,1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='same',
                      strides=(1, 1),
                      kernel_regularizer=l2(1e-6),
                      activation='relu')(out5_4)

        out5=concatenate(inputs=[out5_1,out5_2,out5_3],axis=1)

        out7=Conv2D(filters=256,  # learning rate: 0.1 -> 76%
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out5)
        #out7=pool2(pool_size=(2,2),data_format='channels_first')(out7)

        out8=Conv2D(filters=256,  # learning rate: 0.1 -> 76%
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    weights=None,
                    padding='valid',
                    strides=(1, 1),
                    kernel_regularizer=l2(1e-6),
                    activation='relu')(out7)
        out8=pool2(pool_size=(2,2),data_format='channels_first')(out8)

        out9=Flatten()(out8)


        out10=Dense(units=11,
                   kernel_initializer='normal',
                   kernel_regularizer='l2',
                   activation='softmax')(out9)

        cnn = Model(inputs=input,outputs=out10)

    #temp/resgooglenet764 wrong!!!
    elif iVersion == 4:
        out1=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(input)
        out2=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out1)
        out2=pool2(pool_size=(2,2),data_format='channels_first')(out2)

        out3=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out2)
        out4=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out3)
        out4=add([out2,out4])
        out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

        out5_1=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)

        out5_2=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
        out5_2=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_2)

        out5_3=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
        out5_3=Conv2D(filters=128,kernel_size=(5,5),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_3)

        out5_4=pool2(pool_size=(3,3),strides=(1,1),padding='same',data_format='channels_first')(out4)
        out5_4=Conv2D(filters=128,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_4)

        out5=concatenate(inputs=[out5_1,out5_2,out5_3],axis=1)

    #	out6_1=Conv2D(filters=32,kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5)
    #
    #	out6_2=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5)
    #	out6_2=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out6_2)
    #
    #	out6_3=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5)
    #	out6_3=Conv2D(filters=128,kernel_size=(5,5),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out6_3)
    #
    #	out6_4=pool2(pool_size=(3,3),strides=(1,1),padding='same',data_format='channels_first')(out5)
    #	out6_4=Conv2D(filters=128,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out6_4)
    #
    #	out6=concatenate(inputs=[out6_1,out6_2,out6_3,out6_4],axis=1)

        out7=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
                    activation='relu')(out5)

        out8=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
                    activation='relu')(out7)
        out8=add([out5, out8])
        out8=pool2(pool_size=(2,2),data_format='channels_first')(out8)

        out9=Flatten()(out8)


        out10=Dense(units=11,
                   kernel_initializer='normal',
                   kernel_regularizer='l2',
                   activation='softmax')(out9)

        cnn = Model(inputs=input,outputs=out10)

    #temp/resgooglenet764_v1
    elif iVersion == 5:
        out1=Conv2D(filters=32,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(input)
        out2=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out1)
        out2=pool2(pool_size=(2,2),data_format='channels_first')(out2)

        out3=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out2)
        out4=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out3)
        out4=add([out2,out4])
        out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

        out_3=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
        out_4=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out_3)

        out5_1=Conv2D(filters=64,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out_4)

        out5_2=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out_4)
        out5_2=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_2)

        out5_3=Conv2D(filters=64,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out_4)
        out5_3=Conv2D(filters=128,kernel_size=(5,5),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_3)

        out5_4=pool2(pool_size=(3,3),strides=(1,1),padding='same',data_format='channels_first')(out4)
        out5_4=Conv2D(filters=128,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_4)

        out5=concatenate(inputs=[out5_1,out5_2,out5_3],axis=1)

        sout6=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
                    activation='relu')(out5)

        out7=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
                    activation='relu')(out5)

        out8=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
                    activation='relu')(out7)
        out8=add([sout6, out8])
        out8=pool2(pool_size=(2,2),data_format='channels_first')(out8)

        out9=Flatten()(out8)


        out10=Dense(units=11,
                   kernel_initializer='normal',
                   kernel_regularizer='l2',
                   activation='softmax')(out9)

        cnn = Model(inputs=input,outputs=out10)

    return cnn


def fTrain(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSizes=None, learningRates=None, iEpochs=None):
# grid search on batch_sizes and learning rates
# parse inputs
    batchSizes = 64 if batchSizes is None else batchSizes
    learningRates = 0.01 if learningRates is None else learningRates
    iEpochs = 300 if iEpochs is None else iEpochs
    for iBatch in batchSizes:
        for iLearn in learningRates:
            fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, iBatch, iLearn, iEpochs)


def fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSize=None, learningRate=None, iEpochs=None):
# parse inputs
    batchSize = 64 if batchSize is None else batchSize
    learningRate = 0.01 if learningRate is None else learningRate
    iEpochs = 300 if iEpochs is None else iEpochs

    print('Training CNN GoogleNet')
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)
    model_name = sPath + '/' + sFilename + str(patchSize[0, 0]) + str(patchSize[0, 1]) + '_lr_' + str(
        learningRate) + '_bs_' + str(batchSize)
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '_json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        return

    cnn = createModel(patchSize, 1)
    opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=1), ModelCheckpoint(filepath=model_name+'bestweights.hdf5',monitor='val_acc',verbose=0,save_best_only=True,save_weights_only=False)]
    #callbacks = [ModelCheckpoint(filepath=model_name+'bestweights.hdf5',monitor='val_acc',verbose=0,save_best_only=True,save_weights_only=False)]

    cnn.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
    cnn.summary()
    result = cnn.fit(X_train,
                     y_train,
                     validation_data=[X_test, y_test],
                     epochs=iEpochs,
                     batch_size=batchSize,
                     callbacks=callbacks,
                     verbose=1)

    score_test, acc_test = cnn.evaluate(X_test, y_test, batch_size=batchSize )

    prob_test = cnn.predict(X_test, batchSize, 0)
    y_pred=np.argmax(prob_test,axis=1)
    y_test=np.argmax(y_test,axis=1)
    confusion_mat=confusion_matrix(y_test,y_pred)

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
                             'score_test': score_test,
                             'acc_test': acc_test,
                             'prob_test': prob_test,
                             'confusion_mat':confusion_mat})

def fPredict(X_test, y_test, model_name, sOutPath, patchSize, batchSize):
    weight_name = model_name[0]
    #model_json = model_name[1] + '_json'
    #model_all = model_name[0] + '.hdf5'
    _, sPath = os.path.splitdrive(sOutPath)
    sPath, sFilename = os.path.split(sOutPath)
    #sFilename, sExt = os.path.splitext(sFilename)

    #f = h5py.File(weight_name, 'r+')
    #del f['optimizer_weights']
    #f.close()
    model=load_model(weight_name)
    opti = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]

    #model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
    #model.load_weights(weight_name)
    model.summary()

    score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batchSize)
    prob_pre = model.predict(X_test, batchSize, 0)

    y_pred=np.argmax(prob_pre,axis=1)
    y_test=np.argmax(y_test,axis=1)
    confusion_mat=confusion_matrix(y_test,y_pred)
    # modelSave = model_name[:-5] + '_pred.mat'
    modelSave = sOutPath + '/' + sFilename + '_result.mat'
    sio.savemat(modelSave, {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test, 'confusion_mat':confusion_mat})