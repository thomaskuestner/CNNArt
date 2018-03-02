import os.path
import scipy.io as sio
import numpy as np
import keras
import keras.optimizers
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda, Reshape
from keras.activations import relu, elu, softmax
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.initializers import Constant
from keras.layers import  concatenate, add
from keras.layers.convolutional import Conv3D, UpSampling3D, MaxPooling3D, Cropping3D, ZeroPadding3D
from keras.regularizers import l1_l2,l2
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

def fgetKernelNumber():
    # KN[0] for conv with kernel 1X1X1, KN[1] for conv with kernel 3X3X3 or 5X5X5
    KernelNumber = [32, 64, 128]
    return KernelNumber
def fgetLayerNumIncep():
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

def fTrain(X_train, Y_train, X_test, Y_test, sOutPath, patchSize, batchSizes=None, learningRates=None, iEpochs=None, CV_Patient=0):

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
            fTrainInner(sOutPath, patchSize, learningRate=iLearn, X_train=X_train, y_train=Y_train, X_test=X_test, y_test=Y_test,batchSize=iBatch, iEpochs=iEpochs, CV_Patient=CV_Patient)

def fTrainInner(sOutPath, patchSize, learningRate=0.001, X_train=None, y_train=None, X_test=None, y_test=None,  batchSize=64, iEpochs=299, CV_Patient=0):
    '''train a model with training data X_train with labels Y_train. Validation Data should get the keywords Y_test and X_test'''

    print('Training VNet with inception block')
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    cnn_incep = fCreateModel(patchSize)
    opti, loss = fGetOptimizerAndLoss(optimizer='Adam', learningRate=learningRate)  # loss cat_crosent default
    cnn_incep.compile(optimizer=opti, loss=loss, metrics=['accuracy'])
    cnn_incep.summary()

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath,sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)
    model_name = sPath + '/' + sFilename  + '/' + sFilename + '_VNet_Incep' +'_lr_' + str(learningRate) + '_bs_' + str(batchSize)
    if CV_Patient != 0: model_name = model_name +'_'+ 'CV' + str(CV_Patient)# determine if crossValPatient is used...
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '_json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        print('----------already trained->go to next----------')
        return

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
    callbacks.append(ModelCheckpoint('/no_backup/s1241/checkpoints/checker.hdf5', monitor='val_acc', verbose=0,
       period=5, save_best_only=True))# overrides the last checkpoint, its just for security
    callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-4, verbose=1))

    result =cnn_incep.fit(x = X_train,
                      y = y_train,
                      validation_data=(X_test, y_test),
                      epochs=iEpochs,
                      batch_size=batchSize,
                      callbacks=callbacks,
                      verbose=1)

    print('\nscore and acc on test set:')
    score_test, acc_test = cnn_incep.evaluate(X_test,y_test, batch_size=batchSize, verbose=1)
    print('\npredict class probabillities:')
    prob_test = cnn_incep.predict(X_test, batch_size=batchSize, verbose=1)

    # save model
    json_string = cnn_incep.to_json()
    open(model_json +'.txt', 'w').write(json_string)
    cnn_incep.save_weights(weight_name, overwrite=True)

    # matlab
    acc = result.history['acc']
    loss = result.history['loss']
    val_acc = result.history['val_acc']
    val_loss = result.history['val_loss']
    print('Saving results: ' + model_name)
    sio.savemat(model_name, {'model_settings': model_json,
                             'model': model_all,
                             'acc': acc,
	                         'loss': loss,
	                         'val_acc': val_acc,
	                         'val_loss': val_loss,
	                         'score_test': score_test,
	                         'acc_test': acc_test,
	                         'prob_test': prob_test})

def fPredict(X_test,y_test, model_name, sOutPath, batchSize=64):
    # Takes an already trained model and computes the loss and Accuracy over the samples X with their Labels y

    X_test = np.expand_dims(X_test, axis=1)
    y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32) - 1)]).T

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

    score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batchSize, verbose=1)
    print('loss_test:'+str(score_test)+ '   acc_test:'+ str(acc_test))
    prob_pre = model.predict(X_test, batch_size=batchSize, verbose=1)

    modelSave = sOutPath + '/' + model_name + '_pred.mat'
    print('saving Model:{}'.format(modelSave))
    sio.savemat(modelSave, {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test})
    #model.save(model_all)

def fCreateModel(patchSize,dr_rate=0.0, iPReLU=0, l2_reg=1e-6):
    # Total params: 5,483,161
    Strides = fgetStrides()
    kernelnumber = fgetKernelNumber()
    inp = Input(shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2])))

    after_Incep_1 =  fConvIncep(inp, KB=kernelnumber[0], layernum=fgetLayerNumIncep(), l2_reg=l2_reg)
    after_DownConv_1 = fCreateVNet_DownConv_Block(after_Incep_1, after_Incep_1._keras_shape[1], Strides[0],
                                                     iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Incep_2 = fConvIncep(after_DownConv_1, KB=kernelnumber[1], layernum=fgetLayerNumIncep(), l2_reg=l2_reg)
    after_DownConv_2 = fCreateVNet_DownConv_Block(after_Incep_2, after_Incep_2._keras_shape[1], Strides[1],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Incep_3 = fConvIncep(after_DownConv_2, KB=kernelnumber[2], layernum=fgetLayerNumIncep(), l2_reg=l2_reg)
    after_DownConv_2 = fCreateVNet_DownConv_Block(after_Incep_3, after_Incep_3._keras_shape[1], Strides[2],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    # fully connect layer as dense
    flat_out = Flatten()(after_DownConv_2)
    dropout_out = Dropout(dr_rate)(flat_out)
    dense_out = Dense(units=2,
                          kernel_initializer='normal',
                          kernel_regularizer=l2(l2_reg))(dropout_out)

    outp = Activation('softmax')(dense_out)
    cnn_incep = Model(inputs=inp, outputs=outp)
    return cnn_incep

def fConvIncep(input_t, KB=64, layernum=2, l1_reg=0.0, l2_reg=1e-6, iPReLU=0):
    tower_t = Conv3D(filters=KB,
                     kernel_size=[2,2,1],
                     kernel_initializer='he_normal',
                     weights=None,
                     padding='same',
                     strides=(1, 1, 1),
                     kernel_regularizer=l1_l2(l1_reg, l2_reg),
                     )(input_t)
    incep = fGetActivation(tower_t, iPReLU=iPReLU)

    for counter in range(1,layernum):
        incep = InceptionBlock(incep, l1_reg=l1_reg, l2_reg=l2_reg)

    incepblock_out = concatenate([incep, input_t], axis=1)
    return incepblock_out

def InceptionBlock(inp, l1_reg=0.0, l2_reg=1e-6):
    KN = fgetKernelNumber()
    branch1 = Conv3D(filters=KN[0], kernel_size=(1,1,1), kernel_initializer='he_normal', weights=None,padding='same',
                     strides=(1,1,1),kernel_regularizer=l1_l2(l1_reg, l2_reg),activation='relu')(inp)

    branch3 = Conv3D(filters=KN[0], kernel_size=(1, 1, 1), kernel_initializer='he_normal', weights=None, padding='same',
                     strides=(1, 1, 1), kernel_regularizer=l1_l2(l1_reg, l2_reg), activation='relu')(inp)
    branch3 = Conv3D(filters=KN[2], kernel_size=(3, 3, 3), kernel_initializer='he_normal', weights=None, padding='same',
                     strides=(1, 1, 1), kernel_regularizer=l1_l2(l1_reg, l2_reg), activation='relu')(branch3)

    branch5 = Conv3D(filters=KN[0], kernel_size=(1, 1, 1), kernel_initializer='he_normal', weights=None, padding='same',
                     strides=(1, 1, 1), kernel_regularizer=l1_l2(l1_reg, l2_reg), activation='relu')(inp)
    branch5 = Conv3D(filters=KN[1], kernel_size=(5, 5, 5), kernel_initializer='he_normal', weights=None, padding='same',
                     strides=(1, 1, 1), kernel_regularizer=l1_l2(l1_reg, l2_reg), activation='relu')(branch5)

    branchpool = MaxPooling3D(pool_size=(3,3,3),strides=(1,1,1),padding='same',data_format='channels_first')(inp)
    branchpool = Conv3D(filters=KN[0], kernel_size=(1, 1, 1), kernel_initializer='he_normal', weights=None, padding='same',
                     strides=(1, 1, 1), kernel_regularizer=l1_l2(l1_reg, l2_reg), activation='relu')(branchpool)
    out = concatenate([branch1, branch3, branch5, branchpool], axis=1)
    return out

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

def fGetActivation(input_t,  iPReLU=0):
    init=0.25
    if iPReLU == 1:  # one alpha for each channel
        output_t = PReLU(alpha_initializer=Constant(value=init), shared_axes=[2, 3, 4])(input_t)
    elif iPReLU == 2:  # just one alpha for each layer
        output_t = PReLU(alpha_initializer=Constant(value=init), shared_axes=[2, 3, 4, 1])(input_t)
    else:
        output_t = Activation('relu')(input_t)
    return output_t

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