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
from keras.layers import  concatenate, add, average, GlobalAveragePooling3D
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.regularizers import l1_l2,l2
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import theano.tensor as T
from keras.utils import plot_model

def fgetKernelNumber():
    KernelNumber = [32, 64, 128]
    return KernelNumber
def fgetLayerNumConv():
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
            fTrainInner(sOutPath, patchSize, learningRate=iLearn, X_train=X_train, y_train=Y_train, X_test=X_test, y_test=Y_test,batchSize=iBatch, iEpochs=iEpochs, CV_Patient=CV_Patient,
                        X_train_p2=X_train_p2, y_train_p2=y_train_p2, X_test_p2=X_test_p2, y_test_p2=y_test_p2, patchSize_down=patchSize_down,ScaleFactor=ScaleFactor)

def fTrainInner(sOutPath, patchSize, learningRate=0.001, X_train=None, y_train=None, X_test=None, y_test=None,  batchSize=64, iEpochs=299, CV_Patient=0
                , X_train_p2=None, y_train_p2=None, X_test_p2=None, y_test_p2=None, patchSize_down=None, ScaleFactor=None):
    '''train a model with training data X_train with labels Y_train. Validation Data should get the keywords Y_test and X_test'''

    print('Training VNet with fully convolutional net')
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath,sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)
    model_name = sPath + '/' + sFilename  + '/' + sFilename + '_VNet_FCN_MS' +'_lr_' + str(learningRate) + '_bs_' + str(batchSize)
    if CV_Patient != 0: model_name = model_name +'_'+ 'CV' + str(CV_Patient)# determine if crossValPatient is used...
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '_json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        print('----------already trained->go to next----------')
        return

    model = fCreateModel_MultiFM_2Path(patchSize, patchSize_down)
    plot_model(model, to_file='/no_backup/s1241/MultiScale/modelFCN.png',show_shapes='True')
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
    prob_test = model.predict([X_test, X_test_p2], batch_size=batchSize, verbose=1)

    # result = model.fit(x=X_train,
    #                     y=y_train,
    #                     validation_data=(X_test, y_test),
    #                     epochs=iEpochs,
    #                     batch_size=batchSize,
    #                     callbacks=callbacks,
    #                     verbose=1)
    # print('\nscore and acc on test set:')
    # score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batchSize, verbose=1)
    # print('\npredict class probabillities:')
    # prob_test = model.predict(X_test, batch_size=batchSize, verbose=1)

    # save model
    json_string = model.to_json()
    open(model_json +'.txt', 'w').write(json_string)
    model.save_weights(weight_name, overwrite=True)

    # matlab
    print('Saving results: ' + model_name)
    sio.savemat(model_name, {'model_settings': model_json,
                             'model': model_all,
                             'weights': weight_name,
                             'training_result':result.history,
                             'val_loss': score_test,
                             'val_acc': acc_test,
                             'prob_test': prob_test})

def fPredict(X_test,y_test, model_name, sOutPath, batchSize=64, X_test_p2=None):
    # Takes an already trained model and computes the loss and Accuracy over the samples X with their Labels y

    X_test = np.expand_dims(X_test, axis=1)
    X_test_p2 = np.expand_dims(X_test_p2, axis=1)
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

    score_test, acc_test = model.evaluate([X_test, X_test_p2], y_test, batch_size=batchSize, verbose=1)
    print('loss_test:'+str(score_test)+ '   acc_test:'+ str(acc_test))
    prob_pre = model.predict(X_test, batch_size=batchSize, verbose=1)

    modelSave = sOutPath + '/' + model_name + '_pred.mat'
    print('saving Model:{}'.format(modelSave))
    sio.savemat(modelSave, {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test})
    #model.save(model_all)

def fCreateModel(patchSize,dr_rate=0.0, iPReLU=0, l1_reg=0.0, l2_reg=1e-6):
    # Total params: 1,223,831
    # Replace the dense layer with a convolutional layer with filters=2 for the two classes
    Strides = fgetStrides()
    kernelnumber = fgetKernelNumber()
    inp = Input(shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2])))

    after_Conv_1 = fCreateVNet_Block(inp, kernelnumber[0], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_1 = fCreateVNet_DownConv_Block(after_Conv_1, after_Conv_1._keras_shape[1], Strides[0],
                                                     iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Conv_2 = fCreateVNet_Block(after_DownConv_1, kernelnumber[1], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_2 = fCreateVNet_DownConv_Block(after_Conv_2, after_Conv_2._keras_shape[1], Strides[1],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Conv_3 = fCreateVNet_Block(after_DownConv_2, kernelnumber[2], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_3 = fCreateVNet_DownConv_Block(after_Conv_3, after_Conv_3._keras_shape[1], Strides[2],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    dropout_out = Dropout(dr_rate)(after_DownConv_3)
    fclayer = Conv3D(2,
                       kernel_size=(1,1,1),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l1_l2(l1_reg, l2_reg),
                       )(dropout_out)
    fclayer = GlobalAveragePooling3D()(fclayer)
    outp = Activation('softmax')(fclayer)
    cnn_spp = Model(inputs=inp, outputs=outp)
    return cnn_spp

def fCreateModel_MultiFM(patchSize, dr_rate=0.0, iPReLU=0,l1_reg=0, l2_reg=1e-6):
    # Total params: 1,420,549
    # The dense layer is repleced by a convolutional layer with filters=2 for the two classes
    # The FM from the third down scaled convolutional layer is upsempled by deconvolution and
    # added with the FM from the second down scaled convolutional layer.
    # The combined FM goes through a convolutional layer with filters=2 for the two classes
    # The two predictions are averages as the final result.
    Strides = fgetStrides()
    kernelnumber = fgetKernelNumber()
    inp = Input(shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2])))

    after_Conv_1 = fCreateVNet_Block(inp, kernelnumber[0], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_1 = fCreateVNet_DownConv_Block(after_Conv_1, after_Conv_1._keras_shape[1], Strides[0],
                                                     iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Conv_2 = fCreateVNet_Block(after_DownConv_1, kernelnumber[1], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_2 = fCreateVNet_DownConv_Block(after_Conv_2, after_Conv_2._keras_shape[1], Strides[1],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_Conv_3 = fCreateVNet_Block(after_DownConv_2, kernelnumber[2], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_DownConv_3 = fCreateVNet_DownConv_Block(after_Conv_3, after_Conv_3._keras_shape[1], Strides[2],
                                                   iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    # fully convolution over the FM from the deepest level
    dropout_out1 = Dropout(dr_rate)(after_DownConv_3)
    fclayer1 = Conv3D(2,
                       kernel_size=(1,1,1),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l1_l2(l1_reg, l2_reg),
                       )(dropout_out1)
    fclayer1 = GlobalAveragePooling3D()(fclayer1)
    
    # Upsample FM from the deepest level, add with FM from level 2, 
    UpedFM_Level3 = Conv3DTranspose(filters=97, kernel_size=(3,3,1), strides=(2,2,1), padding='same')(after_DownConv_3)
    conbined_FM_Level23 = add([UpedFM_Level3, after_DownConv_2])    
    fclayer2 = Conv3D(2,
                       kernel_size=(1,1,1),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l1_l2(l1_reg, l2_reg),
                       )(conbined_FM_Level23)
    fclayer2 = GlobalAveragePooling3D()(fclayer2)

    # combine the two predictions using average
    fcl_aver = average([fclayer1, fclayer2])
    predict = Activation('softmax')(fcl_aver)
    cnn_fcl_msfm = Model(inputs=inp, outputs=predict)
    return cnn_fcl_msfm


def fCreateModel_MultiFM_2Path(patchSize, patchSize_down, dr_rate=0.0, iPReLU=0, l1_reg=0, l2_reg=1e-6):
    # Total params: 2,841,098
    # The dense layer is repleced by a convolutional layer with filters=2 for the two classes
    # The FM from the third down scaled convolutional layer is upsempled by deconvolution and
    # added with the FM from the second down scaled convolutional layer.
    # The combined FM goes through a convolutional layer with filters=2 for the two classes
    # The four predictions from the two pathways are averages as the final result.
    Strides = fgetStrides()
    kernelnumber = fgetKernelNumber()
    sharedConv1 = fCreateVNet_Block
    sharedDown1 = fCreateVNet_DownConv_Block
    sharedConv2 = fCreateVNet_Block
    sharedDown2 = fCreateVNet_DownConv_Block
    sharedConv3 = fCreateVNet_Block
    sharedDown3 = fCreateVNet_DownConv_Block

    inp1 = Input(shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2])))
    after_1Conv_1 = sharedConv1(inp1, kernelnumber[0], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_1DownConv_1 = sharedDown1(after_1Conv_1, after_1Conv_1._keras_shape[1], Strides[0],
                                                  iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_1Conv_2 = sharedConv2(after_1DownConv_1, kernelnumber[1], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_1DownConv_2 = sharedDown2(after_1Conv_2, after_1Conv_2._keras_shape[1], Strides[1],
                                                  iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_1Conv_3 = sharedConv3(after_1DownConv_2, kernelnumber[2], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_1DownConv_3 = sharedDown3(after_1Conv_3, after_1Conv_3._keras_shape[1], Strides[2],
                                                  iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    inp2 = Input(shape=(1, int(patchSize_down[0]), int(patchSize_down[1]), int(patchSize_down[2])))
    after_2Conv_1 = sharedConv1(inp2, kernelnumber[0], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_2DownConv_1 = sharedDown1(after_2Conv_1, after_2Conv_1._keras_shape[1], Strides[0],
                                                  iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_2Conv_2 = sharedConv2(after_2DownConv_1, kernelnumber[1], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_2DownConv_2 = sharedDown2(after_2Conv_2, after_2Conv_2._keras_shape[1], Strides[1],
                                                  iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)

    after_2Conv_3 = sharedConv3(after_2DownConv_2, kernelnumber[2], type=fgetLayerNumConv(), l2_reg=l2_reg)
    after_2DownConv_3 = sharedDown3(after_2Conv_3, after_2Conv_3._keras_shape[1], Strides[2],
                                                  iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
    # fully convolution over the FM from the deepest level
    dropout_out1 = Dropout(dr_rate)(after_1DownConv_3)
    fclayer1 = Conv3D(2,
                      kernel_size=(1, 1, 1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='valid',
                      strides=(1, 1, 1),
                      kernel_regularizer=l1_l2(l1_reg, l2_reg),
                      )(dropout_out1)
    fclayer1 = GlobalAveragePooling3D()(fclayer1)

    # Upsample FM from the deepest level, add with FM from level 2,
    UpedFM_1Level3 = Conv3DTranspose(filters=97, kernel_size=(3, 3, 1), strides=(2, 2, 1), padding='same')(
        after_1DownConv_3)
    conbined_FM_1Level23 = add([UpedFM_1Level3, after_1DownConv_2])
    fclayer2 = Conv3D(2,
                      kernel_size=(1, 1, 1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='valid',
                      strides=(1, 1, 1),
                      kernel_regularizer=l1_l2(l1_reg, l2_reg),
                      )(conbined_FM_1Level23)
    fclayer2 = GlobalAveragePooling3D()(fclayer2)

    dropout_out2 = Dropout(dr_rate)(after_2DownConv_3)
    fclayer3 = Conv3D(2,
                      kernel_size=(1, 1, 1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='valid',
                      strides=(1, 1, 1),
                      kernel_regularizer=l1_l2(l1_reg, l2_reg),
                      )(dropout_out2)
    fclayer3 = GlobalAveragePooling3D()(fclayer3)

    # Upsample FM from the deepest level, add with FM from level 2,
    UpedFM_2Level3 = Conv3DTranspose(filters=97, kernel_size=(3, 3, 1), strides=(2, 2, 1), padding='same')(
        after_2DownConv_3)
    conbined_FM_2Level23 = add([UpedFM_2Level3, after_2DownConv_2])
    fclayer4 = Conv3D(2,
                      kernel_size=(1, 1, 1),
                      kernel_initializer='he_normal',
                      weights=None,
                      padding='valid',
                      strides=(1, 1, 1),
                      kernel_regularizer=l1_l2(l1_reg, l2_reg),
                      )(conbined_FM_2Level23)
    fclayer4 = GlobalAveragePooling3D()(fclayer4)
    # combine the two predictions using average
    fcl_aver = average([fclayer1, fclayer2, fclayer3, fclayer4])
    predict = Activation('softmax')(fcl_aver)
    cnn_fcl_2p = Model(inputs=[inp1, inp2], outputs=predict)
    return cnn_fcl_2p

def fCreateVNet_DownConv_Block(input_t, channels, stride, l1_reg=0.0, l2_reg=1e-6, iPReLU=0, dr_rate=0):
    output_t=Dropout(dr_rate)(input_t)
    output_t=Conv3D(channels,
                    kernel_size=stride,
                    strides=stride,
                    weights=None,
                    padding='same',
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
