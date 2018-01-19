import os.path
import scipy.io as sio
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

def fTrain(X_train, Y_train, X_test, Y_test, sOutPath, patchSize,batchSizes=None, learningRates=None, iEpochs=None,sInPaths=None,sInPaths_valid=None, CV_Patient=0, model='motion_head'):#rigid for loops for simplicity
    #add for loops here

    batchSizes = [64] if batchSizes is None else batchSizes
    learningRates = [0.001] if learningRates is None else learningRates
    iEpochs = 300 if iEpochs is None else iEpochs

    for iBatch in batchSizes:
        for iLearn in learningRates:
            cnn = fCreateModel(patchSize, learningRate=iLearn, optimizer='Adam')
            fTrainInner(sOutPath, cnn, learningRate=iLearn, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,batchSize=iBatch, iEpochs=iEpochs)

def fTrainInner(sOutPath, model, learningRate=0.001, patchSize=None, sInPaths=None, sInPaths_valid=None, X_train=None, Y_train=None, X_test=None, Y_test=None,  batchSize=64, iEpochs=299, CV_Patient=0):
    '''train a model with training data X_train with labels Y_train. Validation Data should get the keywords Y_test and X_test'''

    print('Training CNN3D')
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath,sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)
    model_name = sPath + '/' + sFilename + '/' + sFilename +'_lr_' + str(learningRate) + '_bs_' + str(batchSize)
    if CV_Patient != 0: model_name = model_name +'_'+ 'CV' + str(CV_Patient)# determine if crossValPatient is used...
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '_json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        print('----------already trained->go to next----------')
        return


    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
    callbacks.append(ModelCheckpoint('/home/s1222/no_backup/s1222/checkpoints/checker.hdf5', monitor='val_acc', verbose=0,
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

def fPredict(X,y,  sModelPath, sOutPath, batchSize=64):
    """Takes an already trained model and computes the loss and Accuracy over the samples X with their Labels y
    Input:
        X: Samples to predict on. The shape of X should fit to the input shape of the model
        y: Labels for the Samples. Number of Samples should be equal to the number of samples in X
        sModelPath: (String) full path to a trained keras model. It should be *_json.txt file. there has to be a corresponding *_weights.h5 file in the same directory!
        sOutPath: (String) full path for the Output. It is a *.mat file with the computed loss and accuracy stored. 
                    The Output file has the Path 'sOutPath'+ the filename of sModelPath without the '_json.txt' added the suffix '_pred.mat' 
        batchSize: Batchsize, number of samples that are processed at once"""
    sModelPath=sModelPath.replace("_json.txt", "")
    weight_name = sModelPath + '_weights.h5'
    model_json = sModelPath + '_json.txt'
    model_all = sModelPath + '_model.h5'

    # load weights and model (new way)
    model_json= open(model_json, 'r')
    model_string=model_json.read()
    model_json.close()
    model = model_from_json(model_string)

    model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.load_weights(weight_name)


    score_test, acc_test = model.evaluate(X, y, batch_size=batchSize)
    print('loss'+str(score_test)+ '   acc:'+ str(acc_test))
    prob_pre = model.predict(X, batch_size=batchSize, verbose=1)
    print(prob_pre[0:14,:])
    _,sModelFileSave  = os.path.split(sModelPath)

    modelSave = sOutPath +sModelFileSave+ '_pred.mat'
    print('saving Model:{}'.format(modelSave))
    sio.savemat(modelSave, {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test})

def fCreateModel(patchSize, learningRate=1e-3, optimizer='SGD',
                     dr_rate=0.0, input_dr_rate=0.0, max_norm=5, iPReLU=0, l2_reg=1e-6):
# change to functional API
        input_t = Input(shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2])))
        seq_t= Dropout(dr_rate)(input_t)
        seq_t = Conv3D(32,  # numChans
                       kernel_size=(14, 14, 5),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l2(l2_reg),
                       input_shape=(1, int(patchSize[0]), int(patchSize[1]), int(patchSize[2]))
                       )(seq_t)
        seq_t = fGetActivation(seq_t, iPReLU=iPReLU)

        seq_t = Dropout(dr_rate)(seq_t)
        seq_t = Conv3D(64,
                       kernel_size=(7, 7, 3),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l2(l2_reg))(seq_t)

        seq_t = fGetActivation(seq_t, iPReLU=iPReLU)

        seq_t = Dropout(dr_rate)(seq_t)
        seq_t = Conv3D(128,
                       kernel_size=(3, 3, 2),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l2(l2_reg))(seq_t)

        seq_t = fGetActivation(seq_t, iPReLU=iPReLU)

        seq_t = Flatten()(seq_t)

        seq_t = Dropout(dr_rate)(seq_t)
        seq_t = Dense(units=2,
                      kernel_initializer='normal',
                      kernel_regularizer=l2(l2_reg))(seq_t)
        output_t = Activation('softmax')(seq_t)

        opti, loss = fGetOptimizerAndLoss(optimizer, learningRate=learningRate)  # loss cat_crosent default

        cnn = Model(inputs=[input_t], outputs=[output_t])
        cnn.compile(loss=loss, optimizer=opti, metrics=['accuracy'])
        sArchiSpecs = '_l2{}'.format(l2_reg)

        return cnn


####################################################################helpers#############################################
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


def fGetActivation(input_t,  iPReLU=0):
    init=0.25
    if iPReLU == 1:  # one alpha for each channel
        output_t = PReLU(alpha_initializer=Constant(value=init), shared_axes=[2, 3, 4])(input_t)
    elif iPReLU == 2:  # just one alpha for each layer
        output_t = PReLU(alpha_initializer=Constant(value=init), shared_axes=[2, 3, 4, 1])(input_t)
    else:
        output_t = Activation('relu')(input_t)
    return output_t
