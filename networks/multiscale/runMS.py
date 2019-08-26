# -*- coding: utf-8 -*-
"""
Created on Thu Apr 05 14:04:04 2018

@author: cyf
"""
import numpy as np                  # for algebraic operations, matrices
import scipy.io as sio              # I/O
import os.path   # operating system           
from networks.motion.VNetArt.motion_VNetArt import fGetOptimizerAndLoss
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.models import model_from_json
from keras.utils import plot_model
from . import MSnetworks
from utils.Unpatching import fUnpatchLabel, fPatchToImage
from matplotlib import pyplot as plt
import matplotlib as mpl

def frunCNN_MS(dData, sModelIn, lTrain, sOutPath, iBatchSize, iLearningRate, iEpochs, CV_Patient=0, predictImg=[]):
    if lTrain:
        if 'MultiPath' in sModelIn:
            fTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sModelIn, sOutPath, dData['patchSize'], iBatchSize, iLearningRate, iEpochs, CV_Patient=CV_Patient
                   , X_train_p2=dData['X_train_p2'], y_train_p2=dData['y_train_p2'], X_test_p2=dData['X_test_p2'], y_test_p2=dData['y_test_p2'], patchSize_down=dData['patchSize_down'], ScaleFactor=dData['ScaleFactor'])
        else:
            fTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sModelIn, sOutPath, dData['patchSize'], iBatchSize, iLearningRate, iEpochs, CV_Patient=CV_Patient)
    else:  # predicting
        if 'MS' in sModelIn:
            if not dData.has_key('X_test_p2'):
                dData['X_test_p2'] = []
            fPredict(dData['X_test'], dData['y_test'], dData['model_name'], sOutPath, patchSize=dData['patchSize'], batchSize=iBatchSize[0], patchOverlap=dData['patchOverlap'], actualSize=dData['actualSize'], X_test_p2=dData['X_test_p2'], predictImg=predictImg)
        else:
            fPredict(dData['X_test'], dData['y_test'], dData['model_name'], sOutPath, patchSize=dData['patchSize'], batchSize=iBatchSize[0], patchOverlap=dData['patchOverlap'], actualSize=dData['actualSize'], predictImg=predictImg)
        
        
def fTrain(X_train, Y_train, X_test, Y_test, sModelIn, sOutPath, patchSize, batchSizes=None, learningRates=None, iEpochs=None, CV_Patient=0, X_train_p2=None, y_train_p2=None, X_test_p2=None, y_test_p2=None, patchSize_down=None, ScaleFactor=0):

    batchSizes = [64] if batchSizes is None else batchSizes
    learningRates = [0.01] if learningRates is None else learningRates
    iEpochs = 300 if iEpochs is None else iEpochs

    # change the shape of the dataset
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    Y_train = np.asarray([Y_train[:], np.abs(np.asarray(Y_train[:], dtype=np.float32) - 1)]).T
    Y_test = np.asarray([Y_test[:], np.abs(np.asarray(Y_test[:], dtype=np.float32) - 1)]).T    
    if ScaleFactor!=0:
        X_train_p2 = np.expand_dims(X_train_p2, axis=1)
        X_test_p2 = np.expand_dims(X_test_p2, axis=1)
        y_train_p2 = np.asarray([y_train_p2[:], np.abs(np.asarray(y_train_p2[:], dtype=np.float32) - 1)]).T
        y_test_p2 = np.asarray([y_test_p2[:], np.abs(np.asarray(y_test_p2[:], dtype=np.float32) - 1)]).T
        
    for iBatch in batchSizes:
        for iLearn in learningRates:   
            if ScaleFactor==0:
                fTrainInner(sModelIn, sOutPath, patchSize, learningRate=iLearn, X_train=X_train, y_train=Y_train, X_test=X_test, y_test=Y_test,batchSize=iBatch, iEpochs=iEpochs, CV_Patient=CV_Patient)
            else:
                fTrainInner(sModelIn, sOutPath, patchSize,  learningRate=iLearn, X_train=X_train, y_train=Y_train, X_test=X_test, y_test=Y_test,batchSize=iBatch, iEpochs=iEpochs, CV_Patient=CV_Patient
                            ,X_train_p2=X_train_p2, y_train_p2=y_train_p2, X_test_p2=X_test_p2, y_test_p2=y_test_p2, patchSize_down=patchSize_down,ScaleFactor=ScaleFactor)
            
                
def fTrainInner(sModelIn, sOutPath, patchSize, learningRate=0.001, X_train=None, y_train=None, X_test=None, y_test=None,  batchSize=64, iEpochs=299, CV_Patient=0
                , X_train_p2=None, y_train_p2=None, X_test_p2=None, y_test_p2=None, patchSize_down=None, ScaleFactor=0):
    sModelIn = sModelIn[3:]
    print('Training VNet with ' + sModelIn)
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath,sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)
    model_name = sPath + '/' + sFilename  + '/' + sFilename +'_'+ sModelIn +'_lr_' + str(learningRate) + '_bs_' + str(batchSize)
    if CV_Patient != 0: model_name = model_name +'_'+ 'CV' + str(CV_Patient)# determine if crossValPatient is used...
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '_json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        print('----------already trained->go to next----------')
        return
    
    fCreateModel = 'fCreateModel_' + sModelIn
    if ScaleFactor==0: # model with single pathway
        model = getattr(MSnetworks, fCreateModel)(patchSize)
    elif 'VNet_MultiPath' in sModelIn: # dual-pathway VNet without other modules
        model = getattr(MSnetworks, fCreateModel)(patchSize, patchSize_down, ScaleFactor)
    else: # model with 2 pathways and SPP, FCN or Inception modules
        model = getattr(MSnetworks, fCreateModel)(patchSize, patchSize_down)
        
    #plot_model(model, to_file='/MultiScale/model'+sModelIn+'.png',show_shapes='True')
    opti, loss = fGetOptimizerAndLoss(optimizer='Adam', learningRate=learningRate)  # loss cat_crosent default
    model.compile(optimizer=opti, loss=loss, metrics=['accuracy'])
    model.summary()

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
    callbacks.append(ModelCheckpoint('/checkpoints/checker.hdf5', monitor='val_acc', verbose=0,
       period=5, save_best_only=True))# overrides the last checkpoint, its just for security
    callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-4, verbose=1))

    if ScaleFactor==0: # model with single pathway
        result = model.fit(x = X_train,
                          y = y_train,
                          validation_data=(X_test, y_test),
                          epochs=iEpochs,
                          batch_size=batchSize,
                          callbacks=callbacks,
                          verbose=1)
        print('\nscore and acc on test set:')
        score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batchSize, verbose=1)
        print('\npredict class probabillities:')
        prob_test = model.predict(X_test, batch_size=batchSize, verbose=1)
        
    else: # model with 2 pathways
        result = model.fit(x = [X_train, X_train_p2],
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

def fPredict(X_test,y_test, model_name, sOutPath, patchSize=[40,40,10], batchSize=64, patchOverlap=0.5, actualSize=[256, 196, 40], X_test_p2=None, predictImg=[]):
            
    weight_name = sOutPath + '/bestModel/' + model_name + '_weights.h5'
    model_json = sOutPath + '/bestModel/' + model_name + '_json.txt'
    model_all = sOutPath + '/bestModel/' + model_name + '_model.h5'

    # load weights and model (new way)
    model_json= open(model_json, 'r')
    model_string=model_json.read()
    model_json.close()
    model = model_from_json(model_string)

    model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.load_weights(weight_name)

    X_test = np.expand_dims(X_test, axis=1)
    y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32) - 1)]).T
    if ('MS' or 'MultiPath') in model_name:
        X_test_p2 = np.expand_dims(X_test_p2, axis=1)
        score_test, acc_test = model.evaluate([X_test, X_test_p2], y_test, batch_size=batchSize, verbose=1)
        print('loss_test:'+str(score_test)+ '   acc_test:'+ str(acc_test))
        prob_pre = model.predict([X_test, X_test_p2], batch_size=batchSize, verbose=1)
    else:
        score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batchSize, verbose=1)
        print('loss_test:'+str(score_test)+ '   acc_test:'+ str(acc_test))
        prob_pre = model.predict(X_test, batch_size=batchSize, verbose=1)

    modelSave = sOutPath + '/bestModel/' + model_name + '_pred.mat'
    print('saving Model:{}'.format(modelSave))
    sio.savemat(modelSave, {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test})
    #model.save(model_all)

    imglayRef, imglayArt = fUnpatchLabel(prob_pre, patchSize, patchOverlap, actualSize)
    if predictImg==[]:
        img4D =fPatchToImage(actualSize, X_test, patchOverlap) # [Art/Ref, height, width, depth]
    else:
        img4D = predictImg

    x_1d = np.arange(actualSize[0])
    y_1d = np.arange(actualSize[1])
    X, Y = np.meshgrid(x_1d, y_1d)

    for iSlice in range(0, img4D.shape[3], 5):
        fig = plt.figure(dpi=100)
        plt.cla()
        plt.gca().set_aspect('equal')
        plt.xlim(0, actualSize[0])
        plt.ylim(actualSize[1], 0)
        plt.set_cmap(plt.gray())

        # Head: pcolormesh(X, Y,..., vmax=0.5)    Abdomen: pcolormesh(Y, X,..., vmax=0.3)
        plt.subplot(211)
        plt.axis('off')
        plt.pcolormesh(X, Y, np.swapaxes(img4D[0, :, :, iSlice], 0, 1), vmin=0, vmax=0.5)
        plt.pcolormesh(X, Y, np.swapaxes(imglayRef[:, :, iSlice], 0, 1), cmap='jet', alpha=0.2, vmin=0, vmax=1, linestyle='None', rasterized=True)
        plt.title('Slice ' + str(iSlice) + ' without artifacts', fontsize = 26)
        plt.subplot(212)
        plt.axis('off')
        plt.pcolormesh(X, Y, np.swapaxes(img4D[1, :, :, iSlice], 0, 1), vmin=0, vmax=0.5)
        plt.pcolormesh(X, Y, np.swapaxes(imglayArt[:, :, iSlice], 0, 1), cmap='jet', alpha=0.2, vmin=0, vmax=1, linestyle='None', rasterized=True)
        plt.title('Slice ' + str(iSlice) + ' with motion artifacts', fontsize = 26)

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75, top=0.85, hspace=0.2)
        cax = plt.axes([0.8, 0.1, 0.08, 0.75])
        plt.colorbar(cax=cax)
        plt.yticks(fontsize=24)
        fig.set_size_inches(8, 12)
        plt.savefig(sOutPath + '/Overlay/' + model_name + '_Slice' + str(iSlice) + '.png')
        plt.show()