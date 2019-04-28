# -*- coding: utf-8 -*-
"""
----------------------------------
Main function for calling the CNNs
----------------------------------
Created on Wed Jan 27 16:57:10 2016
Copyright: 2016, 2017 Thomas Kuestner (thomas.kuestner@med.uni-tuebingen.de) under Apache2 license
@author: Thomas Kuestner
"""
from tensorflow.python.keras.models import load_model

from config.PATH import LEARNING_OUT

"""Import"""

import sys
import numpy as np  # for algebraic operations, matrices
import h5py
import scipy.io as sio  # I/O
import os.path  # operating system
import argparse
import keras.backend as K

# networks
from networks.motion.CNN2D import *
from networks.motion.CNN3D import *
from networks.motion.MNetArt import *
from networks.motion.VNetArt import *
from networks.multiclass.DenseResNet import *
from networks.multiclass.InceptionNet import *
from networks.multiclass.SENets import *

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

"""functions"""

RUN_CNN_TRAIN_TEST_VALIDATION = 0
RUN_CNN_TRAIN_TEST = 1
RUN_CNN_PREDICT = 2


def fLoadData(conten):
    # prepared in matlab
    print('Loading data')
    for sVarname in ['X_train', 'X_test', 'y_train', 'y_test']:
        if sVarname in conten:
            exec(sVarname + '=conten[sVarname]')
        else:
            exec(sVarname + '= None')

    pIdx = np.random.permutation(np.arange(len(X_train)))
    X_train = X_train[pIdx]
    y_train = y_train[pIdx]
    y_train = np.asarray([y_train[:, 0], np.abs(np.asarray(y_train[:, 0], dtype=np.float32) - 1)]).T
    y_test = np.asarray([y_test[:, 0], np.abs(np.asarray(y_test[:, 0], dtype=np.float32) - 1)]).T
    return X_train, y_train, X_test, y_test


def fRemove_entries(entries, the_dict):
    for key in entries:
        if key in the_dict:
            del the_dict[key]


def fLoadMat(sInPath):
    """Data"""
    if os.path.isfile(sInPath):
        try:
            conten = sio.loadmat(sInPath)
        except:
            f = h5py.File(sInPath, 'r')
            conten = {}
            conten['X_train'] = np.transpose(np.array(f['X_train']), (3, 2, 0, 1))
            conten['X_test'] = np.transpose(np.array(f['X_test']), (3, 2, 0, 1))
            conten['y_train'] = np.transpose(np.array(f['y_train']))
            conten['y_test'] = np.transpose(np.array(f['y_test']))
            conten['patchSize'] = np.transpose(np.array(f['patchSize']))
    else:
        sys.exit('Input file is not existing')
    X_train, y_train, X_test, y_test = fLoadData(conten)  # output order needed for hyperas

    fRemove_entries(('X_train', 'X_test', 'y_train', 'y_test'), conten)
    dData = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    dOut = dData.copy()
    dOut.update(conten)
    return dOut  # output dictionary (similar to conten, but with reshaped X_train, ...)


def fLoadDataForOptim(sInPath):
    if os.path.isfile(sInPath):
        conten = sio.loadmat(sInPath)
    X_train, y_train, X_test, y_test = fLoadData(conten)  # output order needed for hyperas
    return X_train, y_train, X_test, y_test, conten["patchSize"]


# def fLoadAddData(sInPath): # deprecated
#    if os.path.isfile(sInPath):
#        conten = sio.loadmat(sInPath)
#    else:
#        sys.exit('Input file is not existing')
#    for sVarname in conten:
#        if not any(x in sVarname for x in ['X_train', 'X_test', 'y_train', 'y_test'] ):
#            conten[sVarname]

def fRunCNN(dData, sModelIn, lTrain, sParaOptim, sOutPath, iBatchSize, iLearningRate, iEpochs, dlart_handle=None,
            usingSegmentationMasks=False):
    """CNN Models"""
    # check model
    sModel = sModelIn

    # dynamic loading of corresponding model
    cnnModel = __import__(sModel, globals(), locals(), ['createModel', 'fTrain', 'fPredict', 'load_best_model'],
                          0)  # dynamic module loading with specified functions and with absolute importing (level=0) -> work in both Python2 and Python3

    # train (w/ or w/o optimization) and predicting
    if lTrain == RUN_CNN_TRAIN_TEST:  # training
        if sParaOptim == 'hyperas':  # hyperas parameter optimization
            best_run, best_model = optim.minimize(model=cnnModel.fHyperasTrain,
                                                  data=fLoadDataForOptim(args.inPath[0]),
                                                  algo=tpe.suggest,
                                                  max_evals=5,
                                                  trials=Trials())
            X_train, y_train, X_test, y_test, patchSize = fLoadDataForOptim(args.inPath[0])
            score_test, acc_test = best_model.evaluate(X_test, y_test)
            prob_test = best_model.predict(X_test, best_run['batch_size'], 0)

            _, sPath = os.path.splitdrive(sOutPath)
            sPath, sFilename = os.path.split(sPath)
            sFilename, sExt = os.path.splitext(sFilename)
            model_name = sPath + '/' + sFilename + str(patchSize[0, 0]) + str(patchSize[0, 1]) + '_best'
            weight_name = model_name + '_weights.h5'
            model_json = model_name + '.json'
            model_all = model_name + '_model.h5'
            json_string = best_model.to_json()
            open(model_json, 'w').write(json_string)
            # wei = best_model.get_weights()
            best_model.save_weights(weight_name)
            best_model.save(model_all)

            result = best_run['result']
            # acc = result.history['acc']y,
            loss = result.history['loss']
            val_acc = result.history['val_acc']
            val_loss = result.history['val_loss']
            sio.savemat(model_name, {'model_settings': model_json,
                                     'model': model_all,
                                     'weights': weight_name,
                                     'acc': -best_run['loss'],
                                     'loss': loss,
                                     'val_acc': val_acc,
                                     'val_loss': val_loss,
                                     'score_test': score_test,
                                     'acc_test': acc_test,
                                     'prob_test': prob_test})

        elif sParaOptim == 'grid':  # grid search << backward compatibility
            cnnModel.fTrain(X_traind=dData['X_train'],
                            y_traind=dData['y_train'],
                            X_test=dData['X_test'],
                            y_test=dData['y_test'],
                            sOutPath=sOutPath,
                            patchSize=dData['patchSize'],
                            batchSizes=iBatchSize,
                            learningRates=iLearningRate,
                            iEpochs=iEpochs,
                            dlart_handle=dlart_handle)


        else:  # no optimization or grid search (if batchSize|learningRate are arrays)
            if not usingSegmentationMasks:
                cnnModel.fTrain(X_train=dData['X_train'],
                                y_train=dData['y_train'],
                                X_test=dData['X_test'],
                                y_test=dData['y_test'],
                                sOutPath=sOutPath,
                                patchSize=dData['patchSize'],
                                batchSizes=iBatchSize,
                                learningRates=iLearningRate,
                                iEpochs=iEpochs,
                                dlart_handle=dlart_handle)
            else:
                cnnModel.fTrain(X_train=dData['X_train'],
                                y_train=dData['y_train'],
                                Y_segMasks_train=dData['Y_segMasks_train'],
                                X_test=dData['X_test'],
                                y_test=dData['y_test'],
                                Y_segMasks_test=dData['Y_segMasks_test'],
                                sOutPath=sOutPath,
                                patchSize=dData['patchSize'],
                                batchSizes=iBatchSize,
                                learningRates=iLearningRate,
                                iEpochs=iEpochs,
                                dlart_handle=dlart_handle)


    elif lTrain == RUN_CNN_TRAIN_TEST_VALIDATION:
        if sParaOptim == 'hyperas':  # hyperas parameter optimization
            best_run, best_model = optim.minimize(model=cnnModel.fHyperasTrain,
                                                  data=fLoadDataForOptim(args.inPath[0]),
                                                  algo=tpe.suggest,
                                                  max_evals=5,
                                                  trials=Trials())
            X_train, y_train, X_test, y_test, patchSize = fLoadDataForOptim(args.inPath[0])
            score_test, acc_test = best_model.evaluate(X_test, y_test)
            prob_test = best_model.predict(X_test, best_run['batch_size'], 0)

            _, sPath = os.path.splitdrive(sOutPath)
            sPath, sFilename = os.path.split(sPath)
            sFilename, sExt = os.path.splitext(sFilename)
            model_name = sPath + '/' + sFilename + str(patchSize[0, 0]) + str(patchSize[0, 1]) + '_best'
            weight_name = model_name + '_weights.h5'
            model_json = model_name + '.json'
            model_all = model_name + '_model.h5'
            json_string = best_model.to_json()
            open(model_json, 'w').write(json_string)
            # wei = best_model.get_weights()
            best_model.save_weights(weight_name)
            best_model.save(model_all)

            result = best_run['result']
            # acc = result.history['acc']
            loss = result.history['loss']
            val_acc = result.history['val_acc']
            val_loss = result.history['val_loss']
            sio.savemat(model_name, {'model_settings': model_json,
                                     'model': model_all,
                                     'weights': weight_name,
                                     'acc': -best_run['loss'],
                                     'loss': loss,
                                     'val_acc': val_acc,
                                     'val_loss': val_loss,
                                     'score_test': score_test,
                                     'acc_test': acc_test,
                                     'prob_test': prob_test})

        elif sParaOptim == 'grid':  # grid search << backward compatibility
            cnnModel.fTrain(X_traind=dData['X_train'],
                            y_traind=dData['y_train'],
                            X_valid=dData['X_valid'],
                            y_valid=dData['y_valid'],
                            X_test=dData['X_test'],
                            y_test=dData['y_test'],
                            sOutPath=sOutPath,
                            patchSize=dData['patchSize'],
                            batchSizes=iBatchSize,
                            learningRates=iLearningRate,
                            iEpochs=iEpochs,
                            dlart_handle=dlart_handle)

        else:  # no optimization or grid search (if batchSize|learningRate are arrays)
            if not usingSegmentationMasks:
                cnnModel.fTrain(X_train=dData['X_train'],
                                y_train=dData['y_train'],
                                X_valid=dData['X_valid'],
                                y_valid=dData['y_valid'],
                                X_test=dData['X_test'],
                                y_test=dData['y_test'],
                                sOutPath=sOutPath,
                                patchSize=dData['patchSize'],
                                batchSizes=iBatchSize,
                                learningRates=iLearningRate,
                                iEpochs=iEpochs,
                                dlart_handle=dlart_handle)
            else:
                cnnModel.fTrain(X_train=dData['X_train'],
                                y_train=dData['y_train'],
                                Y_segMasks_train=dData['Y_segMasks_train'],
                                X_valid=dData['X_valid'],
                                y_valid=dData['y_valid'],
                                Y_segMasks_valid=dData['Y_segMasks_validation'],
                                X_test=dData['X_test'],
                                y_test=dData['y_test'],
                                Y_segMasks_test=dData['Y_segMasks_test'],
                                sOutPath=sOutPath,
                                patchSize=dData['patchSize'],
                                batchSizes=iBatchSize,
                                learningRates=iLearningRate,
                                iEpochs=iEpochs,
                                dlart_handle=dlart_handle)

    elif lTrain == RUN_CNN_PREDICT:  # predicting
        cnnModel.fPredict(dData['X_test'], dData['y_test'], dData['model_name'], sOutPath, dData['patchSize'],
                          iBatchSize[0])

    _, sPath = os.path.splitdrive(sOutPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)

    model_name = sOutPath + os.sep + sFilename
    model_all = model_name + '_model.h5'
    try:
        model = load_model(model_all)
    except:
        try:
            def dice_coef(y_true, y_pred, epsilon=1e-5):
                dice_numerator = 2.0 * K.sum(y_true * y_pred, axis=[1, 2, 3, 4])
                dice_denominator = K.sum(K.square(y_true), axis=[1, 2, 3, 4]) + K.sum(K.square(y_pred),
                                                                                      axis=[1, 2, 3, 4])

                dice_score = dice_numerator / (dice_denominator + epsilon)
                return K.mean(dice_score, axis=0)

            def dice_coef_loss(y_true, y_pred):
                return 1 - dice_coef(y_true, y_pred)

            model = load_model(model_all,
                               custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
        except:
            model = {}
    return model, model_all


# Main Code
if __name__ == "__main__":  # for command line call
    # input parsing
    # ADD new options here!
    parser = argparse.ArgumentParser(description='''CNN artifact detection''',
                                     epilog='''(c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de''')
    parser.add_argument('-i', '--inPath', nargs=1, type=str, help='input path to *.mat of stored patches',
                        default=LEARNING_OUT + os.sep + 'Datatmp/in.mat')
    parser.add_argument('-o', '--outPath', nargs=1, type=str,
                        help='output path to the file used for storage (subfiles _model, _weights, ... are automatically generated)',
                        default=LEARNING_OUT + os.sep + 'Datatmp/out')
    parser.add_argument('-m', '--model', nargs=1, type=str,
                        choices=['motion_head_CNN2D', 'motion_abd_CNN2D', 'motion_all_CNN2D', 'motion_CNN3D',
                                 'motion_MNetArt', 'motion_VNetArt', 'multi_DenseResNet', 'multi_InceptionNet'],
                        help='select CNN model', default='motion_2DCNN_head')
    parser.add_argument('-t', '--train', dest='train', action='store_true',
                        help='if set -> training | if not set -> prediction')
    parser.add_argument('-p', '--paraOptim', dest='paraOptim', type=str, choices=['grid', 'hyperas', 'none'],
                        help='parameter optimization via grid search, hyper optimization or no optimization',
                        default='none')
    parser.add_argument('-b', '--batchSize', nargs='*', dest='batchSize', type=int, help='batchSize', default=64)
    parser.add_argument('-l', '--learningRates', nargs='*', dest='learningRate', type=int, help='learningRate',
                        default=0.0001)
    parser.add_argument('-e', '--epochs', nargs=1, dest='epochs', type=int, help='epochs', default=300)

    args = parser.parse_args()

    if os.path.isfile(args.outPath[0]):
        print('Warning! Output file is already existing and will be overwritten')

    # load input data
    dData = fLoadMat(args.inPath[0])
    # save path for keras model
    if 'outPath' in dData:
        sOutPath = dData['outPath']
    else:
        sOutPath = args.outPath[0]

    fRunCNN(dData, args.model[0], args.train, args.paraOptim, sOutPath, args.batchSize, args.learningRate,
            args.epochs[0])
