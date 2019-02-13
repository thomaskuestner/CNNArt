"""
+-----------------------------------------------------------------------+
| Main function for calling the CNNs                                    |
| - starting point: data is alrady patched, augmented, splitted, ...    |
+-----------------------------------------------------------------------+
This script performs the calling of the appropriate training/prediction model function
    main.py ==> mainPatches.py ==> model.fTrain()/fPredict()
------------------------------------------------------------------
Copyright: 2016-2018 Thomas Kuestner (thomas.kuestner@med.uni-tuebingen.de) under Apache2 license
@author: Thomas Kuestner
"""

# imports
import sys
import numpy as np                  # for algebraic operations, matrices
import h5py
import scipy.io as sio              # I/O
import os.path                      # operating system
import argparse

# networks
from networks.motion.CNN2D import *
from networks.motion.CNN3D import *
from networks.motion.MNetArt import *
from networks.motion.VNetArt import *
from networks.multiclass.CNN2D.DenseResNet import *
from networks.multiclass.CNN2D.InceptionNet import *
from correction.networks.motion import *

#from hyperopt import Trials, STATUS_OK, tpe
#from hyperas import optim



"""functions"""
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
    y_train= np.asarray([y_train[:,0], np.abs(np.asarray(y_train[:,0],dtype=np.float32)-1)]).T
    y_test= np.asarray([y_test[:,0], np.abs(np.asarray(y_test[:,0],dtype=np.float32)-1)]).T
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
            f = h5py.File(sInPath,'r')
            conten = {}
            conten['X_train'] = np.transpose(np.array(f['X_train']), (3,2,0,1))
            conten['X_test'] = np.transpose(np.array(f['X_test']), (3,2,0,1))
            conten['y_train'] = np.transpose(np.array(f['y_train']))
            conten['y_test'] = np.transpose(np.array(f['y_test']))
            conten['patchSize'] = np.transpose(np.array(f['patchSize']))
    else:
        sys.exit('Input file is not existing')
    X_train, y_train, X_test, y_test = fLoadData(conten) # output order needed for hyperas

    fRemove_entries(('X_train', 'X_test', 'y_train', 'y_test'), conten )
    dData = {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}
    dOut = dData.copy()
    dOut.update(conten)
    return dOut # output dictionary (similar to conten, but with reshaped X_train, ...)

def fLoadDataForOptim(sInPath):
     if os.path.isfile(sInPath):
        conten = sio.loadmat(sInPath)
     X_train, y_train, X_test, y_test = fLoadData(conten) # output order needed for hyperas
     return X_train, y_train, X_test, y_test, conten["patchSize"]

#def fLoadAddData(sInPath): # deprecated
#    if os.path.isfile(sInPath):
#        conten = sio.loadmat(sInPath)
#    else:
#        sys.exit('Input file is not existing')
#    for sVarname in conten:
#        if not any(x in sVarname for x in ['X_train', 'X_test', 'y_train', 'y_test'] ):
#            conten[sVarname]

def fRunCNN(dData, sModelIn, lTrain, sParaOptim, sOutPath, iBatchSize, iLearningRate, iEpochs, CV_Patient=0):
    """CNN Models"""
    # check model
    if 'motion' in sModelIn:
        if 'CNN2D' in sModelIn:
            sModel = 'networks.motion.CNN2D.' + sModelIn
        elif 'motion_CNN3D' in sModelIn:
            sModel = 'networks.motion.CNN3D.' + sModelIn
        elif 'motion_MNetArt' in sModelIn:
            sModel = 'networks.motion.MNetArt.' + sModelIn
        elif 'motion_VNetArt' in sModelIn:
            sModel = 'networks.motion.VNetArt.' + sModelIn
    elif 'multi' in sModelIn:
        if 'multi_DenseResNet' in sModelIn:
            sModel = 'networks.multiclass.DenseResNet.' + sModelIn
        elif 'multi_InceptionNet' in sModelIn:
            sModel = 'networks.multiclass.InceptionNet.' + sModelIn
    else:
        sys.exit("Model is not supported")

    # dynamic loading of corresponding model
    cnnModel = __import__(sModel, globals(), locals(), ['createModel', 'fTrain', 'fPredict'], 0)  # dynamic module loading with specified functions and with absolute importing (level=0) -> work in both Python2 and Python3

    # train (w/ or w/o optimization) and predicting
    if lTrain:  # training
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
            model_json = model_name + '_json'
            model_all = model_name + '_model.h5'
            json_string = best_model.to_json()
            open(model_json, 'w').write(json_string)
            # wei = best_model.get_weights()
            best_model.save_weights(weight_name)
            # best_model.save(model_all)

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
            cnnModel.fTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'], iBatchSize, iLearningRate, iEpochs, CV_Patient=CV_Patient)

        else:# no optimization or grid search (if batchSize|learningRate are arrays)
            cnnModel.fTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'], iBatchSize, iLearningRate, iEpochs, CV_Patient=CV_Patient)

    else:  # predicting
        cnnModel.fPredict(dData['X_test'], dData['y_test'], dData['model_name'], sOutPath, patchSize=dData['patchSize'], batchSize=iBatchSize[0], patchOverlap=dData['patchOverlap'], actualSize=dData['actualSize'], iClass=dData['iClass'])

def fRunCNNCorrection(dData, dHyper, dParam):
    sModelIn = dHyper['sCorrection']
    if 'motion' in sModelIn:
        if '2D' in sModelIn:
            sModel = 'correction.networks.motion.VAE2D.' + sModelIn
        elif '3D' in sModelIn:
            sModel = 'correction.networks.motion.VAE3D.' + sModelIn
    else:
        sys.exit("Model is not supported")

    # dynamic loading of corresponding model
    # dynamic module loading with specified functions and with absolute importing (level=0) ->
    # work in both Python2 and Python3
    model = __import__(sModel, globals(), locals(), ['createModel', 'fTrain', 'fPredict'], 0)

    if dParam['lTrain']:
        # perform training
        model.fTrain(dData, dParam, dHyper)
    else:
        # perform prediction
        model.fPredict(dData['test_ref'], dData['test_art'], dParam, dHyper)

def fMainCNN(X_train, X_val, X_test, label_train, label_val, label_test, lTrain=True, param=None):
    # X_train, X_val, X_test: numpy arrays with training, validation, test data
    # label_train, label_val, label_test: corresponding labels
    # lTrain: perform training (true) or prediction with trained architecture (false)
    # param: dictionary containing all set UI parameters and further requested configs

    # link input data
    dData['X_train'] = X_train
    dData['X_test'] = X_test
    dData['y_train'] = label_train
    dData['y_test'] = label_test

    # set output
    sOutPath = param['sOutPath']

    fRunCNN(dData,param['sModel'], lTrain, param['paraOptim'], sOutPath, param['batchSize'], param['learningRate'], param['epochs'])

# Main Code
# calling it as a script -> requires a mat/h5 file with the stored data
if __name__ == "__main__": # for command line call
    # input parsing
    parser = argparse.ArgumentParser(description='''CNN artifact detection''', epilog='''(c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de''')
    parser.add_argument('-i','--inPath', nargs = 1, type = str, help='input path to *.mat of stored patches', default= '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/in.mat')
    parser.add_argument('-o','--outPath', nargs = 1, type = str, help='output path to the file used for storage (subfiles _model, _weights, ... are automatically generated)', default= '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/out' )
    parser.add_argument('-m','--model', nargs = 1, type = str, choices =['motion_head_CNN2D', 'motion_abd_CNN2D', 'motion_all_CNN2D', 'motion_CNN3D', 'motion_MNetArt', 'motion_VNetArt', 'multi_DenseResNet', 'multi_InceptionNet'], help='select CNN model', default='motion_2DCNN_head' )
    parser.add_argument('-t','--train', dest='train', action='store_true', help='if set -> training | if not set -> prediction' )
    parser.add_argument('-p','--paraOptim', dest='paraOptim', type = str, choices = ['grid','hyperas','none'], help='parameter optimization via grid search, hyper optimization or no optimization', default = 'none')
    parser.add_argument('-b', '--batchSize', nargs='*', dest='batchSize', type=int, help='batchSize', default=64)
    parser.add_argument('-l', '--learningRates', nargs='*', dest='learningRate', type=int, help='learningRate', default=0.0001)
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

    fRunCNN(dData,args.model[0], args.train, args.paraOptim, sOutPath, args.batchSize, args.learningRate, args.epochs[0])
