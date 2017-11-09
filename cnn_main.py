# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:57:10 2016

@author: Thomas Kuestner
"""


"""Import"""

import sys
import numpy as np                  # for algebraic operations, matrices
import h5py
#import scipy as sp                  # numerical things, optimization, integrals
import scipy.io as sio              # I/O
import os.path                      # operating system
#import theano.tensor as T           # define, optimze, evaluate multidim arrays
#import keras                        # CNN
#import matplotlib.pyplot as plt     # for plotting
import argparse

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim


#from keras.models import Sequential
#from keras.layers.core import Dense, Activation, Flatten#, Layer  Dropout, Flatten
#from keras.layers import containers

#from keras.layers.convolutional import Convolution2D
#from keras.layers.convolutional import MaxPooling2D as pool2
#from keras.layers.convolutional import ZeroPadding2D as zero2d
#from keras.models import model_from_json
#from keras.regularizers import l2#, activity_l2
#from theano import function

#from keras.optimizers import SGD

"""functions"""    
def fLoadData(conten):
    # prepared in matlab
    print 'Loading data'
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
        

# input parsing
parser = argparse.ArgumentParser(description='''CNN feature learning''', epilog='''(c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de''')
parser.add_argument('-i','--inPath', nargs = 1, type = str, help='input path to *.mat of stored patches', default= '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/in.mat')
parser.add_argument('-o','--outPath', nargs = 1, type = str, help='output path to the file used for storage (subfiles _model, _weights, ... are automatically generated)', default= '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/out' )
parser.add_argument('-m','--model', nargs = 1, type = str, choices =['motion_head', 'motion_abd', 'motion_all', 'shim', 'noise'], help='select CNN model', default='motion' )
parser.add_argument('-t','--train', dest='train', action='store_true', help='if set -> training | if not set -> prediction' )
parser.add_argument('-p','--paraOptim', dest='paraOptim', type = str, choices = ['grid','hyperas','none'], help='parameter optimization via grid search, hyper optimization or no optimization', default = 'none')

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


"""CNN Models"""
# dynamic loading of corresponding model
cnnModel = __import__(args.model[0], globals(), locals(), ['createModel', 'fTrain', 'fPredict'], -1) # dynamic module loading with specified functions and with relative implict importing (level=-1) -> only in Python2 

# train (w/ or w/o optimization) and predicting
if args.train: # training
    if args.paraOptim == 'hyperas': # hyperas parameter optimization
        best_run, best_model = optim.minimize(model=cnnModel.fHyperasTrain,
                                              data=fLoadDataForOptim(args.inPath[0]),
                                              algo=tpe.suggest,
                                              max_evals=5,
                                              trials=Trials())
        X_train, y_train, X_test, y_test, patchSize = fLoadDataForOptim(args.inPath[0])
        score_test, acc_test = best_model.evaluate(X_test, y_test)
        prob_test = best_model.predict(X_test, best_run['batch_size'], 0)
        
        _, sPath = os.path.splitdrive(sOutPath)
        sPath,sFilename = os.path.split(sPath)
        sFilename, sExt = os.path.splitext(sFilename)
        model_name = sPath + '/' + sFilename + str(patchSize[0,0]) + str(patchSize[0,1]) +'_best'
        weight_name = model_name + '_weights.h5'
        model_json = model_name + '_json'
        model_all = model_name + '_model.h5'
        json_string = best_model.to_json()
        open(model_json, 'w').write(json_string)
        #wei = best_model.get_weights()
        best_model.save_weights(weight_name)
        #best_model.save(model_all)
        
        result = best_run['result']
        #acc = result.history['acc']
        loss = result.history['loss']
        val_acc = result.history['val_acc']
        val_loss = result.history['val_loss']
        sio.savemat(model_name,{'model_settings':model_json,
                                    'model':model_all,
                                    'weights':weight_name,
                                    'acc':-best_run['loss'],
                                    'loss': loss,
                                    'val_acc':val_acc,
                                    'val_loss':val_loss,
                                    'score_test':score_test,
                                    'acc_test':acc_test,
                                    'prob_test':prob_test})

    elif args.paraOptim == 'grid': # grid search
        #cnnModel.fGridTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'], [64,128], [0.1, 0.01, 0.05, 0.005, 0.001], 300)
        #cnnModel.fGridTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'], [128], [0.1, 0.01, 0.05, 0.005, 0.001], 300)
        #cnnModel.fGridTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'], [64], [0.1, 0.01, 0.005, 0.001, 0.0001], 300)
		cnnModel.fGridTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'], [64], [0.001, 0.0001], 300)


    else: # no optimization
        cnnModel.fTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'], 128, 0.01, 300)
        
else: # predicting
    cnnModel.fPredict(dData['X_test'],dData['y_test'],dData['model_name'], sOutPath, dData['patchSize'], 64)






                         
#}

# LOOK AT KERNEL
#imshow(cnn.layers[0].W.get_value()[3,0,:,:])
