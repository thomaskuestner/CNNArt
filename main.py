"""
+---------------------------------------------------------------+
| Main function/script for calling the CNNs                     |
| - starting point: raw or DICOM data                           |
| - data is patched, augmented, splitted, ...                   |
+---------------------------------------------------------------+
This script performs the calling of the appropriate training/prediction model function
    main.py ==> model.fTrain()/fPredict()
------------------------------------------------------------------
Copyright: 2016-2019 Thomas Kuestner (thomas.kuestner@med.uni-tuebingen.de) under Apache2 license
@author: Thomas Kuestner
"""

# imports
import sys
import numpy as np                  # for algebraic operations, matrices
import h5py
import scipy.io as sio              # I/O
import os.path                      # operating system
import argparse

# utils
from DatabaseInfo import DatabaseInfo
import utils.DataPreprocessing as datapre
import utils.Training_Test_Split as ttsplit
import utils.scaling as scaling

# networks
from networks.motion.CNN2D import *
from networks.motion.CNN3D import *
from networks.motion.MNetArt import *
from networks.motion.VNetArt import *
from networks.multiclass.CNN2D.DenseResNet import *
from networks.multiclass.CNN2D.InceptionNet import *
from correction.networks.motion import *
from networks.FullyConvolutionalNetworks.motion import *

# VAE correction network
import correction.main_correction as correction

# multi-scale
from utils.calculateInputOfPath2 import fcalculateInputOfPath2
from networks.multiscale.runMS import frunCNN_MS

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
    elif 'FCN' in sModelIn:
        sModel = 'networks.FullyConvolutionalNetworks.motion.' + sModelIn # TODO: may require to adapt patching and data augmentation from GUI/PyQt/DLart/dlart.py
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
    # == DEPRECATED == #

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
    # == DEPRECATED == #

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


def fParseConfig(sFile):
    # get config file
    with open(sFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg


def fDeprecatedMain(args):
    # == DEPRECATED == #

    if os.path.isfile(args.outPath[0]):
        print('Warning! Output file is already existing and will be overwritten')

    # replace by input arguments -> deprecated call -> backward compatibility
    # load input data
    dData = fLoadMat(args.inPath[0])
    # save path for keras model
    if 'outPath' in dData:
        sOutPath = dData['outPath']
    else:
        sOutPath = args.outPath[0]

    fRunCNN(dData, args.model[0], args.train, args.paraOptim, sOutPath, args.batchSize, args.learningRate, args.epochs[0])


def fTrainArtDetection():
    # training for artifact detection

    # check if file is already existing -> skip patching
    if glob.glob(sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + '*_input.mat'):  # deprecated
        sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + '_input.mat'
        try:
            conten = sio.loadmat(sDatafile)
        except:
            f = h5py.File(sDatafile, 'r')
            conten = {}
            conten['X_train'] = np.transpose(np.array(f['X_train']), (3, 2, 0, 1))
            conten['X_test'] = np.transpose(np.array(f['X_test']), (3, 2, 0, 1))
            conten['y_train'] = np.transpose(np.array(f['y_train']))
            conten['y_test'] = np.transpose(np.array(f['y_test']))
            conten['patchSize'] = np.transpose(np.array(f['patchSize']))

        X_train = conten['X_train']
        X_test = conten['X_test']
        y_train = conten['y_train']
        y_test = conten['y_test']

    elif glob.glob(sDatafile):
        with h5py.File(sDatafile, 'r') as hf:
            X_train = hf['X_train'][:]
            X_test = hf['X_test'][:]
            y_train = hf['y_train'][:]
            y_test = hf['y_test'][:]
            patchSize = hf['patchSize'][:]
            if sTrainingMethod == "MultiScaleSeparated":
                X_train_p2 = hf['X_train_p2'][:]
                X_test_p2 = hf['X_test_p2'][:]
                y_train_p2 = hf['y_train_p2'][:]
                y_test_p2 = hf['y_test_p2'][:]
                patchSize_down = hf['patchSize_down'][:]

    else:  # perform patching
        X_train = []
        scpatchSize = [0 for i in range(len(patchSize))]

        if sTrainingMethod == "None" or sTrainingMethod == "ScaleJittering":
            lScaleFactor = [1]
        if sTrainingMethod == "MultiScaleSeparated":
            lScaleFactor = lScaleFactor[:-1]

        #   images will be split into pathces with size scpatchSize and then scaled to patchSize
        for iscalefactor in lScaleFactor:
            # Calculate the patchsize according to scale factor and training method
            scpatchSize = patchSize
            if iscalefactor != 1:
                if sTrainingMethod == "MultiScaleSeparated":
                    scpatchSize = fcalculateInputOfPath2(patchSize, iscalefactor, cfg['network'])
                elif sTrainingMethod == "MultiScaleTogether":
                    scpatchSize = [int(psi / iscalefactor) for psi in patchSize]

            if len(scpatchSize) == 3:
                dAllPatches = np.zeros((0, scpatchSize[0], scpatchSize[1], scpatchSize[2]))
            else:
                dAllPatches = np.zeros((0, scpatchSize[0], scpatchSize[1]))
            dAllLabels = np.zeros(0)
            dAllPats = np.zeros((0, 1))
            lDatasets = cfg['selectedDatabase']['dataref'] + cfg['selectedDatabase']['dataart']
            iLabels = cfg['selectedDatabase']['labelref'] + cfg['selectedDatabase']['labelart']
            for ipat, pat in enumerate(dbinfo.lPats):
                if os.path.exists(dbinfo.sPathIn + os.sep + pat + os.sep + dbinfo.sSubDirs[1]):
                    for iseq, seq in enumerate(lDatasets):
                        # patches and labels of reference/artifact
                        tmpPatches, tmpLabels = datapre.fPreprocessData(
                            os.path.join(dbinfo.sPathIn, pat, dbinfo.sSubDirs[1], seq), scpatchSize,
                            cfg['patchOverlap'], 1, cfg['sLabeling'], sTrainingMethod=sTrainingMethod,
                            range_norm=cfg['range'])
                        dAllPatches = np.concatenate((dAllPatches, tmpPatches), axis=0)
                        dAllLabels = np.concatenate((dAllLabels, iLabels[iseq] * tmpLabels), axis=0)
                        dAllPats = np.concatenate((dAllPats, ipat * np.ones((tmpLabels.shape[0], 1), dtype=np.int)),
                                                  axis=0)
                else:
                    pass
            print('Start splitting')
            # perform splitting: sp for split
            if cfg['sSplitting'] == 'crossvalidation_data':
                spX_train, spy_train, spX_test, spy_test = ttsplit.fSplitDataset(dAllPatches, dAllLabels, dAllPats,
                                                                                 cfg['sSplitting'], scpatchSize,
                                                                                 cfg['patchOverlap'], cfg['dSplitval'],
                                                                                 '', nfolds=nFolds)
            else:
                spX_train, spy_train, spX_test, spy_test = ttsplit.fSplitDataset(dAllPatches, dAllLabels, dAllPats,
                                                                                 cfg['sSplitting'], scpatchSize,
                                                                                 cfg['patchOverlap'], cfg['dSplitval'],
                                                                                 '')
            print('Start scaling')
            # perform scaling: sc for scale
            scX_train, scX_test, scedpatchSize = scaling.fscaling(spX_train, spX_test, scpatchSize, iscalefactor)
            if sTrainingMethod == "MultiScaleSeparated":
                X_train_p2 = scX_train
                X_test_p2 = scX_test
                y_train_p2 = spy_train
                y_test_p2 = spy_test
                patchSize_down = scedpatchSize
                X_train_cut, X_test_cut = scaling.fcutMiddelPartOfPatch(spX_train, spX_test, scpatchSize, patchSize)
                X_train = X_train_cut
                X_test = X_test_cut
                y_train = spy_train
                y_test = spy_test
            else:
                if len(X_train) == 0:
                    X_train = scX_train
                    X_test = scX_test
                    y_train = spy_train
                    y_test = spy_test
                else:
                    X_train = np.concatenate((X_train, scX_train), axis=1)
                    X_test = np.concatenate((X_test, scX_test), axis=1)
                    y_train = np.concatenate((y_train, spy_train), axis=1)
                    y_test = np.concatenate((y_test, spy_test), axis=1)

        print('Start saving')
        # save to file (deprecated)
        if lSave:
            # sio.savemat(sOutPath + os.sep + sFSname + str(patchSize[0]) + str(patchSize[1]) + '_input.mat', {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'patchSize': cfg['patchSize']})
            with h5py.File(sDatafile, 'w') as hf:
                hf.create_dataset('X_train', data=X_train)
                hf.create_dataset('X_test', data=X_test)
                hf.create_dataset('y_train', data=y_train)
                hf.create_dataset('y_test', data=y_test)
                hf.create_dataset('patchSize', data=patchSize)
                hf.create_dataset('patchOverlap', data=cfg['patchOverlap'])
                if sTrainingMethod == "MultiScaleSeparated":
                    hf.create_dataset('X_train_p2', data=X_train_p2)
                    hf.create_dataset('X_test_p2', data=X_test_p2)
                    hf.create_dataset('y_train_p2', data=y_train_p2)
                    hf.create_dataset('y_test_p2', data=y_test_p2)
                    hf.create_dataset('patchSize_down', data=patchSize_down)

    # perform training
    for iFold in range(0, len(X_train)):
        if len(X_train) != 1:
            CV_Patient = iFold + 1
        else:
            CV_Patient = 0
        if 'MultiPath' in cfg['network']:
            frunCNN_MS(
                {'X_train': X_train[iFold], 'y_train': y_train[iFold], 'X_test': X_test[iFold], 'y_test': y_test[iFold],
                 'patchSize': patchSize
                    , 'X_train_p2': X_train_p2[iFold], 'y_train_p2': y_train_p2[iFold], 'X_test_p2': X_test_p2[iFold],
                 'y_test_p2': y_test_p2[iFold], 'patchSize_down': patchSize_down, 'ScaleFactor': lScaleFactor[0]}
                , cfg['network'], lTrain, sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'], CV_Patient)
        elif 'MS' in cfg['network']:
            frunCNN_MS(
                {'X_train': X_train[iFold], 'y_train': y_train[iFold], 'X_test': X_test[iFold], 'y_test': y_test[iFold],
                 'patchSize': patchSize}
                , cfg['network'], lTrain, sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'], CV_Patient)
        else:
            fRunCNN(
                {'X_train': X_train[iFold], 'y_train': y_train[iFold], 'X_test': X_test[iFold], 'y_test': y_test[iFold],
                 'patchSize': patchSize}, cfg['network'], lTrain, cfg['sOpti'], sOutPath, cfg['batchSize'], cfg['lr'],
                cfg['epochs'], CV_Patient)


def fPredictArtDetection():
    # prediction
    sNetworktype = cfg['network'].split("_")
    if len(sPredictModel) == 0:
        sPredictModel = cfg['selectedDatabase']['bestmodel'][sNetworktype[2]]

    if sTrainingMethod == "MultiScaleSeparated":
        patchSize = fcalculateInputOfPath2(cfg['patchSize'], cfg['lScaleFactor'][0], cfg['network'])

    if len(patchSize) == 3:
        X_test = np.zeros((0, patchSize[0], patchSize[1], patchSize[2]))
        y_test = np.zeros((0))
        allImg = np.zeros((len(cfg['lPredictImg']), cfg['correction']['actualSize'][0],
                           cfg['correction']['actualSize'][1], cfg['correction']['actualSize'][2]))
    else:
        X_test = np.zeros((0, patchSize[0], patchSize[1]))
        y_test = np.zeros(0)

    for iImg in range(0, len(cfg['lPredictImg'])):
        # patches and labels of reference/artifact
        tmpPatches, tmpLabels = datapre.fPreprocessData(cfg['lPredictImg'][iImg], patchSize, cfg['patchOverlap'], 1,
                                                        cfg['sLabeling'], sTrainingMethod=sTrainingMethod)
        X_test = np.concatenate((X_test, tmpPatches), axis=0)
        y_test = np.concatenate((y_test, cfg['lLabelPredictImg'][iImg] * tmpLabels), axis=0)
        allImg[iImg] = datapre.fReadData(cfg['lPredictImg'][iImg])

    if sTrainingMethod == "MultiScaleSeparated":
        X_test_p1 = scaling.fcutMiddelPartOfPatch(X_test, X_test, patchSize, cfg['patchSize'])
        X_train_p2, X_test_p2, scedpatchSize = scaling.fscaling([X_test], [X_test], patchSize, cfg['lScaleFactor'][0])
        frunCNN_MS({'X_test': X_test_p1, 'y_test': y_test, 'patchSize': patchSize, 'X_test_p2': X_test_p2[0],
                    'model_name': sPredictModel, 'patchOverlap': cfg['patchOverlap'],
                    'actualSize': cfg['correction']['actualSize']}, cfg['network'], lTrain, sOutPath, cfg['batchSize'],
                   cfg['lr'], cfg['epochs'], predictImg=allImg)
    elif 'MS' in cfg['network']:
        frunCNN_MS({'X_test': X_test, 'y_test': y_test, 'patchSize': cfg['patchSize'], 'model_name': sPredictModel,
                    'patchOverlap': cfg['patchOverlap'], 'actualSize': cfg['correction']['actualSize']}, cfg['network'],
                   lTrain, sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'], predictImg=allImg)
    else:
        fRunCNN({'X_train': [], 'y_train': [], 'X_test': X_test, 'y_test': y_test, 'patchSize': patchSize,
                             'model_name': sPredictModel, 'patchOverlap': cfg['patchOverlap'],
                             'actualSize': cfg['correction']['actualSize']}, cfg['network'], lTrain, cfg['sOpti'],
                            sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'])


# Main Code
# calling it as a script -> requires a mat/h5 file with the stored data
if __name__ == "__main__": # for command line call
    # input parsing
    parser = argparse.ArgumentParser(description='''CNN artifact detection''', epilog='''(c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de''')
    parser.add_argument('-c', '--config', nargs = 1, type = str, help='path to config file', default= 'config/param.yml')
    parser.add_argument('-i','--inPath', nargs = 1, type = str, help='input path to *.mat of stored patches', default= '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/in.mat')
    parser.add_argument('-o','--outPath', nargs = 1, type = str, help='output path to the file used for storage (subfiles _model, _weights, ... are automatically generated)', default= '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/out' )
    parser.add_argument('-m','--model', nargs = 1, type = str, choices =['motion_head_CNN2D', 'motion_abd_CNN2D', 'motion_all_CNN2D', 'motion_CNN3D', 'motion_MNetArt', 'motion_VNetArt', 'multi_DenseResNet', 'multi_InceptionNet'], help='select CNN model', default='motion_2DCNN_head' )
    parser.add_argument('-t','--train', dest='train', action='store_true', help='if set -> training | if not set -> prediction' )
    parser.add_argument('-p','--paraOptim', dest='paraOptim', type = str, choices = ['grid','hyperas','none'], help='parameter optimization via grid search, hyper optimization or no optimization', default = 'none')
    parser.add_argument('-b', '--batchSize', nargs='*', dest='batchSize', type=int, help='batchSize', default=64)
    parser.add_argument('-l', '--learningRates', nargs='*', dest='learningRate', type=int, help='learningRate', default=0.0001)
    parser.add_argument('-e', '--epochs', nargs=1, dest='epochs', type=int, help='epochs', default=300)

    args = parser.parse_args()
    # TODO: add a check for old version
    # fDeprecatedMain(args)

    # parse input
    cfg = fParseConfig(args.config[0])

    lTrain = cfg['lTrain']  # training or prediction
    lSave = cfg['lSave']  # save intermediate test, training sets
    lCorrection = cfg['lCorrection']  # artifact correction or classification
    sPredictModel = cfg['sPredictModel']  # choose trained model used in prediction
    # initiate info objects
    # default database: MRPhysics with ['newProtocol','dicom_sorted']
    dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])
    sTrainingMethod = cfg['sTrainingMethod']  # options of multiscale
    lScaleFactor = cfg['lScaleFactor']

    # load/create input data
    patchSize = cfg['patchSize']
    if cfg['sSplitting'] == 'normal':
        sFSname = 'normal'
    elif cfg['sSplitting'] == 'crossvalidation_data':
        sFSname = 'crossVal_data'
        nFolds = cfg['nFolds']
    elif cfg['sSplitting'] == 'crossvalidation_patient':
        sFSname = 'crossVal'

    # set ouput path
    sOutsubdir = cfg['subdirs'][2]
    sOutPath = cfg['selectedDatabase']['pathout'] + os.sep + ''.join(map(str, patchSize)).replace(" ",
                                                                                                  "") + os.sep + sOutsubdir + str(
        patchSize[0]) + str(patchSize[1])  # + str(ind_split) + '_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
    if len(patchSize) == 3:
        sOutPath = sOutPath + str(patchSize[2])
    if sTrainingMethod != "None":
        if sTrainingMethod != "ScaleJittering":
            sOutPath = sOutPath + '_sf' + ''.join(map(str, lScaleFactor)).replace(" ", "").replace(".", "")
            sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + 'sf' + ''.join(
                map(str, lScaleFactor)).replace(" ", "").replace(".", "") + '.h5'
        else:
            sOutPath = sOutPath + '_sj'
            sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + 'sj' + '.h5'
    else:
        sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + '.h5'

    if lCorrection:
        #########################
        ## Artifact Correction ##
        #########################
        correction.run(cfg, dbinfo)

    elif lTrain:
        ########################
        ## artifact detection ##
        ## ---- training ---- ##
        ########################
        fTrainArtDetection()


    else:
        ########################
        ## artifact detection ##
        ## --- prediction --- ##
        ########################
        fPredictArtDetection()
