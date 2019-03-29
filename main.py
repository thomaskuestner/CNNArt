"""
+---------------------------------------------------------------+
| Main function/script for calling the CNNs                     |
| - starting point: raw or DICOM data                           |
+---------------------------------------------------------------+
This script performs the loading of the data, patching, augmentation and sSplitting
    main.py ==> mainPatches.py ==> model.fTrain()/fPredict()
------------------------------------------------------------------
Copyright: 2016-2018 Thomas Kuestner (thomas.kuestner@med.uni-tuebingen.de) under Apache2 license
@author: Thomas Kuestner
"""

# imports
import os
import glob
import yaml
import numpy as np
import scipy.io as sio
import h5py
from DatabaseInfo import DatabaseInfo
import utils.DataPreprocessing as datapre
import utils.Training_Test_Split as ttsplit
import mainPatches
import utils.scaling as scaling
import correction.main_correction as correction
from utils.calculateInputOfPath2 import fcalculateInputOfPath2
from networks.multiscale.runMS import frunCNN_MS

# get config file
with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

lTrain = cfg['lTrain'] # training or prediction
lSave = cfg['lSave'] # save intermediate test, training sets
lCorrection = cfg['lCorrection'] # artifact correction or classification
sPredictModel = cfg['sPredictModel'] # choose trained model used in prediction
# initiate info objects
# default database: MRPhysics with ['newProtocol','dicom_sorted']
dbinfo = DatabaseInfo(cfg['MRdatabase'],cfg['subdirs'])
sTrainingMethod = cfg['sTrainingMethod'] # options of multiscale
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

sOutsubdir = cfg['subdirs'][2]
sOutPath = cfg['selectedDatabase']['pathout'] + os.sep + ''.join(map(str,patchSize)).replace(" ", "") + os.sep + sOutsubdir + str(patchSize[0]) + str(patchSize[1]) # + str(ind_split) + '_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
if len(patchSize) == 3:
    sOutPath = sOutPath + str(patchSize[2])
if sTrainingMethod != "None":
    if sTrainingMethod != "ScaleJittering":
        sOutPath = sOutPath+ '_sf' + ''.join(map(str, lScaleFactor)).replace(" ", "").replace(".", "")
        sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + 'sf' + ''.join(map(str, lScaleFactor)).replace(" ", "").replace(".", "") + '.h5'
    else:
        sOutPath = sOutPath + '_sj'
        sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + 'sj' + '.h5'
else:
    sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str,patchSize)).replace(" ", "") + '.h5'

if lCorrection:
    #########################
    ## Artifact Correction ##
    #########################
    correction.run(cfg, dbinfo)

elif lTrain:
    ##############
    ## training ##
    ##############
    # check if file is already existing -> skip patching
    if glob.glob(sOutPath + os.sep + sFSname + ''.join(map(str,patchSize)).replace(" ", "") + '*_input.mat'): # deprecated
        sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str,patchSize)).replace(" ", "") + '_input.mat'
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

    else: # perform patching
        X_train = []
        scpatchSize = [0 for i in range(len(patchSize))]

        if sTrainingMethod == "None" or sTrainingMethod == "ScaleJittering":
            lScaleFactor = [1]
        if sTrainingMethod == "MultiScaleSeparated" :
            lScaleFactor = lScaleFactor[:-1]

        #   images will be split into pathces with size scpatchSize and then scaled to patchSize
        for iscalefactor in lScaleFactor:
            # Calculate the patchsize according to scale factor and training method
            scpatchSize = patchSize
            if iscalefactor != 1:
                if sTrainingMethod == "MultiScaleSeparated":
                    scpatchSize = fcalculateInputOfPath2(patchSize, iscalefactor, cfg['network'])
                elif sTrainingMethod == "MultiScaleTogether":
                    scpatchSize = [int(psi/iscalefactor) for psi in patchSize]

            if len(scpatchSize)==3:
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
                        tmpPatches, tmpLabels  = datapre.fPreprocessData(os.path.join(dbinfo.sPathIn, pat, dbinfo.sSubDirs[1], seq), scpatchSize, cfg['patchOverlap'], 1, cfg['sLabeling'], sTrainingMethod=sTrainingMethod, range_norm=cfg['range'])
                        dAllPatches = np.concatenate((dAllPatches, tmpPatches), axis=0)
                        dAllLabels = np.concatenate((dAllLabels, iLabels[iseq]*tmpLabels), axis=0)
                        dAllPats = np.concatenate((dAllPats, ipat*np.ones((tmpLabels.shape[0],1), dtype=np.int)), axis=0)
                else:
                    pass
            print('Start splitting')
            # perform splitting: sp for split
            if cfg['sSplitting'] == 'crossvalidation_data':
                spX_train, spy_train, spX_test, spy_test = ttsplit.fSplitDataset(dAllPatches, dAllLabels, dAllPats, cfg['sSplitting'], scpatchSize, cfg['patchOverlap'], cfg['dSplitval'], '', nfolds = nFolds)
            else:
                spX_train, spy_train, spX_test, spy_test = ttsplit.fSplitDataset(dAllPatches, dAllLabels, dAllPats, cfg['sSplitting'], scpatchSize, cfg['patchOverlap'], cfg['dSplitval'], '')
            print('Start scaling')
            # perform scaling: sc for scale
            scX_train, scX_test, scedpatchSize= scaling.fscaling(spX_train, spX_test, scpatchSize, iscalefactor)
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
    for iFold in range(0,len(X_train)):
        if len(X_train) != 1:
            CV_Patient = iFold + 1
        else:
            CV_Patient = 0
        if 'MultiPath' in cfg['network']:
            frunCNN_MS({'X_train': X_train[iFold], 'y_train': y_train[iFold], 'X_test': X_test[iFold], 'y_test': y_test[iFold], 'patchSize': patchSize
                        , 'X_train_p2': X_train_p2[iFold], 'y_train_p2': y_train_p2[iFold], 'X_test_p2': X_test_p2[iFold],'y_test_p2': y_test_p2[iFold], 'patchSize_down': patchSize_down, 'ScaleFactor': lScaleFactor[0]}
                        ,  cfg['network'], lTrain, sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'], CV_Patient)
        elif 'MS' in cfg['network']:
            frunCNN_MS({'X_train': X_train[iFold], 'y_train': y_train[iFold], 'X_test': X_test[iFold], 'y_test': y_test[iFold], 'patchSize': patchSize}
                        ,  cfg['network'], lTrain, sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'], CV_Patient)
        else:
            mainPatches.fRunCNN({'X_train': X_train[iFold], 'y_train': y_train[iFold], 'X_test': X_test[iFold], 'y_test': y_test[iFold], 'patchSize': patchSize}, cfg['network'], lTrain, cfg['sOpti'], sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'], CV_Patient)

else:
    ################
    ## prediction ##
    ################
    sNetworktype = cfg['network'].split("_")
    if len(sPredictModel) == 0:
        sPredictModel = cfg['selectedDatabase']['bestmodel'][sNetworktype[2]]

    if sTrainingMethod == "MultiScaleSeparated":
        patchSize = fcalculateInputOfPath2(cfg['patchSize'], cfg['lScaleFactor'][0], cfg['network'])

    if len(patchSize) == 3:
        X_test = np.zeros((0, patchSize[0], patchSize[1], patchSize[2]))
        y_test = np.zeros((0))
        allImg = np.zeros((len(cfg['lPredictImg']), cfg['correction']['actualSize'][0], cfg['correction']['actualSize'][1], cfg['correction']['actualSize'][2]))
    else:
        X_test = np.zeros((0, patchSize[0], patchSize[1]))
        y_test = np.zeros(0)

    for iImg in range(0, len(cfg['lPredictImg'])):
        # patches and labels of reference/artifact
        tmpPatches, tmpLabels  = datapre.fPreprocessData(cfg['lPredictImg'][iImg], patchSize, cfg['patchOverlap'], 1, cfg['sLabeling'], sTrainingMethod=sTrainingMethod)
        X_test = np.concatenate((X_test, tmpPatches), axis=0)
        y_test = np.concatenate((y_test, cfg['lLabelPredictImg'][iImg]*tmpLabels), axis=0)
        allImg[iImg] = datapre.fReadData(cfg['lPredictImg'][iImg])

    if sTrainingMethod == "MultiScaleSeparated":
        X_test_p1 = scaling.fcutMiddelPartOfPatch(X_test, X_test, patchSize, cfg['patchSize'])
        X_train_p2, X_test_p2, scedpatchSize = scaling.fscaling([X_test], [X_test], patchSize, cfg['lScaleFactor'][0])
        frunCNN_MS({'X_test': X_test_p1, 'y_test': y_test, 'patchSize': patchSize, 'X_test_p2': X_test_p2[0], 'model_name': sPredictModel, 'patchOverlap': cfg['patchOverlap'],'actualSize': cfg['correction']['actualSize']}, cfg['network'], lTrain, sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'], predictImg=allImg)
    elif 'MS' in cfg['network']:
        frunCNN_MS({'X_test': X_test, 'y_test': y_test, 'patchSize': cfg['patchSize'], 'model_name': sPredictModel, 'patchOverlap': cfg['patchOverlap'], 'actualSize': cfg['correction']['actualSize']},  cfg['network'], lTrain, sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'], predictImg=allImg)
    else:
        mainPatches.fRunCNN({'X_train': [], 'y_train': [], 'X_test': X_test, 'y_test': y_test, 'patchSize': patchSize, 'model_name': sPredictModel, 'patchOverlap': cfg['patchOverlap'], 'actualSize': cfg['correction']['actualSize']}, cfg['network'], lTrain, cfg['sOpti'], sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'])
