# demo file
import os
import glob
import yaml
import numpy as np
import scipy.io as sio
import h5py
from DatabaseInfo import DatabaseInfo
import utils.DataPreprocessing as datapre
import utils.Training_Test_Split as ttsplit
import cnn_main
import utils.scaling as scaling
import correction.main_correction as correction

with open('config' + os.sep + 'paramnormal.yml', 'r') as ymlfile:
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
elif cfg['sSplitting'] == 'crossvalidation_patient':
    sFSname = 'crossVal'

sOutsubdir = cfg['subdirs'][2]
sOutPath = cfg['selectedDatabase']['pathout'] + os.sep + ''.join(map(str,patchSize)).replace(" ", "") + os.sep + sOutsubdir + str(patchSize[0]) + str(patchSize[1]) # + str(ind_split) + '_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
if len(patchSize) == 3:
    sOutPath = sOutPath + str(patchSize[2])
if sTrainingMethod == "scalingPrior":
    sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + 'sf' + ''.join(map(str, lScaleFactor)).replace(" ", "").replace(".", "") + '.h5'
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

    else: # perform patching
        X_train = []
        scpatchSize = patchSize
        if sTrainingMethod != "scalingPrior":
            lScaleFactor = [1]
        # Else perform scaling:
        #   images will be split into pathces with size scpatchSize and then scaled to patchSize
        for iscalefactor in lScaleFactor:
            scpatchSize = [int(psi/iscalefactor) for psi in patchSize]
            if len(patchSize) == 3:
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
                        tmpPatches, tmpLabels  = datapre.fPreprocessData(os.path.join(dbinfo.sPathIn, pat, dbinfo.sSubDirs[1], seq), scpatchSize, cfg['patchOverlap'], 1, cfg['sLabeling'])
                        dAllPatches = np.concatenate((dAllPatches, tmpPatches), axis=0)
                        dAllLabels = np.concatenate((dAllLabels, iLabels[iseq]*tmpLabels), axis=0)
                        dAllPats = np.concatenate((dAllPats, ipat*np.ones((tmpLabels.shape[0],1), dtype=np.int)), axis=0)
                else:
                    pass
            print('Start splitting')
            # perform splitting: sp for split
            spX_train, spy_train, spX_test, spy_test = ttsplit.fSplitDataset(dAllPatches, dAllLabels, dAllPats, cfg['sSplitting'], scpatchSize, cfg['patchOverlap'], cfg['dSplitval'], '')
            print('Start scaling')
            # perform scaling: sc for scale
            scX_train, scX_test= scaling.fscaling(spX_train, spX_test, scpatchSize, iscalefactor)
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

    # perform training
    for iFold in range(0,len(X_train)):
        if len(X_train) != 1:
            CV_Patient = iFold + 1
        else:
            CV_Patient = 0
        cnn_main.fRunCNN({'X_train': X_train[iFold], 'y_train': y_train[iFold], 'X_test': X_test[iFold], 'y_test': y_test[iFold], 'patchSize': patchSize}, cfg['network'], lTrain, cfg['sOpti'], sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'], CV_Patient)

else: 
    ################
    ## prediction ##
    ################
    if glob.glob(sDatafile):
        with h5py.File(sDatafile, 'r') as hf:
            X_test = hf['X_test'][:]
            y_test = hf['y_test'][:]
            patchSize = hf['patchSize'][:]
    else:
        X_test = np.zeros((0, patchSize[0], patchSize[1]))
        y_test = np.zeros(0)
        for iImg in range(0, len(cfg['lPredictImg'])):
            # patches and labels of reference/artifact
            tmpPatches, tmpLabels  = datapre.fPreprocessData(cfg['lPredictImg'][iImg], cfg['patchSize'], cfg['patchOverlap'], 1, cfg['sLabeling'])
            X_test = np.concatenate((X_test, tmpPatches), axis=0)
            y_test = np.concatenate((y_test, cfg['lLabelPredictImg'][iImg]*tmpLabels), axis=0)
    
    sNetworktype = cfg['network'].split("_")
    if len(sPredictModel) == 0:
        sPredictModel = cfg['selectedDatabase']['bestmodel'][sNetworktype[2]]
    for iFold in range(0, len(X_test)):
        if len(X_test) != 1:
            CV_Patient = iFold + 1
        else:
            CV_Patient = 0
        cnn_main.fRunCNN({'X_train': [], 'y_train': [], 'X_test': X_test[iFold], 'y_test': y_test[iFold], 'patchSize': patchSize, 'model_name': sPredictModel }, cfg['network'], lTrain, cfg['sOpti'], sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'], CV_Patient)
