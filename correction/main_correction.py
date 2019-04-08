# external import
import glob
import os
import h5py
import numpy as np

# internal import
import utils.DataPreprocessing as datapre
import main as cnn_main

def run(cfg, dbinfo):
    """
    the main interface of correction program
    @param cfg: the configuration file loaded from config/param.yml
    @param dbinfo: database related info
    """
    # load parameters form config file and define the corresponding output path
    patchSize = cfg['patchSize']

    sOutsubdir = cfg['subdirs'][3]
    sOutPath = cfg['selectedDatabase']['pathout'] + os.sep \
               + ''.join(map(str, patchSize)).replace(" ", "") + os.sep + sOutsubdir

    if cfg['sSplitting'] == 'normal':
        sFSname = 'normal'
        sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + '.h5'
    elif cfg['sSplitting'] == 'crossvalidation_data':
        sFSname = 'crossVal_data'
        sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + '.h5'
    elif cfg['sSplitting'] == 'crossvalidation_patient':
        sFSname = 'crossVal'
        sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + '_' + \
                    cfg['correction']['test_patient'] + '.h5'

    # if h5 file exists then load the dataset
    if glob.glob(sDatafile):
        with h5py.File(sDatafile, 'r') as hf:
            train_ref = hf['train_ref'][:]
            train_art = hf['train_art'][:]
            test_ref = hf['test_ref'][:]
            test_art = hf['test_art'][:]
            patchSize = hf['patchSize'][:]

    else:
        # perform patching and splitting
        train_ref, test_ref, train_art, test_art = datapre.fPreprocessDataCorrection(cfg, dbinfo)

        # save to h5 file
        if cfg['lSave']:
            with h5py.File(sDatafile, 'w') as hf:
                hf.create_dataset('train_ref', data=train_ref)
                hf.create_dataset('test_ref', data=test_ref)
                hf.create_dataset('train_art', data=train_art)
                hf.create_dataset('test_art', data=test_art)
                hf.create_dataset('patchSize', data=patchSize)
                hf.create_dataset('patchOverlap', data=cfg['patchOverlap'])

    dHyper = cfg['correction']
    dParam = {'batchSize': cfg['batchSize'], 'patchSize': patchSize, 'patchOverlap': cfg['patchOverlap'],
              'learningRate': cfg['lr'], 'epochs': cfg['epochs'], 'lTrain': cfg['lTrain'], 'lSave': cfg['lSave'],
              'sOutPath': sOutPath, 'lSaveIndividual': cfg['lSaveIndividual']}

    if len(train_ref) == 1:
        dData = {'train_ref': train_ref[0], 'test_ref': test_ref[0],
                 'train_art': train_art[0], 'test_art': test_art[0]}
        cnn_main.fRunCNNCorrection(dData, dHyper, dParam)
    else:
        for patient_index in range(len(train_ref)):
            dData = {'train_ref': train_ref[patient_index], 'test_ref': test_ref[patient_index],
                     'train_art': train_art[patient_index], 'test_art': test_art[patient_index]}
            cnn_main.fRunCNNCorrection(dData, dHyper, dParam)
