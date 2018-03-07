# external import
import glob
import os
import h5py
import numpy as np

# internal import
import utils.DataPreprocessing as datapre
import cnn_main

def run(cfg, dbinfo):
    """
    the main interface of correction program
    @param cfg: the configuration file loaded from config/param.yml
    @param dbinfo: database related info
    """
    # load parameters form config file and define the corresponding output path
    patchSize = cfg['patchSize']

    if cfg['sSplitting'] == 'normal':
        sFSname = 'normal'
    elif cfg['sSplitting'] == 'crossvalidation_data':
        sFSname = 'crossVal_data'
    elif cfg['sSplitting'] == 'crossvalidation_patient':
        sFSname = 'crossVal'

    sModelIn = cfg['correction']['sCorrection']
    sOutsubdir = cfg['subdirs'][3]
    sOutPath = cfg['selectedDatabase']['pathout'] + os.sep \
               + ''.join(map(str, patchSize)).replace(" ", "") + os.sep + sOutsubdir
    sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + '.h5'

    # if h5 file exists then load the dataset

    if cfg['lTrain']:
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

        dHyper = {'batchSize': cfg['batchSize'], 'learningRate': cfg['lr'], 'epochs': cfg['epochs'],
                  'kl_weight': cfg['correction']['kl_weight'], 'perceptual_weight': cfg['correction']['perceptual_weight'],
                  'pixel_weight': cfg['correction']['pixel_weight'], 'pl_network': cfg['correction']['pl_network'],
                  'loss_model': cfg['correction']['loss_model']}
        for iFold in range(len(train_ref)):
            dData = {'train_ref': train_ref[iFold], 'test_ref': test_ref[iFold], 'train_art': train_art[iFold], 'test_art': test_art[iFold]}
            cnn_main.fRunCNNCorrection(dData, sModelIn, patchSize, sOutPath, dHyper, cfg['lTrain'], cfg['lSave'], cfg['correction']['unpatch'], cfg['patchOverlap'])

    else:
        dHyper = {'batchSize': cfg['batchSize'], 'bestModel': cfg['correction']['bestModel'],
                  'kl_weight': cfg['correction']['kl_weight'], 'perceptual_weight': cfg['correction']['perceptual_weight'],
                  'pixel_weight': cfg['correction']['pixel_weight'], 'pl_network': cfg['correction']['pl_network'],
                  'loss_model': cfg['correction']['loss_model']}

        with h5py.File(sDatafile, 'r') as hf:
            dData = hf['test_art'][:]

        for i in range(dData.shape[0]):
            cnn_main.fRunCNNCorrection(dData[i], sModelIn, patchSize, sOutPath, dHyper, cfg['lTrain'], cfg['lSave'], cfg['correction']['unpatch'], cfg['patchOverlap'])

