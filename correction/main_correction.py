# external import
import glob
import os
import h5py

# internal import
import utils.DataPreprocessing as datapre
import utils.Training_Test_Split as ttsplit

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

    sOutsubdir = cfg['subdirs'][3]
    sOutPath = cfg['selectedDatabase']['pathout'] + os.sep \
               + ''.join(map(str, patchSize)).replace(" ", "") + os.sep + sOutsubdir
    sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + '.h5'

    # if h5 file exists then load the dataset
    if glob.glob(sDatafile):
        with h5py.File(sDatafile, 'r') as hf:
            train_ref = hf['train_ref'][:]
            train_art = hf['train_art'][:]
            test_ref = hf['test_ref'][:]
            test_art = hf['test_art'][:]
            patchSize = hf['patchSize'][:]

    else:
        # perform patching
        dRefPatches, dArtPatches, dAllPats = datapre.fPreprocessDataCorrection(cfg, dbinfo)

        # perform splitting
        train_ref, test_ref, train_art, test_art = ttsplit.fSplitDatasetCorrection(cfg['sSplitting'], dRefPatches, dArtPatches, dAllPats, cfg['dSplitval'], cfg['nFolds'])

        # save to h5 file
        if cfg['lSave']:
            with h5py.File(sDatafile, 'w') as hf:
                hf.create_dataset('train_ref', data=train_ref)
                hf.create_dataset('test_ref', data=test_ref)
                hf.create_dataset('train_art', data=train_art)
                hf.create_dataset('test_art', data=test_art)
                hf.create_dataset('patchSize', data=patchSize)
