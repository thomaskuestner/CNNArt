# external import
import glob
import os
import h5py
import numpy as np

# internal import
import utils.DataPreprocessing as datapre

def fPatching(cfg, dbinfo):
    patchSize = cfg['patchSize']
    dAllPatches = np.zeros((0, patchSize[0], patchSize[1]))
    dAllLabels = np.zeros(0)
    dAllPats = np.zeros((0, 1))

    lDatasets = cfg['selectedDatabase']['dataref'] + cfg['selectedDatabase']['dataart']
    iLabels = cfg['selectedDatabase']['labelref'] + cfg['selectedDatabase']['labelart']
    for ipat, pat in enumerate(dbinfo.lPats):
        if os.path.exists(dbinfo.sPathIn + os.sep + pat + os.sep + dbinfo.sSubDirs[1]):
            for iseq, seq in enumerate(lDatasets):
                # patches and labels of reference/artifact
                tmpPatches, tmpLabels = datapre.fPreprocessData(
                    os.path.join(dbinfo.sPathIn, pat, dbinfo.sSubDirs[1], seq), patchSize, cfg['patchOverlap'], 1,
                    'volume')
                dAllPatches = np.concatenate((dAllPatches, tmpPatches), axis=0)
                dAllLabels = np.concatenate((dAllLabels, iLabels[iseq] * tmpLabels), axis=0)
                dAllPats = np.concatenate((dAllPats, ipat * np.ones((tmpLabels.shape[0], 1), dtype=np.int)), axis=0)
        else:
            pass

    iRefPatches = np.where(dAllLabels == iLabels[0])[0]
    iCorPatches = np.where(dAllLabels == iLabels[1])[0]
    dRefPatches, dCorPatches = dAllLabels[iRefPatches], dAllLabels[iCorPatches]

    return dRefPatches, dCorPatches

def fSplitDataset():
    pass

def run(cfg, dbinfo):
    patchSize = cfg['patchSize']
    sOutsubdir = cfg['subdirs'][3]
    sOutPath = cfg['selectedDatabase']['pathout'] + os.sep \
               + ''.join(map(str, patchSize)).replace(" ", "") + os.sep + sOutsubdir
    sDatafile = sOutPath + os.sep + ''.join(map(str, patchSize)).replace(" ", "") + '.h5'

    # if h5 file exists
    if glob.glob(sDatafile):
        with h5py.File(sDatafile, 'r') as hf:
            X_train_reference = hf['X_train_reference'][:]
            X_train_corrupted = hf['X_train_corrupted'][:]
            X_test_reference = hf['X_test_reference'][:]
            X_test_corrupted = hf['X_test_corrupted'][:]
            patchSize = hf['patchSize'][:]

    else:
        # perform patching
        dRefPatches, dCorPatches = fPatching(cfg, dbinfo)
	print(dRefPatches.shape)
	print(dCorPatches.shape)
        # perform splitting




