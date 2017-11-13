# demo file
import os
import yaml
import dicom
import dicom_numpy
import numpy as np
import DatabaseInfo
import utils.DataPreprocessing as datapre
import utils.Training_Test_Split as ttsplit

##### Training<
# parse parameters
lTrain = True
lSave = False # save intermediate test, training sets
with open(os.path.join('config', os.sep, 'param.yml'), 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# initiate info objects
# default database: MRPhysics with ['newProtocol','dicom_sorted']
dbinfo = DatabaseInfo()


# load/create input data
patchSize = cfg['patchSize']
if cfg['sSplitting'] == 'normal':
    sFSname = 'normal'
elif cfg['sSplitting'] == 'crossvalidation_data':
    sFSname = 'crossVal_data'
elif cfg['sSplitting'] == 'crossvalidation_patient':
    sFSname = 'crossVal'

sDatafile = cfg[cfg['selectedModel']]['pathout'] + os.sep + str(patchSize[0]) + str(patchSize[1]) + os.sep + sFSname # + str(ind_split) + '_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
dAllPatches = []
dAllLabels = []
dAllPats = []
lDatasets = cfg[cfg['selectedModel']]['dataref'] + cfg[cfg['selectedModel']]['dataart']
iLabels = cfg[cfg['selectedModel']]['labelref'] + cfg[cfg['selectedModel']]['labelart']
for ipat, pat in enumerate(dbinfo.lPats):
    for iseq, seq in enumerate(lDatasets):
        # patches and labels of reference/artifact
        tmpPatches, tmpLabels  = datapre.fPreprocessData(os.path.join(dbinfo.sPathIn, os.sep, pat, os.sep, dbinfo.sSubDirs[1], os.sep, seq), cfg['patchSize'], cfg['patchOverlap'], 1 )
        dAllPatches = np.concatenate((dAllPatches, tmpPatches), axis=2)
        dAllLabels = np.concatenate((dAllLabels, iLabels[iseq]*tmpLabels), axis=0)
        dAllPats = np.concatenate((dAllPats,ipat*np.ones((tmpLabels.shape[0],1), dtype=np.int)), axis=0)

# perform splitting
X_train, y_train, X_test, y_test = ttsplit.fSplitDataset(dAllPatches, dAllLabels, dAllPats, cfg['sSplitting'], cfg['patchSize'], cfg['patchOverlap'], cfg['dSplitval'], '')

# perform training
