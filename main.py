# demo file
import os
import yaml
import dicom
import dicom_numpy
import numpy as np
import DatabaseInfo
import utils.DataPreprocessing as datapre

##### Training
# parse parameters
lTrain = true
with open(os.path.join('config', os.sep, 'param.yml'), 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# initiate info objects
# default database: MRPhysics with ['newProtocol','dicom_sorted']
dbinfo = DatabaseInfo()

dPatchesRef = []
dLabelsRef = []
dPatchesArt = []
dLabelsArt = []
for pat in dbinfo.lPats:
    # patches and labels of reference
    tmpPatchesRef, tmpLabelsRef  = datapre.fPreprocessData(os.path.join(dbinfo.sPathIn, os.sep, pat, os.sep, dbinfo.sSubDirs[1], os.sep, cfg[cfg['selectedModel']]['dataref']), cfg['patchSize'], cfg['patchOverlap'], 1 )
    # patches and labels of artifact
    tmpPatchesArt, tmpLabelsArt = datapre.fPreprocessData(os.path.join(dbinfo.sPathIn, os.sep, pat, os.sep, dbinfo.sSubDirs[1], os.sep, cfg[cfg['selectedModel']]['dataart']), cfg['patchSize'], cfg['patchOverlap'], 1)

    dPatchesRef = np.concatenate((dPatchesRef,tmpPatchesRef), axis=2)
    dLabelsRef = np.concatenate((dLabelsRef, tmpLabelsRef), axis=2)
    dPatchesArt = np.concatenate((dPatchesArt, tmpPatchesArt), axis=2)
    dLabelsArt = np.concatenate((dLabelsArt, tmpLabelsArt), axis=2)

# perform splitting

# perform training