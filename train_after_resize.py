
import h5py
import yaml
import os
import cnn_main

with open('config' + os.sep + 'paramnormal.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

forig = h5py.File('/home/s1241/no_backup/s1241/CNNArt/MultiScale/Headcross/8080/testout/crossVal8080to4040.h5', 'r')
X_train = forig['X_train']
X_test = forig['X_test']
y_train = forig['y_train']
y_test = forig['y_test']
patchSize = forig['patchSize']
patchOverlap = forig['patchOverlap']
OrigPatchSize = forig['OrigPatchSize']

sOutsubdir = cfg['subdirs'][2]
sOutPath = cfg['selectedDatabase']['pathout'] + os.sep + ''.join(map(str,OrigPatchSize)).replace(" ", "") + os.sep + sOutsubdir

lSave = False # save intermediate test, training sets

# Select True for Training and False for Predicting
lTrain = True

for iFold in range(0, len(X_train)):
    if cfg['sSplitting'] == 'crossvalidation_patient':
        CV_Patient = iFold + 1
    cnn_main.fRunCNN({'X_train': X_train[iFold], 'y_train': y_train[iFold], 'X_test': X_test[iFold], 'y_test': y_test[iFold], 'patchSize': patchSize}, cfg['network'], lTrain, cfg['sOpti'], sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'], CV_Patient = 0)

