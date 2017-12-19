
import h5py
import yaml
import numpy as np
from PIL import Image
import scipy.io as sio
import os
import cnn_main


# f84 = h5py.File('/home/s1241/no_backup/s1241/CNNArt/MultiScale/Headnormal/8080/testout/normal8080to4040.h5','r')
# X_train84=f84['X_train']
# Slice84 = X_train84[0][100] * 200
# print(Slice84)
# img84=Image.fromarray(Slice84)
# img84.show()
#
# f80 = h5py.File('/home/s1241/no_backup/s1241/CNNArt/MultiScale/Headnormal/8080/testout/normal8080.h5','r')
# X_train80=f80['X_train']
# Slice80 = X_train80[0][100] * 200
# print(Slice80)
# img80=Image.fromarray(Slice80)
# img80.show()

# f20 = h5py.File('/home/s1241/no_backup/s1241/CNNArt/MultiScale/Headnormal/2020/testout/normal2020.h5','r')
# X_train20=f20['X_train']
# Slice20 = X_train20[0][200] * 200
# print(Slice20)
# img20=Image.fromarray(Slice20)
# img20.show()
#
# f24 = h5py.File('/home/s1241/no_backup/s1241/CNNArt/MultiScale/Headnormal/2020/testout/normal2020to4040.h5','r')
# X_train24=f24['X_train']
# Slice24 = X_train24[0][200] * 200
# print(Slice24)
# img24=Image.fromarray(Slice24)
# img24.show()

lTrain = True
lSave = False # save intermediate test, training sets
with open('config' + os.sep + 'paramnormal.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

forig = h5py.File('/home/s1241/no_backup/s1241/CNNArt/MultiScale/Headnormal/8080/testout/normal8080to4040.h5', 'r')
X_train = forig['X_train']
X_test = forig['X_test']
y_train = forig['y_train']
y_test = forig['y_test']
patchSize = forig['patchSize']
patchOverlap = forig['patchOverlap']
OrigPatchSize = forig['OrigPatchSize']

sOutsubdir = cfg['subdirs'][2]
sOutPath = cfg['selectedDatabase']['pathout'] + os.sep + ''.join(map(str,OrigPatchSize)).replace(" ", "") + os.sep + sOutsubdir
for iFold in range(0,len(X_train)):
    cnn_main.fRunCNN({'X_train': X_train[iFold], 'y_train': y_train[iFold], 'X_test': X_test[iFold], 'y_test': y_test[iFold], 'patchSize': patchSize}, cfg['network'], lTrain, cfg['sOpti'], sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'])
