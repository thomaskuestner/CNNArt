import h5py
import numpy as np
import keras
from fSetGPU import*

PathIn = "/home/d1224/no_backup/d1224/PatchbasedLabeling Results/Prediction_Data/Beckent2_Move_ma_without_2D.h5"
PathOut = "/home/d1224/no_backup/d1224/PredictResults/"
sModelPath = "/home/d1224/no_backup/d1224/Kopft1_05_withoutlabel_testma_val_ab_4040_lr_0.001_bs_64.mat"
dData = dict()
patchSize = [40, 40]
with h5py.File(PathIn, 'r') as hf:
    X_test = hf['AllPatches'][:]
    y_test = hf['AllLabels'][:]

if X_test.shape[0] == patchSize[0] and X_test.shape[1] == patchSize[1]:
    X_test = np.transpose(X_test, (2, 0, 1))

y_test = y_test == 1
y_test = y_test.astype(int)
X_test = np.expand_dims(X_test, axis=1) # axis = 1
y_test = keras.utils.to_categorical(y_test)
#y_test= np.asarray([y_test, np.abs(np.asarray(y_test,dtype=np.float32)-1)]).T
print(y_test)
patchSize = np.array(([40, 40]), dtype = np.float32)
dData['X_test'] = X_test
dData['y_test'] = y_test
dData['patchSize'] = patchSize
model = 'motion_head'
batchSize = 64
fSetGPU()
cnnModel = __import__(model, globals(), locals(), ['createModel', 'fTrain', 'fPredict'], -1)
cnnModel.fPredict(dData['X_test'], dData['y_test'], sModelPath, PathOut, batchSize)