import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import seaborn as sn
import pandas as pd
from sklearn import metrics




path = 'D:/med_data/MRPhysics/MA Results/2D_64x64/Multiclass SE-ResNet-56_2D_64x64_2018-04-07_13-13/model_predictions.mat'

mat = sio.loadmat(path)

Y_test = mat['Y_test']

prob_pre = mat['prob_pre']

indexes = np.argmax(prob_pre, axis=1)

onehots = np.zeros((prob_pre.shape[0], prob_pre.shape[1]))

for i in range(indexes.shape[0]):
    onehots[i, indexes[i]] = 1


acc = metrics.accuracy_score(Y_test, onehots)
precision = metrics.precision_score(Y_test, onehots, average='weighted', labels=np.unique(onehots))
#f1score = metrics.f1_score(Y_test, onehots, average='weighted', labels=np.unique(onehots))

print(acc)
print(precision)
#print(f1score)


