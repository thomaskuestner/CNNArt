import numpy as np
from keras.layers import *
from matplotlib import pyplot as plt

# from sklearn.model_selection import KFold
#
# X = np.ones([100, 10, 10])
# y = np.zeros([100, 1])
# kf = KFold(n_splits=5)
# kf.get_n_splits(X)
#
# print(kf)
#
# for train_index, test_index in kf.split(X):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]

voxel = np.zeros((2,3,5))
voxel[0,0,0] = 1
voxel[0,1,0] = 1

voxel2 = np.ones((2, 3, 5))
voxel2 = voxel2-voxel

asdf = voxel2.shape

print("hi" if 10 == 23 else "hoho")


