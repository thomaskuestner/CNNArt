import numpy as np
from keras.layers import *

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
y_train = [1, 1, 0, 0, 1, 1]

a = np.asarray([y_train[:], np.abs(np.asarray(y_train[:], dtype=np.float32)-1)]).T

print(a)