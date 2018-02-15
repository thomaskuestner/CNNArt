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

imgs = np.ones([9, 32, 32, 1])
img = imgs[2].reshape(32, 32)

plt.imshow(img.reshape(32, 32), cmap='gray')
plt.show()
print()


