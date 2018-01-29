import numpy as np



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

dict = {1: [1, 1, 3], 2:[3, 5, 6], 3:[3, 5, 5]}

asdf = np.zeros([100, 20, 20])

a = []
a.append(dict[1])
a.append(dict[2])

a = np.asarray(a)

print(a.shape)




print(a)