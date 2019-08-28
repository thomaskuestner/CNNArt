import numpy as np
import tensorflow as tf

def freshape_tensor(X_train, Y_segMasks_train):
    Y_segMasks_train = tf.where(tf.math.equal(Y_segMasks_train, 3), tf.ones_like(Y_segMasks_train), Y_segMasks_train)
    Y_segMasks_train = tf.where(tf.math.equal(Y_segMasks_train, 2), tf.zeros_like(Y_segMasks_train), Y_segMasks_train)

    y_labels_train = tf.expand_dims(Y_segMasks_train, axis=-1)
    y_labels_train = tf.where(tf.math.equal(y_labels_train, 0), -tf.ones_like(y_labels_train), y_labels_train)
    # y_labels_train = tf.where(tf.math.equal(y_labels_train, 1), tf.ones_like(y_labels_train), tf.zeros_like(y_labels_train))

    y_labels_train = tf.math.reduce_sum(y_labels_train, axis=1)
    y_labels_train = tf.math.reduce_sum(y_labels_train, axis=1)
    y_labels_train = tf.math.reduce_sum(y_labels_train, axis=1)
    y_labels_train = tf.where(tf.math.greater_equal(y_labels_train, 0), tf.ones_lile(y_labels_train), tf.zeros_like(y_labels_train))
    Y_train = tf.one_hot(y_labels_train, 2)

    X_train = tf.expand_dims(X_train, axis=-1)
    Y_segMasks_train = tf.expand_dims(Y_segMasks_train, axis=-1)

    return X_train, Y_train, Y_segMasks_train

def freshape_numpy(X_train, Y_segMasks_train):
    Y_segMasks_train[Y_segMasks_train == 3] = 1
    Y_segMasks_train[Y_segMasks_train == 2] = 0
    y_labels_train = np.expand_dims(Y_segMasks_train, axis=-1)
    y_labels_train[y_labels_train == 0] = -1
    y_labels_train[y_labels_train == 1] = 1
    y_labels_train = np.sum(y_labels_train, axis=1)
    y_labels_train = np.sum(y_labels_train, axis=1)
    y_labels_train = np.sum(y_labels_train, axis=1)
    y_labels_train[y_labels_train >= 0] = 1
    y_labels_train[y_labels_train < 0] = 0
    Y_train = []
    for i in range(y_labels_train.shape[0]):
        Y_train.append([1, 0] if y_labels_train[i].all() == 0 else [0, 1])
    Y_train = np.asarray(Y_train)
    X_train = np.expand_dims(X_train, axis=-1)
    Y_segMasks_train = np.expand_dims(Y_segMasks_train, axis=-1)
    return X_train, Y_train, Y_segMasks_train