import os
#os.environ["CUDA_DEVICE_ORDER"]="0000:02:00.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices)

import os.path
import scipy.io as sio
import numpy as np
import math
import keras
from keras.layers import Input
import keras.backend as K
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.activations import softmax
from keras.layers import concatenate
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Model
from keras.models import Sequential
from keras.layers import UpSampling3D
from keras.layers.convolutional import Convolution2D
from keras.layers import LeakyReLU
from keras.layers import Softmax
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.regularizers import l2  # , activity_l2

from keras.optimizers import SGD
from utils.image_preprocessing import ImageDataGenerator
from matplotlib import pyplot as plt

from utils.Label import *

from sklearn.metrics import classification_report, confusion_matrix



def predict_model(X_test, Y_test, sModelPath, batch_size=32, classMappings=None):
    X_test = np.expand_dims(X_test, axis=-1)

    # pathes
    _, sPath = os.path.splitdrive(sModelPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)

    # load weights and model
    with open(sModelPath + os.sep + sFilename + '.json', 'r') as fp:
        model_string = fp.read()

    model = model_from_json(model_string)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.load_weights(sModelPath + os.sep + sFilename + '_weights.h5')

    # evaluate model on test data
    score_test, acc_test = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('loss' + str(score_test) + '   acc:' + str(acc_test))

    # predict test dataset
    probability_predictions = model.predict(X_test, batch_size=batch_size, verbose=1, steps=None)

    #classification report
    target_names = []
    for i in sorted(classMappings):
        target_names.append(Label.LABEL_STRINGS[i])

    classification_summary = classification_report(np.argmax(Y_test, axis=1),
                                                   np.argmax(probability_predictions, axis=1),
                                                   target_names=target_names, digits=4)

    #confusion matrix
    confusionMatrix = confusion_matrix(y_true=np.argmax(Y_test, axis=1),
                                       y_pred=np.argmax(probability_predictions, axis=1),
                                       labels=range(int(probability_predictions.shape[1])))


    prediction = {
        'predictions': probability_predictions,
        'score_test': score_test,
        'acc_test': acc_test,
        'classification_report': classification_summary,
        'confusion_matrix': confusionMatrix
    }

    return prediction

