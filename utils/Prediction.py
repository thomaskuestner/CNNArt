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


def predict_segmentation_model(X_test, y=None, Y_segMasks_test=None, sModelPath=None, sOutPath=None, batch_size=64):
    """Takes an already trained model and computes the loss and Accuracy over the samples X with their Labels y
        Input:
            X: Samples to predict on. The shape of X should fit to the input shape of the model
            y: Labels for the Samples. Number of Samples should be equal to the number of samples in X
            sModelPath: (String) full path to a trained keras model. It should be *_json.txt file. there has to be a corresponding *_weights.h5 file in the same directory!
            sOutPath: (String) full path for the Output. It is a *.mat file with the computed loss and accuracy stored.
                        The Output file has the Path 'sOutPath'+ the filename of sModelPath without the '_json.txt' added the suffix '_pred.mat'
            batchSize: Batchsize, number of samples that are processed at once"""

    X_test = np.expand_dims(X_test, axis=-1)
    Y_segMasks_test_foreground = np.expand_dims(Y_segMasks_test, axis=-1)
    Y_segMasks_test_background = np.ones(Y_segMasks_test_foreground.shape) - Y_segMasks_test_foreground
    Y_segMasks_test = np.concatenate((Y_segMasks_test_background, Y_segMasks_test_foreground), axis=-1)

    _, sPath = os.path.splitdrive(sModelPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)

    listdir = os.listdir(sModelPath)

    #sModelPath = sModelPath.replace("_json.txt", "")
    #weight_name = sModelPath + '_weights.h5'
    #model_json = sModelPath + '_json.txt'
    #model_all = sModelPath + '_model.h5'

    # load weights and model (new way)
    with open(sModelPath + os.sep + sFilename + '.json', 'r') as fp:
        model_string = fp.read()

    model = model_from_json(model_string)

    model.summary()

    model.compile(loss=dice_coef_loss, optimizer=keras.optimizers.Adam(), metrics=[dice_coef])
    model.load_weights(sModelPath+ os.sep + sFilename+'_weights.h5')

    score_test, acc_test = model.evaluate(X_test, Y_segMasks_test, batch_size=2)
    print('loss' + str(score_test) + '   acc:' + str(acc_test))

    prob_pre = model.predict(X_test, batch_size=batch_size, verbose=1)

    predictions = {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test}

    return predictions



def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)