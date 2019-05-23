"""
+-----------------------------------------------------------------------+
| Main template function for calling the CNNs                           |
| - starting point: data is alrady patched, augmented, splitted, ...    |
+-----------------------------------------------------------------------+

Copyright: 2016-2018 Thomas Kuestner (thomas.kuestner@med.uni-tuebingen.de) under Apache2 license
@author: Thomas Kuestner
"""


"""Import"""
import sys
#import numpy as np                  # for algebraic operations, matrices
#import h5py
#import os.path                      # operating system


# --------------------------------------------
# option A: link to your networks to be loaded
# --------------------------------------------
from networks.motion.CNN3D import *

def fMainCNN(X_train, X_val, X_test, label_train, label_val, label_test, lTrain=true, param):
    # X_train, X_val, X_test: numpy arrays with training, validation, test data patches (already patched, augmented and split)
    # label_train, label_val, label_test: corresponding labels
    # lTrain: perform training (true) or prediction with trained architecture (false)
    # param: dictionary containing all set UI parameters and further requested configs

    # ------------------------------------------------------
    # option B: dynamic loading of corresponding model
    # functions fTrain and fPredict must exist or adapt this
    # ------------------------------------------------------
    cnnModel = __import__(sModel, globals(), locals(), ['fTrain', 'fPredict'], 0)  # dynamic module loading with specified functions and with absolute importing (level=0) -> work in both Python2 and Python3

    cnnModel.fTrain(_YOUR_DEFINED_INTERFACE_)
