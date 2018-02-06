# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 15:59:36 2017
@author: Sebastian Milde, Thomas Kuestner
"""

import math
import pandas as pd
import numpy as np
import h5py
import inspect
import dis
from sklearn.model_selection import KFold
import os
from DeepLearningArt.DLArt_GUI import dlart
import keras.backend as K

def expecting():
    """Return how many values the caller is expecting"""
    f = inspect.currentframe()
    f = f.f_back.f_back
    c = f.f_code
    i = f.f_lasti
    bytecode = c.co_code
    instruction = bytecode[i+3]
    if instruction == dis.opmap['UNPACK_SEQUENCE']:
        howmany = bytecode[i+4]
        return howmany
    elif instruction == dis.opmap['POP_TOP']:
        return 0
    return 1

def fSplitDataset(allPatches, allY, allPats, sSplitting, patchSize, patchOverlap, testTrainingDatasetRatio=0, validationTrainRatio=0, outPutPath=None, nfolds = 0):
    # TODO: adapt path
    iReturn = expecting()
    #iReturn = 1000

    # 2D or 3D patching?
    if len(patchSize) == 2:
        #2D patches are used
        if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1]:
            allPatches = np.transpose(allPatches, (2, 0, 1))
    elif len(patchSize) == 3:
        #3D patches are used
        if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1] and allPatches.shape[2] == patchSize[2]:
            allPatches = np.transpose(allPatches, (3, 0, 1, 2))

    if sSplitting == dlart.DeepLearningArtApp.SIMPLE_RANDOM_SAMPLE_SPLITTING:
        # splitting
        indexSlices = range(allPatches.shape[0])
        indexSlices = np.random.permutation(indexSlices)

        if len(patchSize)==2:
            #2D patching
            allPatches = allPatches[indexSlices, :, :]
        elif len(patchSize)==3:
            #3D patching
            allPatches = allPatches[indexSlices, :, :, :]

        allY = allY[indexSlices]

        #num of samples in test set and validation set
        numAllPatches = allPatches.shape[0]
        numSamplesTest = math.floor(testTrainingDatasetRatio*numAllPatches)
        numSamplesValidation = math.floor(validationTrainRatio*(numAllPatches-numSamplesTest))

        if len(patchSize) == 2:
            #2D patching
            # subarrays as no-copy views (array slices)
            X_test = allPatches[:numSamplesTest, :, :]
            X_valid = allPatches[numSamplesTest:(numSamplesTest+numSamplesValidation), :, :]
            X_train = allPatches[(numSamplesTest+numSamplesValidation):, :, :]

        elif len(patchSize) == 3:
            # 3D patching
            # subarrays as no-copy views (array slices)
            X_test = allPatches[:numSamplesTest, :, :, :]
            X_talid = allPatches[numSamplesTest:(numSamplesTest + numSamplesValidation), :, :, :]
            X_train = allPatches[(numSamplesTest + numSamplesValidation):, :, :, :]

        y_test = allY[:numSamplesTest]
        y_valid = allY[numSamplesTest:(numSamplesTest + numSamplesValidation)]
        y_train = allY[(numSamplesTest + numSamplesValidation):]

        # #random samples
        # nPatches = allPatches.shape[0]
        # dVal = math.floor(split_ratio * nPatches)
        # rand_num = np.random.permutation(np.arange(nPatches))
        # rand_num = rand_num[0:int(dVal)].astype(int)
        # print(rand_num)
        #
        # #do splitting
        # X_test = allPatches[rand_num, :, :]
        # y_test = allY[rand_num]
        # X_train = allPatches
        # X_train = np.delete(X_train, rand_num, axis=0)
        # y_train = allY
        # y_train = np.delete(y_train, rand_num)
        # print(X_train.shape)
        # print(X_test.shape)
        # print(y_train.shape)
        # print(y_test.shape)
        # #!!!! train dataset is not randomly shuffeled!!!

        if iReturn == 0:
            if len(patchSize) == 3:
                folder = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + str(patchSize[2])
                Path = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + str(
                    patchSize[2]) + os.sep + 'normal_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
            else:
                folder = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1])
                Path = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + os.sep + 'normal_' + str(
                    patchSize[0]) + str(patchSize[1]) + '.h5'

            if os.path.isdir(folder):
                pass
            else:
                os.makedirs(folder)

            print(Path)
            with h5py.File(Path, 'w') as hf:
                hf.create_dataset('X_train', data=X_train)
                hf.create_dataset('X_test', data=X_test)
                hf.create_dataset('y_train', data=y_train)
                hf.create_dataset('y_test', data=y_test)
                hf.create_dataset('patchSize', data=patchSize)
                hf.create_dataset('patchOverlap', data=patchOverlap)
        else:
            # if len(patchSize) == 2:
            #     # 2D patches are used
            #     if allPatches.shape[1] == patchSize[0] and allPatches.shape[2] == patchSize[1]:
            #         X_train = np.transpose(X_train, (1, 2, 0))
            #         X_valid = np.transpose(X_valid, (1, 2, 0))
            #         X_test = np.transpose(X_test, (1, 2, 0))
            # elif len(patchSize) == 3:
            #     # 3D patches are used
            #     if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1] and allPatches.shape[2] == patchSize[2]:
            #         X_train = np.transpose(X_train, (1, 2, 3, 0))
            #         X_valid = np.transpose(X_valid, (1, 2, 3, 0))
            #         X_test = np.transpose(X_test, (1, 2, 3, 0))

            return [X_train], [y_train], [X_valid], [y_valid], [X_test], [y_test] # embed in a 1-fold list


    elif sSplitting == dlart.DeepLearningArtApp.CROSS_VALIDATION_SPLITTING:
        # split into test/train sets
        #shuffle
        indexSlices = range(allPatches.shape[0])
        indexSlices = np.random.permutation(indexSlices)

        allPatches = allPatches[indexSlices, :, :]
        allY = allY[indexSlices]

        # num of samples in test set
        numAllPatches = allPatches.shape[0]
        numSamplesTest = math.floor(testTrainingDatasetRatio*numAllPatches)

        # subarrays as no-copy views (array slices)
        xTest = allPatches[:numSamplesTest, :, :]
        yTest = allY[:numSamplesTest]

        xTrain = allPatches[numSamplesTest:, :, :]
        yTrain = allY[numSamplesTest:]

        # split training dataset into n folds
        if nfolds == 0:
            kf = KFold(n_splits=len(allPats))
        else:
            kf = KFold(n_splits=nfolds)

        #ind_split = 0
        X_trainFold = []
        X_testFold = []
        y_trainFold = []
        y_testFold = []

        for train_index, test_index in kf.split(xTrain):
            X_train, X_test = xTrain[train_index], xTrain[test_index]
            y_train, y_test = yTrain[train_index], yTrain[test_index]

            if iReturn == 0:
                if len(patchSize) == 3:
                    folder = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + str(patchSize[2])
                    Path = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + str(
                        patchSize[2]) + os.sep + 'crossVal_data' + str(ind_split) + '_' + str(patchSize[0]) + str(
                        patchSize[1]) + str(patchSize[2]) + '.h5'
                else:
                    folder = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1])
                    Path = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + os.sep + 'crossVal_data' + str(
                        ind_split) + '_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
                if os.path.isdir(folder):
                    pass
                else:
                    os.makedirs(folder)

                with h5py.File(Path, 'w') as hf:
                    hf.create_dataset('X_train', data=X_train)
                    hf.create_dataset('X_test', data=X_test)
                    hf.create_dataset('y_train', data=y_train)
                    hf.create_dataset('y_test', data=y_test)
                    hf.create_dataset('patchSize', data=patchSize)
                    hf.create_dataset('patchOverlap', data=patchOverlap)
            else:
                X_trainFold.append(X_train)
                X_testFold.append(X_test)
                y_trainFold.append(y_train)
                y_testFold.append(y_test)

            #ind_split += 1

        X_trainFold = np.asarray(X_trainFold)
        X_testFold = np.asarray(X_testFold)
        y_trainFold = np.asarray(y_trainFold)
        y_testFold = np.asarray(y_testFold)

        if iReturn > 0:
            return X_trainFold, y_trainFold, X_testFold, y_testFold, xTest, yTest



    elif sSplitting == dlart.DeepLearningArtApp.PATIENT_CROSS_VALIDATION_SPLITTING:
        unique_pats = len(allPats)

        X_trainFold = []
        X_testFold = []
        y_trainFold = []
        y_testFold = []

        for ind_split in unique_pats:
            train_index = np.where(allPats != ind_split)[0]
            test_index = np.where(allPats == ind_split)[0]
            X_train, X_test = allPatches[train_index], allPatches[test_index]
            y_train, y_test = allY[train_index], allY[test_index]

            if iReturn == 0:
                if len(patchSize) == 3:
                    folder = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + str(patchSize[2])
                    Path = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + str(
                        patchSize[2]) + os.sep + 'crossVal' + str(ind_split) + '_' + str(patchSize[0]) + str(
                        patchSize[1]) + str(patchSize[2]) + '.h5'
                else:
                    folder = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1])
                    Path = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + os.sep + 'crossVal' + str(
                        ind_split) + '_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
                if os.path.isdir(folder):
                    pass
                else:
                    os.makedirs(folder)

                with h5py.File(Path, 'w') as hf:
                    hf.create_dataset('X_train', data=X_train)
                    hf.create_dataset('X_test', data=X_test)
                    hf.create_dataset('y_train', data=y_train)
                    hf.create_dataset('y_test', data=y_test)
                    hf.create_dataset('patchSize', data=patchSize)
                    hf.create_dataset('patchOverlap', data=patchOverlap)

            else:
                X_trainFold.append(X_train)
                X_testFold.append(X_test)
                y_trainFold.append(y_train)
                y_testFold.append(y_test)


        X_trainFold = np.asarray(X_trainFold, dtype='f')
        X_testFold = np.asarray(X_testFold, dtype='f')
        y_trainFold = np.asarray(y_trainFold, dtype='f')
        y_testFold = np.asarray(y_testFold, dtype='f')

        if iReturn > 0:
            return X_trainFold, y_trainFold, X_testFold, y_testFold