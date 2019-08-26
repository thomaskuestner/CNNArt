# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 15:59:36 2017
@author: Sebastian Milde, Thomas Kuestner
"""

import dis
import inspect

import math
import numpy as np
from sklearn.model_selection import KFold



def expecting():
    """Return how many values the caller is expecting"""
    f = inspect.currentframe()
    f = f.f_back.f_back
    c = f.f_code
    i = f.f_lasti
    bytecode = c.co_code
    instruction = bytecode[i + 3]
    if instruction == dis.opmap['UNPACK_SEQUENCE']:
        howmany = bytecode[i + 4]
        return howmany
    elif instruction == dis.opmap['POP_TOP']:
        return 0
    return 1


def fSplitDataset(allPatches, allY, allPats, allTestPats, sSplitting, patchSize, patchOverlap, testTrainingDatasetRatio=0,
                  validationTrainRatio=0, outPutPath=None, nfolds=0, isRandomShuffle=True):
    # TODO: adapt path
    # iReturn = expecting()
    # iReturn = 1000
    iReturn = 1

    # 2D or 3D patching?
    if len(patchSize) == 2:
        # 2D patches are used
        if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1]:
            allPatches = np.transpose(allPatches, (2, 0, 1))
    elif len(patchSize) == 3:
        # 3D patches are used
        if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1] and allPatches.shape[2] == \
                patchSize[2]:
            allPatches = np.transpose(allPatches, (3, 0, 1, 2))

    if sSplitting == 'SIMPLE_RANDOM_SAMPLE_SPLITTING':
        # splitting
        indexSlices = range(allPatches.shape[0])

        if isRandomShuffle:
            indexSlices = np.random.permutation(indexSlices)

        if len(patchSize) == 2:
            # 2D patching
            allPatches = allPatches[indexSlices, :, :]
        elif len(patchSize) == 3:
            # 3D patching
            allPatches = allPatches[indexSlices, :, :, :]

        shapeAllY = allY.shape

        if len(shapeAllY) > 1:
            if allY.shape[0] == patchSize[0] and allY.shape[1] == patchSize[1]:
                allY = np.transpose(allY, (2, 0, 1))

        allY = allY[indexSlices]

        # num of samples in test set and validation set
        numAllPatches = allPatches.shape[0]
        numSamplesTest = math.floor(testTrainingDatasetRatio * numAllPatches)
        numSamplesValidation = math.floor(validationTrainRatio * (numAllPatches - numSamplesTest))

        if len(patchSize) == 2:
            # 2D patching
            # subarrays as no-copy views (array slices)
            X_test = allPatches[:numSamplesTest, :, :]
            X_valid = allPatches[numSamplesTest:(numSamplesTest + numSamplesValidation), :, :]
            X_train = allPatches[(numSamplesTest + numSamplesValidation):, :, :]

        elif len(patchSize) == 3:
            # 3D patching
            # subarrays as no-copy views (array slices)
            X_test = allPatches[:numSamplesTest, :, :, :]
            X_valid = allPatches[numSamplesTest:(numSamplesTest + numSamplesValidation), :, :, :]
            X_train = allPatches[(numSamplesTest + numSamplesValidation):, :, :, :]

        y_test = allY[:numSamplesTest]
        y_valid = allY[numSamplesTest:(numSamplesTest + numSamplesValidation)]
        y_train = allY[(numSamplesTest + numSamplesValidation):]

        return [X_train], [y_train], [X_valid], [y_valid], [X_test], [y_test]  # embed in a 1-fold list

    elif sSplitting == 'CROSS_VALIDATION_SPLITTING':
        # split into test/train sets
        # shuffle
        indexSlices = range(allPatches.shape[0])
        indexSlices = np.random.permutation(indexSlices)

        allPatches = allPatches[indexSlices, :, :]
        allY = allY[indexSlices]

        # num of samples in test set
        numAllPatches = allPatches.shape[0]
        numSamplesTest = math.floor(testTrainingDatasetRatio * numAllPatches)

        # subarrays as no-copy views (array slices)
        xTest = allPatches[:numSamplesTest, :, :]
        yTest = allY[:numSamplesTest]

        xTrain = allPatches[numSamplesTest:, :, :]
        yTrain = allY[numSamplesTest:]

        # split training dataset into n folds
        if nfolds == 0:
            kf = KFold(n_splits=len(allTestPats))
        else:
            kf = KFold(n_splits=nfolds)

        ind_split = 0
        X_trainFold = []
        X_testFold = []
        y_trainFold = []
        y_testFold = []

        for train_index, test_index in kf.split(xTrain):
            X_train, X_test = xTrain[train_index], xTrain[test_index]
            y_train, y_test = yTrain[train_index], yTrain[test_index]

            X_trainFold.append(X_train)
            X_testFold.append(X_test)
            y_trainFold.append(y_train)
            y_testFold.append(y_test)

            ind_split += 1

        X_trainFold = np.asarray(X_trainFold)
        X_testFold = np.asarray(X_testFold)
        y_trainFold = np.asarray(y_trainFold)
        y_testFold = np.asarray(y_testFold)

        return [X_trainFold], [y_trainFold], [X_testFold], [y_testFold], [xTest], [yTest]


    elif sSplitting == 'PATIENT_CROSS_VALIDATION_SPLITTING':
        #unique_pats = len(allTestPats)

        X_trainFold = []
        X_testFold = []
        y_trainFold = []
        y_testFold = []

        for ind_split in allTestPats:
            train_index = np.where(allPats != ind_split)[0]
            test_index = np.where(allPats == ind_split)[0]
            X_train, X_test = allPatches[train_index], allPatches[test_index]
            y_train, y_test = allY[train_index], allY[test_index]

            X_trainFold.append(X_train)
            X_testFold.append(X_test)
            y_trainFold.append(y_train)
            y_testFold.append(y_test)

        if validationTrainRatio > 0:
            iAll = np.arange(len(X_trainFold))
            iSel = np.random.choice(len(X_trainFold), np.around(validationTrainRatio * len(X_trainFold)))
            X_valFold = np.asarray(X_trainFold[iSel], dtype='f')
            y_valFold = np.asarray(y_trainFold[iSel], dtype='f')
            iRem = np.delete(iAll, iSel)

            X_trainFold = np.asarray(X_trainFold[iRem], dtype='f')
            X_testFold = np.asarray(X_testFold, dtype='f')
            y_trainFold = np.asarray(y_trainFold[iRem], dtype='f')
            y_testFold = np.asarray(y_testFold, dtype='f')

        else:
            X_trainFold = np.asarray(X_trainFold, dtype='f')
            X_testFold = np.asarray(X_testFold, dtype='f')
            y_trainFold = np.asarray(y_trainFold, dtype='f')
            y_testFold = np.asarray(y_testFold, dtype='f')
            X_valFold = np.asarray([])
            y_valFold = np.asarray([])

        if iReturn > 0:
            return [X_trainFold], [y_trainFold], [X_valFold], [y_valFold], [X_testFold], [y_testFold]


def fSplitSegmentationDataset(allPatches, allY, allSegmentationMasks, allPats, allTestPats, sSplitting, patchSize, patchOverlap,
                              testTrainingDatasetRatio=0, validationTrainRatio=0, outPutPath=None, nfolds=0,
                              isRandomShuffle=True):
    # TODO: adapt path
    # iReturn = expecting()
    # iReturn = 1000
    iReturn = 1

    # 2D or 3D patching?
    if len(patchSize) == 2:
        # 2D patches are used
        if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1]:
            allPatches = np.transpose(allPatches, (2, 0, 1))
            allSegmentationMasks = np.transpose(allSegmentationMasks, (2, 0, 1))
    elif len(patchSize) == 3:
        # 3D patches are used
        if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1] and allPatches.shape[2] == \
                patchSize[2]:
            allPatches = np.transpose(allPatches, (3, 0, 1, 2))
            allSegmentationMasks = np.transpose(allSegmentationMasks, (3, 0, 1, 2))

    if sSplitting == 'SIMPLE_RANDOM_SAMPLE_SPLITTING':
        # splitting
        indexSlices = range(allPatches.shape[0])

        if isRandomShuffle:
            indexSlices = np.random.permutation(indexSlices)

        if len(patchSize) == 2:
            # 2D patching
            allPatches = allPatches[indexSlices, :, :]
            allSegmentationMasks = allSegmentationMasks[indexSlices, :, :]
        elif len(patchSize) == 3:
            # 3D patching
            allPatches = allPatches[indexSlices, :, :, :]
            allSegmentationMasks = allSegmentationMasks[indexSlices, :, :, :]

        shapeAllY = allY.shape

        if len(shapeAllY) > 1:
            if allY.shape[0] == patchSize[0] and allY.shape[1] == patchSize[1]:
                allY = np.transpose(allY, (2, 0, 1))

        allY = allY[indexSlices]

        # num of samples in test set and validation set
        numAllPatches = allPatches.shape[0]
        numSamplesTest = math.floor(testTrainingDatasetRatio * numAllPatches)
        numSamplesValidation = math.floor(validationTrainRatio * (numAllPatches - numSamplesTest))

        if len(patchSize) == 2:
            # 2D patching
            # subarrays as no-copy views (array slices)
            X_test = allPatches[:numSamplesTest, :, :]
            Y_segMasks_test = allSegmentationMasks[:numSamplesTest, :, :]

            X_valid = allPatches[numSamplesTest:(numSamplesTest + numSamplesValidation), :, :]
            Y_segMasks_valid = allSegmentationMasks[numSamplesTest:(numSamplesTest + numSamplesValidation), :, :]

            X_train = allPatches[(numSamplesTest + numSamplesValidation):, :, :]
            Y_segMasks_train = allSegmentationMasks[(numSamplesTest + numSamplesValidation):, :, :]


        elif len(patchSize) == 3:
            # 3D patching
            # subarrays as no-copy views (array slices)
            X_test = allPatches[:numSamplesTest, :, :, :]
            Y_segMasks_test = allSegmentationMasks[:numSamplesTest, :, :, :]

            X_valid = allPatches[numSamplesTest:(numSamplesTest + numSamplesValidation), :, :, :]
            Y_segMasks_valid = allSegmentationMasks[numSamplesTest:(numSamplesTest + numSamplesValidation), :, :, :]

            X_train = allPatches[(numSamplesTest + numSamplesValidation):, :, :, :]
            Y_segMasks_train = allSegmentationMasks[(numSamplesTest + numSamplesValidation):, :, :, :]

        y_test = allY[:numSamplesTest]
        y_valid = allY[numSamplesTest:(numSamplesTest + numSamplesValidation)]
        y_train = allY[(numSamplesTest + numSamplesValidation):]

        return [X_train], [y_train], [Y_segMasks_train], [X_valid], [y_valid], [Y_segMasks_valid], [X_test], [
            y_test], [Y_segMasks_test]  # embed in a 1-fold list


    elif sSplitting == 'CROSS_VALIDATION_SPLITTING':
        # split into test/train sets
        # shuffle
        indexSlices = range(allPatches.shape[0])
        indexSlices = np.random.permutation(indexSlices)

        allPatches = allPatches[indexSlices, :, :]
        allY = allY[indexSlices]

        # num of samples in test set
        numAllPatches = allPatches.shape[0]
        numSamplesTest = math.floor(testTrainingDatasetRatio * numAllPatches)

        # subarrays as no-copy views (array slices)
        xTest = allPatches[:numSamplesTest, :, :]
        yTest = allY[:numSamplesTest]

        xTrain = allPatches[numSamplesTest:, :, :]
        yTrain = allY[numSamplesTest:]

        # split training dataset into n folds
        if nfolds == 0:
            kf = KFold(n_splits=len(allTestPats))
        else:
            kf = KFold(n_splits=nfolds)

        ind_split = 0
        X_trainFold = []
        X_testFold = []
        y_trainFold = []
        y_testFold = []

        for train_index, test_index in kf.split(xTrain):
            X_train, X_test = xTrain[train_index], xTrain[test_index]
            y_train, y_test = yTrain[train_index], yTrain[test_index]

            X_trainFold.append(X_train)
            X_testFold.append(X_test)
            y_trainFold.append(y_train)
            y_testFold.append(y_test)

            ind_split += 1

        X_trainFold = np.asarray(X_trainFold)
        X_testFold = np.asarray(X_testFold)
        y_trainFold = np.asarray(y_trainFold)
        y_testFold = np.asarray(y_testFold)

        return [X_trainFold], [y_trainFold], [X_testFold], [y_testFold], [xTest], [yTest]

    elif sSplitting == 'PATIENT_CROSS_VALIDATION_SPLITTING':
        #unique_pats = len(allTestPats)

        #for ind_split in allTestPats:
        #    train_index = np.where(allPats != ind_split)[0]
        #    test_index = np.where(allPats == ind_split)[0]
        test_index = np.in1d(allPats, allTestPats)
        train_index = [not x for x in test_index]

        X_train = allPatches[train_index, :, :, :]
        X_test = allPatches[test_index, :, :, :]
        Y_segMasks_train = allSegmentationMasks[train_index, :, :, :]
        Y_segMasks_test = allSegmentationMasks[test_index, :, :, :]
        y_train, y_test = allY[train_index], allY[test_index]

        if validationTrainRatio > 0:
            iAll = np.arange(X_train.shape[0])
            iSel = np.random.choice(X_train.shape[0], np.around(validationTrainRatio * X_train.shape[0]))
            X_valFold = X_train[iSel, :, :, :]
            y_valFold = y_train[iSel]
            Y_segMasks_valFold = Y_segMasks_train[iSel, :, :, :]

            iRem = np.delete(iAll, iSel)
            X_trainFold = X_train[iRem, :, :, :]
            Y_segMasks_trainFold = Y_segMasks_train[iRem, :, :, :]
            y_trainFold = y_train[iRem]

        else:
            X_trainFold = X_train
            X_valFold = np.asarray([])
            Y_segMasks_trainFold = Y_segMasks_train
            Y_segMasks_valFold = np.asarray([])
            y_trainFold = y_train
            y_valFold = np.asarray([])

        if iReturn > 0:
            return [X_trainFold], [y_trainFold], [Y_segMasks_trainFold], [X_valFold], [y_valFold], [Y_segMasks_valFold], [X_test], [y_test], [Y_segMasks_test]


def TransformDataset(allPatches, allY, patchSize, patchOverlap, isRandomShuffle=True, isUsingSegmentation=False, allSegmentationMasks=None):

    if len(patchSize) == 2:
        # 2D patches are used
        if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1]:
            allPatches = np.transpose(allPatches, (2, 0, 1))
            if isUsingSegmentation:
                allSegmentationMasks = np.transpose(allSegmentationMasks, (2, 0, 1))
    elif len(patchSize) == 3:
        # 3D patches are used
        if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1] and allPatches.shape[2] == \
                patchSize[2]:
            allPatches = np.transpose(allPatches, (3, 0, 1, 2))
            if isUsingSegmentation:
                allSegmentationMasks = np.transpose(allSegmentationMasks, (3, 0, 1, 2))

    indexSlices = range(allPatches.shape[0])

    if isRandomShuffle:
        indexSlices = np.random.permutation(indexSlices)

    if len(patchSize) == 2:
        # 2D patching
        allPatches = allPatches[indexSlices, :, :]
        if isUsingSegmentation:
            allSegmentationMasks = allSegmentationMasks[indexSlices, :, :]
    elif len(patchSize) == 3:
        # 3D patching
        allPatches = allPatches[indexSlices, :, :, :]
        if isUsingSegmentation:
            allSegmentationMasks = allSegmentationMasks[indexSlices, :, :, :]

    shapeAllY = allY.shape
    if len(shapeAllY) > 1:
        if allY.shape[0] == patchSize[0] and allY.shape[1] == patchSize[1]:
            allY = np.transpose(allY, (2, 0, 1))

    allY = allY[indexSlices]

    X_data = allPatches
    if isUsingSegmentation:
        Y_segMasks_data = allSegmentationMasks

    y_data = allY

    if isUsingSegmentation:
        return [X_data], [y_data], [Y_segMasks_data]
    else:
        return [X_data], [y_data]
