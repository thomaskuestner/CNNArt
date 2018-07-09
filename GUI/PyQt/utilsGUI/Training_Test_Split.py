# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 15:59:36 2017

@author: Sebastian Milde, Thomas Kuestner
"""

import math
import numpy as np
import h5py
import inspect
import dis
from sklearn.model_selection import KFold
import os
from GUI.PyQt.DLArt_GUI import dlart
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

    # TODO: adapt path
    iReturn = expecting()

    # 2D or 3D patching?
    if len(patchSize) == 2:
        #2D patches are used
        if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1]:
            allPatches = np.transpose(allPatches, (2, 0, 1))
        #3D patches are used
        if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1] and allPatches.shape[2] == patchSize[2]:
            allPatches = np.transpose(allPatches, (3, 0, 1, 2))

    if sSplitting == dlart.DeepLearningArtApp.SIMPLE_RANDOM_SAMPLE_SPLITTING:
        # splitting
    else:
        if isRandomShuffle:
            indexSlices = np.random.permutation(indexSlices)

        if len(patchSize)==2:
            #2D patching
            allPatches = allPatches[indexSlices, :, :]
        elif len(patchSize)==3:
            #3D patching
            allPatches = allPatches[indexSlices, :, :, :]

        shapeAllY = allY.shape

        if len(shapeAllY) > 1:


        allY = allY[indexSlices]

        #num of samples in test set and validation set
        numSamplesTest = math.floor(testTrainingDatasetRatio*numAllPatches)
        numSamplesValidation = math.floor(validationTrainRatio*(numAllPatches-numSamplesTest))

        if len(patchSize) == 2:
            #2D patching
            # subarrays as no-copy views (array slices)
            X_test = allPatches[:numSamplesTest, :, :]
            X_valid = allPatches[numSamplesTest:(numSamplesTest+numSamplesValidation), :, :]
            X_train = allPatches[(numSamplesTest+numSamplesValidation):, :, :]
    if sSplitting == "normal":
        elif len(patchSize) == 3:
            # 3D patching
            # subarrays as no-copy views (array slices)
            X_test = allPatches[:numSamplesTest, :, :, :]
            X_valid = allPatches[numSamplesTest:(numSamplesTest + numSamplesValidation), :, :, :]
            X_train = allPatches[(numSamplesTest + numSamplesValidation):, :, :, :]
        print("Done")
        y_test = allY[:numSamplesTest]
        y_valid = allY[numSamplesTest:(numSamplesTest + numSamplesValidation)]
        y_train = allY[(numSamplesTest + numSamplesValidation):]

        # #random samples
        nPatches = allPatches.shape[0]
        dVal = math.floor(split_ratio * nPatches)
        rand_num = np.random.permutation(np.arange(nPatches))
        rand_num = rand_num[0:int(dVal)].astype(int)
        print(rand_num)
        if len(patchSize) == 3:
            X_test = allPatches[rand_num, :, :, :]
        else:
        # #do splitting
            X_test = allPatches[rand_num, :, :]
        y_test = allY[rand_num]
        X_train = allPatches
        X_train = np.delete(X_train, rand_num, axis=0)
        y_train = allY
        y_train = np.delete(y_train, rand_num)
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        # #!!!! train dataset is not randomly shuffeled!!!

        if iReturn == 0:
            if len(patchSize) == 3:
                folder = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + str(patchSize[2])
                Path = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + str(
            else:
                folder = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1])
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

    elif sSplitting == "crossvalidation_data":
        if nfolds == 0:
        else:
            kf = KFold(n_splits=nfolds)

        ind_split = 0
        X_trainFold = []
        X_testFold = []
        y_trainFold = []
        y_testFold = []


            if iReturn == 0:
                if len(patchSize) == 3:
                    folder = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + str(patchSize[2])
                    Path = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + str(
                        patchSize[1]) + str(patchSize[2]) + '.h5'
                else:
                    folder = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1])
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

            ind_split += 1

        X_trainFold = np.asarray(X_trainFold)
        X_testFold = np.asarray(X_testFold)
        y_trainFold = np.asarray(y_trainFold)
        y_testFold = np.asarray(y_testFold)

        if iReturn > 0:

    elif sSplitting == "crossvalidation_patient":

    elif sSplitting == dlart.DeepLearningArtApp.PATIENT_CROSS_VALIDATION_SPLITTING:

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
                        patchSize[1]) + str(patchSize[2]) + '.h5'
                else:
                    folder = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1])
                    Path = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + os.sep + 'crossVal' + str(
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

def fSplitDatasetCorrection(sSplitting, dRefPatches, dArtPatches, allPats, split_ratio, nfolds, test_index):
    """
    Split dataset with three options:
    1. normal: randomly split data according to the split_ratio without cross validation
def fSplitSegmentationDataset(allPatches, allY, allSegmentationMasks, allPats, sSplitting, patchSize, patchOverlap, testTrainingDatasetRatio=0, validationTrainRatio=0, outPutPath=None, nfolds = 0, isRandomShuffle=True):
    # TODO: adapt path
    iReturn = expecting()
    #iReturn = 1000

    # 2D or 3D patching?
    if len(patchSize) == 2:
        #2D patches are used
        if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1]:
            allPatches = np.transpose(allPatches, (2, 0, 1))
            allSegmentationMasks = np.transpose(allSegmentationMasks, (2, 0, 1))
    elif len(patchSize) == 3:
        #3D patches are used
        if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1] and allPatches.shape[2] == patchSize[2]:
            allPatches = np.transpose(allPatches, (3, 0, 1, 2))
            allSegmentationMasks = np.transpose(allSegmentationMasks, (3, 0, 1, 2))

    if sSplitting == dlart.DeepLearningArtApp.SIMPLE_RANDOM_SAMPLE_SPLITTING:
        # splitting
        indexSlices = range(allPatches.shape[0])

        if isRandomShuffle:
    3. crossvalidation_patient: perform crossvalidation with separate patient data
        if len(patchSize)==2:
            #2D patching
            allPatches = allPatches[indexSlices, :, :]
            allSegmentationMasks = allSegmentationMasks[indexSlices, :, :]
        elif len(patchSize)==3:
            #3D patching
            allPatches = allPatches[indexSlices, :, :, :]
            allSegmentationMasks = allSegmentationMasks[indexSlices, :, :, :]
    @param sSplitting: splitting mode 'normal', 'crossvalidation_data' or 'crossvalidation_patient'
        shapeAllY = allY.shape
    @param dRefPatches: reference patches
        if len(shapeAllY) > 1:
            if allY.shape[0] == patchSize[0] and allY.shape[1] == patchSize[1]:
                allY = np.transpose(allY, (2, 0, 1))
    @param dArtPatches: artifact patches
    @param allPats: patient index
        allY = allY[indexSlices]

    @param split_ratio: the ratio to split test data
        numAllPatches = allPatches.shape[0]
        numSamplesTest = math.floor(testTrainingDatasetRatio*numAllPatches)
        numSamplesValidation = math.floor(validationTrainRatio*(numAllPatches-numSamplesTest))
    @param nfolds: folds for cross validation
        if len(patchSize) == 2:
            #2D patching
            # subarrays as no-copy views (array slices)
            X_test = allPatches[:numSamplesTest, :, :]
            Y_segMasks_test = allSegmentationMasks[:numSamplesTest, :, :]
    @return: testing and training data for both reference and artifact images
            X_valid = allPatches[numSamplesTest:(numSamplesTest+numSamplesValidation), :, :]
            Y_segMasks_valid = allSegmentationMasks[numSamplesTest:(numSamplesTest+numSamplesValidation), :, :]
    """
            X_train = allPatches[(numSamplesTest+numSamplesValidation):, :, :]
            Y_segMasks_train = allSegmentationMasks[(numSamplesTest+numSamplesValidation):, :, :]
    train_ref_fold = []
    test_ref_fold = []
        elif len(patchSize) == 3:
            # 3D patching
            # subarrays as no-copy views (array slices)
            X_test = allPatches[:numSamplesTest, :, :, :]
            Y_segMasks_test = allSegmentationMasks[:numSamplesTest, :, :, :]
    train_art_fold = []
            X_valid = allPatches[numSamplesTest:(numSamplesTest + numSamplesValidation), :, :, :]
            Y_segMasks_valid = allSegmentationMasks[numSamplesTest:(numSamplesTest + numSamplesValidation), :, :, :]
    test_art_fold = []
            X_train = allPatches[(numSamplesTest + numSamplesValidation):, :, :, :]
            Y_segMasks_train = allSegmentationMasks[(numSamplesTest + numSamplesValidation):, :, :, :]
        y_test = allY[:numSamplesTest]
        y_valid = allY[numSamplesTest:(numSamplesTest + numSamplesValidation)]
        y_train = allY[(numSamplesTest + numSamplesValidation):]
    if sSplitting == 'normal':
        # #random samples
        nPatches = dRefPatches.shape[0]
        dVal = math.floor(split_ratio * nPatches)
        rand_num = np.random.permutation(np.arange(nPatches))
        rand_num = rand_num[0:int(dVal)].astype(int)
        #
        # #do splitting
        test_ref_fold.append(dRefPatches[rand_num, :, :])
        # y_test = allY[rand_num]
        # X_train = allPatches
        train_ref_fold.append(np.delete(dRefPatches, rand_num, axis=0))
        # y_train = allY
        test_art_fold.append(dArtPatches[rand_num, :, :])
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

            return [X_train], [y_train], [Y_segMasks_train], [X_valid], [y_valid], [Y_segMasks_valid], [X_test], [y_test], [Y_segMasks_test] # embed in a 1-fold list


    elif sSplitting == dlart.DeepLearningArtApp.CROSS_VALIDATION_SPLITTING:
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

    if sSplitting == "crossvalidation_data":
        if nfolds == 0:
        else:
            kf = KFold(n_splits=nfolds)

        #ind_split = 0
        X_trainFold = []
        X_testFold = []
        y_trainFold = []
        y_testFold = []


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
                X_testFold.append(X_test)

            #ind_split += 1

        X_trainFold = np.asarray(X_trainFold)
        y_trainFold = np.asarray(y_trainFold)
        y_testFold = np.asarray(y_testFold)
    elif sSplitting == 'crossvalidation_patient':
            return X_trainFold, y_trainFold, X_testFold, y_testFold, xTest, yTest



    elif sSplitting == dlart.DeepLearningArtApp.PATIENT_CROSS_VALIDATION_SPLITTING:

        X_trainFold = []
        y_trainFold = []
        y_testFold = []
            unique_pats = [test_index]
        for ind_split in unique_pats:
            train_index = np.where(allPats != ind_split)[0]
            test_index = np.where(allPats == ind_split)[0]

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
                y_trainFold.append(y_train)



        if iReturn > 0:
