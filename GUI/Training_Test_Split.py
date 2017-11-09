# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 15:59:36 2017

@author: Sebastian Milde
"""

import math
import numpy as np
import h5py
from sklearn.model_selection import KFold
import os

def fSplitDataset(allPatches, allY, sSplitting, patchSize, patchOverlap, split_ratio):
    sFolder = 'C:/Users/Sebastian Milde/Pictures/Universitaet/Masterarbeit/Data_train_test/'
    if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1]:
        allPatches = np.transpose(allPatches, (2, 0, 1))
        print(allPatches.shape)

    if sSplitting == "normal":
        print("Done")
        nPatches = allPatches.shape[0]
        dVal = math.floor(split_ratio * nPatches)
        rand_num = np.random.permutation(np.arange(nPatches))
        rand_num = rand_num[0:int(dVal)].astype(int)
        print(rand_num)

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

        folder = sFolder + 'normal/' + str(patchSize[0]) + str(patchSize[1])
        if os.path.isdir(folder):
            pass
        else:
            os.makedirs(folder)

        Path = sFolder + 'normal/' + str(patchSize[0]) + str(patchSize[1]) + '/normal_data' +  str(patchSize[0]) + str(patchSize[1]) +'.h5'
        print(Path)
        with h5py.File(Path, 'w') as hf:
            hf.create_dataset('X_train', data=X_train)
            hf.create_dataset('X_test', data=X_test)
            hf.create_dataset('y_train', data=y_train)
            hf.create_dataset('y_test', data=y_test)
            hf.create_dataset('patchSize', data=patchSize)
            hf.create_dataset('patchOverlap', data=patchOverlap)

    elif sSplitting == "crossvalidation_data":
        kf = KFold(n_splits = 15)
        ind_split = 0
        for train_index, test_index in kf.split(allPatches):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = allPatches[train_index], allPatches[test_index]
            y_train, y_test = allY[train_index], allY[test_index]
            print(X_train.shape, X_test.shape)
            print(y_train.shape, y_test.shape)
            folder = sFolder + 'crossvalidation_data/' + str(patchSize[0]) + str(patchSize[1])
            Path = sFolder + 'crossvalidation_data/' + str(patchSize[0]) + str(patchSize[1]) + '/crossVal_data' + str(ind_split) + '_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
            os.makedirs(folder)
            ind_split += 1
            with h5py.File(Path, 'w') as hf:
                hf.create_dataset('X_train', data=X_train)
                hf.create_dataset('X_test', data=X_test)
                hf.create_dataset('y_train', data=y_train)
                hf.create_dataset('y_test', data=y_test)
                hf.create_dataset('patchSize', data=patchSize)
                hf.create_dataset('patchOverlap', data=patchOverlap)

    elif sSplitting == "crossvalidation_patient":
        kf = KFold(n_splits=15)
        ind_split = 0
        for train_index, test_index in kf.split(allPatches):
            X_train, X_test = allPatches[train_index], allPatches[test_index]
            y_train, y_test = allY[train_index], allY[test_index]
            print(X_train.shape, X_test.shape)
            print(y_train.shape, y_test.shape)
            folder = sFolder + 'crossvalidation_patient/' + str(patchSize[0]) + str(patchSize[1])
            Path = sFolder + 'crossvalidation_patient/' + str(patchSize[0]) + str(patchSize[1]) + '/crossVal_data' + str(
                ind_split) + '_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
            os.makedirs(folder)
            ind_split += 1
            with h5py.File(Path, 'w') as hf:
                hf.create_dataset('X_train', data=X_train)
                hf.create_dataset('X_test', data=X_test)
                hf.create_dataset('y_train', data=y_train)
                hf.create_dataset('y_test', data=y_test)
                hf.create_dataset('patchSize', data=patchSize)
                hf.create_dataset('patchOverlap', data=patchOverlap)