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

def fSplitDataset(resultFolder, proband_list, model_list, allPatches, allY, sSplitting, patchSize, patchOverlap, split_ratio):
    if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1]:
        allPatches = np.transpose(allPatches, (2, 0, 1))

    if sSplitting == "normal_rand":
        nPatches = allPatches.shape[0]
        dVal = math.floor(split_ratio * nPatches)
        rand_num = np.random.permutation(np.arange(nPatches))
        rand_num_test = rand_num[0:int(dVal)].astype(int)
        rand_num_train = rand_num[int(dVal):nPatches].astype(int)

        X_test = allPatches[rand_num_test, :, :]
        y_test = allY[rand_num_test]
        X_train = allPatches
        X_train = np.delete(X_train, rand_num_test, axis=0)
        y_train = allY
        y_train = np.delete(y_train, rand_num_test)

        folder = resultFolder + 'normal/' + str(patchSize[0]) + str(patchSize[1])
        if os.path.isdir(folder):
            pass
        else:
            os.makedirs(folder)

        Path = resultFolder + 'normal/' + str(patchSize[0]) + str(patchSize[1]) + '/Becken10_normal_data' +  str(patchSize[0]) + str(patchSize[1]) +'.h5'
        print(Path)
        with h5py.File(Path, 'w') as hf:
            hf.create_dataset('X_train', data=X_train)
            hf.create_dataset('X_test', data=X_test)
            hf.create_dataset('y_train', data=y_train)
            hf.create_dataset('y_test', data=y_test)
            hf.create_dataset('patchSize', data=patchSize)
            hf.create_dataset('patchOverlap', data=patchOverlap)
            hf.create_dataset('test_index', data=rand_num_test)
            hf.create_dataset('train_index', data=rand_num_train)
            hf.create_dataset('proband_list', data=proband_list)
            hf.create_dataset('model_list', data=model_list)

    elif sSplitting == "normal":
        nPatches = allPatches.shape[0]
        print(allPatches.shape)
        dVal = 18720#math.floor(split_ratio * nPatches)
        num_ind = np.arange(nPatches)
        num_ind_test = num_ind[0:int(dVal)].astype(int)
        num_ind_train = num_ind[int(dVal):nPatches].astype(int)

        X_test = allPatches[num_ind_test, :, :]
        y_test = allY[num_ind_test]
        X_train = allPatches
        X_train = np.delete(X_train, num_ind_test, axis=0)
        y_train = allY
        y_train = np.delete(y_train, num_ind_test)

        folder = resultFolder + 'normal/' + str(patchSize[0]) + str(patchSize[1])
        if os.path.isdir(folder):
            pass
        else:
            os.makedirs(folder)

        Path = resultFolder + 'normal/AllData_Move_05_label05_val_ab' +  str(patchSize[0]) + str(patchSize[1]) +'.h5'
        print(Path)
        with h5py.File(Path, 'w') as hf:
            hf.create_dataset('X_train', data=X_train)
            hf.create_dataset('X_test', data=X_test)
            hf.create_dataset('y_train', data=y_train)
            hf.create_dataset('y_test', data=y_test)
            hf.create_dataset('patchSize', data=patchSize)
            hf.create_dataset('patchOverlap', data=patchOverlap)
            hf.create_dataset('test_index', data=num_ind_test)
            hf.create_dataset('train_index', data=num_ind_train)
            hf.create_dataset('proband_list', data=proband_list)
            hf.create_dataset('model_list', data=model_list)

    elif sSplitting == "crossvalidation_data":
        kf = KFold(n_splits = 15)
        ind_split = 0
        for train_index, test_index in kf.split(allPatches):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = allPatches[train_index], allPatches[test_index]
            y_train, y_test = allY[train_index], allY[test_index]
            print(X_train.shape, X_test.shape)
            print(y_train.shape, y_test.shape)
            folder = resultFolder + 'crossvalidation_data/' + str(patchSize[0]) + str(patchSize[1])
            Path = resultFolder + 'crossvalidation_data/' + str(patchSize[0]) + str(patchSize[1]) + '/crossVal_data' + str(ind_split) + '_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
            os.makedirs(folder)
            ind_split += 1
            with h5py.File(Path, 'w') as hf:
                hf.create_dataset('X_train', data=X_train)
                hf.create_dataset('X_test', data=X_test)
                hf.create_dataset('y_train', data=y_train)
                hf.create_dataset('y_test', data=y_test)
                hf.create_dataset('patchSize', data=patchSize)
                hf.create_dataset('patchOverlap', data=patchOverlap)
                hf.create_dataset('test_index', data=test_index)
                hf.create_dataset('train_index', data=train_index)
                hf.create_dataset('proband_list', data=proband_list)
                hf.create_dataset('model_list', data=model_list)

    elif sSplitting == "crossvalidation_patient":
        kf = KFold(n_splits=15)
        ind_split = 0
        for train_index, test_index in kf.split(allPatches):
            X_train, X_test = allPatches[train_index], allPatches[test_index]
            y_train, y_test = allY[train_index], allY[test_index]
            print(X_train.shape, X_test.shape)
            print(y_train.shape, y_test.shape)
            folder = resultFolder + 'crossvalidation_patient/' + str(patchSize[0]) + str(patchSize[1])
            Path = resultFolder + 'crossvalidation_patient/' + str(patchSize[0]) + str(patchSize[1]) + '/crossVal_data' + str(
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
                hf.create_dataset('proband_list', data=proband_list)
                hf.create_dataset('model_list', data=model_list)

def fSplitDataset3D(resultFolder, proband_list, model_list, allPatches, allY, sSplitting, patchSize, patchOverlap, split_ratio):
    if allPatches.shape[0] == patchSize[0] and allPatches.shape[1] == patchSize[1]:
        allPatches = np.transpose(allPatches, (3, 0, 1, 2))

    if sSplitting == "normal_rand":
        nPatches = allPatches.shape[0]
        dVal = math.floor(split_ratio * nPatches)
        rand_num = np.random.permutation(np.arange(nPatches))
        rand_num_test = rand_num[0:int(dVal)].astype(int)
        rand_num_train = rand_num[int(dVal):nPatches].astype(int)

        X_test = allPatches[rand_num_test, :, :, :]
        y_test = allY[rand_num_test]
        X_train = allPatches
        X_train = np.delete(X_train, rand_num_test, axis=0)
        y_train = allY
        y_train = np.delete(y_train, rand_num_test)

        folder = resultFolder + 'normal/' + str(patchSize[0]) + str(patchSize[1])
        if os.path.isdir(folder):
            pass
        else:
            os.makedirs(folder)

        Path = resultFolder + 'normal/' + str(patchSize[0]) + str(patchSize[1]) + '/normal_data' +  str(patchSize[0]) + str(patchSize[1]) +'.h5'
        print(Path)
        with h5py.File(Path, 'w') as hf:
            hf.create_dataset('X_train', data=X_train)
            hf.create_dataset('X_test', data=X_test)
            hf.create_dataset('y_train', data=y_train)
            hf.create_dataset('y_test', data=y_test)
            hf.create_dataset('patchSize', data=patchSize)
            hf.create_dataset('patchOverlap', data=patchOverlap)
            hf.create_dataset('rand_num_test', data=rand_num_test)
            hf.create_dataset('rand_num_train', data=rand_num_train)
            hf.create_dataset('proband_list', data=proband_list)
            hf.create_dataset('model_list', data=model_list)

    elif sSplitting == "normal":
        nPatches = allPatches.shape[0]
        print(allPatches.shape)
        dVal = 1980#math.floor(split_ratio * nPatches)
        num_ind = np.arange(nPatches)
        num_ind_test = num_ind[0:dVal].astype(int)
        num_ind_train = num_ind[int(dVal):nPatches].astype(int)

        X_test = allPatches[num_ind_test, :, :, :]
        y_test = allY[num_ind_test]
        X_train = allPatches
        X_train = np.delete(X_train, num_ind_test, axis=0)
        y_train = allY
        y_train = np.delete(y_train, num_ind_test)

        folder = resultFolder + 'normal/' + str(patchSize[0]) + str(patchSize[1])
        if os.path.isdir(folder):
            pass
        else:
            os.makedirs(folder)

        Path = resultFolder + 'normal_3D/Beckent2_Move_05_label05_val_ab_test_ma_' +  str(patchSize[0]) + str(patchSize[1]) +'3D.h5'
        print(Path)
        with h5py.File(Path, 'w') as hf:
            hf.create_dataset('X_train', data=X_train)
            hf.create_dataset('X_test', data=X_test)
            hf.create_dataset('y_train', data=y_train)
            hf.create_dataset('y_test', data=y_test)
            hf.create_dataset('patchSize', data=patchSize)
            hf.create_dataset('patchOverlap', data=patchOverlap)
            hf.create_dataset('test_index', data=num_ind_test)
            hf.create_dataset('train_index', data=num_ind_train)
            hf.create_dataset('proband_list', data=proband_list)
            hf.create_dataset('model_list', data=model_list)


    elif sSplitting == "crossvalidation_data":
        kf = KFold(n_splits = 15)
        ind_split = 0
        for train_index, test_index in kf.split(allPatches):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = allPatches[train_index], allPatches[test_index]
            y_train, y_test = allY[train_index], allY[test_index]
            print(X_train.shape, X_test.shape)
            print(y_train.shape, y_test.shape)
            folder = resultFolder + 'crossvalidation_data/' + str(patchSize[0]) + str(patchSize[1])
            Path = resultFolder + 'crossvalidation_data/' + str(patchSize[0]) + str(patchSize[1]) + '/crossVal_data' + str(ind_split) + '_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
            if os.path.isdir(folder):
                pass
            else:
                os.makedirs(folder)
            ind_split += 1
            with h5py.File(Path, 'w') as hf:
                hf.create_dataset('X_train', data=X_train)
                hf.create_dataset('X_test', data=X_test)
                hf.create_dataset('y_train', data=y_train)
                hf.create_dataset('y_test', data=y_test)
                hf.create_dataset('patchSize', data=patchSize)
                hf.create_dataset('patchOverlap', data=patchOverlap)
                hf.create_dataset('proband_list', data=proband_list)
                hf.create_dataset('model_list', data=model_list)

    elif sSplitting == "crossvalidation_patient":
        kf = KFold(n_splits=15)
        ind_split = 0
        for train_index, test_index in kf.split(allPatches):
            X_train, X_test = allPatches[train_index], allPatches[test_index]
            y_train, y_test = allY[train_index], allY[test_index]
            print(X_train.shape, X_test.shape)
            print(y_train.shape, y_test.shape)
            folder = resultFolder + 'crossvalidation_patient/' + str(patchSize[0]) + str(patchSize[1])
            Path = resultFolder + 'crossvalidation_patient/' + str(patchSize[0]) + str(patchSize[1]) + '/crossVal_data' + str(
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
                hf.create_dataset('proband_list', data=proband_list)
                hf.create_dataset('model_list', data=model_list)