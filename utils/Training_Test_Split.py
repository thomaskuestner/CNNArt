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

def fSplitDataset(allPatches, allY, allPats, sSplitting, patchSize, patchOverlap, split_ratio, sFolder, nfolds = 0):
    # TODO: adapt path
    iReturn = expecting()

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

        if iReturn == 0:
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
            return [X_train], [y_train], [X_test], [y_test] # embed in a 1-fold list

    elif sSplitting == "crossvalidation_data":
        if nfolds == 0:
            kf = KFold(n_splits=len(np.unique(allPats)))
        else:
            kf = KFold(n_splits=nfolds)
        ind_split = 0
        X_trainFold = []
        X_testFold = []
        y_trainFold = []
        y_testFold = []

        for train_index, test_index in kf.split(allPatches):
            X_train, X_test = allPatches[train_index], allPatches[test_index]
            y_train, y_test = allY[train_index], allY[test_index]

            if iReturn == 0:
                folder = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1])
                Path = sFolder + os.sep + str(patchSize[0]) + str(patchSize[1]) + os.sep + 'crossVal_data' + str(ind_split) + '_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
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
            return X_trainFold, y_trainFold, X_testFold, y_testFold

    elif sSplitting == "crossvalidation_patient":
        unique_pats = np.unique(allPats)

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