
import numpy as np
from scipy import interpolate
import math
import time

def fScaleOnePatch(dPatch, randPatchSize, PatchSize):
    xaxis = np.linspace(0, PatchSize[0], randPatchSize[0])
    yaxis = np.linspace(0, PatchSize[1], randPatchSize[1])
    zaxis = np.linspace(0, PatchSize[2], randPatchSize[2])
    inter_train0 = np.mgrid[0:PatchSize[0], 0:PatchSize[1], 0:PatchSize[2]]
    inter_train1 = np.rollaxis(inter_train0, 0, 4)
    inter_train = np.reshape(inter_train1, [inter_train0.size // 3, 3])
    scaleddPatch = interpolate.interpn((xaxis, yaxis, zaxis), dPatch, inter_train, method='linear', bounds_error=False, fill_value=0)
    reshdPatch = np.reshape(scaleddPatch, [PatchSize[0], PatchSize[1], PatchSize[2]])
    return reshdPatch

def fscaling(X_train, X_test, scpatchSize, iscalefactor) :
    if len(scpatchSize) == 3:
        scX_train, scX_test, afterSize = fscaling3D(X_train, X_test, scpatchSize, iscalefactor)
    else:
        scX_train, scX_test, afterSize = fscaling2D(X_train, X_test, scpatchSize, iscalefactor)

    return scX_train, scX_test, afterSize

def fscaling2D(X_train, X_test, scpatchSize, iscalefactor) :
    start = time.clock()
    afterSize = np.round(np.multiply(scpatchSize, iscalefactor)).astype(int)

    # Prepare for the using of scipy.interpolation: create the coordinates of grid
    if iscalefactor == 1:
        return X_train, X_test, scpatchSize
    else:
        xaxis = np.linspace(0, afterSize[0], scpatchSize[0])
        yaxis = np.linspace(0, afterSize[1], scpatchSize[1])

    dAllx_train = None
    dAllx_test = None

    for ifold in range(len(X_train)):
        lenTrain = X_train[ifold].shape[0]
        lenTest = X_test[ifold].shape[0]

        # in batches
        BatchTrain = BatchTest = 20
        for icand in range(15,26):
            if lenTrain % icand == 0:
                BatchTrain = icand
            if lenTest % icand == 0:
                BatchTest = icand
        dx_Train = None
        dx_Test = None
        stepTrain = -((0 - lenTrain) // BatchTrain)
        stepTest = -((0 - lenTest) // BatchTest)

        for ibatch in range(BatchTrain):
            indTrain = int(stepTrain*ibatch)
            if (indTrain+stepTrain) < lenTrain:
                inter_train0=np.mgrid[0:stepTrain, 0:afterSize[0], 0:afterSize[1]]
                values_train = X_train[ifold][indTrain:(indTrain+stepTrain)]
                zaxis_train = np.arange(stepTrain)
            else:
                inter_train0 = np.mgrid[0:(lenTrain-indTrain), 0:afterSize[0], 0:afterSize[1]]
                values_train = X_train[ifold][indTrain:lenTrain]
                zaxis_train = np.arange(lenTrain-indTrain)
            inter_train1=np.rollaxis(inter_train0, 0, 4)
            inter_train=np.reshape(inter_train1, [inter_train0.size//3, 3]) # 3 for the dimension of coordinates
            upedTrain=interpolate.interpn((zaxis_train, xaxis, yaxis), values_train, inter_train, method='linear',bounds_error=False, fill_value=0)
            if dx_Train is None:
                dx_Train = upedTrain
            else:
                dx_Train = np.concatenate((dx_Train,upedTrain), axis=0)

        dFoldx_train=np.reshape(dx_Train,[1, lenTrain, afterSize[0], afterSize[1]])

        if dAllx_train is None:
            dAllx_train = dFoldx_train
        else:
            dAllx_train = np.concatenate((dAllx_train, dFoldx_train), axis=0)

        for ibatch in range(BatchTest):
            indTest = int(stepTest * ibatch)
            if (indTest + stepTest) < lenTest:
                inter_test0 = np.mgrid[0:stepTest, 0:afterSize[0], 0:afterSize[1]]
                values_test = X_train[ifold][indTest:(indTest + stepTest)]
                zaxis_test = np.arange(stepTest)
            else:
                inter_test0 = np.mgrid[0:(lenTest - indTest), 0:afterSize[0], 0:afterSize[1]]
                values_test = X_train[ifold][indTest:lenTest]
                zaxis_test = np.arange(lenTest - indTest)
            inter_test1=np.rollaxis(inter_test0, 0, 4)
            inter_test=np.reshape(inter_test1, [inter_test0.size//3, 3]) # 3 for the dimension of coordinates
            upedTest=interpolate.interpn((zaxis_test, xaxis, yaxis), values_test, inter_test, method='linear',bounds_error=False, fill_value=0)
            if dx_Test is None:
                dx_Test = upedTest
            else:
                dx_Test = np.concatenate((dx_Test,upedTest), axis=0)

        dFoldx_test = np.reshape(dx_Test, [1, lenTest, afterSize[0], afterSize[1]])

        if dAllx_test is None:
            dAllx_test = dFoldx_test
        else:
            dAllx_test = np.concatenate((dAllx_test, dFoldx_test), axis=0)
    stop = time.clock()
    print(stop - start)
    return dAllx_train, dAllx_test, afterSize


def fscaling3D(X_train, X_test, scpatchSize, iscalefactor):
    afterSize = np.ceil(np.multiply(scpatchSize, iscalefactor)).astype(int)

    # Prepare for the using of scipy.interpolation: create the coordinates of grid
    if iscalefactor == 1:
        return X_train, X_test, scpatchSize
    else:
        xaxis = np.linspace(0, afterSize[0], scpatchSize[0])
        yaxis = np.linspace(0, afterSize[1], scpatchSize[1])
        zaxis = np.linspace(0, afterSize[2], scpatchSize[2])

    dAllx_train = None
    dAllx_test = None

    for ifold in range(len(X_train)):
        lenTrain = X_train[ifold].shape[0]
        lenTest = X_test[ifold].shape[0]

        start = time.clock()

        # no batch
        inter_train0 = np.mgrid[0:lenTrain, 0:afterSize[0], 0:afterSize[1], 0:afterSize[2]]
        inter_train1 = np.rollaxis(inter_train0, 0, 5)
        inter_train = np.reshape(inter_train1, [inter_train0.size // 4, 4])  # 4 for the dimension of coordinates

        zaxis_train = np.arange(lenTrain)

        upedTrain = interpolate.interpn((zaxis_train, xaxis, yaxis, zaxis),
                                        X_train[ifold],
                                        inter_train, method='linear', bounds_error=False, fill_value=0)
        dFoldx_train = np.reshape(upedTrain, [1, lenTrain, afterSize[0], afterSize[1], afterSize[2]])


        inter_test0 = np.mgrid[0:lenTest, 0:afterSize[0], 0:afterSize[1], 0:afterSize[2]]
        inter_test1 = np.rollaxis(inter_test0, 0, 5)
        inter_test = np.reshape(inter_test1, [inter_test0.size // 4, 4])  # 4 for the dimension of coordinates

        zaxis_test = np.arange(lenTest)

        upedTest = interpolate.interpn((zaxis_test, xaxis, yaxis, zaxis),
                                       X_test[ifold],
                                       inter_test, method='linear', bounds_error=False, fill_value=0)
        dFoldx_test = np.reshape(upedTest, [1, lenTest, afterSize[0], afterSize[1], afterSize[2]])

        stop = time.clock()
        print(stop-start)

        if dAllx_train is None:
            dAllx_train = dFoldx_train
        else:
            dAllx_train = np.concatenate((dAllx_train, dFoldx_train), axis=0)

        if dAllx_test is None:
            dAllx_test = dFoldx_test
        else:
            dAllx_test = np.concatenate((dAllx_test, dFoldx_test), axis=0)

    return dAllx_train, dAllx_test, afterSize

def fcutMiddelPartOfPatch(X_train_sp, X_test_sp, scpatchSize, patchSize):
    cropStart = [(scpatchSize[idim]-patchSize[idim])//2 for idim in range(len(patchSize))]
    if np.array(X_train_sp).ndim == 4:
        if len(patchSize) == 2:
            X_train = np.array(X_train_sp)[:, cropStart[0]:cropStart[0] + patchSize[0], cropStart[1]:cropStart[1] + patchSize[1]]
        else:
            X_train = np.array(X_train_sp)[:, cropStart[0]:cropStart[0] + patchSize[0], cropStart[1]:cropStart[1] + patchSize[1], cropStart[2]:cropStart[2] + patchSize[2]]
        return X_train
    else:
        if len(patchSize) == 2:
            X_train = np.array(X_train_sp)[:, :, cropStart[0]:cropStart[0] + patchSize[0], cropStart[1]:cropStart[1] + patchSize[1]]
            X_test = np.array(X_test_sp)[:, :, cropStart[0]:cropStart[0] + patchSize[0], cropStart[1]:cropStart[1] + patchSize[1]]
        else:
            X_train = np.array(X_train_sp)[:, :, cropStart[0]:cropStart[0] + patchSize[0], cropStart[1]:cropStart[1] + patchSize[1], cropStart[2]:cropStart[2] + patchSize[2]]
            X_test = np.array(X_test_sp)[:, :, cropStart[0]:cropStart[0] + patchSize[0], cropStart[1]:cropStart[1] + patchSize[1], cropStart[2]:cropStart[2] + patchSize[2]]
        return X_train, X_test

def fcutMiddelPartOfOnePatch(Patch, fromPatchSize, toPatchSize):
    cropStart = [(fromPatchSize[idim]-toPatchSize[idim])//2 for idim in range(len(toPatchSize))]
    if len(toPatchSize) == 2:
        toPatch = np.array(Patch)[cropStart[0]:cropStart[0] + toPatchSize[0], cropStart[1]:cropStart[1] + toPatchSize[1]]
    else:
        toPatch = np.array(Patch)[cropStart[0]:cropStart[0] + toPatchSize[0], cropStart[1]:cropStart[1] + toPatchSize[1], cropStart[2]:cropStart[2] + toPatchSize[2]]
    return toPatch
