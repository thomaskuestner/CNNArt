
import numpy as np
from scipy import interpolate
import math
import time

def fscaling(X_train, X_test, scpatchSize, iscalefactor) :
    if len(scpatchSize) == 3:
        scX_train, scX_test = fscaling3D(X_train, X_test, scpatchSize, iscalefactor)
    else:
        scX_train, scX_test = fscaling2D(X_train, X_test, scpatchSize, iscalefactor)

    return scX_train, scX_test

def fscaling2D(X_train, X_test, scpatchSize, iscalefactor) :
    start = time.clock()
    afterSize = int(np.round(scpatchSize[0] * iscalefactor))

    # Prepare for the using of scipy.interpolation: create the coordinates of grid
    if iscalefactor == 1:
        return X_train, X_test
    else:
        xaxis = np.linspace(0, afterSize, scpatchSize[0])
        yaxis = np.linspace(0, afterSize, scpatchSize[1])

    dAllx_train = None
    dAllx_test = None

    for ifold in range(len(X_train)):
        lenTrain = X_train[ifold].shape[0]
        lenTest = X_test[ifold].shape[0]

        # in batches
        Batch=20
        dx_Train = None
        dx_Test = None
        stepTrain = int(math.ceil(lenTrain/Batch))
        stepTest = int(math.ceil(lenTest / Batch))

        for ibatch in range(Batch):
            indTrain = int(stepTrain*ibatch)
            if (indTrain+stepTrain) < lenTrain:
                inter_train0=np.mgrid[0:stepTrain, 0:afterSize, 0:afterSize]
                values_train = X_train[ifold][indTrain:(indTrain+stepTrain)]
                zaxis_train = np.arange(stepTrain)
            else:
                inter_train0 = np.mgrid[0:(lenTrain-indTrain), 0:afterSize, 0:afterSize]
                values_train = X_train[ifold][indTrain:lenTrain]
                zaxis_train = np.arange(lenTrain-indTrain)
            inter_train1=np.rollaxis(inter_train0, 0, 4)
            inter_train=np.reshape(inter_train1, [inter_train0.size//3, 3]) # 3 for the dimension of coordinates
            upedTrain=interpolate.interpn((zaxis_train, xaxis, yaxis), values_train, inter_train, method='linear',bounds_error=False, fill_value=0)
            if dx_Train is None:
                dx_Train = upedTrain
            else:
                dx_Train = np.concatenate((dx_Train,upedTrain), axis=0)

        dFoldx_train=np.reshape(dx_Train,[1, lenTrain, afterSize, afterSize])

        if dAllx_train is None:
            dAllx_train = dFoldx_train
        else:
            dAllx_train = np.concatenate((dAllx_train, dFoldx_train), axis=0)

        for ibatch in range(Batch):
            indTest = int(stepTest * ibatch)
            if (indTest + stepTest) < lenTest:
                inter_test0 = np.mgrid[0:stepTest, 0:afterSize, 0:afterSize]
                values_test = X_train[ifold][indTest:(indTest + stepTest)]
                zaxis_test = np.arange(stepTest)
            else:
                inter_test0 = np.mgrid[0:(lenTest - indTest), 0:afterSize, 0:afterSize]
                values_test = X_train[ifold][indTest:lenTest]
                zaxis_test = np.arange(lenTest - indTest)
            inter_test1=np.rollaxis(inter_test0, 0, 4)
            inter_test=np.reshape(inter_test1, [inter_test0.size//3, 3]) # 3 for the dimension of coordinates
            upedTest=interpolate.interpn((zaxis_test, xaxis, yaxis), values_test, inter_test, method='linear',bounds_error=False, fill_value=0)
            if dx_Test is None:
                dx_Test = upedTest
            else:
                dx_Test = np.concatenate((dx_Test,upedTest), axis=0)

        dFoldx_test = np.reshape(dx_Test, [1, lenTest, afterSize, afterSize])

        if dAllx_test is None:
            dAllx_test = dFoldx_test
        else:
            dAllx_test = np.concatenate((dAllx_test, dFoldx_test), axis=0)
    stop = time.clock()
    print(stop - start)
    return dAllx_train, dAllx_test


def fscaling3D(X_train, X_test, scpatchSize, iscalefactor):
    afterSize = np.rint(np.multiply(scpatchSize, iscalefactor)).astype(int)

    # Prepare for the using of scipy.interpolation: create the coordinates of grid
    if iscalefactor == 1:
        return X_train, X_test
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

    return dAllx_train, dAllx_test


