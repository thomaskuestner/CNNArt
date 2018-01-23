import h5py
import numpy as np
from scipy import interpolate
import time

def fscaling(X_train, X_test, scpatchSize, iscalefactor) :
    if len(scpatchSize) == 3:
        scX_train, scX_test = fscaling3D(X_train, X_test, scpatchSize, iscalefactor)
    else:
        scX_train, scX_test = fscaling2D(X_train, X_test, scpatchSize, iscalefactor)

    return scX_train, scX_test

def fscaling2D(X_train, X_test, scpatchSize, iscalefactor) :

    afterSize = int(scpatchSize[0] * iscalefactor)

    # Prepare for the using of scipy.interpolation: create the coordinates of grid
    if iscalefactor == 1:
        return X_train, X_test
    else:
        xaxis = np.arange(0, afterSize, iscalefactor)
        yaxis = np.arange(0, afterSize, iscalefactor)

    dAllx_train = None
    dAllx_test = None

    for ifold in range(len(X_train)):
        lenTrain = X_train[ifold].shape[0]
        lenTest = X_test[ifold].shape[0]
        zaxis_train = np.arange(lenTrain)
        zaxis_test = np.arange(lenTest)

        # in batches
        Batch=20
        dx_Train = None
        for ibatch in range(Batch):
            inter_train0=np.mgrid[lenTrain//Batch*ibatch:lenTrain//Batch*(ibatch+1), 0:afterSize, 0:afterSize]
            inter_train1=np.rollaxis(inter_train0, 0, 4)
            inter_train=np.reshape(inter_train1, [inter_train0.size//3, 3]) # 3 for the dimension of coordinates

            zaxis_train=np.arange(lenTrain//Batch)

            upedTrain=interpolate.interpn((zaxis_train, xaxis, yaxis), X_train[ifold][lenTrain//Batch*ibatch:lenTrain//Batch*(ibatch+1)], inter_train, method='linear',bounds_error=False, fill_value=0)
            if dx_Train is None:
                dx_Train = upedTrain
            else:
                dx_Train = np.concatenate((dx_Train,upedTrain), axis=0)

        dFoldx_train=np.reshape(dx_Train,[1, lenTrain, afterSize, afterSize])

        if dAllx_train is None:
            dAllx_train = dFoldx_train
        else:
            dAllx_train = np.concatenate((dAllx_train, dFoldx_train), axis=0)


        dx_Test = None
        for ibatch in range(Batch):

            inter_test0=np.mgrid[lenTest//Batch*ibatch:lenTest//Batch*(ibatch+1), 0:afterSize, 0:afterSize]
            inter_test1=np.rollaxis(inter_test0, 0, 4)
            inter_test=np.reshape(inter_test1, [inter_test0.size//3, 3]) # 3 for the dimension of coordinates

            zaxis_test=np.arange(lenTest//Batch)
            
            upedTest=interpolate.interpn((zaxis_test, xaxis, yaxis), X_test[ifold][lenTest//Batch*ibatch:lenTest//Batch*(ibatch+1)], inter_test, method='linear',bounds_error=False, fill_value=0)
            if dx_Test is None:
                dx_Test = upedTest
            else:
                dx_Test = np.concatenate((dx_Test,upedTest), axis=0)

        dFoldx_test = np.reshape(dx_Test, [1, lenTest, afterSize, afterSize])

        if dAllx_test is None:
            dAllx_test = dFoldx_test
        else:
            dAllx_test = np.concatenate((dAllx_test, dFoldx_test), axis=0)

    return dAllx_train, dAllx_test


def fscaling3D(X_train, X_test, scpatchSize, iscalefactor):
    afterSize = np.multiply(scpatchSize, iscalefactor).astype(int)

    # Prepare for the using of scipy.interpolation: create the coordinates of grid
    if iscalefactor == 1:
        return X_train, X_test
    else:
        xaxis = np.arange(0, afterSize[0], iscalefactor)
        yaxis = np.arange(0, afterSize[1], iscalefactor)
        zaxis = np.arange(0, afterSize[2], iscalefactor)

    dAllx_train = None
    dAllx_test = None

    for ifold in range(len(X_train)):
        lenTrain = X_train[ifold].shape[0]
        lenTest = X_test[ifold].shape[0]
        zaxis_train = np.arange(lenTrain)
        zaxis_test = np.arange(lenTest)

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


        # # in batches
        # Batch = 2
        # dx_Train = None
        # for ibatch in range(Batch):
        #     inter_train0 = np.mgrid[lenTrain // Batch * ibatch:lenTrain // Batch * (ibatch + 1), 0:afterSize[0], 0:afterSize[1], 0:afterSize[2]]
        #     inter_train1 = np.rollaxis(inter_train0, 0, 5)
        #     inter_train = np.reshape(inter_train1, [inter_train0.size // 4, 4])  # 4 for the dimension of coordinates
        #
        #     zaxis_train = np.arange(lenTrain // Batch)
        #
        #     upedTrain = interpolate.interpn((zaxis_train, xaxis, yaxis, zaxis),X_train[ifold][lenTrain // Batch * ibatch:lenTrain // Batch * (ibatch + 1)],
        #                                     inter_train, method='linear', bounds_error=False, fill_value=0)
        #     if dx_Train is None:
        #         dx_Train = upedTrain
        #     else:
        #         dx_Train = np.concatenate((dx_Train, upedTrain), axis=0)
        #
        # dFoldx_train = np.reshape(dx_Train, [1, lenTrain, afterSize[0], afterSize[1], afterSize[2]])
        #
        #
        # dx_Test = None
        # for ibatch in range(Batch):
        #
        #     inter_test0 = np.mgrid[lenTest // Batch * ibatch:lenTest // Batch * (ibatch + 1), 0:afterSize[0], 0:afterSize[1], 0:afterSize[2]]
        #     inter_test1 = np.rollaxis(inter_test0, 0, 5)
        #     inter_test = np.reshape(inter_test1, [inter_test0.size // 4, 4])  # 4 for the dimension of coordinates
        #
        #     zaxis_test = np.arange(lenTest // Batch)
        #
        #     upedTest = interpolate.interpn((zaxis_test, xaxis, yaxis, zaxis),
        #                                    X_test[ifold][lenTest // Batch * ibatch:lenTest // Batch * (ibatch + 1)],
        #                                    inter_test, method='linear', bounds_error=False, fill_value=0)
        #     if dx_Test is None:
        #         dx_Test = upedTest
        #     else:
        #         dx_Test = np.concatenate((dx_Test, upedTest), axis=0)
        #
        # dFoldx_test = np.reshape(dx_Test, [1, lenTest, afterSize[0], afterSize[1], afterSize[2]])

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


