import h5py
import numpy as np
from scipy import interpolate

def fscaling(X_train, X_test, scpatchSize, iscalefactor) :

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
        # Up scaling the training set
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


        # Up scaling the test set
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



