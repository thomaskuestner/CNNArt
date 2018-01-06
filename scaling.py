
import h5py
import numpy as np
from scipy import interpolate
import cnn_main
import time

    # Prepare for the using of scipy.interpolation: create the coordinates of grid
        xaxis = np.arange(0, afterSize, 2)
        yaxis = np.arange(0, afterSize, 2)
    else:
        xaxis = np.arange(0, afterSize, 0.5)
        yaxis = np.arange(0, afterSize, 0.5)
    sDatafile = sDatafile[:-3] + 'to' + ''.join(str(afterSize)).replace(" ", "") + ''.join(str(afterSize)).replace(" ", "") + '.h5'
    dAllx_train = None
    dAllx_test = None

    for ifold in range(len(X_train)):
        lenTrain = X_train[ifold].shape[0]
        lenTest = X_test[ifold].shape[0]
        zaxis_train = np.arange(lenTrain)
        zaxis_test = np.arange(lenTest)

        Batch=20
        dx_Train = None
        for ibatch in range(Batch):
            inter_train0=np.mgrid[lenTrain//Batch*ibatch:lenTrain//Batch*(ibatch+1), 0:afterSize, 0:afterSize]
            inter_train1=np.rollaxis(inter_train0, 0, 4)
            inter_train=np.reshape(inter_train1, [inter_train0.size//3, 3]) # 3 for the dimension of coordinates

            zaxis_train=np.arange(lenTrain//Batch)

            upedTrain=interpolate.interpn((zaxis_train, xaxis, yaxis), X_train[0][lenTrain//Batch*ibatch:lenTrain//Batch*(ibatch+1)], inter_train, method='linear',bounds_error=False, fill_value=0)
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

            upedTest=interpolate.interpn((zaxis_test, xaxis, yaxis), X_test[0][lenTest//Batch*ibatch:lenTest//Batch*(ibatch+1)], inter_test, method='linear',bounds_error=False, fill_value=0)
            if dx_Test is None:
                dx_Test = upedTest
            else:
                dx_Test = np.concatenate((dx_Test,upedTest), axis=0)

        dFoldx_test = np.reshape(dx_Test, [1, lenTest, afterSize, afterSize])

        if dAllx_test is None:
            dAllx_test = dFoldx_test
        else:
            dAllx_test = np.concatenate((dAllx_test, dFoldx_test), axis=0)

    patchSize = [afterSize, afterSize]
    return dAllx_train, dAllx_test, patchSize

    
    X_train=forig['X_train']
    X_test=forig['X_test']
    y_train=forig['y_train']
    y_test=forig['y_test']
    patchSize=forig['patchSize']
    patchOverlap=forig['patchOverlap']
    
    NX_train, NX_test, NewpatchSize = fscalling(X_train, X_test, patchSize)
    OrigPatchSize = patchSize
    patchSize = NewpatchSize
    
        hf.create_dataset('X_train', data= NX_train)
        hf.create_dataset('X_test', data= NX_test)
        hf.create_dataset('y_train', data=y_train)
        hf.create_dataset('y_test', data=y_test)
        hf.create_dataset('patchSize', data=patchSize)
        hf.create_dataset('patchOverlap', data=patchOverlap)
        hf.create_dataset('OrigPatchSize', data=OrigPatchSize)


