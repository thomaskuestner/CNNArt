# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:38:33 2018

@author: cyf
"""
import sys
import os.path
import scipy.io as sio
import numpy as np
# networks
from networks.motion.CNN2D import *
from networks.motion.CNN3D import *
from networks.motion.MNetArt import *
from networks.motion.VNetArt import *
from networks.multiclass.DenseResNet import *
from networks.multiclass.InceptionNet import *
from correction.networks.motion import *

def fcalculateInputOfPath2(patchSize, scaleFactor, sModelIn):

    # check model
    if 'motion' in sModelIn:
        if 'CNN2D' in sModelIn:
            sModel = 'networks.motion.CNN2D.' + sModelIn
        elif 'motion_CNN3D' in sModelIn:
            sModel = 'networks.motion.CNN3D.' + sModelIn
        elif 'motion_MNetArt' in sModelIn:
            sModel = 'networks.motion.MNetArt.' + sModelIn
        elif 'motion_VNetArt' in sModelIn:
            sModel = 'networks.motion.VNetArt.' + sModelIn
    elif 'multi' in sModelIn:
        if 'multi_DenseResNet' in sModelIn:
            sModel = 'networks.multiclass.DenseResNet.' + sModelIn
        elif 'multi_InceptionNet' in sModelIn:
            sModel = 'networks.multiclass.InceptionNet.' + sModelIn
    else:
        sys.exit("Model is not supported")
        
    cnnModel = __import__(sModel, globals(), locals(), ['createModel', 'fTrain', 'fPredict'], 0)
    Kernels = cnnModel.fgetKernels()
    Strides = cnnModel.fgetStrides()
    layerNum = cnnModel.fgetLayerNum()
    FMperLayerPath1 = [[0]*len(patchSize) for i in range(layerNum)]
    FMperLayerPath2 = [[0]*len(patchSize) for i in range(layerNum)]
    InputOfPath2 = [0 for i in range(len(patchSize))]
    patchSize_p2 = [0 for i in range(len(patchSize))]
    for idimension in range(len(patchSize)): # 2D/3D patch
        # calculate the size of the output feature map of pathway with unscaled patch
        FMperLayerPath1[0][idimension] = (patchSize[idimension] - Kernels[0][idimension])/Strides[0][idimension]+1
        for ilayer in range(1, layerNum):
            FMperLayerPath1[ilayer][idimension] = (FMperLayerPath1[ilayer-1][idimension] - Kernels[ilayer][idimension])/Strides[ilayer][idimension]+1
        # calculate the size of the output feature map of pathway with scaled patch    
        FMperLayerPath2[2][idimension] = int(np.ceil(FMperLayerPath1[2][idimension]*scaleFactor))
        for ilayer in range(layerNum-1,-1,-1):
            FMperLayerPath2[ilayer-1][idimension] = (FMperLayerPath2[ilayer][idimension]-1) * Strides[ilayer][idimension] + Kernels[ilayer][idimension]
        # calculate the size of the input of pathway with scaled patch
        patchSize_p2[idimension] = (FMperLayerPath2[0][idimension] -1) * Strides[0][idimension] + Kernels[0][idimension]
        InputOfPath2[idimension] = int(patchSize_p2[idimension]/scaleFactor)
    InputOfPath2[-1] = max(InputOfPath2[-1],patchSize[-1])
    return InputOfPath2