# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:38:33 2018

@author: cyf
"""
import numpy as np
from networks.multiscale.MSnetworks import fgetKernels, fgetStrides, fgetLayerNum


def fcalculateInputOfPath2(patchSize, scaleFactor, sModelIn):

    Kernels = fgetKernels()
    Strides = fgetStrides()
    layerNum = fgetLayerNum()
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