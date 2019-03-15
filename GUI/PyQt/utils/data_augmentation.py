from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import h5py
from matplotlib import pyplot as plt
import os
import random

# loading hdf5 dataset
from config.PATH import DLART_OUT_PATH


def main():
    path = DLART_OUT_PATH + os.sep + "Patients-1_Datasets-1_3D_SegMask_64x64x32_Overlap-0.7_Labeling-mask_Split-simpleRand"

    try:
        with h5py.File(path + os.sep + "datasets.hdf5", 'r') as hf:
            X_train = hf['X_train'][:]
            X_validation = hf['X_validation'][:]
            X_test = hf['X_test'][:]
            Y_train = hf['Y_train'][:]
            Y_validation = hf['Y_validation'][:]
            Y_test = hf['Y_test'][:]
            #if self.usingSegmentationMasksForPrediction:
             #   Y_segMasks_train = hf['Y_segMasks_train'][:]
              #  Y_segMasks_validation = hf['Y_segMasks_validation'][:]
               # Y_segMasks_test = hf['Y_segMasks_test'][:]
    except:
        raise TypeError("Can't read HDF5 dataset!")


    mr_matrix = X_test[10,:,:,:]

    img_index = 10
    pad_ratio = 0.2
    padded_pixel = int(np.round(pad_ratio*mr_matrix.shape[1]))

    # plot original
    img = mr_matrix[:,:,img_index]
    plt.subplot(331)
    plt.imshow(img)

    # mr_matrix_ud = np.flip(mr_matrix, axis=0)
    # mr_matrix_lr = np.flip(mr_matrix, axis=1)
    # mr_matrix_pad = np.pad(mr_matrix,
    #                        ((padded_pixel, padded_pixel), (padded_pixel, padded_pixel), (0, 0)),
    #                        'constant',
    #                        constant_values=0)

    # x, y, z flip
    mr_matrix_ud = np.flip(X_test, axis=1)
    mr_matrix_lr = np.flip(X_test, axis=2)
    mr_matrix_fb = np.flip(X_test, axis=3)

    mr_matrix_pad = np.pad(X_test,
                           ((0, 0), (padded_pixel, padded_pixel), (padded_pixel, padded_pixel), (padded_pixel, padded_pixel)),
                           'constant',
                           constant_values=0)

    a, b = augment_3D_Data(X_test, X_test.copy(), xFlip=True, yFlip=True, zFlip=True, xShift=True, yShift=True, zShift=True)




def augment_3D_Data(patches, labels, xFlip=False, yFlip=False, zFlip=False, xShift=False, yShift=False, zShift=False):
    pad_ratio = 0.20
    xPad = int(np.round(pad_ratio*patches.shape[0]))
    yPad = int(np.round(pad_ratio*patches.shape[1]))
    zPad = int(np.round(pad_ratio*patches.shape[2]))

    xHeight = patches.shape[0]
    yHeight = patches.shape[1]
    zHeight = patches.shape[2]

    for i in range(patches.shape[-1]):

        #plt.subplot(131)
        #plt.imshow(patches[i, :, :, 12])

        if xShift and yShift and zShift:
            shifted_patch = np.pad(patches[:, :, :, i],
                             ((xPad, xPad), (yPad, yPad), (zPad, zPad)),
                             'constant',
                             constant_values=0)

            shifted_label = np.pad(labels[:, :, :, i],
                                   ((xPad, xPad), (yPad, yPad), (zPad, zPad)),
                                   'constant',
                                   constant_values=0)

            iX = np.random.random_integers(0, 2*xPad-1)
            iY = np.random.random_integers(0, 2*yPad-1)
            iZ = np.random.random_integers(0, 2*zPad-1)

            shifted_patch = shifted_patch[iX:iX+xHeight, iY:iY+yHeight, iZ:iZ+zHeight]
            shifted_label = shifted_label[iX:iX+xHeight, iY:iY+yHeight, iZ:iZ+zHeight]
            patches[:, :, :, i] = shifted_patch
            labels[:, :, :, i] = shifted_label

        #plt.subplot(132)
        #plt.imshow(patches[i, :, :, 12])

        # flips
        if xFlip and yFlip and zFlip:
            randi = np.random.random_integers(-1, 2)
            if randi != -1:
                pat = patches[:, :, :, i]
                pat2 = np.flip(pat, axis=randi)
                patches[:, :, :, i]=pat2

                label = labels[:, :, :, i]
                label2 = np.flip(label, axis=randi)
                labels[:, :, :, i] = label2

            #plt.subplot(133)
            #plt.imshow(patches[i, :, :, 12])

        #plt.show()

    return patches, labels




#if __name__ == "__main__":
 #   main()