"""
@author: Sebastian Milde, Thomas Kuestner
"""
import numpy as np
import math

#########################################################################################################################################
#Function: fRigidPatching                                                                                                               #
#The function fRigidPatching is responsible for splitting the dicom numpy array in patches depending on the patchSize and the           #
#patchOverlap. Besides the function creates an 1D array with the corresponding labels.                                                  #
#                                                                                                                                       #
#Input: dicom_numpy_array ---> 3D dicom array (height, width, number of slices)                                                         #
#       patchSize ---> size of patches, example: [40, 40], patchSize[0] = height, patchSize[1] = weight, height and weight can differ   #
#       patchOverlap ---> the ratio for overlapping, example: 0.25                                                                      #
#       mask_numpy_array ---> 3D mask array contains information about the areas of artefacts. movement-artefact = 1, shim-artefact = 2 #
#                             noise-artefact = 3                                                                                        #
#       ratio_labeling ---> set the ratio of the number of 'Pixel-Artefacts' to the whole number of pixels of one patch                 #
#Output: dPatches ---> 3D-Numpy-Array, which contain all Patches.                                                                       #
#        dLabels ---> 1D-Numpy-Array with all corresponding labels                                                                      #
#########################################################################################################################################

def fRigidPatching(dicom_numpy_array, patchSize, patchOverlap, mask_numpy_array, ratio_labeling):
    dPatches = None
    move_artefact = False
    shim_artefact = False
    noise_artefact = False
    dLabels = []

    dOverlap = np.multiply(patchSize, patchOverlap)
    dNotOverlap = np.round(np.multiply(patchSize, (1 - patchOverlap)))
    size_zero_pad = np.array(([math.ceil((dicom_numpy_array.shape[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[
        0], math.ceil((dicom_numpy_array.shape[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1]]))
    zero_pad = np.array(([int(size_zero_pad[0]) - dicom_numpy_array.shape[0], int(size_zero_pad[1]) - dicom_numpy_array.shape[1]]))
    zero_pad_part = np.array(([int(math.ceil(zero_pad[0] / 2)), int(math.ceil(zero_pad[1] / 2))]))
    Img_zero_pad = np.lib.pad(dicom_numpy_array, (
    (zero_pad_part[0], zero_pad[0] - zero_pad_part[0]), (zero_pad_part[1], zero_pad[1] - zero_pad_part[1]), (0, 0)),
                              mode='constant')
    Mask_zero_pad = np.lib.pad(mask_numpy_array, (
    (zero_pad_part[0], zero_pad[0] - zero_pad_part[0]), (zero_pad_part[1], zero_pad[1] - zero_pad_part[1]), (0, 0)),
                              mode='constant')

    for iZ in range(0, dicom_numpy_array.shape[2], 1):
        for iY in range(0, int(size_zero_pad[0] - dOverlap[0]), int(dNotOverlap[0])):
            for iX in range(0, int(size_zero_pad[1] - dOverlap[1]), int(dNotOverlap[1])):
                dPatch = Img_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ]
                dPatch = dPatch[:, :, np.newaxis]

                if dPatches is None:
                    dPatches = dPatch
                else:
                    dPatches = np.concatenate((dPatches, dPatch), axis=2)

                dPatch_mask = Mask_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ]
                patch_number_value = patchSize[0] * patchSize[1]

                if np.count_nonzero((dPatch_mask==1).astype(np.int)) > int(ratio_labeling*patch_number_value):
                    move_artefact = True
                if np.count_nonzero((dPatch_mask==2).astype(np.int)) > int(ratio_labeling*patch_number_value):
                    shim_artefact = True
                if np.count_nonzero((dPatch_mask==3).astype(np.int)) > int(ratio_labeling*patch_number_value):
                    noise_artefact = True

                label = [0]

                if move_artefact == True and shim_artefact != True and noise_artefact != True:
                    label = [1]
                elif move_artefact != True and shim_artefact == True and noise_artefact != True:
                    label = [2]
                elif move_artefact != True and shim_artefact != True and noise_artefact == True:
                    label = [3]
                elif move_artefact == True and shim_artefact == True and noise_artefact != True:
                    label = [4]
                elif move_artefact == True and shim_artefact != True and noise_artefact == True:
                    label = [5]
                elif move_artefact != True and shim_artefact == True and noise_artefact == True:
                    label = [6]
                elif move_artefact == True and shim_artefact == True and noise_artefact == True:
                    label = [7]

                dLabels = np.concatenate((dLabels, label))

                move_artefact = False
                shim_artefact = False
                noise_artefact = False

    return dPatches, dLabels