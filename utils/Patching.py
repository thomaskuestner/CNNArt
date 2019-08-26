"""
@author: Sebastian Milde, Thomas Kuestner
"""
import numpy as np
import math
import random
from utils.scaling import fScaleOnePatch
import tensorflow as tf

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
#        dLabels ---> 1D-Numpy-Array with all corresponding labels
                                                                     #
#########################################################################################################################################

def fRigidPatching(dicom_numpy_array, patchSize, patchOverlap, mask_numpy_array, ratio_labeling, sLabeling):
    move_artefact = False
    shim_artefact = False
    noise_artefact = False

    dOverlap = np.multiply(patchSize, patchOverlap)
    dNotOverlap = np.round(np.multiply(patchSize, (1 - patchOverlap)))
    size_zero_pad = np.array(([math.ceil((dicom_numpy_array.shape[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[
        0], math.ceil((dicom_numpy_array.shape[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1]]))
    zero_pad = np.array(([int(math.ceil(size_zero_pad[0])) - dicom_numpy_array.shape[0], int(math.ceil(size_zero_pad[1])) - dicom_numpy_array.shape[1]]))
    zero_pad_part = np.array(([int(math.ceil(zero_pad[0] / 2)), int(math.ceil(zero_pad[1] / 2))]))
    Img_zero_pad = np.lib.pad(dicom_numpy_array, (
    (zero_pad_part[0], zero_pad[0] - zero_pad_part[0]), (zero_pad_part[1], zero_pad[1] - zero_pad_part[1]), (0, 0)),
                              mode='constant')
    nbPatches = int(((size_zero_pad[0]-patchSize[0])/((1-patchOverlap)*patchSize[0])+1)*((size_zero_pad[1]-patchSize[1])/((1-patchOverlap)*patchSize[1])+1)*dicom_numpy_array.shape[2])
    dPatches = np.zeros((patchSize[0], patchSize[1], nbPatches), dtype=float) #dtype=np.float32
    dLabels = np.zeros((nbPatches), dtype = float) #dtype = float
    idxPatch = 0
    if sLabeling == 'volume':
        for iZ in range(0, dicom_numpy_array.shape[2], 1):
            for iY in range(0, int(size_zero_pad[0] - dOverlap[0]), int(dNotOverlap[0])):
                for iX in range(0, int(size_zero_pad[1] - dOverlap[1]), int(dNotOverlap[1])):
                    dPatch = Img_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ]
                    dPatches[:,:,idxPatch] = dPatch
                    idxPatch += 1

        #print(idxPatch)
        dPatches = dPatches[:, :, 0:idxPatch]
        dLabels = np.ones((dPatches.shape[2]))
    elif sLabeling == 'patch':
        Mask_zero_pad = np.lib.pad(mask_numpy_array, (
        (zero_pad_part[0], zero_pad[0] - zero_pad_part[0]), (zero_pad_part[1], zero_pad[1] - zero_pad_part[1]), (0, 0)),
                                  mode='constant')

        for iZ in range(0, dicom_numpy_array.shape[2], 1):
            for iY in range(0, int(size_zero_pad[0] - dOverlap[0]), int(dNotOverlap[0])):
                for iX in range(0, int(size_zero_pad[1] - dOverlap[1]), int(dNotOverlap[1])):
                    dPatch = Img_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ]
                    dPatches[:,:,idxPatch] = dPatch

                    dPatch_mask = Mask_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ]
                    patch_number_value = patchSize[0] * patchSize[1]

                    if np.count_nonzero((dPatch_mask==1).astype(np.int)) > int(ratio_labeling*patch_number_value):
                        move_artefact = True
                    if np.count_nonzero((dPatch_mask==2).astype(np.int)) > int(ratio_labeling*patch_number_value):
                        shim_artefact = True
                    if np.count_nonzero((dPatch_mask==3).astype(np.int)) > int(ratio_labeling*patch_number_value):
                        noise_artefact = True

                    label = 0

                    if move_artefact == True and shim_artefact != True and noise_artefact != True:
                        label = 1
                    elif move_artefact != True and shim_artefact == True and noise_artefact != True:
                        label = 2
                    elif move_artefact != True and shim_artefact != True and noise_artefact == True:
                        label = 3
                    elif move_artefact == True and shim_artefact == True and noise_artefact != True:
                        label = 4
                    elif move_artefact == True and shim_artefact != True and noise_artefact == True:
                        label = 5
                    elif move_artefact != True and shim_artefact == True and noise_artefact == True:
                        label = 6
                    elif move_artefact == True and shim_artefact == True and noise_artefact == True:
                        label = 7

                    dLabels[idxPatch] = label
                    idxPatch += 1

                    move_artefact = False
                    shim_artefact = False
                    noise_artefact = False

        dPatches = dPatches[:, :, 0:idxPatch]
        dLabels = dLabels[0:idxPatch]
    return dPatches, dLabels
##########################################################################################################################################
# In case of 3D patches:                                                                                                                 #
#Input: dicom_numpy_array ---> 4D dicom array (height, width, lengh, number of slices)                                                   #
#       patchSize ---> size of patches, example: [40, 40, 10], patchSize[0] = height, patchSize[1] = weight, height and weight can differ#
#       patchOverlap ---> the ratio for overlapping, example: 0.25                                                                       #
#       mask_numpy_array ---> 4D mask array contains information about the areas of artefacts. movement-artefact = 1, shim-artefact = 2  #
#                             noise-artefact = 3                                                                                         #
#       ratio_labeling ---> set the ratio of the number of 'Pixel-Artefacts' to the whole number of pixels of one patch                  #
#Output: dPatches ---> 4D-Numpy-Array, which contain all Patches.                                                                        #
#        dLabels ---> 1D-Numpy-Array with all corresponding labels                                                                       #
##########################################################################################################################################
def fRigidPatching3D(dicom_numpy_array, patchSize, patchOverlap, mask_numpy_array, ratio_labeling, sLabeling, sTrainingMethod='None'):

    move_artefact = False
    shim_artefact = False
    noise_artefact = False


    dOverlap = np.multiply(patchSize, patchOverlap)
    dNotOverlap = np.ceil(np.multiply(patchSize, (1 - patchOverlap)))
    size_zero_pad = np.array([math.ceil((dicom_numpy_array.shape[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[
        0], math.ceil((dicom_numpy_array.shape[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1], math.ceil((dicom_numpy_array.shape[2] - dOverlap[2]) / (dNotOverlap[2])) * dNotOverlap[2] + dOverlap[2]])
    zero_pad = np.array([int(math.ceil(size_zero_pad[0])) - dicom_numpy_array.shape[0], int(math.ceil(size_zero_pad[1])) - dicom_numpy_array.shape[1], int(math.ceil(size_zero_pad[2])) - dicom_numpy_array.shape[2]])
    zero_pad_part = np.array([int(math.ceil(zero_pad[0] / 2)), int(math.ceil(zero_pad[1] / 2)), int(math.ceil(zero_pad[2] / 2))])

    Img_zero_pad = np.lib.pad(dicom_numpy_array, ((zero_pad_part[0], zero_pad[0] - zero_pad_part[0]), (zero_pad_part[1], zero_pad[1] - zero_pad_part[1]), (zero_pad_part[2], zero_pad[2] - zero_pad_part[2])),
                              mode='constant')

    nbPatches = ((size_zero_pad[0]-patchSize[0])/((1-patchOverlap)*patchSize[0])+1)*((size_zero_pad[1]-patchSize[1])/((1-patchOverlap)*patchSize[1])+1)*((size_zero_pad[2]-patchSize[2])/(np.round((1-patchOverlap)*patchSize[2]))+1)
    dPatches = np.zeros((patchSize[0], patchSize[1], patchSize[2], int(nbPatches)), dtype=float)
    dLabels = np.zeros((int(nbPatches)), dtype = int) #float
    idxPatch = 0

    if sLabeling == 'volume' and sTrainingMethod == 'ScaleJittering':
        for iZ in range(0, int(size_zero_pad[2] - dOverlap[2]), int(dNotOverlap[2])):
            for iY in range(0, int(size_zero_pad[0] - dOverlap[0]), int(dNotOverlap[0])):
                for iX in range(0, int(size_zero_pad[1] - dOverlap[1]), int(dNotOverlap[1])):
                    if (iX>=int(size_zero_pad[1] - dOverlap[1]-patchSize[1])) or(iY>=int(size_zero_pad[0] - dOverlap[0]-patchSize[0])) or (iZ>=int(size_zero_pad[2] - dOverlap[2]-patchSize[2])):
                        randPatchSize = patchSize
                    else:
                        randPatchSize = np.round(np.multiply(patchSize, (np.random.rand(1) + 1))).astype(int)
                    dPatch = Img_zero_pad[iY:iY + randPatchSize[0], iX:iX + randPatchSize[1], iZ:iZ + randPatchSize[2]]
                    scaleddPatch = fScaleOnePatch(dPatch, randPatchSize, patchSize)
                    dPatches[:, :, :, idxPatch] = scaleddPatch
                    idxPatch += 1
        dPatches = dPatches[:, :, :, 0:idxPatch]
        dLabels = np.ones((dPatches.shape[3]))

    elif sLabeling == 'volume':
        for iZ in range(0, int(size_zero_pad[2] - dOverlap[2]), int(dNotOverlap[2])):
            for iY in range(0, int(size_zero_pad[0] - dOverlap[0]), int(dNotOverlap[0])):
                for iX in range(0, int(size_zero_pad[1] - dOverlap[1]), int(dNotOverlap[1])):
                    dPatch = Img_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ:iZ + patchSize[2]]
                    dPatches[:,:,:,idxPatch] = dPatch
                    idxPatch += 1

        dPatches = dPatches[:, :, :, 0:idxPatch]
        dLabels = np.ones((dPatches.shape[3]))
    elif sLabeling == 'patch':
        Mask_zero_pad = np.lib.pad(mask_numpy_array, (
            (zero_pad_part[0], zero_pad[0] - zero_pad_part[0]), (zero_pad_part[1], zero_pad[1] - zero_pad_part[1]), (0, 0)),
                                   mode='constant')
        for iZ in range(0, int(size_zero_pad[2] - dOverlap[2]), int(dNotOverlap[2])):
            for iY in range(0, int(size_zero_pad[0] - dOverlap[0]), int(dNotOverlap[0])):
                for iX in range(0, int(size_zero_pad[1] - dOverlap[1]), int(dNotOverlap[1])):
                    dPatch = Img_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ:iZ + patchSize[2]]
                    dPatches[:,:,:,idxPatch] = dPatch

                    dPatch_mask = Mask_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ:iZ + patchSize[2]]
                    patch_number_value = patchSize[0] * patchSize[1] * patchSize[2]

                    if np.count_nonzero((dPatch_mask==1).astype(np.int)) > int(ratio_labeling*patch_number_value):
                        move_artefact = True
                    if np.count_nonzero((dPatch_mask==2).astype(np.int)) > int(ratio_labeling*patch_number_value):
                        shim_artefact = True
                    if np.count_nonzero((dPatch_mask==3).astype(np.int)) > int(ratio_labeling*patch_number_value):
                        noise_artefact = True

                    label = 0

                    if move_artefact == True and shim_artefact != True and noise_artefact != True:
                        label = 1
                    elif move_artefact != True and shim_artefact == True and noise_artefact != True:
                        label = 2
                    elif move_artefact != True and shim_artefact != True and noise_artefact == True:
                        label = 3
                    elif move_artefact == True and shim_artefact == True and noise_artefact != True:
                        label = 4
                    elif move_artefact == True and shim_artefact != True and noise_artefact == True:
                        label = 5
                    elif move_artefact != True and shim_artefact == True and noise_artefact == True:
                        label = 6
                    elif move_artefact == True and shim_artefact == True and noise_artefact == True:
                        label = 7

                    dLabels[idxPatch] = label
                    idxPatch += 1

                    move_artefact = False
                    shim_artefact = False
                    noise_artefact = False
                    
        dPatches = dPatches[:, :, :, 0:idxPatch]
        dLabels = dLabels[0:idxPatch]
    return dPatches, dLabels

########################################################################################################################
# @ author : Shanqi Yang
# input : 3D image tensor
# output : lists of pathes cropped from this 3D tensor
# calls compute_patch_indices to calculate the index lists where the image will be cropped
########################################################################################################################

def get_patches(image, num_patches, image_shape = [316, 260, 320],
                patch_size=[64, 64, 64], overlap = 32, start = [0, 0, 0]):

# Order: to define where to start to choose the index, from the start side or from the end side
    order = random.choice([False, True])
    index_lists = compute_patch_indices(image_shape = image_shape,
                                        patch_size = patch_size,
                                        overlap = overlap,
                                        start = start,
                                        order = order)
    assert num_patches == len(index_lists)
    patches_collection = []
    #print('index list')
    #for item in index_lists:
    #    print(item)
    for index in index_lists:
        patch = tf.slice(image, index, patch_size)
        #patch = image[index[0]:(index[0] + patch_size[0]),index[1]: (index[1]+ patch_size[1]), index[2]:(index[2]+ patch_size[2])]
        patches_collection.append(patch)

    patches_collection = tf.stack(patches_collection)
    assert patches_collection.get_shape().dims == [num_patches, patch_size[0], patch_size[1], patch_size[2]]
    return patches_collection

########################################################################################################################
# @ author : Shanqi Yang
# input : the information about the image, and how it is supposed to be cropped
# output : lists of indexs list
# calls compute_patch_indices to calculate the index lists where the image will be cropped
########################################################################################################################

def compute_patch_indices(image_shape, patch_size, overlap, start = [0, 0, 0], order = True):
    if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(image_shape))
        #print(overlap)

    stop = [(i-j) for i, j  in zip(image_shape, patch_size)]
    step = patch_size - overlap
    index_list = get_set_of_patch_indices(start, stop, step, order)
    return get_random_indexs (image_shape, patch_size, index_list)

# order is for fetching those near the bounds
# if fetch in True mode, then those near the stop won't be fetched
# if fetch in False mode , then those near the start won't be fetched

def get_set_of_patch_indices(start, stop, step, order = False):
    if order:
        return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                          start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)
    else:
        return np.asarray(np.mgrid[stop[0]:start[0]:-step[0], stop[1]:start[1]:-step[1],
                           stop[2]:start[2]:-step[2]].reshape(3, -1).T, dtype=np.int)

def get_random_indexs (image_shape, patch_size, index_list):

    index0bound = image_shape[0] - patch_size[0]
    index1bound = image_shape[1] - patch_size[1]
    index2bound = image_shape[2] - patch_size[2]

    for index in index_list:
        newIndex0 = index[0] + random.randint(-10, 10)
        newIndex1 = index[1] + random.randint(-10, 10)
        newIndex2 = index[2] + random.randint(-10, 10)

        index[0] = newIndex0 if (newIndex0 <= index0bound and newIndex0 >= 0) else index[0]
        index[1] = newIndex1 if (newIndex1 <= index1bound and newIndex1 >= 0) else index[1]
        index[2] = newIndex2 if (newIndex2 <= index2bound and newIndex2 >= 0) else index[2]

    return index_list
