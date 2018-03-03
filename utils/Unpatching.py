import numpy as np
import math

#########################################################################################################################################
#Function: fUnpatch2D                                                                                                                   #
#The function fUnpatch2D has the task to reconstruct the probability-images. Every patch contains the probability of every class.       #
#To visualize the probabilities it is important to reconstruct the probability-images. This function is used for 2D patching.           #                                                                                                                                    #
#Input: prob_list ---> list of probabilities of every Patch. The column describes the classes, the row describes the probability of     #
#                      every class                                                                                                      #
#       patchSize ---> size of patches, example: [40, 40, 10], patchSize[0] = height, patchSize[1] = weight, patchSize[2] = depth       #
#       patchOverlap ---> the ratio for overlapping, example: 0.25                                                                      #                                                                        #
#       actualSize ---> the actual size of the chosen mrt-layer: example: ab, t1_tse_tra_Kopf_0002; actual size = [256, 196, 40]        #
#       iClass --->  the number of the class, example: ref = 0, artefact = 1                                                            #
#Output: unpatchImg ---> 3D-Numpy-Array, which contains the probability of every image pixel.                                           #
#########################################################################################################################################

def fUnpatch2D(prob_list, patchSize, patchOverlap, actualSize, iClass):
    iCorner = [0, 0, 0]
    dOverlap = np.round(np.multiply(patchSize, patchOverlap))
    dNotOverlap = [patchSize[0] - dOverlap[0], patchSize[1] - dOverlap[1]]
    paddedSize = [int(math.ceil((actualSize[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[
        0]), int(math.ceil((actualSize[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1]), actualSize[2]]
    unpatchImg = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
    print(unpatchImg.shape)
    numVal = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
    for iIndex in range(0, prob_list.shape[0], 1):
        print(iIndex)
        lMask = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
        lMask[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]), iCorner[2]] = 1
        unpatchImg[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]), iCorner[2]] = np.add(unpatchImg[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]), iCorner[2]], prob_list[iIndex,iClass])
        lMask = lMask == 1
        numVal[lMask] = numVal[lMask] + 1

        iCorner[0] =int(iCorner[0]+dNotOverlap[0])
        if iCorner[0] + patchSize[0] - 1 > paddedSize[0]:
            iCorner[0] = 0
            iCorner[1] = int(iCorner[1] + dNotOverlap[1])

        if iCorner[1] + patchSize[1] - 1 > paddedSize[1]:
            iCorner[1] = 0
            iCorner[0] = 0
            iCorner[2] = iCorner[2] + 1

    unpatchImg = np.divide(unpatchImg, numVal)

    if paddedSize == actualSize:
        pass
    else:
        pad_y = (paddedSize[0]-actualSize[0])/2
        pad_x = (paddedSize[1]-actualSize[1])/2
        unpatchImg = unpatchImg[pad_y:paddedSize[0] - (paddedSize[0]-actualSize[0]-pad_y), pad_x:paddedSize[1] - (paddedSize[1]-actualSize[1]-pad_x), : ]

    return unpatchImg


#########################################################################################################################################
#Function: fUnpatch3D                                                                                                                   #
#The function fUnpatch3D has the task to reconstruct the probability-images. Every patch contains the probability of every class.       #
#To visualize the probabilities it is inportant to reconstruct the probability-images. This function is used for 3D patching.           #                                                                                                                                    #
#Input: prob_list ---> list of probabilities of every Patch. The column describes the classes, the row describes the probability of     #
#                      every class                                                                                                      #
#       patchSize ---> size of patches, example: [40, 40, 10], patchSize[0] = height, patchSize[1] = weight, patchSize[2] = depth       #
#       patchOverlap ---> the ratio for overlapping, example: 0.25                                                                      #                                                                        #
#       actualSize ---> the actual size of the chosen mrt-layer: example: ab, t1_tse_tra_Kopf_0002; actual size = [256, 196, 40]        #
#       iClass --->  the number of the class, example: ref = 0, artefact = 1                                                            #
#Output: unpatchImg ---> 3D-Numpy-Array, which contains the probability of every image pixel.                                           #
#########################################################################################################################################

def fUnpatch3D(prob_list, patchSize, patchOverlap, actualSize, iClass):
    iCorner = [0, 0, 0]
    dOverlap = np.round(np.multiply(patchSize, patchOverlap))
    dNotOverlap = [patchSize[0] - dOverlap[0], patchSize[1] - dOverlap[1], patchSize[2] - dOverlap[2]]
    paddedSize = [int(math.ceil((actualSize[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[
        0]), int(math.ceil((actualSize[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1]),
                  int(math.ceil((actualSize[2] - dOverlap[2]) / (dNotOverlap[2])) * dNotOverlap[2] + dOverlap[2])]
    unpatchImg = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
    numVal = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))

    for iIndex in range(0, prob_list.shape[0], 1):
        print(iIndex)
        lMask = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
        lMask[iCorner[0]: iCorner[0] + patchSize[0], iCorner[1]: iCorner[1] + patchSize[1], iCorner[2]: iCorner[2] + patchSize[2]] = 1
        unpatchImg[iCorner[0]: iCorner[0] + patchSize[0], iCorner[1]: iCorner[1] + patchSize[1], iCorner[2]: iCorner[2] + patchSize[2]] = np.add(unpatchImg[iCorner[0]: iCorner[0] + patchSize[0], iCorner[1]: iCorner[1] + patchSize[1], iCorner[2]: iCorner[2] + patchSize[2]], prob_list[iIndex,iClass])
        lMask = lMask == 1
        numVal[lMask] = numVal[lMask] + 1

        iCorner[0] =int(iCorner[0]+dNotOverlap[0])
        if iCorner[0] + patchSize[0] - 1 > paddedSize[0]:
            iCorner[0] = 0
            iCorner[1] = int(iCorner[1] + dNotOverlap[1])

        if iCorner[1] + patchSize[1] - 1 > paddedSize[1]:
            iCorner[1] = 0
            iCorner[0] = 0
            iCorner[2] = int(iCorner[2] + dNotOverlap[2])

    unpatchImg = np.divide(unpatchImg, numVal)

    if paddedSize == actualSize:
        pass
    else:
        pad_y = (paddedSize[0]-actualSize[0])/2
        pad_x = (paddedSize[1]-actualSize[1])/2
        pad_z = (paddedSize[2]-actualSize[2])/2
        unpatchImg = unpatchImg[pad_y:paddedSize[0] - (paddedSize[0]-actualSize[0]-pad_y), pad_x:paddedSize[1] - (paddedSize[1]-actualSize[1]-pad_x), pad_z:paddedSize[2] - (paddedSize[2]-actualSize[2]-pad_z) ]

    return unpatchImg


# rigid unpatching
def fRigidUnpatching(PatchSize, PatchOverlap, dImg, prob_test):
    dActSize = np.round(PatchOverlap * PatchSize)
    iPadSize_x = math.ceil(dImg.shape[1] / dActSize[1]) * dActSize[1]
    iPadSize_y = math.ceil(dImg.shape[0] / dActSize[0]) * dActSize[0]
    iPadCut_x = iPadSize_x - dImg.shape[1]
    iPadCut_y = iPadSize_y - dImg.shape[0]
    dOverlay = np.zeros((int(iPadSize_y), int(iPadSize_x), dImg.shape[2]))
    x_max = int(2*iPadSize_x / PatchSize[0])
    y_max = int(2*iPadSize_y / PatchSize[1])
    x_index = x_max - 1
    y_index = y_max - 1
    patch_nmb_lay = x_index*y_index

    for iZ in range(0,dImg.shape[2], 1):
        for iX in range(0, x_max, 1):
            for iY in range(0, y_max, 1):
                if iX == 0 and iY == 0 or iX == x_index and iY == y_index or iX == x_index and iY == 0 or iX == 0 and iY == y_index:
                    num_1 = get_first_index(iX, iY, iZ, patch_nmb_lay, x_index, y_index)
                    dOverlay[iY * dActSize[0]:iY * dActSize[0] + dActSize[0],
                    iX * dActSize[1]:iX * dActSize[1] + dActSize[1], iZ] = prob_test[num_1]

                elif (iX == 0 or iX == x_index) and 0 < iY < y_index:
                    num_1 = get_first_index(iX, iY, iZ, patch_nmb_lay, x_index, y_index)
                    num_2 = num_1 - 1
                    dOverlay[iY * dActSize[0]:iY * dActSize[0] + dActSize[0],
                    iX * dActSize[1]:iX * dActSize[1] + dActSize[1], iZ] = (prob_test[num_1] + prob_test[num_2]) / 2

                elif (iY == 0 or iY == y_index) and 0 < iX < x_index:
                    num_1 = get_first_index(iX, iY, iZ, patch_nmb_lay, x_index, y_index)
                    num_2 = num_1 - y_index
                    dOverlay[iY * dActSize[0]:iY * dActSize[0] + dActSize[0],
                    iX * dActSize[1]:iX * dActSize[1] + dActSize[1], iZ] = (prob_test[num_1] + prob_test[num_2]) / 2

                else:
                    num_1 = get_first_index(iX, iY, iZ, patch_nmb_lay, x_index, y_index)
                    num_2 = num_1 - 1
                    num_3 = num_1 - y_index
                    num_4 = num_2 - y_index
                    dOverlay[iY * dActSize[0]:iY * dActSize[0] + dActSize[0],
                    iX * dActSize[1]:iX * dActSize[1] + dActSize[1], iZ] = (prob_test[num_1] + prob_test[num_2] + prob_test[
                        num_3] + prob_test[num_4]) / 4

    dOverlay = dOverlay[iPadCut_y / 2:iPadSize_y - iPadCut_y / 2, iPadCut_x / 2:iPadSize_x - iPadCut_x / 2, :]

    return dOverlay

def get_first_index(iX, iY, iZ, patch_nmb_layer, x_index, y_index):
    num = iZ*patch_nmb_layer + iX * y_index + iY
    if iY == y_index and not iX == x_index:
        num = num - 1
    elif iX == x_index and not iY == y_index:
        num = num - y_index
    elif iX == x_index and iY == y_index:
        num = num - y_index - 1

    return num

def fRigidUnpatchingCorrection(actual_size, allPatches):
    patch_size = [allPatches.shape[1], allPatches.shape[2]]
    height, width = actual_size[0], actual_size[1]
    num_rows, num_cols = int(math.ceil(height*1.0/patch_size[0])), int(math.ceil(width*1.0/patch_size[1]))
    num_slices = allPatches.shape[0]/(num_rows * num_cols)

    allPatches = np.reshape(allPatches, (num_slices, -1, patch_size[0], patch_size[1]))
    unpatchImg = np.zeros((num_slices, height, width))

    for slice in range(num_slices):
        for row in range(num_rows):
            if row == (num_rows - 1):
                for col in range(num_cols):
                    index = row * num_cols + col
                    if col == (num_cols - 1):
                        unpatchImg[slice, row * patch_size[0]:, col * patch_size[1]:] = allPatches[slice, index, :height - row * patch_size[0], :width - col * patch_size[1]]
                    else:
                        unpatchImg[slice, row * patch_size[0]:, col * patch_size[1]:(col + 1) * patch_size[1]] = allPatches[slice, index, :height - row * patch_size[0], :]
            else:
                for col in range(num_cols):
                    index = row * num_cols + col
                    if col == (num_cols - 1):
                        unpatchImg[slice, row * patch_size[0]:(row + 1) * patch_size[0], col * patch_size[1]:] = allPatches[slice, index, :, :width - col * patch_size[1]]
                    else:
                        unpatchImg[slice, row * patch_size[0]:(row + 1) * patch_size[0],
                        col * patch_size[1]:(col + 1) * patch_size[1]] = allPatches[slice, index]

    return unpatchImg