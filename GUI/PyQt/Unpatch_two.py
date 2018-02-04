import numpy as np
import math

#########################################################################################################################################
#Module: Unpatching                                                                                                                     #
#The module Unpatching is responsible for reconstructing the probability-images. To reconstruct the images the means of all             #
#probabilities from overlapping patches are calculated and are assigned to every pixel-image. It's important to consider the order of   #
#dimensions within the algorithm of the module RigidPatching. In this case the order is: weight(x), height(y), depth(z)                 #
#The Unpatching-module contains two function:                                                                                           #
#fUnpatch2D: For 2D Patch-Splitting                                                                                                     #
#fUnpatch3D: For 3D Patch-Splitting                                                                                                     #
#########################################################################################################################################

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
    numVal = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
    # print prob_list.shape[0]
    for iIndex in range(0, prob_list.shape[0], 1):
        print(iIndex)
        lMask = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
        print(lMask.shape)
        print(iCorner[2])
        #print(lMask[iCorner[0]: iCorner[0] + patchSize[0], iCorner[1]: iCorner[1] + patchSize[1], iCorner[2]].shape)
        lMask[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]), iCorner[2]] = 1
        unpatchImg[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]), iCorner[2]] = np.add(unpatchImg[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]), iCorner[2]], prob_list[iIndex,iClass])
        lMask = lMask == 1
        numVal[lMask] = numVal[lMask] + 1

        iCorner[1] =int(iCorner[1]+dNotOverlap[1])
        if iCorner[1] + patchSize[1] - 1 > paddedSize[1]:
            iCorner[1] = 0
            iCorner[0] = int(iCorner[0] + dNotOverlap[0])

        if iCorner[0] + patchSize[0] - 1 > paddedSize[0]:
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
        lMask = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
        lMask[iCorner[0]: iCorner[0] + patchSize[0], iCorner[1]: iCorner[1] + patchSize[1], iCorner[2]: iCorner[2] + patchSize[2]] = 1
        unpatchImg[iCorner[0]: iCorner[0] + patchSize[0], iCorner[1]: iCorner[1] + patchSize[1], iCorner[2]: iCorner[2] + patchSize[2]] = np.add(unpatchImg[iCorner[0]: iCorner[0] + patchSize[0], iCorner[1]: iCorner[1] + patchSize[1], iCorner[2]: iCorner[2] + patchSize[2]], prob_list[iIndex,iClass])
        lMask = lMask == 1
        numVal[lMask] = numVal[lMask] + 1

        iCorner[1] =int(iCorner[1]+dNotOverlap[1])
        if iCorner[1] + patchSize[1] - 1 > paddedSize[1]:
            iCorner[1] = 0
            iCorner[0] = int(iCorner[0] + dNotOverlap[0])

        if iCorner[0] + patchSize[0] - 1 > paddedSize[0]:
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




