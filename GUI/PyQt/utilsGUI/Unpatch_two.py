import numpy as np
import math

def fUnpatch2D(prob_list, patchSize, patchOverlap, actualSize):
    iCorner = [0, 0, 0]
    dOverlap = np.round(np.multiply(patchSize, patchOverlap))
    dNotOverlap = [patchSize[0] - dOverlap[0], patchSize[1] - dOverlap[1]]
    paddedSize = [int(math.ceil((actualSize[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[
        0]), int(math.ceil((actualSize[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1]), actualSize[2]]
    unpatchImg = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
    # numVal = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
    # print prob_list.shape[0]
    prob_list = np.argmax(prob_list, 1)
    print(prob_list.shape)
    for iIndex in range(0, prob_list.shape[0], 1):
        # print(iIndex)
        lMask = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
        # print(lMask.shape)
        # print(iCorner[2])
        #print(lMask[iCorner[0]: iCorner[0] + patchSize[0], iCorner[1]: iCorner[1] + patchSize[1], iCorner[2]].shape)
        lMask[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]), iCorner[2]] = 1
        unpatchImg[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]), iCorner[2]]  = prob_list[iIndex]
        # lMask = lMask == 1
        # numVal[lMask] = numVal[lMask] + 1

        iCorner[0] =int(iCorner[0]+dNotOverlap[0])
        if iCorner[0] + patchSize[0] - 1 > paddedSize[0]:
            iCorner[0] = 0
            iCorner[1] = int(iCorner[1] + dNotOverlap[1])

        if iCorner[1] + patchSize[1] - 1 > paddedSize[1]:
            iCorner[1] = 0
            iCorner[0] = 0
            iCorner[2] = iCorner[2] + 1

    # unpatchImg = np.divide(unpatchImg, numVal)

    if paddedSize == actualSize:
        pass
    else:
        pad_y = math.ceil((paddedSize[0]-actualSize[0])/2)
        pad_x = math.ceil((paddedSize[1]-actualSize[1])/2)
        unpatchImg = unpatchImg[pad_y:paddedSize[0] - (paddedSize[0]-actualSize[0]-pad_y), pad_x:paddedSize[1] - (paddedSize[1]-actualSize[1]-pad_x), : ]

    return unpatchImg
