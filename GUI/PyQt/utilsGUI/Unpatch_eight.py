import numpy as np
import math

def UnpatchArte8(IndexType, patchSize, patchOverlap, actualSize):
    dOverlap = np.round(np.multiply(patchSize, patchOverlap))    # only 2 elements
    dNotOverlap = [patchSize[0] - dOverlap[0], patchSize[1] - dOverlap[1]]
    paddedSize = [int(math.ceil((actualSize[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[
        0]), int(math.ceil((actualSize[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1]), actualSize[2]]      # (196, 256, 40)-[200, 260, 40]

    iCorner = [0, 0, 0]
    unpatchImg1 = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
    for iIndex in range(0, IndexType.size, 1):
        if IndexType[iIndex] == 1 or IndexType[iIndex] == 4 or IndexType[iIndex] == 5 or IndexType[iIndex] == 7:
            unpatchImg1[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]), iCorner[2]]\
                = np.add(unpatchImg1[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]), iCorner[2]], 1.)

        iCorner[0] =int(iCorner[0]+dNotOverlap[0])
        if iCorner[0] + patchSize[0] - 1 > paddedSize[0]:
            iCorner[0] = 0
            iCorner[1] = int(iCorner[1] + dNotOverlap[1])

        if iCorner[1] + patchSize[1] - 1 > paddedSize[1]:
            iCorner[1] = 0
            iCorner[0] = 0
            iCorner[2] = iCorner[2] + 1
    unpatchImg1[unpatchImg1 > 0] = 1

    iCorner = [0, 0, 0]
    unpatchImg2 = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
    for iIndex in range(0, IndexType.size, 1):
        if IndexType[iIndex] == 2 or IndexType[iIndex] == 4 or IndexType[iIndex] == 6 or IndexType[iIndex] == 7:
            unpatchImg2[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]),
            iCorner[2]] = np.add(
                unpatchImg2[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]),
                iCorner[2]], 1.)

        iCorner[0] =int(iCorner[0]+dNotOverlap[0])
        if iCorner[0] + patchSize[0] - 1 > paddedSize[0]:
            iCorner[0] = 0
            iCorner[1] = int(iCorner[1] + dNotOverlap[1])

        if iCorner[1] + patchSize[1] - 1 > paddedSize[1]:
            iCorner[1] = 0
            iCorner[0] = 0
            iCorner[2] = iCorner[2] + 1
    unpatchImg2[unpatchImg2 > 0] = 1

    iCorner = [0, 0, 0]
    unpatchImg3 = np.zeros((paddedSize[0], paddedSize[1], paddedSize[2]))
    for iIndex in range(0, IndexType.size, 1):
        if IndexType[iIndex] == 3 or IndexType[iIndex] == 5 or IndexType[iIndex] == 6 or IndexType[iIndex] == 7:
            unpatchImg3[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]),
            iCorner[2]] = np.add(
                unpatchImg3[iCorner[0]: iCorner[0] + int(patchSize[0]), iCorner[1]: iCorner[1] + int(patchSize[1]),
                iCorner[2]], 1.)

        iCorner[0] =int(iCorner[0]+dNotOverlap[0])
        if iCorner[0] + patchSize[0] - 1 > paddedSize[0]:
            iCorner[0] = 0
            iCorner[1] = int(iCorner[1] + dNotOverlap[1])

        if iCorner[1] + patchSize[1] - 1 > paddedSize[1]:
            iCorner[1] = 0
            iCorner[0] = 0
            iCorner[2] = iCorner[2] + 1
    unpatchImg3[unpatchImg3 > 0] = 1


    if paddedSize == actualSize:
        pass
    else:
        pad_y = math.ceil((paddedSize[0]-actualSize[0])/2) # add math.ceil for python3
        pad_x = math.ceil((paddedSize[1]-actualSize[1])/2) #
        unpatchImg1 = unpatchImg1[pad_y:paddedSize[0] - (paddedSize[0]-actualSize[0]-pad_y),
                      pad_x:paddedSize[1] - (paddedSize[1]-actualSize[1]-pad_x), : ]
        unpatchImg2 = unpatchImg2[pad_y:paddedSize[0] - (paddedSize[0] - actualSize[0] - pad_y),
                     pad_x:paddedSize[1] - (paddedSize[1] - actualSize[1] - pad_x), :]
        unpatchImg3 = unpatchImg3[pad_y:paddedSize[0] - (paddedSize[0] - actualSize[0] - pad_y),
                 pad_x:paddedSize[1] - (paddedSize[1] - actualSize[1] - pad_x), :]

        unpatchImg1 = unpatchImg1 + 1 # motion
        unpatchImg2 = unpatchImg2 + 1 # shim
        unpatchImg3 = unpatchImg3 + 1 # noise

    return unpatchImg1, unpatchImg2, unpatchImg3