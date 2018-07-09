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

def fRigidUnpatchingCorrection2D(actual_size, allPatches, patchOverlap, mode='overwritten'):
    patch_size = [allPatches.shape[1], allPatches.shape[2]]
    height, width = actual_size[0], actual_size[1]
    dOverlap = np.multiply(patch_size, patchOverlap).astype(int)
    dNotOverlap = np.round(np.multiply(patch_size, (1 - patchOverlap))).astype(int)

    height_pad = int(math.ceil((height - dOverlap[0]) * 1.0 / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[0])
    width_pad = int(math.ceil((width - dOverlap[1]) * 1.0 / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1])

    num_rows = int(math.ceil((height_pad-patch_size[0])*1.0/dNotOverlap[0])+1)
    num_cols = int(math.ceil((width_pad-patch_size[1])*1.0/dNotOverlap[1])+1)
    num_slices = allPatches.shape[0]/(num_rows * num_cols)

    allPatches = np.reshape(allPatches, (num_slices, -1, patch_size[0], patch_size[1]))
    unpatchImg = np.zeros((num_slices, height_pad, width_pad))
    dividor_grid = np.zeros((num_slices, height_pad, width_pad))

    if mode == 'overwritten':
        for slice in range(num_slices):
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    unpatchImg[slice, row * dNotOverlap[0]:row * dNotOverlap[0] + patch_size[0], col * dNotOverlap[1]:col * dNotOverlap[1] + patch_size[1]] = allPatches[slice, index]

    elif mode == 'average':
        for slice in range(num_slices):
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    unpatchImg[slice, row * dNotOverlap[0]:row * dNotOverlap[0] + patch_size[0], col * dNotOverlap[1]:col * dNotOverlap[1] + patch_size[1]] += allPatches[slice, index]
                    dividor_grid[slice, row * dNotOverlap[0]:row * dNotOverlap[0] + patch_size[0], col * dNotOverlap[1]:col * dNotOverlap[1] + patch_size[1]] = np.add(
                    dividor_grid[slice, row * dNotOverlap[0]:row * dNotOverlap[0] + patch_size[0], col * dNotOverlap[1]:col * dNotOverlap[1] + patch_size[1]], 1.0)

        unpatchImg = np.divide(unpatchImg, dividor_grid)

    unpatchImg_cropped = unpatchImg[:, (height_pad - height) / 2: height_pad - (height_pad - height) / 2,
                         (width_pad - width) / 2: width_pad - (width_pad - width) / 2]

    unpatchImg_cropped = (unpatchImg_cropped + 1) * 255 / 2
    return unpatchImg_cropped


def fRigidUnpatchingCorrection3D(actual_size, allPatches, patchOverlap, mode='overwritten'):
    patch_size = [allPatches.shape[1], allPatches.shape[2], allPatches.shape[3]]
    height, width, depth = actual_size[0], actual_size[1], actual_size[2]

    dOverlap = np.multiply(patch_size, patchOverlap).astype(int)
    dNotOverlap = np.ceil(np.multiply(patch_size, (1 - patchOverlap))).astype(int)

    height_pad = int(math.ceil((height - dOverlap[0]) * 1.0 / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[0])
    width_pad = int(math.ceil((width - dOverlap[1]) * 1.0 / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1])
    depth_pad = int(math.ceil((depth - dOverlap[2]) * 1.0 / (dNotOverlap[2])) * dNotOverlap[2] + dOverlap[2])

    num_rows = int(math.ceil((height_pad-patch_size[0])*1.0/dNotOverlap[0])+1)
    num_cols = int(math.ceil((width_pad-patch_size[1])*1.0/dNotOverlap[1])+1)
    num_slices = int(math.ceil((depth_pad-patch_size[2])*1.0/dNotOverlap[2])+1)

    unpatchImg = np.zeros((depth_pad, height_pad, width_pad))
    dividor_grid = np.zeros((depth_pad, height_pad, width_pad))

    allPatches = np.transpose(allPatches, (0, 3, 1, 2))
    allPatches = np.reshape(allPatches, (num_slices, -1, patch_size[2], patch_size[0], patch_size[1]))

    if mode == 'overwritten':
        for slice in range(num_slices):
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    unpatchImg[slice * dNotOverlap[2]:slice * dNotOverlap[2] + patch_size[2],
                    row * dNotOverlap[0]:row * dNotOverlap[0] + patch_size[0],
                    col * dNotOverlap[1]:col * dNotOverlap[1] + patch_size[1]] = allPatches[slice, index]

    elif mode == 'average':
        for slice in range(num_slices):
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    unpatchImg[slice * dNotOverlap[2]:slice * dNotOverlap[2] + patch_size[2],
                    row * dNotOverlap[0]:row * dNotOverlap[0] + patch_size[0],
                    col * dNotOverlap[1]:col * dNotOverlap[1] + patch_size[1]] += allPatches[slice, index]
                    dividor_grid[slice * dNotOverlap[2]:slice * dNotOverlap[2] + patch_size[2], row * dNotOverlap[0]:row * dNotOverlap[0] + patch_size[0],col * dNotOverlap[1]:col * dNotOverlap[1] + patch_size[1]] = \
                        np.add(dividor_grid[slice * dNotOverlap[2]:slice * dNotOverlap[2] + patch_size[2], row * dNotOverlap[0]:row * dNotOverlap[0] + patch_size[0],col * dNotOverlap[1]:col * dNotOverlap[1] + patch_size[1]], 1)

        unpatchImg = np.divide(unpatchImg, dividor_grid)

    unpatchImg_cropped = unpatchImg[(depth_pad - depth)/2:depth_pad - (depth_pad - depth)/2,
                         (height_pad - height) / 2: height_pad - (height_pad - height) / 2,
                         (width_pad - width) / 2: width_pad - (width_pad - width) / 2]

    unpatchImg_cropped = (unpatchImg_cropped - np.min(unpatchImg_cropped)) * 2094 / (np.max(unpatchImg_cropped) - np.min(unpatchImg_cropped))
    return unpatchImg_cropped

def fPatchToImage(actual_size, allPatches, patchOverlap):
    patch_size = [allPatches.shape[-3], allPatches.shape[-2], allPatches.shape[-1]]
    dOverlap = np.multiply(patch_size, patchOverlap).astype(int)
    dNotOverlap = np.round(np.multiply(patch_size, (1 - patchOverlap))).astype(int)

    height, width, depth = actual_size[1], actual_size[0], actual_size[2]
    width_pad = int(math.ceil((width - dOverlap[0]) * 1.0 / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[0])
    height_pad = int(math.ceil((height - dOverlap[1]) * 1.0 / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1])
    depth_pad = int(math.ceil((depth - dOverlap[2]) * 1.0 / (dNotOverlap[2])) * dNotOverlap[2] + dOverlap[2])

    num_rows, num_cols, num_slices = int(math.ceil((height_pad - patch_size[1]) * 1.0 / dNotOverlap[1]) + 1), int(
        math.ceil((width_pad - patch_size[0]) * 1.0 / dNotOverlap[0]) + 1), int(
        math.ceil((depth_pad - patch_size[2]) * 1.0 / dNotOverlap[2]) + 1)
    num_4a = allPatches.shape[0] / (num_rows * num_cols * num_slices)
    allPatches = np.reshape(allPatches, (num_4a, -1, patch_size[0], patch_size[1], patch_size[2]))
    unpatchImg = np.zeros((num_4a, width_pad, height_pad, depth_pad))
    dividor_grid = np.zeros((num_4a, width_pad, height_pad, depth_pad))

    for i4a in range(num_4a):
        for slice in range(num_slices):
            for col in range(num_cols):
                for row in range(num_rows):
                    index = slice * num_cols * num_rows + col * num_rows + row
                    unpatchImg[i4a, col * dNotOverlap[0]:col * dNotOverlap[0] + patch_size[0], row * dNotOverlap[1]:row * dNotOverlap[1] + patch_size[1], slice * dNotOverlap[2]:slice * dNotOverlap[2] + patch_size[2]] += allPatches[i4a, index]
                    dividor_grid[i4a, col * dNotOverlap[0]:col * dNotOverlap[0] + patch_size[0], row * dNotOverlap[1]:row * dNotOverlap[1] + patch_size[1], slice * dNotOverlap[2]:slice * dNotOverlap[2] + patch_size[2]] = np.add(
                    dividor_grid[i4a, col * dNotOverlap[0]:col * dNotOverlap[0] + patch_size[0], row * dNotOverlap[1]:row * dNotOverlap[1] + patch_size[1], slice * dNotOverlap[2]:slice * dNotOverlap[2] + patch_size[2]], 1.0)

    unpatchImg = np.divide(unpatchImg, dividor_grid)
    unpatchImg_cropped = unpatchImg[:,(width_pad - width) / 2: width_pad - (width_pad - width) / 2, (height_pad - height) / 2: height_pad - (height_pad - height) / 2, (depth_pad - depth) / 2: depth_pad - (depth_pad - depth) / 2]
    return unpatchImg_cropped


def fUnpatchLabel(prob_list, patchSize, patchOverlap, actualSize, iClass=0):
    # If iClass=0: the value 0 is the label of reference image, and show the possibility of Artifact at the same time
    # If iClass=1, the first half unpatchImg[0] is label of image with artifact, the rest unpatchImg[1] for reference images

    dOverlap = np.multiply(patchSize, patchOverlap).astype(int)
    # dNotOverlap = np.round(np.multiply(patchSize, (1 - patchOverlap))).astype(int)
    dNotOverlap = np.subtract(patchSize, dOverlap)
    paddedSize = [int(math.ceil((actualSize[0] - dOverlap[0]) * 1.0/ dNotOverlap[0]) * dNotOverlap[0] + dOverlap[0]), int(math.ceil((actualSize[1] - dOverlap[1]) * 1.0/ (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1]), int(math.ceil((actualSize[2] - dOverlap[2]) * 1.0/ (dNotOverlap[2])) * dNotOverlap[2] + dOverlap[2])]

    num_rows, num_cols, num_slices = int(math.ceil((paddedSize[1] - patchSize[1]) * 1.0 / dNotOverlap[1]) + 1), int(
        math.ceil((paddedSize[0] - patchSize[0]) * 1.0 / dNotOverlap[0]) + 1), int(
        math.ceil((paddedSize[2] - patchSize[2]) * 1.0 / dNotOverlap[2]) + 1)
    num_4a = prob_list.shape[0] / (num_rows * num_cols * num_slices)
    prob_list = np.reshape(prob_list, (num_4a, -1, 2))
    unpatchImg = np.zeros((num_4a, paddedSize[0], paddedSize[1], paddedSize[2]))
    numVal = np.zeros((num_4a, paddedSize[0], paddedSize[1], paddedSize[2]))

    for i4a in range(num_4a):
        iCorner = [iClass, 0, 0, 0]
        for iIndex in range(prob_list.shape[1]):
            unpatchImg[i4a, iCorner[1]: iCorner[1] + patchSize[0], iCorner[2]: iCorner[2] + patchSize[1], iCorner[3]: iCorner[3] + patchSize[2]] = np.add(unpatchImg[i4a, iCorner[1]: iCorner[1] + patchSize[0], iCorner[2]: iCorner[2] + patchSize[1], iCorner[3]: iCorner[3] + patchSize[2]], prob_list[i4a, iIndex, iClass])
            numVal[i4a, iCorner[1]: iCorner[1] + patchSize[0], iCorner[2]: iCorner[2] + patchSize[1], iCorner[3]: iCorner[3] + patchSize[2]] = np.add(
            numVal[i4a, iCorner[1]: iCorner[1] + patchSize[0], iCorner[2]: iCorner[2] + patchSize[1], iCorner[3]: iCorner[3] + patchSize[2]], 1.0)

            iCorner[1] =int(iCorner[1]+dNotOverlap[0])
            if iCorner[1] + patchSize[0] - 1 > paddedSize[0]:
                iCorner[1] = 0
                iCorner[2] = int(iCorner[2] + dNotOverlap[1])

            if iCorner[2] + patchSize[1] - 1 > paddedSize[1]:
                iCorner[2] = 0
                iCorner[1] = 0
                iCorner[3] = int(iCorner[3] + dNotOverlap[2])

    unpatchImg = np.divide(unpatchImg, numVal)

    if paddedSize == actualSize:
        pass
    else:
        pad_y = (paddedSize[0]-actualSize[0])/2
        pad_x = (paddedSize[1]-actualSize[1])/2
        pad_z = (paddedSize[2]-actualSize[2])/2
        unpatchImg = unpatchImg[:, pad_y:actualSize[0]+pad_y, pad_x:actualSize[1]+pad_x, pad_z:actualSize[2]+pad_z]

    return unpatchImg[0], unpatchImg[1]