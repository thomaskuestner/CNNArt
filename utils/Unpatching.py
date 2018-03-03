import numpy as np
import math

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