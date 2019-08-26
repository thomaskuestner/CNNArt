def fPatch(imgIn, patch_size=[40, 40, 10], stride=10):
    iSize = imgIn.shape
    if len(iSize) < 3:
        l2Dimg = True
        iSize = iSize + (1,)
    else:
        l2Dimg = False

    if len(patch_size) < 3:
        l2DPatching = True
    else:
        l2DPatching = False

    patches = []

    if l2DPatching:  # 2D patching on a 2D/3D image
        for h in range(0, iSize[2]):
            for i in range(0, iSize[0] - patch_size[0] + 1, stride):
                for j in range(0, iSize[1] - patch_size[1] + 1, stride):
                    if l2Dimg:
                        x = imgIn[i:i + patch_size[0], j:j + patch_size[1]]
                    else:
                        x = imgIn[i:i + patch_size[0], j:j + patch_size[1], h]
                    patches.append(x)
    else:  # 3D patching on a 2D/3D image
        for h in range(0, iSize[2] - patch_size[2] + 1, stride):
            for i in range(0, iSize[0] - patch_size[0] + 1, stride):
                for j in range(0, iSize[1] - patch_size[1] + 1, stride):
                    if l2Dimg:
                        x = imgIn[i:i + patch_size[0], j:j + patch_size[1]]
                    else:
                        x = imgIn[i:i + patch_size[0], j:j + patch_size[1], h:h+patch_size[2]]
                    patches.append(x)

    return np.array(patches, dtype=imgIn.dtype), iSize


def fUnpatch(patchesIn, iSize, patch_size=[40, 40, 10], stride=10, overlap_mode='avg'):
    if len(iSize) < 3:
        l2Dimg = True
        iSize = iSize + (1,)
    else:
        l2Dimg = False

    if len(patch_size) < 3:
        l2DPatching = True
    else:
        l2DPatching = False

    img = np.zeros(iSize, dtype=patchesIn.dtype)
    iScale = np.zeros(iSize, dtype=patchesIn.dtype)
    iCnt = 0

    if l2DPatching:  # 2D patching on a 2D/3D image
        for h in range(0, iSize[2]):
            for i in range(0, iSize[0] - patch_size[0] + 1, stride):
                for j in range(0, iSize[1] - patch_size[1] + 1, stride):
                    #lMask = np.zeros(iSize)
                    #lMask[i:i + patch_size, j:j + patch_size, h] = 1
                    #lMask = lMask == 1

                    # TODO: should work for both 2D and 3D image since last dimension is appended -> crop it away later?
                    if overlap_mode == 'avg':
                        img[i:i + patch_size[0], j:j + patch_size[1], h] += patchesIn[iCnt, :, :]
                    else:
                        img[i:i + patch_size[0], j:j + patch_size[1], h] = patchesIn[iCnt, :, :]
                    iScale[i:i + patch_size[0], j:j + patch_size[1], h] += 1
                    #iScale[lMask] = iScale[lMask] + 1
                    #iScale[i:i + patch_size, j:j + patch_size, h] = float(min(i + 1, patch_size, iSize[0] - i) * min(j + 1, patch_size, iSize[1] - j))
                    iCnt += 1
    else:
        for h in range(0, iSize[2] - patch_size[2] + 1, stride):
            for i in range(0, iSize[0] - patch_size[0] + 1, stride):
                for j in range(0, iSize[1] - patch_size[1] + 1, stride):

                    # TODO: should work for both 2D and 3D image since last dimension is appended -> crop it away later?
                    if overlap_mode == 'avg':
                        img[i:i + patch_size[0], j:j + patch_size[1], h:h + patch_size[2]] += patchesIn[iCnt, :, :]
                    else:
                        img[i:i + patch_size[0], j:j + patch_size[1], h:h + patch_size[2]] = patchesIn[iCnt, :, :]
                    iScale[i:i + patch_size[0], j:j + patch_size[1], h:h + patch_size[2]] += 1
                    iCnt += 1

    if overlap_mode == 'avg':
        iScale[iScale == 0] = 1
        img = np.divide(img, iScale)

    return img, iScale
