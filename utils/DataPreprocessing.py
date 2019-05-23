"""
@author: Sebastian Milde, Thomas Kuestner
"""

import numpy as np
import pydicom as dicom
import dicom_numpy
import os
import shelve
from utils.Patching import*
import cProfile
import utils.Training_Test_Split as ttsplit
import utils.scaling as scaling


def fPreprocessData(pathDicom, patchSize, patchOverlap, ratio_labeling, sLabeling, sTrainingMethod='None', range_norm = [0, 1]):
    # set variables
    model = os.path.basename(os.path.dirname(pathDicom))
    dir = os.path.dirname(os.path.dirname(pathDicom))
    if os.path.basename(dir) == 'dicom_sorted':
        proband = os.path.basename(os.path.dirname(dir))
    else:
        proband = os.path.basename(dir)
    #pathDicom = mrt_Folder + "/" + proband + "/dicom_sorted/" + model + "/"
    # Creation of dicom_numpy_array and mask_numpy_array
    dicom_numpy_array = create_DICOM_Array(os.path.join(pathDicom, ''))
    mask_numpy_array = create_MASK_Array(proband, model, dicom_numpy_array.shape[0], dicom_numpy_array.shape[1], dicom_numpy_array.shape[2])
    # Normalisation
    scale_dicom_numpy_array = (dicom_numpy_array - np.min(dicom_numpy_array)) * (range_norm[1] - range_norm[0]) / (np.max(dicom_numpy_array) - np.min(dicom_numpy_array)) + range_norm[0]

    # RigidPatching
    if len(patchSize) == 3: # 3D patches
        dPatches, dLabel = fRigidPatching3D(scale_dicom_numpy_array, patchSize, patchOverlap, mask_numpy_array,ratio_labeling, sLabeling, sTrainingMethod)
        dPatches = np.transpose(dPatches, (3, 0, 1, 2))
    else:
        dPatches, dLabel = fRigidPatching(scale_dicom_numpy_array, patchSize, patchOverlap, mask_numpy_array,ratio_labeling, sLabeling)
        dPatches = np.transpose(dPatches, (2, 0, 1))

    return dPatches, dLabel

def fPreprocessDataCorrection(cfg, dbinfo):
    """
    Perform patching to reference and artifact images according to given patch size.
    @param cfg: the configuration file loaded from config/param.yml
    @param dbinfo: database related info
    @return: patches from reference and artifact images and an array which stores the corresponding patient index
    """
    train_ref = []
    test_ref = []
    train_art = []
    test_art = []

    sTrainingMethod = cfg['sTrainingMethod']  # options of multiscale
    patchSize = cfg['patchSize']
    range_norm = cfg['range']
    lScaleFactor = cfg['lScaleFactor']

    scpatchSize = patchSize
    if sTrainingMethod != "scalingPrior":
        lScaleFactor = [1]
    # Else perform scaling:
    #   images will be split into pathces with size scpatchSize and then scaled to patchSize
    for iscalefactor in lScaleFactor:
        lDatasets = cfg['selectedDatabase']['dataref'] + cfg['selectedDatabase']['dataart']
        scpatchSize = [int(psi / iscalefactor) for psi in patchSize]
        if len(patchSize) == 3:
            dRefPatches = np.empty((0, scpatchSize[0], scpatchSize[1], scpatchSize[2]))
            dArtPatches = np.empty((0, scpatchSize[0], scpatchSize[1], scpatchSize[2]))
        else:
            dRefPatches = np.empty((0, scpatchSize[0], scpatchSize[1]))
            dArtPatches = np.empty((0, scpatchSize[0], scpatchSize[1]))

        dRefPats = np.empty((0, 1))
        dArtPats = np.empty((0, 1))

        for ipat, pat in enumerate(dbinfo.lPats):
            if os.path.exists(dbinfo.sPathIn + os.sep + pat + os.sep + dbinfo.sSubDirs[1]):
                for iseq, seq in enumerate(lDatasets):
                    # patches and labels of reference/artifact
                    if os.path.exists(dbinfo.sPathIn + os.sep + pat + os.sep + dbinfo.sSubDirs[1] + os.sep + seq):
                        tmpPatches, tmpLabels = fPreprocessData(os.path.join(dbinfo.sPathIn, pat, dbinfo.sSubDirs[1], seq), scpatchSize, cfg['patchOverlap'], 1, 'volume', range_norm)

                        if iseq < len(lDatasets)/2:
                            dRefPatches = np.concatenate((dRefPatches, tmpPatches), axis=0)
                            dRefPats = np.concatenate((dRefPats, ipat * np.ones((tmpPatches.shape[0], 1), dtype=np.int)), axis=0)
                        else:
                            dArtPatches = np.concatenate((dArtPatches, tmpPatches), axis=0)
                            dArtPats = np.concatenate((dArtPats, ipat * np.ones((tmpPatches.shape[0], 1), dtype=np.int)), axis=0)
            else:
                pass

        assert(dRefPatches.shape == dArtPatches.shape and dRefPats.shape == dArtPats.shape)

        # perform splitting
        print('Start splitting')
        if cfg['correction']['test_patient'] in dbinfo.lPats:
            test_index = dbinfo.lPats.index(cfg['correction']['test_patient'])
        else:
            test_index = -1
        train_ref_sp, test_ref_sp, train_art_sp, test_art_sp = ttsplit.fSplitDatasetCorrection(cfg['sSplitting'],
                                                                                               dRefPatches, dArtPatches,
                                                                                               dRefPats,
                                                                                               cfg['dSplitval'],
                                                                                               cfg['nFolds'],
                                                                                               test_index)

        print('Start scaling')
        # perform scaling: sc for scale
        train_ref_sc, test_ref_sc, _ = scaling.fscaling(train_ref_sp, test_ref_sp, scpatchSize, iscalefactor)
        train_art_sc, test_art_sc, _ = scaling.fscaling(train_art_sp, test_art_sp, scpatchSize, iscalefactor)

        if len(train_ref) == 0:
            train_ref = train_ref_sc
            test_ref = test_ref_sc
            train_art = train_art_sc
            test_art = test_art_sc
        else:
            train_ref = np.concatenate((train_ref, train_ref_sc), axis=1)
            test_ref = np.concatenate((test_ref, test_ref_sc), axis=1)
            train_art = np.concatenate((train_art, train_art_sc), axis=1)
            test_art = np.concatenate((test_art, test_art_sc), axis=1)

    return train_ref, test_ref, train_art, test_art

def mask_rectangle(x_coo1, y_coo1, x_coo2, y_coo2, layer_mask, art_no):
    x_coo1 = round(x_coo1)
    y_coo1 = round(y_coo1)
    x_coo2 = round(x_coo2)
    y_coo2 = round(y_coo2)
    layer_mask[min(y_coo1, y_coo2):max(y_coo1, y_coo2) + 1, min(x_coo1, x_coo2):max(x_coo1, x_coo2) + 1] = art_no

    return layer_mask

def mask_ellipse(x_coo1, y_coo1, x_coo2, y_coo2, layer_mask, art_no):
    x_coo1 = round(x_coo1)
    y_coo1 = round(y_coo1)
    x_coo2 = round(x_coo2)
    y_coo2 = round(y_coo2)
    b_y, a_x = abs((y_coo2 - y_coo1) / 2), abs((x_coo2 - x_coo1) / 2)
    y_m, x_m = min(y_coo1, y_coo2) + b_y - 1, min(x_coo1, x_coo2) + a_x - 1
    y_height = layer_mask.shape[0]
    x_width = layer_mask.shape[1]
    print(y_m, x_m, y_height, x_width)
    y, x = np.ogrid[-y_m:y_height - y_m, -x_m:x_width - x_m]
    mask = b_y * b_y * x * x + a_x * a_x * y * y <= a_x * a_x * b_y * b_y
    layer_mask[mask] = art_no

    return layer_mask

def mask_lasso(p, layer_mask, art_no):
    pix1 = np.arange(layer_mask.shape[1])
    pix2 = np.arange(layer_mask.shape[0])
    xv, yv = np.meshgrid(pix1, pix2)
    pix = np.vstack((xv.flatten(), yv.flatten())).T

    ind = p.contains_points(pix, radius=1)
    lin = np.arange(layer_mask.size)
    newArray = layer_mask.flatten()
    newArray[lin[ind]] = art_no
    mask_lay = newArray.reshape(layer_mask.shape)

    return mask_lay

def create_DICOM_Array(PathDicom):
    filenames_list = []

    file_list = os.listdir(PathDicom)
    for file in file_list:
        filenames_list.append(PathDicom + file)
    datasets = [dicom.read_file(f) \
                for f in filenames_list]
    try:
        voxel_ndarray, _ = dicom_numpy.combine_slices(datasets)
        voxel_ndarray = voxel_ndarray.astype(float)
        voxel_ndarray = np.swapaxes(voxel_ndarray, 0, 1)
        print(voxel_ndarray.dtype)
        # voxel_ndarray = voxel_ndarray[:-1:]
        # print(voxel_ndarray.shape)
    except dicom_numpy.DicomImportException:
        # invalid DICOM data
        raise

    print(voxel_ndarray.shape)
    return voxel_ndarray

def create_MASK_Array(proband, model, mrt_height, mrt_width, mrt_depth):
    mask = np.zeros((mrt_height, mrt_width, mrt_depth))
    # TODO: adapt path
    try:
        loadMark = shelve.open("../Markings/" + proband + ".slv")
        print(model)
        if loadMark.has_key(model):
            marks = loadMark[model]
            for key in marks:
                num = int(key.find("_"))
                img_no = int(key[0:num])
                key2 = key[num + 1:len(key)]
                num = int(key2.find("_"))
                str_no = key2[0:num]
                tool_no = int(str_no[0])
                art_no = int(str_no[1])
                mask_lay = mask[:, :, img_no]
                p = marks[key]
                print(p)
                if tool_no == 1:
                    mask_lay = mask_rectangle(p[0], p[1], p[2], p[3], mask_lay, art_no)
                elif tool_no == 2:
                    mask_lay = mask_ellipse(p[0], p[1], p[2], p[3], mask_lay, art_no)
                elif tool_no == 3:
                    mask_lay = mask_lasso(p, mask_lay, art_no)

                mask[:, :, img_no] = mask_lay
        else:
            pass

        loadMark.close()
        return mask
    except:
        return mask

def fReadData(pathDicom):
    # set variables
    model = os.path.basename(os.path.dirname(pathDicom))
    dir = os.path.dirname(os.path.dirname(pathDicom))
    if os.path.basename(dir) == 'dicom_sorted':
        proband = os.path.basename(os.path.dirname(dir))
    else:
        proband = os.path.basename(dir)

    dicom_numpy_array = create_DICOM_Array(os.path.join(pathDicom, ''))
    range_norm = [0, 1]
    scale_dicom_numpy_array = (dicom_numpy_array - np.min(dicom_numpy_array)) * (range_norm[1] - range_norm[0]) / (
    np.max(dicom_numpy_array) - np.min(dicom_numpy_array))

    return scale_dicom_numpy_array
