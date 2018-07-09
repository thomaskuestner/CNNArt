import numpy as np
import dicom
import dicom_numpy
import os
import shelve
from GUI.PyQt.utilsGUI.RigidPatching import *
import cProfile
import json
from matplotlib import path

def fPreprocessData(Path_Markings, mrt_Folder,proband, model, patchSize, patchOverlap, ratio_labeling, dimension):

    # Creation of dicom_numpy_array and mask_numpy_array
    dicom_numpy_array = create_DICOM_Array(mrt_Folder, proband, model)
    mask_numpy_array = create_MASK_Array(Path_Markings,proband, model, dicom_numpy_array.shape[0], dicom_numpy_array.shape[1],
                                     dicom_numpy_array.shape[2])
    # Normalisation
    range_norm = [0, 1]
    scale_dicom_numpy_array = (dicom_numpy_array - np.min(dicom_numpy_array)) * (range_norm[1] - range_norm[0]) / (np.max(dicom_numpy_array) - np.min(dicom_numpy_array))

    # RigidPatching
    if dimension == '2D':
        dPatches, dLabel, nbPatches = fRigidPatching(scale_dicom_numpy_array, patchSize, patchOverlap, mask_numpy_array,
                                                     ratio_labeling)
    else:
        dPatches, dLabel, nbPatches = fRigidPatching3D(scale_dicom_numpy_array, patchSize, patchOverlap, mask_numpy_array,
                                                     ratio_labeling)

    return dPatches, dLabel, nbPatches


def mask_rectangle(x_coo1, y_coo1, x_coo2, y_coo2, layer_mask, art_no):
    x_coo1 = round(x_coo1)
    y_coo1 = round(y_coo1)
    x_coo2 = round(x_coo2)
    y_coo2 = round(y_coo2)
    layer_mask[int(min(y_coo1, y_coo2)):int(max(y_coo1, y_coo2)) + 1, int(min(x_coo1, x_coo2)):int(max(x_coo1, x_coo2)) + 1] = art_no

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

def create_DICOM_Array(mrt_Folder, proband, model):
    filenames_list = []
    PathDicom = mrt_Folder + "/" + proband + "/dicom_sorted/" + model + "/"
    file_list = os.listdir(PathDicom)
    for file in file_list:
        filenames_list.append(PathDicom + file)
    datasets = [dicom.read_file(f) \
                for f in filenames_list]
    try:
        voxel_ndarray, _ = dicom_numpy.combine_slices(datasets)
        print(voxel_ndarray.dtype)
        voxel_ndarray = voxel_ndarray.astype(float) #float
        voxel_ndarray = np.swapaxes(voxel_ndarray, 0, 1)
        print(voxel_ndarray.dtype)
        # voxel_ndarray = voxel_ndarray[:-1:]
        # print(voxel_ndarray.shape)
    except dicom_numpy.DicomImportException:
        # invalid DICOM data
        raise

    return voxel_ndarray

###########################################################################################################################
# Function: create_MASK_Array                                                                                             #
# This function creates an 3D array with all labels(artefact types)                                                       #
# Input: Path_mark ----> Path for all saved markings of dicom images                                                      #
#        proband ----> name of chosen proband, example: '01_ab'                                                           #
#        model ----> name of model, example: 't1_tse_tra_Kopf_0002'                                                       #
#        mrt_height ----> the height of the dicom image                                                                   #
#        mrt_width ----> the width of the dicom image                                                                     #
#        mrt_depth ----> the number of slices of the dicom 3D array                                                       #
# Output: mask ----> 3D array with all labels                                                                             #
###########################################################################################################################

def create_MASK_Array(pathMarking, proband, model, mrt_height, mrt_width, mrt_depth):
    mask = np.zeros((mrt_height, mrt_width, mrt_depth))

    #JSON file
    with open(pathMarking, 'r') as fp:
        loadMark = json.load(fp)

    #loadMark = shelve.open(Path_mark + proband + ".dumbdbm.slv")

    if model in loadMark:
        marks = loadMark[model]
        for key in marks:
            num = int(key.find("_"))
            img_no = int(key[0:num])
            key2 = key[num + 1:len(key)]
            num = int(key2.find("_"))
            str_no = key2[0:num]
            tool_no = int(str_no[0])
            art_no = int(str_no[1])
            #print(mask[:, :, img_no].shape)
            mask_lay = mask[:, :, img_no]
            p = marks[key]
            if tool_no == 1:
                # p has to be an ndarray
                p = np.asarray(p['points'])
                mask_lay = mask_rectangle(p[0], p[1], p[2], p[3], mask_lay, art_no)
            elif tool_no == 2:
                # p has to be an ndarray
                p = np.asarray(p['points'])
                mask_lay = mask_ellipse(p[0], p[1], p[2], p[3], mask_lay, art_no)
            elif tool_no == 3:
                # p has to be a matplotlib path
                p = path.Path(np.asarray(p['vertices']), p['codes'])
                mask_lay = mask_lasso(p, mask_lay, art_no)
            mask[:, :, img_no] = mask_lay
    else:
        pass

    #loadMark.close()  # used for shelve
    #print(mask.dtype)

    return mask