"""
@author: Sebastian Milde, Thomas Kuestner
"""

import numpy as np
import dicom
import dicom_numpy
import os
import shelve
from utils.Patching import*
import cProfile


def fPreprocessData(pathDicom, patchSize, patchOverlap, ratio_labeling):
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
    #mask_numpy_array = create_MASK_Array(proband, model, dicom_numpy_array.shape[0], dicom_numpy_array.shape[1], dicom_numpy_array.shape[2])
    # Normalisation
    range_norm = [0, 1]
    scale_dicom_numpy_array = (dicom_numpy_array - np.min(dicom_numpy_array)) * (range_norm[1] - range_norm[0]) / (np.max(dicom_numpy_array) - np.min(dicom_numpy_array))

    # RigidPatching
    dPatches, dLabel = fRigidPatching(scale_dicom_numpy_array, patchSize, patchOverlap, mask_numpy_array, ratio_labeling)

    return dPatches, dLabel

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

    return voxel_ndarray

def create_MASK_Array(proband, model, mrt_height, mrt_width, mrt_depth):
    mask = np.zeros((mrt_height, mrt_width, mrt_depth))
    # TODO: adapt path
    loadMark = shelve.open("C:/Users/Sebastian Milde/Pictures/Universitaet/Masterarbeit/Markings/" + proband + ".slv")
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

