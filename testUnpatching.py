from utils.Unpatching import *
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
import dicom
import dicom_numpy as dicom_np

path = 'D:/med_data/MRPhysics/MA Results/Output_Learning-9.3.18/Multiclass SE-ResNet-56_2D_64x64_2018-03-07_11-48/model_predictions.mat'
mat = sio.loadmat(path)

datapath = 'D:/med_data/MRPhysics/DeepLearningArt_Output/Datasets/Patients-1_Datasets-1_2D_64x64_Overlap-0.5_Labeling-mask_Split-simpleRand'
with h5py.File(datapath + os.sep + "datasets.hdf5", 'r') as hf:
    X_test = hf['X_test'][:]
    Y_test = hf['Y_test'][:]

#########################################################################################
#########################################################################################
currentDataDir = 'D:/med_data/MRPhysics/newProtocol/12_ms/dicom_sorted/t1_tse_tra_fs_Becken_Motion_0010'

fileNames = os.listdir(currentDataDir)
fileNames = [os.path.join(currentDataDir, f) for f in fileNames]

# read DICOMS
dicomDataset = [dicom.read_file(f) for f in fileNames]

# Combine DICOM Slices to a single 3D image (voxel)
try:
    voxel_ndarray, ijk_to_xyz = dicom_np.combine_slices(dicomDataset)
    voxel_ndarray = voxel_ndarray.astype(float)
    voxel_ndarray = np.swapaxes(voxel_ndarray, 0, 1)
except dicom_np.DicomImportException as e:
    #invalid DICOM data
    raise
#########################################################################################
#########################################################################################


confusion_matrix = mat['confusion_matrix']
predictions = mat['prob_pre']

#unpatched_img = fUnpatch2D(predictions, [64, 64], 0.5, [240, 320, 40], 1)

#sio.savemat('D:/med_data/unpatched.mat', {'unpatched_img': unpatched_img})

unpatched_img = sio.loadmat('D:/med_data/unpatched.mat')
unpatched_img = unpatched_img['unpatched_img']

index = 25

img = np.squeeze(unpatched_img[:,:, index])
slice = np.squeeze(voxel_ndarray[:,:, index])

f, ax = plt.subplots(figsize=(11, 9))
plt.imshow(slice, cmap='gray')
plt.imshow(img, cmap='jet', interpolation='nearest', alpha=.4)

plt.colorbar(ax=ax)
plt.show(block=True)