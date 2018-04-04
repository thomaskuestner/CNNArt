from utils.Unpatching import *
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
import dicom
import dicom_numpy as dicom_np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import time

path = 'D:/med_data/MRPhysics/MA Results/FCN/FCN 3D-VResFCN-Upsampling small_3D_80x80x16_2018-03-09_13-10/model_predictions.mat'
mat = sio.loadmat(path)

########################################################################################################################
currentDataDir = 'D:/med_data/MRPhysics/newProtocol/16_mj/dicom_sorted/t1_tse_tra_Kopf_Motion_0003'

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
########################################################################################################################

######
patches = 'D:/med_data/MRPhysics/DeepLearningArt_Output/Datasets/Patients-1_Datasets-1_3D_SegMask_80x80x16_Overlap-0.6_Labeling-mask_Split-simpleRand/datasets.hdf5'

with h5py.File(patches, 'r') as hf:
    X_train = hf['X_train'][:]
    X_validation = hf['X_validation'][:]
    X_test = hf['X_test'][:]
    Y_train = hf['Y_train'][:]
    Y_validation = hf['Y_validation'][:]
    Y_test = hf['Y_test'][:]
    Y_segMasks_train = hf['Y_segMasks_train'][:]
    Y_segMasks_validation = hf['Y_segMasks_validation'][:]
    Y_segMasks_test = hf['Y_segMasks_test'][:]


predictions = mat['prob_pre']

# i = 29
# j = 10
# for i in range(0, 80):
#     for j in range(0, 1):
#         plt.cla()
#         plt.subplot(141)
#         plt.imshow(np.squeeze(X_test[i, :,:, j]), cmap='gray')
#         plt.subplot(142)
#         plt.imshow(np.squeeze(Y_segMasks_test[i, :, :, j]), cmap='gray')
#         plt.subplot(143)
#         plt.imshow(np.squeeze(predictions[i, :, :, j, 1]), cmap='gray')
#         plt.subplot(144)
#         plt.imshow(np.squeeze(predictions[i, :, :, j, 0]), cmap='gray')
#         plt.show()
#         time.sleep(0.2)

#####

# colormap for visualization of classification results
colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]  # green -> yellow -> red
cmap_name = 'artifact_map_colors'
artifact_colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
colormap = artifact_colormap(np.arange(artifact_colormap.N))
# Set alpha
colormap[:,-1] = np.linspace(0, 0.3, artifact_colormap.N)
# Create new colormap
colormap = ListedColormap(colormap)


one_colors = [(1,0,0), (1,0,0)]
one_color_colormap = LinearSegmentedColormap.from_list('one_color', one_colors, N=100)
one_color_colormap = one_color_colormap(np.arange(one_color_colormap.N))
one_color_colormap[:,-1] = np.linspace(0, 0.2, 100)
one_color_colormap = ListedColormap(one_color_colormap)


unpatched_img_foreground = fUnpatchSegmentation(predictions, [80, 80, 16], 0.6, [256, 196, 40], 1)
unpatched_img_background = fUnpatchSegmentation(predictions, [80, 80, 16], 0.6, [256, 196, 40], 0)
unpatched_img_mask = fUnpatchSegmentation(Y_segMasks_test, [80, 80, 16], 0.6, [256, 196, 40], 1000)

index = 25

unpatched_img_foreground = np.squeeze(unpatched_img_foreground[:,:,index])
unpatched_img_background = np.squeeze(unpatched_img_background[:,:,index])
unpatched_img_mask = np.squeeze(unpatched_img_mask[:,:,index])

slice = np.squeeze(voxel_ndarray[:,:,index]);

ax2 = plt.subplot(131)
plt.imshow(slice, cmap='gray')
plt.imshow(unpatched_img_mask, cmap=one_color_colormap, interpolation='nearest', alpha=1.)
plt.title('Ground Truth Segmentation Mask')


ax1 = plt.subplot(132)
plt.imshow(slice, cmap='gray')
plt.imshow(unpatched_img_foreground, cmap=colormap, interpolation='nearest', alpha=1.)
plt.title('Predicted Foreground')
plt.colorbar(ax=ax1)

ax = plt.subplot(133)
plt.imshow(slice, cmap='gray')
plt.imshow(unpatched_img_background, cmap=colormap, interpolation='nearest', alpha=1.)
plt.title('Predicted Background')


plt.colorbar(ax=ax)
plt.show(block=True)