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
import copy

path = 'D:/med_data/MRPhysics/MA Results/2D_64x64/Multiclass SE-ResNet-56_2D_64x64_2018-03-07_11-48/model_predictions.mat'
mat = sio.loadmat(path)

datapath = 'D:/med_data/MRPhysics/DeepLearningArt_Output/Datasets/Patients-1_Datasets-1_2D_64x64_Overlap-0.85_Labeling-mask_Split-simpleRand'
with h5py.File(datapath + os.sep + "datasets.hdf5", 'r') as hf:
    X_test = hf['X_test'][:]
    Y_test = hf['Y_test'][:]

#########################################################################################
#########################################################################################
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
#########################################################################################
#########################################################################################
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
one_color_colormap[:,-1] = np.linspace(0, 0.25, 100)

red_colormap = ListedColormap(one_color_colormap)
blue_colormap = copy.deepcopy(red_colormap)
blue_colormap.colors[:,0] = 0
blue_colormap.colors[:,1] = 1

###########################################################################################

confusion_matrix = mat['confusion_matrix']
predictions = mat['prob_pre']

#unpatched_img = fMulticlassUnpatch2D(predictions, [64, 64], 0.85, [256, 200, 40])
#unpatched_img_0 = fUnpatch2D(predictions, [64, 64], 0.85, [256, 200, 40], 0)
#unpatched_img_1 = fUnpatch2D(predictions, [64, 64], 0.85, [256, 200, 40], 1)

#sio.savemat('D:/med_data/unpatched.mat', {'unpatched_img': unpatched_img})

unpatched_img = sio.loadmat('D:/med_data/unpatched.mat')
unpatched_img = unpatched_img['unpatched_img']

index = 30

img_0 = np.squeeze(unpatched_img[:,:, 0, index])
img_1 = np.squeeze(unpatched_img[:,:, 1, index])

slice = np.squeeze(voxel_ndarray[:,:, index])

f, ax = plt.subplots(figsize=(11, 9))
plt.imshow(slice, cmap='gray')
#plt.imshow(img_0, cmap=blue_colormap, interpolation='nearest', vmin=0, vmax=1)
plt.colorbar()
plt.imshow(img_1, cmap=red_colormap, interpolation='nearest', vmin=0, vmax=1)
plt.colorbar()

plt.show(block=True)