import os
import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np
import dicom_numpy as dicom_np
import dicom
from skimage import data, img_as_float
from skimage import exposure
import keras.backend as K

def histogram_equalization(x):
    # if np.random.random() < 0.5:
    x = exposure.equalize_hist(x)
    return x

def contrast_stretching(x):
    # if np.random.random() < 0.5:
    p2, p98 = np.percentile(x, (2, 98))
    x = exposure.rescale_intensity(x, in_range=(p2, p98))
    return x

def adaptive_equalization(x):
    # if np.random.random() < 0.5:
    x = exposure.equalize_adapthist(x, clip_limit=0.03)
    return x

def zca(x):
    x = np.asarray(x, dtype=K.floatx())
    x = np.copy(x)
    flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
    sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
    u, s, _ = linalg.svd(sigma)
    principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 1e-6))), u.T)

    flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
    whitex = np.dot(flatx, principal_components)
    x = np.reshape(whitex, x.shape)
    return x

currentDataDir3 = 'D:/med_data/MRPhysics/newProtocol/01_ab/dicom_sorted/t1_tse_tra_Kopf_0002'
#currentDataDir = 'D:/med_data/MRPhysics/newProtocol/16_mj/dicom_sorted/t2_tse_tra_fs_Becken_0009'
#currentDataDir = 'D:/med_data/MRPhysics/newProtocol/16_mj/dicom_sorted/t1_tse_tra_Kopf_Motion_0003'
#currentDataDir = 'D:/med_data/MRPhysics/newProtocol/01_ab/dicom_sorted/t1_tse_tra_Kopf_Motion_0003'
currentDataDir2 = 'D:/med_data/MRPhysics/newProtocol/04_dc/dicom_sorted/t1_tse_tra_Kopf_Motion_0003'
currentDataDir = 'D:/med_data/MRPhysics/newProtocol/04_dc/dicom_sorted/t1_tse_tra_Kopf_0002'

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




fileNames = os.listdir(currentDataDir2)
fileNames = [os.path.join(currentDataDir2, f) for f in fileNames]
# read DICOMS
dicomDataset = [dicom.read_file(f) for f in fileNames]
# Combine DICOM Slices to a single 3D image (voxel)
try:
    voxel_ndarray2, ijk_to_xyz = dicom_np.combine_slices(dicomDataset)
    voxel_ndarray2 = voxel_ndarray2.astype(float)
    voxel_ndarray2 = np.swapaxes(voxel_ndarray2, 0, 1)
except dicom_np.DicomImportException as e:
    #invalid DICOM data
    raise





fileNames = os.listdir(currentDataDir3)
fileNames = [os.path.join(currentDataDir3, f) for f in fileNames]
# read DICOMS
dicomDataset = [dicom.read_file(f) for f in fileNames]
# Combine DICOM Slices to a single 3D image (voxel)
try:
    voxel_ndarray3, ijk_to_xyz = dicom_np.combine_slices(dicomDataset)
    voxel_ndarray3 = voxel_ndarray3.astype(float)
    voxel_ndarray3 = np.swapaxes(voxel_ndarray3, 0, 1)
except dicom_np.DicomImportException as e:
    #invalid DICOM data
    raise







voxel_ndarray = np.concatenate((voxel_ndarray, voxel_ndarray2, voxel_ndarray3), axis=-1)

# normalization of DICOM voxel
rangeNorm = [0,1]
norm_voxel_ndarray = (voxel_ndarray-np.min(voxel_ndarray))*(rangeNorm[1]-rangeNorm[0])/(np.max(voxel_ndarray)-np.min(voxel_ndarray))

img = norm_voxel_ndarray[40:140,40:140, 26]
plt.subplot(221)
plt.title('Reference', fontsize=14)
plt.imshow(img, cmap='gray')
plt.axis('off')


voxel = np.transpose(norm_voxel_ndarray, (2, 0, 1))
voxel = np.expand_dims(voxel, axis=-1)
voxel = voxel[:, 40:140, 40:140, :]
voxel = zca(voxel)
plt.subplot(222)
plt.title('ZCA Whitening', fontsize=14)
img2 = voxel[26, :,:,0]
plt.imshow(img2, cmap='gray')
plt.axis('off')

img = np.cov(img)
plt.subplot(223)
plt.title('Covariance Reference', fontsize=14)
plt.imshow(img, cmap='gray')
plt.axis('off')

img2 = np.cov(img2)
img2 = (img2-np.min(img2))*(rangeNorm[1]-rangeNorm[0])/(np.max(img2)-np.min(img2))
plt.subplot(224)
plt.title('Covariance ZCA Whitening', fontsize=14)
plt.imshow(img2, cmap='gray')
plt.axis('off')


# img = adaptive_equalization(img)
# plt.subplot(224)
# plt.title('Adaptive Equalization', fontsize=14)
# plt.imshow(img, cmap='gray')
# plt.axis('off')

plt.show()

print()




