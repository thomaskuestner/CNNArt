import os
from matplotlib import pyplot as plt
import dicom
import dicom_numpy
from Unpatching import*
import scipy.io as sio

PathDicom = "C:/Users/Sebastian Milde/Pictures/MRT/ab/dicom_sorted/t1_tse_tra_Kopf_0002"

#load Dicom_Array
files = sorted([os.path.join(PathDicom, file) for file in os.listdir(PathDicom)], key=os.path.getctime)
datasets = [dicom.read_file(f) \
                        for f in files]

try:
    voxel_ndarray, pixel_space = dicom_numpy.combine_slices(datasets)
    voxel_ndarray_matlab = voxel_ndarray.transpose(1, 0, 2)
except dicom_numpy.DicomImportException:
    # invalid DICOM data
    raise

dx, dy, dz = 1.0, 1.0, pixel_space[2][2] #pixel_space[0][0], pixel_space[1][1], pixel_space[2][2]
voxel_ndarray = np.swapaxes(voxel_ndarray, 0, 1)
print(voxel_ndarray.shape)
pixel_array = voxel_ndarray[:, :, 0]
print(pixel_array.shape)
#pixel_array = np.swapaxes(pixel_array, 0, 1)
print(pixel_array[0:20,0:20])
imgOverlay = np.zeros((196,256))#np.random.rand(196,256)
imgOverlay[0:50, 0:255] = 1
PatchSize = np.array((40.0, 40.0))
PatchOverlay = 0.5
Path = "C:/Users/Sebastian Milde/Documents/MATLAB/IQA/Codes_FeatureLearning/bestModels/abdomen_4040_lr_0.0001_bs_64.mat"
conten = sio.loadmat(Path)
prob_test = conten['prob_test']
prob_test = prob_test[0:8000, 0]
imglay = fUnpatch(PatchSize, PatchOverlay, voxel_ndarray, prob_test)
x_1d = dx * np.arange(voxel_ndarray.shape[0])
y_1d = dy * np.arange(voxel_ndarray.shape[1])
z_1d = dz * np.arange(voxel_ndarray.shape[2])
fig = plt.figure(dpi=100)
ax = plt.gca()
plt.cla()
plt.gca().set_aspect('equal')  # plt.axes().set_aspect('equal')
plt.xlim(0, voxel_ndarray.shape[0]*dx)
plt.ylim(voxel_ndarray.shape[1]*dy, 0)
plt.set_cmap(plt.gray())
plt.pcolormesh(x_1d, y_1d, np.swapaxes(pixel_array, 0, 1), vmin=0, vmax=2094) #x_1d, y_1d, np.swapaxes(pixel_array, 0, 1)
plt.pcolormesh(x_1d, y_1d, np.swapaxes(imglay[:,:,5], 0, 1), cmap = 'YlGnBu', alpha = 0.2, vmin=0, vmax=1, linewidth = 0, rasterized = True) #jet, alpha = 0.2, YlGnBu, x_1d, y_1d, np.swapaxes(imgOverlay, 0, 1)
plt.colorbar(pad = 0.5, shrink = 1.0, aspect = 5)
plt.show()
