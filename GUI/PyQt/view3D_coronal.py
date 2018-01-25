from __future__ import print_function

from UnpatchNew import*
import scipy.io as sio
import matplotlib as mpl
from collections import Counter

import dicom
import numpy as np
import os
import scipy.ndimage
import dicom_numpy
from matplotlib import pyplot as plt

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))  # scan is list
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

data_path = "C:\\Users\hansw\Videos\\artefacts\\03\dicom_sorted\\t1_tse_tra_Kopf_Motion_0003"
id = 0
patient = load_scan(data_path)

image = np.stack([s.pixel_array for s in patient])

voxel_ndarray, spacing = resample(image, patient, [1, 1, 1])
voxel_ndarray = np.swapaxes(voxel_ndarray, 0, 2)   # to seb

class IndexTracker(object):
    def __init__(self, X, Y, Z):
        self.fig, self.ax = plt.subplots(1, 1)
        #self.ax.set_title('Axial View')

        self.X = X
        self.slices = X.shape[1]
        self.ind = self.slices//2
        self.Y = Y
        self.Z = Z
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        plt.cla() # not clf
        self.update()

    def update(self):
        self.im1 = self.ax.imshow(np.swapaxes(self.X[:, self.ind, :], 0, 1), cmap='gray', vmin=0, vmax=2094)

        self.cmap1 = mpl.colors.ListedColormap(['blue', 'purple','cyan', 'yellow', 'green'])
        self.im2 = self.ax.imshow(np.swapaxes(self.Y[:, self.ind, :], 0, 1), cmap=self.cmap1, alpha=.3, vmin=1, vmax=6, extent=[0,196,132,0])

        plt.rcParams['hatch.color'] = 'r'
        self.im3 = self.ax.contourf(np.transpose(self.Z[:, self.ind, :]), hatches=[None, '//', '\\', 'XX'], colors='none', edges='r', levels=np.arange(5), extent=[0,196,132,0])

        #plt.xlim(0, voxel_ndarray.shape[0] * dx)
        #plt.ylim(voxel_ndarray.shape[1] * dy, 0)
        plt.axis('off')
        self.ax.set_ylabel('slice %s' % self.ind)
        self.fig.canvas.draw_idle()

#fig, ax = plt.subplots(1, 1)    ### 2 output?
PathDicom = "C:\\Users\hansw\Videos\\artefacts\\03\dicom_sorted\\t1_tse_tra_Kopf_Motion_0003"
files = sorted([os.path.join(PathDicom, file) for file in os.listdir(PathDicom)], key=os.path.getctime)
datasets = [dicom.read_file(f) \
                        for f in files]
try:
    voxel_ndarray0, pixel_space = dicom_numpy.combine_slices(datasets)
except dicom_numpy.DicomImportException:
    raise
dx, dy, dz = 1.0, 1.0, pixel_space[2][2]
print(pixel_space)

PatchSize = np.array((40.0, 40.0))     # only 2 elements
PatchOverlay = 0.5
Path = "Pred_result.mat"
conten = sio.loadmat(Path)
prob_test = conten['prob_pre']

IndexType = np.argmax(prob_test, 1)
IndexType[IndexType==0] = 1
IndexType[(IndexType>1) & (IndexType<4)] = 2
IndexType[(IndexType>6) & (IndexType<9)] = 3
IndexType[(IndexType>3) & (IndexType<7)] = 4
IndexType[IndexType>8] = 5

a = Counter(IndexType).most_common(1)
domain = a[0][0]

PType = np.delete(prob_test,[1,3,5,6,8,10],1) # only 5 region left
PArte = np.delete(prob_test,[0,2,4,7,9],1)
PArte[:,[4,5]] = PArte[:,[5,4]]
PNew = np.concatenate((PType, PArte), axis=1)
IndexArte = np.argmax(PNew,1)

IType = UnpatchType(IndexType, domain, PatchSize, PatchOverlay, voxel_ndarray0.shape)
IArte = UnpatchArte(IndexArte, PatchSize, PatchOverlay, voxel_ndarray0.shape)

tracker = IndexTracker(voxel_ndarray, IType, IArte)

plt.colorbar(tracker.im2)
#plt.axis('off')
plt.show()