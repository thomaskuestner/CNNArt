from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import dicom
import dicom_numpy
from UnpatchNew import*
import scipy.io as sio
import matplotlib as mpl
from collections import Counter

class IndexTracker(object):
    def __init__(self, X, Y, Z):
        self.fig, self.ax = plt.subplots(1, 1)
        #self.ax.set_title('Axial View')

        self.X = X
        self.slices = X.shape[2]
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
        self.im1 = self.ax.imshow(np.swapaxes(self.X[:, :, self.ind], 0, 1), cmap='gray', vmin=0, vmax=2094)

        self.cmap1 = mpl.colors.ListedColormap(['blue', 'purple','cyan', 'yellow', 'green'])
        self.im2 = self.ax.imshow(np.swapaxes(self.Y[:, :, self.ind], 0, 1), cmap=self.cmap1, alpha=.3, vmin=1, vmax=6)

        plt.rcParams['hatch.color'] = 'r'
        self.im3 = self.ax.contourf(np.transpose(self.Z[:, :, self.ind]), hatches=[None, '//', '\\', 'XX'], colors='none', edges='r', levels=np.arange(5))

        self.ax.set_ylabel('slice %s' % self.ind)
        self.fig.canvas.draw_idle()


#fig, ax = plt.subplots(1, 1)    ### 2 output?

PathDicom = "C:\\Users\hansw\Videos\\artefacts\\03\dicom_sorted\\t1_tse_tra_Kopf_Motion_0003"
files = sorted([os.path.join(PathDicom, file) for file in os.listdir(PathDicom)], key=os.path.getctime)
datasets = [dicom.read_file(f) \
                        for f in files]
try:
    voxel_ndarray, pixel_space = dicom_numpy.combine_slices(datasets)
except dicom_numpy.DicomImportException:
    raise
dx, dy, dz = 1.0, 1.0, pixel_space[2][2]

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

IType = UnpatchType(IndexType, domain, PatchSize, PatchOverlay, voxel_ndarray.shape)
IArte = UnpatchArte(IndexArte, PatchSize, PatchOverlay, voxel_ndarray.shape)

tracker = IndexTracker(voxel_ndarray, IType, IArte)

plt.xlim(0, voxel_ndarray.shape[0] * dx)
plt.ylim(voxel_ndarray.shape[1] * dy, 0)
plt.colorbar(tracker.im2)

plt.show()