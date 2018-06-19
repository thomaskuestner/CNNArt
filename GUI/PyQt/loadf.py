import os
import dicom
import dicom_numpy
import numpy as np
import scipy.ndimage
from PyQt5 import QtCore

class loadImage(QtCore.QThread):
    trigger0 = QtCore.pyqtSignal()
    def __init__(self, PathDicom):
        super(loadImage, self).__init__()
        self.PathDicom = PathDicom

    def run(self):
        self.sscan = self.load_scan(self.PathDicom)
        if self.sscan:
            self.simage = np.stack([s.pixel_array for s in self.sscan])
            self.voxel_ndarray = np.swapaxes(self.simage, 0, 2)
            spacing = map(float, ([self.sscan[0].SliceThickness] + self.sscan[0].PixelSpacing))
            spacing = np.array(list(spacing))
            new_spacing = [1, 1, 1]
            resize_factor = spacing / new_spacing
            new_real_shape = self.simage.shape * resize_factor
            self.new_shape = np.round(new_real_shape)
            self.new_shape[0], self.new_shape[2] = self.new_shape[2], self.new_shape[0]
            self.new_shape=self.new_shape.tolist()
        self.trigger0.emit()

    def load_scan(self, path):
        if path:
            slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key=lambda x: int(x.InstanceNumber))
            try:
                slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
            except:
                slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

            for s in slices:
                s.SliceThickness = slice_thickness
            return slices
