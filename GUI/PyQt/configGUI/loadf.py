import os
import dicom
import numpy as np
from PyQt5 import QtCore
from PIL import Image
import nibabel as nib
import scipy.io as sio

VALID_IMG_FORMAT = ('.CUR', '.ICNS', '.SVG', '.TGA', '.BMP', '.WEBP', '.GIF',
                    '.JPG', '.JPEG', '.PNG', '.PBM', '.PGM', '.PPM', '.TIFF',
                    '.XBM')  # Image formats supported by Qt
VALID_DCM_FORMAT = ('.IMA', '.DCM')  # Image formats supported by dicom reading


class loadImage(QtCore.QThread):
    trigger = QtCore.pyqtSignal()

    def __init__(self, pathDicom):
        # pathDicom is a folder with dicom images: load images in this folder
        # pathDicom is a single file: ima/dcm, .npy, .mat, .nii, .jpg/.tif/.png/other image format
        # pathDicom is a array: print this image direcctly
        # array is 4D: time, slice, row, column
        # array is 5D: time, slice, row, column, channel
        # every image will be stored in self.voxel_ndarray
        super(loadImage, self).__init__()
        self.PathDicom = pathDicom
        self.voxel_ndarray = []
        self.run()

    def run(self):

        if type(self.PathDicom) is str:
            if os.path.isdir(self.PathDicom):
                try:
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
                        self.new_shape = list(self.new_shape)
                except:
                    pass
            elif os.path.isfile(self.PathDicom):
                if self.PathDicom.upper().endswith(VALID_IMG_FORMAT):
                    self.load_img(self.PathDicom)
                elif self.PathDicom.upper().endswith(VALID_DCM_FORMAT):
                    self.load_dcm(self.PathDicom)
                elif self.PathDicom.upper().endswith('.NII'):
                    self.load_nii(self.PathDicom)
                elif self.PathDicom.upper().endswith('.NPY'):
                    self.load_npy(self.PathDicom)
                elif self.PathDicom.upper().endswith('.MAT'):
                    self.load_mat(self.PathDicom)
                self.voxel_ndarray = self.voxel_ndarray.reshape(self.new_shape)

        elif type(self.PathDicom) is np.ndarray:
            # here is pathdicom a numpy array
            self.new_shape = list(self.PathDicom.shape)
            if len(self.new_shape) < 3:
                self.new_shape.append(1)
            elif len(self.new_shape) == 4:
                self.new_shape.insert(0, 1)
            self.voxel_ndarray = self.PathDicom.reshape(self.new_shape)

        self.trigger.emit()

    def load_scan(self, path):
        if path:
            slices = []
            for s in os.listdir(path):
                if '.directory' in s:
                    pass
                else:
                    slice = dicom.read_file(path + '/' + s, force=True)
                    slices.append(slice)
            slices.sort(key=lambda x: int(x.InstanceNumber))
            try:
                slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
            except:
                slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
            for s in slices:
                s.SliceThickness = slice_thickness
            return slices

    def load_img(self, image):
        img = Image.open(image)
        img.load()
        self.voxel_ndarray = np.asarray(img, dtype="int32")
        self.new_shape = list(self.voxel_ndarray.shape)
        if len(self.new_shape) < 3:
            self.new_shape.append(1)
        self.new_shape[0], self.new_shape[1] = self.new_shape[1], self.new_shape[0]
        self.voxel_ndarray = np.swapaxes(self.voxel_ndarray, 0, 1)

    def load_dcm(self, dcmImg):
        self.voxel_ndarray = np.asarray(dicom.read_file(dcmImg).pixel_array, dtype="int32")
        self.new_shape = list(self.voxel_ndarray.shape)
        if len(self.new_shape) < 3:
            self.new_shape.append(1)
        self.new_shape[0], self.new_shape[1] = self.new_shape[1], self.new_shape[0]
        self.voxel_ndarray = np.swapaxes(self.voxel_ndarray, 0, 1)

    def load_nii(self, nii):
        nibImg = nib.load(nii)
        self.voxel_ndarray = np.asarray(nibImg.get_data(), dtype="int32")
        self.new_shape = list(self.voxel_ndarray.shape)
        if len(self.new_shape) < 3:
            self.new_shape.append(1)

    def load_npy(self, npy):
        npyImg = np.load(npy)
        for item in npyImg:
            if '__' not in item and 'readme' not in item:
                arrays = npyImg[item]
                self.voxel_ndarray = [np.expand_dims(array, axis=0) for array in arrays]
        self.voxel_ndarray = np.concatenate(self.voxel_ndarray)
        self.voxel_ndarray = np.swapaxes(self.voxel_ndarray, 0, 2)
        self.voxel_ndarray = np.swapaxes(self.voxel_ndarray, 0, 1)
        self.new_shape = list(self.voxel_ndarray.shape)
        if len(self.new_shape) < 3:
            self.new_shape.append(1)

    def load_mat(self, mat):
        matImg = sio.loadmat(mat)

        ashapelist = []
        self.voxel_ndarray = []
        for item in matImg:
            if not '__' in item and not 'readme' in item:
                arrays = matImg[item]
                ashapelist.append(len(list(arrays.shape)))
        if 3 in ashapelist:
            for item in matImg:
                if not '__' in item and not 'readme' in item:
                    arrays = matImg[item]
                    if len(list(arrays.shape)) >= 3:
                        try:
                            self.voxel_ndarray.append([np.expand_dims(array, axis=-1) for array in arrays])
                            self.voxel_ndarray = np.concatenate(self.voxel_ndarray, axis=-1)
                        except:
                            self.voxel_ndarray = ([np.expand_dims(array, axis=-1) for array in arrays])
                            self.voxel_ndarray = np.concatenate(self.voxel_ndarray, axis=-1)
        elif max(ashapelist) == 2:
            for item in matImg:
                if not '__' in item and not 'readme' in item:
                    arrays = matImg[item]
                    ashape = list(arrays.shape)
                    if len(list(arrays.shape)) == 2:
                        arrays = arrays.reshape(ashape)
                        try:
                            self.voxel_ndarray.append([np.expand_dims(array, axis=-1) for array in arrays])
                            self.voxel_ndarray = np.concatenate(self.voxel_ndarray, axis=-1)
                        except:
                            self.voxel_ndarray = ([np.expand_dims(array, axis=-1) for array in arrays])
                            self.voxel_ndarray = np.concatenate(self.voxel_ndarray, axis=-1)

        self.voxel_ndarray = np.swapaxes(self.voxel_ndarray, 0, 2)
        self.voxel_ndarray = np.swapaxes(self.voxel_ndarray, 1, 2)

        self.new_shape = list(self.voxel_ndarray.shape)
        if len(self.new_shape) < 3:
            self.new_shape.append(1)
        elif len(self.new_shape) == 4:
            self.new_shape.insert(0, 1)

