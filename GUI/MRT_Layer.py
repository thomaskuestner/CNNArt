import numpy as np

class MRT_Layer:

    def __init__(self, mrt_width, mrt_height, number_mrt, artefact, sModel, x_1d, y_1d, z_1d, Dicom_array, model_name, Matlab_Array):
        self.__patchDictionary = {}
        self.__mrt_width = mrt_width
        self.__mrt_height = mrt_height
        self.__number_mrt = number_mrt
        self.__artefact = artefact
        self.__sModel = sModel
        self.x_1d = x_1d
        self.y_1d = y_1d
        self.z_1d = z_1d
        self.__Dicom_array = Dicom_array
        self.__Matlab_Array = Matlab_Array
        self.__model_name = model_name
        self.__current_Number = 0
        self.__mask = np.zeros((mrt_height, mrt_width, number_mrt))

        pix1_width = self.__mrt_width
        pix2_height = self.__mrt_height
        pix1 = np.arange(pix1_width)
        pix2 = np.arange(pix2_height)
        xv, yv = np.meshgrid(pix1, pix2)
        self.__pix = np.vstack((xv.flatten(), yv.flatten())).T

    def get_patchDictionary(self):
        return self.__patchDictionary

    def get_mrt_height(self):
        return self.__mrt_height

    def get_mrt_width(self):
        return self.__mrt_width

    def get_current_Number(self):
        return self.__current_Number

    def get_mask(self):
        return self.__mask

    def get_number_mrt(self):
        return self.__number_mrt

    def get_Dicom_array(self):
        return self.__Dicom_array

    def get_Matlab_Array(self):
        return self.__Matlab_Array

    def get_model_name(self):
        return self.__model_name

    def get_smodel(self):
        return self.__sModel

    def get_pix(self):
        return self.__pix

    def get_x_arange(self):
        return self.x_1d

    def get_y_arange(self):
        return self.y_1d

    def get_z_arange(self):
        return self.z_1d

    def get_current_Slice(self):
        return self.__Dicom_array[:,:,self.__current_Number]

    def change_mask(self, x_start, x_end, y_start, y_end, value):
        self.__mask[self.__current_Number,y_start:y_end, x_start:x_end] = value

    def set_patchDictionary(self):
        firstPatch = {self.__current_Number:1}
        self.__patchDictionary.update(firstPatch)

    def set_mask(self, mask):
        self.__mask = mask

    def increase_current_Number(self):
        self.__current_Number += 1

    def decrease_current_Number(self):
        self.__current_Number -= 1

    def current_Number(self, value):
        self.__current_Number = value

    def get_number_patch(self):
        return self.__patchDictionary[self.__current_Number]

    def update_mask(self, mask_lay):
        self.__mask[:,:,self.__current_Number] = mask_lay

    def get_partMask(self):
        partMask = self.__mask[:,:, self.__current_Number]
        return partMask

    def addPatch(self):
        new_numb_patch = self.__patchDictionary[self.__current_Number] + 1
        self.__patchDictionary[self.__current_Number] = new_numb_patch

    def keyAvailable(self):
        bKey = False
        if self.__patchDictionary.has_key(self.__current_Number):
            bKey = True

        return bKey