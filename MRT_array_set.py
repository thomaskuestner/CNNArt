import numpy as np

class MRT_array:

    def __init__(self, mrt_width, mrt_height, dicom_set, labeling_mask):
        self.__mrt_artefact_name = []
        self.__mrt_proband_list = []
        self.__mrt_width = mrt_width
        self.__mrt_height = mrt_height
        self.__dicom_set = dicom_set
        self.__labeling_mask = labeling_mask

    def concatenate_arrays(self, dicom_array, mask_array):
        self.__dicom_set = np.concatenate((self.__dicom_set, dicom_array), axis=2)
        self.__labeling_mask = np.concatenate((self.__labeling_mask, mask_array), axis=2)

    def get_mrt_width(self):
        return self.__mrt_width

    def get_mrt_height(self):
        return self.__mrt_height

    def get_dicom_set(self):
        return self.__dicom_set

    def get_labeling_mask(self):
        return  self.__labeling_mask

