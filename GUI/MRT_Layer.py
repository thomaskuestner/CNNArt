import numpy as np

class MRT_Layer():

    def __init__(self, mrt_width, mrt_height, number_mrt, proband, sModel, x_1d, y_1d, z_1d, v_min, v_max, Dicom_array, model_name):
        self.__mrt_width = mrt_width
        self.__mrt_height = mrt_height
        self.__number_mrt = number_mrt
        self.__proband = proband
        self.__model = sModel
        self.x_1d = x_1d
        self.y_1d = y_1d
        self.z_1d = z_1d
        self.v_min = v_min
        self.v_max = v_max
        self.__Dicom_array = Dicom_array
        self.__model_name = model_name
        self.__current_Number = 0

    def get_mrt_height(self): #important
        return self.__mrt_height

    def get_mrt_width(self): #important
        return self.__mrt_width

    def get_current_Number(self): #important
        return self.__current_Number

    def get_number_mrt(self): #important
        return self.__number_mrt

    def get_Dicom_array(self): #important
        return self.__Dicom_array

    def get_model_name(self): #important
        return self.__model_name

    def get_model(self):
        return self.__model

    def get_proband(self):
        return self.__proband

    def get_x_arange(self): #important
        return self.x_1d

    def get_y_arange(self): #important
        return self.y_1d

    def get_z_arange(self): #important
        return self.z_1d

    def get_v_min(self):
        return self.v_min

    def get_v_max(self):
        return self.v_max

    def get_current_Slice(self): #important
        return self.__Dicom_array[:,:,self.__current_Number]

    def increase_current_Number(self): #important
        self.__current_Number += 1

    def decrease_current_Number(self): #important
        self.__current_Number -= 1

    def set_v_min(self, v_min):
        self.v_min = v_min

    def set_v_max(self, v_max):
        self.v_max = v_max

