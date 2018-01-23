"""
Created on Monday January 08 7:40 2018
@author: Sebastian Milde
"""

import numpy as np

#########################################################################################################################################
#Module: MRT_Layer                                                                                                                      #
#The module MRT_Layer is a class module with several methods. If an image acquisition is loaded, an object, named MRT_Layer is created  #
#with the following attributes:                                                                                                         #
#  mrt_width: The width of MRT-Image                                                                                                    #
#  mrt_height: The height of MRT-Image                                                                                                  #
#  number_mrt: The number of slices of the image acquisition                                                                            #
#  proband:  the name abbreviation of the proband, e.g 01_ab                                                                            #
#  model: e.g. t1_tse_tra_Kopf_Motion_0003                                                                                              #
#  x_1d: The arange of the x-dimension for plotting the dicom image with pcolormesh                                                     #
#  y_1d: The arange of the y-dimension for plotting the dicom image with pcolormesh                                                     #
#  z_1d: The arange of the z-dimension for plotting the dicom image with pcolormesh                                                     #
#  v_min: The min value of the colorbar                                                                                                 #
#  v_max: The max value of the colorbar                                                                                                 #
#  Dicom_array: The 3D array with all values of the DICOM acquisition                                                                   #
#  model_name: the name of the model, which is created in a menu for switching the Layers within the GUI                                #
#  current_number: saves the current number of the slice within a DICOM  acquisition                                                    #
# The module is responsible for managing the loaded MRT image acquisition.                                                              #
#########################################################################################################################################
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

    def get_mrt_height(self):
        return self.__mrt_height

    def get_mrt_width(self):
        return self.__mrt_width

    def get_current_Number(self):
        return self.__current_Number

    def get_number_mrt(self):
        return self.__number_mrt

    def get_Dicom_array(self):
        return self.__Dicom_array

    def get_model_name(self):
        return self.__model_name

    def get_model(self):
        return self.__model

    def get_proband(self):
        return self.__proband

    def get_x_arange(self):
        return self.x_1d

    def get_y_arange(self):
        return self.y_1d

    def get_z_arange(self):
        return self.z_1d

    def get_v_min(self):
        return self.v_min

    def get_v_max(self):
        return self.v_max

    def get_current_Slice(self):
        return self.__Dicom_array[:,:,self.__current_Number]

    def increase_current_Number(self):
        self.__current_Number += 1

    def decrease_current_Number(self):
        self.__current_Number -= 1

    def set_v_min(self, v_min):
        self.v_min = v_min

    def set_v_max(self, v_max):
        self.v_max = v_max
