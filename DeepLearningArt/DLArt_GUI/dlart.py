import sys

#sys.path.append("C:/Users/Yannick/'Google Drive'/30_Content/CNNArt/utils")
from DeepLearningArt.DLArt_GUI.RigidPatching import *
from DeepLearningArt.DLArt_GUI.DataPreprocessing import *
from utils.Training_Test_Split import *

#from RigidPatching import *
#from DataPreprocessing import *

import os

#from Dataset import Dataset
from DeepLearningArt.DLArt_GUI.Dataset import Dataset
from utils.Label import Label
import tensorflow as tf
import numpy as np
import dicom as dicom
import dicom_numpy as dicom_np

import shelve

class DeepLearningArtApp():
    datasets = {
        't1_tse_tra_Kopf_0002': Dataset('t1_tse_tra_Kopf_0002', None,'ref', 'head', 't1'),
        't1_tse_tra_Kopf_Motion_0003': Dataset('t1_tse_tra_Kopf_Motion_0003', None, 'motion','head', 't1'),
        't1_tse_tra_fs_mbh_Leber_0004': Dataset('t1_tse_tra_fs_mbh_Leber_0004',None,'ref','abdomen', 't1'),
        't1_tse_tra_fs_mbh_Leber_Motion_0005': Dataset('t1_tse_tra_fs_mbh_Leber_Motion_0005', None, 'motion', 'abdomen', 't1'),
        't2_tse_tra_fs_navi_Leber_0006': Dataset('t2_tse_tra_fs_navi_Leber_0006',None,'ref','abdomen', 't2'),
        't2_tse_tra_fs_navi_Leber_Shim_xz_0007': Dataset('t2_tse_tra_fs_navi_Leber_Shim_xz_0007', None, 'shim', 'abdomen', 't2'),
        't1_tse_tra_fs_Becken_0008': Dataset('t1_tse_tra_fs_Becken_0008', None, 'ref', 'pelvis', 't1'),
        't2_tse_tra_fs_Becken_0009': Dataset('t2_tse_tra_fs_Becken_0009', None, 'ref', 'pelvis', 't2'),
        't1_tse_tra_fs_Becken_Motion_0010': Dataset('t1_tse_tra_fs_Becken_Motion_0010', None, 'motion', 'pelvis', 't1'),
        't2_tse_tra_fs_Becken_Motion_0011': Dataset('t2_tse_tra_fs_Becken_Motion_0011', None, 'motion', 'pelvis', 't2'),
        't2_tse_tra_fs_Becken_Shim_xz_0012': Dataset('t2_tse_tra_fs_Becken_Shim_xz_0012', None, 'shim', 'pelvis', 't2')
    }

    modelSubDir = "dicom_sorted"

    # constants labeling modes
    MASK_LABELING = 0
    PATCH_LABELING = 1

    # constants patching modes
    PATCHING_2D = 0
    PATCHING_3D = 1

    def __init__(self):
        # attributes for paths and database
        self.pathDatabase = "D:" + os.sep + "med_data" + os.sep + "MRPhysics" + os.sep + "newProtocol"
        self.selectedPatients = ''
        self.selectedDatasets = ''
        self.pathOutputPatching = "D:" + os.sep + "med_data" + os.sep + "MRPhysics" + os.sep + "DeepLearningArt_Output"
        self.markingsPath = "D:" + os.sep + "med_data" + os.sep + "MRPhysics" + os.sep + "Markings"

        # attributes for patching
        self.patchSizeX = 40
        self.patchSizeY = 40
        self.patchSizeZ = 5
        self.patchOverlapp = 0.6

        #attributes for labeling
        self.labelingMode = ''

        #attributes for patching
        self.patchingMode = DeepLearningArtApp.PATCHING_2D


    def generateDataset(self):
        dAllPatches = np.zeros((40, 40, 0))
        dAllLabels = np.zeros(0)

        for patient in self.selectedPatients:
            for dataset in self.selectedDatasets:
                currentDataDir = self.pathDatabase + os.sep + patient + os.sep + self.modelSubDir + os.sep + dataset
                if os.path.exists(currentDataDir):
                    # get list with all paths of dicoms for current patient and current dataset
                    fileNames = tf.gfile.ListDirectory(currentDataDir)
                    fileNames = [os.path.join(currentDataDir, f) for f in fileNames]

                    # read DICOMS
                    dicomDataset = [dicom.read_file(f) for f in fileNames]

                    # Combine DICOM Slices to a single 3D image (voxel)
                    try:
                        voxel_ndarray, ijk_to_xyz = dicom_np.combine_slices(dicomDataset)
                    except dicom_np.DicomImportException as e:
                        #invalid DICOM data
                        raise

                    # normalization of DICOM voxel
                    rangeNorm = [0,1]
                    norm_voxel_ndarray = (voxel_ndarray-np.min(voxel_ndarray))*(rangeNorm[1]-rangeNorm[0])/(np.max(voxel_ndarray)-np.min(voxel_ndarray))

                    # 2D or 3D patching?
                    if self.patchingMode == DeepLearningArtApp.PATCHING_2D:
                        # 2D patching
                        # mask labeling or path labeling
                        if self.labelingMode == DeepLearningArtApp.MASK_LABELING:
                            # path to marking file
                            currentMarkingsPath = self.getMarkingsPath() + os.sep + patient + ".json"
                            # get the markings mask
                            labelMask_ndarray = create_MASK_Array(currentMarkingsPath, patient, dataset, voxel_ndarray.shape[0],
                                                                  voxel_ndarray.shape[1], voxel_ndarray.shape[2])

                            #compute 2D Mask labling patching
                            dPatches, dLabels = fRigidPatching_maskLabeling(norm_voxel_ndarray,
                                    [self.patchSizeX, self.patchSizeY], self.patchOverlapp, labelMask_ndarray, 1,
                                     DeepLearningArtApp.datasets[dataset])

                        elif self.labelingMode == DeepLearningArtApp.PATCH_LABELING:
                            # get label
                            datasetLabel = DeepLearningArtApp.datasets[dataset].getDatasetLabel()

                            #compute 2D patch labeling patching
                            dPatches, dLabels = fRigidPatching_patchLabeling(norm_voxel_ndarray,
                                                                             [self.patchSizeX, self.patchSizeY],
                                                                             self.patchOverlapp, 1)
                            dLabels = dLabels*datasetLabel
                        else:
                            print("We do not know what labeling mode you want to use :p")
                    elif self.patchingMode == DeepLearningArtApp.PATCHING_3D:
                        # 3D Patching
                        print("Do 3D patching......")

                dAllPatches = np.concatenate((dAllPatches, dPatches), axis=2)
                dAllLabels = np.concatenate((dAllLabels, dLabels), axis=0)

        [X_train], [y_train], [X_test], [y_test]= fSplitDataset(dAllPatches, dAllLabels, None, 'normal',
                [self.patchSizeX, self.patchSizeY], self.patchOverlapp, 0.1, self.pathOutputPatching, nfolds=0)

        print("")



    def getAllDicomsPathList(self):
        '''

        :return: a list with all paths of dicoms from the selected patients and selected datasets
        '''
        allDicomsPathList = []
        for patient in self.selectedPatients:
            for dataset in self.selectedDatasets:
                curDataDir = self.pathDatabase + os.sep + patient + os.sep + self.modelSubDir + os.sep + dataset
                if os.path.exists(curDataDir):  # check if path exists... especially for the dicom_sorted subdir!!!!!
                    fileNames = tf.gfile.ListDirectory(curDataDir)
                    fileNames = [os.path.join(curDataDir, f) for f in fileNames]
                    allDicomsPathList = allDicomsPathList + fileNames
        return allDicomsPathList


    def setLabelingMode(self, mode):
        if mode == DeepLearningArtApp.MASK_LABELING or mode == DeepLearningArtApp.PATCH_LABELING:
            self.labelingMode = mode

    def getLabelingMode(self):
        return self.labelingMode

    def setMarkingsPath(self, path):
        self.markingsPath = path

    def getMarkingsPath(self):
        return self.markingsPath

    def setPatchSizeX(self, s):
        self.patchSizeX = s

    def getPatchSizeX(self):
        return self.patchSizeX

    def setPatchSizeY(self, s):
        self.patchSizeY = s

    def getPatchSizeY(self):
        return self.patchSizeY

    def setPatchSizeZ(self, s):
        self.patchSizeZ = s

    def getPatchSizeZ(self):
        return self.patchSizeZ

    def setPatchOverlapp(self, o):
        self.patchOverlapp = o

    def getPatchOverlapp(self):
        return self.patchOverlapp

    def setPathToDatabase(self, pathToDB):
        self.pathDatabase = pathToDB

    def getPathToDatabase(self):
        return self.pathDatabase

    def setOutputPathForPatching(self, outPath):
        self.pathOutputPatching = outPath

    def getOutputPathForPatching(self):
        return self.pathOutputPatching

    def setSelectedPatients(self, pats):
        self.selectedPatients = pats

    def getSelectedPatients(self):
        return self.selectedPatients

    def setSelectedDatasets(self, sets):
        self.selectedDatasets = sets

    def getSelectedDatasets(self):
        return self.selectedDatasets

    def setPatchingMode(self, mode):
        if mode == DeepLearningArtApp.PATCHING_2D or mode == DeepLearningArtApp.PATCHING_3D:
            self.patchingMode = mode

    def getPatchingMode(self):
        return self.patchingMode