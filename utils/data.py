import numpy as np
import os
from utils.RigidPatching import *
from utils.Training_Test_Split_FCN import *
import pydicom
import dicom_numpy as dicom_np
import tensorflow as tf
import h5py
import json
from matplotlib import path

class Data:

    def __init__(self, cfg):
        self.patchSize = cfg['patchSize']
        self.patchSizeX = self.patchSize[0]
        self.patchSizeY = self.patchSize[1]
        self.patchSizeZ = self.patchSize[2]
        if len(self.patchSize) > 2:
            self.patchingMode = 'PATCHING_3D'
        else:
            self.patchingMode = 'PATCHING_2D'

        self.labelingMode = 'MASK_LABELING'  # voxel-wise labeling
        self.usingSegmentationMasks = True  # voxel-wise classification
        self.splittingMode = 'SIMPLE_RANDOM_SAMPLE_SPLITTING'  # = "crossvalidation_patient"
        self.storeMode = 'STORE_HDF5'  # 'STORE_DISABLED', 'STORE_HDF5', 'STORE_PATCH_BASED', 'STORE_TFRecord' (TODO)
        self.pathOutput = cfg['sOutputPath']
        self.splittingMode = False  # True = DIY self-splitting
        self.database = cfg['sDatabase']
        self.iGPU = cfg['iGPU']
        self.usingArtifacts = True
        self.usingBodyRegions = True
        self.usingTWeighting = True



    def getAllDicomsPathList(self):
        '''

        :return: a list with all paths of dicoms from the selected patients and selected datasets
        '''
        allDicomsPathList = []  # TODO parsing!
        for patient in self.selectedPatients:
            for dataset in self.selectedDatasets:
                curDataDir = self.pathDatabase + os.sep + patient + os.sep + self.modelSubDir + os.sep + dataset
                if os.path.exists(curDataDir):  # check if path exists... especially for the dicom_sorted subdir!!!!!
                    fileNames = tf.gfile.ListDirectory(curDataDir)
                    fileNames = [os.path.join(curDataDir, f) for f in fileNames]
                    allDicomsPathList = allDicomsPathList + fileNames
        return allDicomsPathList


    def generateDataset(self):
        """
        method performs the splitting of the datasets to the learning datasets (training, validation, test)
        and handles the storage of datasets
        :return:
        """
        self.X_test = []
        self.X_validation = []
        self.X_train = []
        self.Y_test = []
        self.Y_validation = []
        self.Y_train = []

        if self.patchingMode == 'PATCHING_2D':
            dAllPatches = np.zeros((self.patchSizeX, self.patchSizeY, 0))
            dAllLabels = np.zeros(0)
            if self.usingSegmentationMasks:
                dAllSegmentationMaskPatches = np.zeros((self.patchSizeX, self.patchSizeY, 0))
        elif self.patchingMode == 'PATCHING_3D':
            dAllPatches = np.zeros((self.patchSizeX, self.patchSizeY, self.patchSizeZ, 0))
            dAllLabels = np.zeros(0)
            if self.usingSegmentationMasks:
                dAllSegmentationMaskPatches = np.zeros((self.patchSizeX, self.patchSizeY, self.patchSizeZ, 0))
        else:
            raise IOError("We do not know your patching mode...")

        # stuff for storing

        # outPutFolder name:
        outPutFolder = "Patients-" + str(len(self.selectedPatients)) + "_" + \
                       "Datasets-" + str(len(self.selectedDatasets)) + "_" + \
                       ("2D" if self.patchingMode == 'PATCHING_2D' else "3D") + \
                       ('_SegMask_' if self.usingSegmentationMasks else '_') + \
                       str(self.patchSizeX) + "x" + str(self.patchSizeY)
        if self.patchingMode == 'PATCHING_3D':
            outPutFolder = outPutFolder + "x" + str(self.patchSizeZ)

        outPutFolder = outPutFolder + "_Overlap-" + str(self.patchOverlap) + "_" + \
                       "Labeling-" + ("patch" if self.labelingMode == 'PATCH_LABELING' else "mask")

        if self.splittingMode == 'SIMPLE_RANDOM_SAMPLE_SPLITTING':
            outPutFolder = outPutFolder + "_Split-simpleRand"
        elif self.splittingMode == 'CROSS_VALIDATION_SPLITTING':
            outPutFolder = outPutFolder + "_Split-crossVal"
        elif self.splittingMode == 'SIMPLE_RANDOM_SAMPLE_SPLITTING':
            outPutFolder = outPutFolder + "Split-patientCrossVal"

        outputFolderPath = self.pathOutput + os.sep + outPutFolder

        if not os.path.exists(outputFolderPath):
            os.makedirs(outputFolderPath)

        # create dataset summary
        self.datasetName = outPutFolder
        self.datasetForPrediction = outputFolderPath
        self.createDatasetInfoSummary(outPutFolder, outputFolderPath)

        if self.storeMode == 'STORE_PATCH_BASED':
            self.outPutFolderDataPath = outputFolderPath + os.sep + "data"
            if not os.path.exists(self.outPutFolderDataPath):
                os.makedirs(self.outPutFolderDataPath)

            labelDict = {}

        # for storing patch based
        iPatchToDisk = 0

        for patient in self.selectedPatients:
            for dataset in self.selectedDatasets:
                currentDataDir = self.pathDatabase + os.sep + patient + os.sep + self.modelSubDir + os.sep + dataset

                if os.path.exists(currentDataDir):
                    # get list with all paths of dicoms for current patient and current dataset
                    fileNames = os.listdir(currentDataDir)
                    fileNames = [os.path.join(currentDataDir, f) for f in fileNames]

                    # read DICOMS
                    dicomDataset = [pydicom.read_file(f) for f in fileNames]

                    # Combine DICOM Slices to a single 3D image (voxel)
                    try:
                        voxel_ndarray, ijk_to_xyz = dicom_np.combine_slices(dicomDataset)
                        voxel_ndarray = voxel_ndarray.astype(float)
                        voxel_ndarray = np.swapaxes(voxel_ndarray, 0, 1)
                    except dicom_np.DicomImportException as e:
                        # invalid DICOM data
                        raise

                    # normalization of DICOM voxel
                    rangeNorm = [0, 1]
                    norm_voxel_ndarray = (voxel_ndarray - np.min(voxel_ndarray)) * (rangeNorm[1] - rangeNorm[0]) / (
                            np.max(voxel_ndarray) - np.min(voxel_ndarray))

                    # sort array
                    newnparray = np.zeros(shape=norm_voxel_ndarray.shape)
                    for i in range(norm_voxel_ndarray.shape[-1]):
                        newnparray[:, :, norm_voxel_ndarray.shape[-1] - 1 - i] = norm_voxel_ndarray[:, :, i]

                    norm_voxel_ndarray = newnparray

                    # 2D or 3D patching?
                    if self.patchingMode == 'PATCHING_2D':
                        # 2D patching
                        # mask labeling or path labeling
                        if self.labelingMode == 'MASK_LABELING':
                            # path to marking file
                            if self.database == 'MRPhysics':
                                currentMarkingsPath = self.markingsPath + os.sep + patient + ".json"
                                # get the markings mask
                                labelMask_ndarray = self.create_MASK_Array(currentMarkingsPath, patient, dataset,
                                                                      voxel_ndarray.shape[0],
                                                                      voxel_ndarray.shape[1], voxel_ndarray.shape[2])
                            elif self.database == 'NAKO_IQA':
                                # TODO: load from nrrd

                            # compute 2D Mask labling patching
                            dPatches, dLabels = fRigidPatching_maskLabeling(norm_voxel_ndarray,
                                                                            [self.patchSizeX, self.patchSizeY],
                                                                            self.patchOverlap,
                                                                            labelMask_ndarray, 0.5,
                                                                            self.datasets[dataset])

                            # convert to float32
                            dPatches = np.asarray(dPatches, dtype=np.float32)
                            dLabels = np.asarray(dLabels, dtype=np.float32)

                            ############################################################################################
                            if self.usingSegmentationMasks:
                                dPatchesOfMask, dLabelsMask = fRigidPatching_maskLabeling(labelMask_ndarray,
                                                                                          [self.patchSizeX,
                                                                                           self.patchSizeY],
                                                                                          self.patchOverlap,
                                                                                          labelMask_ndarray, 0.5,
                                                                                          self.datasets[
                                                                                              dataset])

                                dPatchesOfMask = np.asarray(dPatchesOfMask, dtype=np.float32)

                            ############################################################################################


                        elif self.labelingMode == 'PATCH_LABELING':
                            # get label
                            datasetLabel = self.datasets[dataset].getDatasetLabel()

                            # compute 2D patch labeling patching
                            dPatches, dLabels = fRigidPatching_patchLabeling(norm_voxel_ndarray,
                                                                             [self.patchSizeX, self.patchSizeY],
                                                                             self.patchOverlap, 1)
                            dLabels = dLabels * datasetLabel

                            # convert to float32
                            dPatches = np.asarray(dPatches, dtype=np.float32)
                            dLabels = np.asarray(dLabels, dtype=np.float32)
                    elif self.patchingMode == 'PATCHING_3D':
                        # 3D Patching
                        if self.labelingMode == 'MASK_LABELING':
                            # path to marking file
                            if self.database == 'MRPhysics':
                                currentMarkingsPath = self.markingsPath + os.sep + patient + ".json"
                                # get the markings mask
                                labelMask_ndarray = self.create_MASK_Array(currentMarkingsPath, patient, dataset,
                                                                      voxel_ndarray.shape[0],
                                                                      voxel_ndarray.shape[1], voxel_ndarray.shape[2])
                            elif self.database == 'NAKO_IQA':
                                # TODO: load from nrrd

                            # compute 3D Mask labling patching
                            dPatches, dLabels = fRigidPatching3D_maskLabeling(norm_voxel_ndarray,
                                                                              [self.patchSizeX, self.patchSizeY,
                                                                               self.patchSizeZ],
                                                                              self.patchOverlap,
                                                                              labelMask_ndarray,
                                                                              0.5,
                                                                              self.datasets[dataset])

                            # convert to float32
                            dPatches = np.asarray(dPatches, dtype=np.float32)
                            dLabels = np.asarray(dLabels, dtype=np.float32)

                            ############################################################################################
                            if self.usingSegmentationMasks:
                                dPatchesOfMask, dLabelsMask = fRigidPatching3D_maskLabeling(labelMask_ndarray,
                                                                                            [self.patchSizeX,
                                                                                             self.patchSizeY,
                                                                                             self.patchSizeZ],
                                                                                            self.patchOverlap,
                                                                                            labelMask_ndarray, 0.5,
                                                                                            self.datasets[
                                                                                                dataset])
                                dPatchesOfMask = np.asarray(dPatchesOfMask, dtype=np.byte)
                            ############################################################################################

                        elif self.labelingMode == 'PATCH_LABELING':
                            print("3D local patch labeling not available until now!")

                    else:
                        print("We do not know what labeling mode you want to use :p")

                    if self.storeMode == 'STORE_PATCH_BASED':
                        # patch based storage
                        if self.patchingMode == 'PATCHING_3D':
                            for i in range(0, dPatches.shape[3]):
                                patchSlice = np.asarray(dPatches[:, :, :, i], dtype=np.float32)
                                np.save((self.outPutFolderDataPath + os.sep + "X" + str(iPatchToDisk) + ".npy"),
                                        patchSlice, allow_pickle=False)
                                labelDict["Y" + str(iPatchToDisk)] = int(dLabels[i])
                                iPatchToDisk += 1
                        else:
                            for i in range(0, dPatches.shape[2]):
                                patchSlice = np.asarray(dPatches[:, :, i], dtype=np.float32)
                                np.save((self.outPutFolderDataPath + os.sep + "X" + str(iPatchToDisk) + ".npy"),
                                        patchSlice, allow_pickle=False)
                                labelDict["Y" + str(iPatchToDisk)] = int(dLabels[i])
                                iPatchToDisk += 1

                    else:
                        # concatenate all patches in one array
                        if self.patchingMode == 'PATCHING_2D':
                            dAllPatches = np.concatenate((dAllPatches, dPatches), axis=2)
                            dAllLabels = np.concatenate((dAllLabels, dLabels), axis=0)
                            if self.usingSegmentationMasks:
                                dAllSegmentationMaskPatches = np.concatenate(
                                    (dAllSegmentationMaskPatches, dPatchesOfMask), axis=2)
                        elif self.patchingMode == 'PATCHING_3D':
                            dAllPatches = np.concatenate((dAllPatches, dPatches), axis=3)
                            dAllLabels = np.concatenate((dAllLabels, dLabels), axis=0)
                            if self.usingSegmentationMasks:
                                dAllSegmentationMaskPatches = np.concatenate(
                                    (dAllSegmentationMaskPatches, dPatchesOfMask), axis=3)

        # dataset splitting
        # store mode
        if self.storeMode != 'STORE_DISABLED':
            # H5py store mode
            if self.storeMode == 'STORE_HDF5':

                if self.patchingMode == 'PATCHING_2D':
                    if not self.usingSegmentationMasks:
                        [self.X_train], [self.Y_train], [self.X_validation], [self.Y_validation], [self.X_test], [
                            self.Y_test] \
                            = fSplitDataset(dAllPatches, dAllLabels, allPats=self.selectedPatients,
                                            sSplitting=self.splittingMode,
                                            patchSize=[self.patchSizeX, self.patchSizeY],
                                            patchOverlap=self.patchOverlap,
                                            testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                            validationTrainRatio=self.trainValidationRatio,
                                            outPutPath=self.pathOutputPatching,
                                            nfolds=0, isRandomShuffle=self.isRandomShuffle)
                    else:
                        # do segmentation mask split
                        [self.X_train], [self.Y_train], [self.Y_segMasks_train], \
                        [self.X_validation], [self.Y_validation], [self.Y_segMasks_validation], \
                        [self.X_test], [self.Y_test], [self.Y_segMasks_test] \
                            = fSplitSegmentationDataset(dAllPatches, dAllLabels, dAllSegmentationMaskPatches,
                                                        allPats=self.selectedPatients,
                                                        sSplitting=self.splittingMode,
                                                        patchSize=[self.patchSizeX, self.patchSizeY],
                                                        patchOverlap=self.patchOverlap,
                                                        testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                                        validationTrainRatio=self.trainValidationRatio,
                                                        outPutPath=self.pathOutputPatching,
                                                        nfolds=0, isRandomShuffle=self.isRandomShuffle)

                    # store datasets with h5py
                    self.datasetOutputPath = outputFolderPath
                    with h5py.File(outputFolderPath + os.sep + 'datasets.hdf5', 'w') as hf:
                        hf.create_dataset('X_train', data=self.X_train)
                        hf.create_dataset('X_validation', data=self.X_validation)
                        hf.create_dataset('X_test', data=self.X_test)
                        hf.create_dataset('Y_train', data=self.Y_train)
                        hf.create_dataset('Y_validation', data=self.Y_validation)
                        hf.create_dataset('Y_test', data=self.Y_test)
                        if self.usingSegmentationMasks:
                            hf.create_dataset('Y_segMasks_train', data=self.Y_segMasks_train)
                            hf.create_dataset('Y_segMasks_validation', data=self.Y_segMasks_validation)
                            hf.create_dataset('Y_segMasks_test', data=self.Y_segMasks_test)

                elif self.patchingMode == 'PATCHING_3D':
                    if not self.usingSegmentationMasks:
                        [self.X_train], [self.Y_train], [self.X_validation], [self.Y_validation], [self.X_test], [
                            self.Y_test] \
                            = fSplitDataset(dAllPatches, dAllLabels, allPats=self.selectedPatients,
                                            sSplitting=self.splittingMode,
                                            patchSize=[self.patchSizeX, self.patchSizeY, self.patchSizeZ],
                                            patchOverlap=self.patchOverlap,
                                            testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                            validationTrainRatio=self.trainValidationRatio,
                                            outPutPath=self.pathOutputPatching,
                                            nfolds=0, isRandomShuffle=self.isRandomShuffle)
                    else:
                        [self.X_train], [self.Y_train], [self.Y_segMasks_train], \
                        [self.X_validation], [self.Y_validation], [self.Y_segMasks_validation], \
                        [self.X_test], [self.Y_test], [self.Y_segMasks_test] \
                            = fSplitSegmentationDataset(dAllPatches,
                                                        dAllLabels,
                                                        dAllSegmentationMaskPatches,
                                                        allPats=self.selectedPatients,
                                                        sSplitting=self.splittingMode,
                                                        patchSize=[self.patchSizeX, self.patchSizeY,
                                                                   self.patchSizeZ],
                                                        patchOverlap=self.patchOverlap,
                                                        testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                                        validationTrainRatio=self.trainValidationRatio,
                                                        outPutPath=self.pathOutputPatching,
                                                        nfolds=0, isRandomShuffle=self.isRandomShuffle)

                    # store datasets with h5py
                    self.datasetOutputPath = outputFolderPath
                    with h5py.File(outputFolderPath + os.sep + 'datasets.hdf5', 'w') as hf:
                        hf.create_dataset('X_train', data=self.X_train)
                        hf.create_dataset('X_validation', data=self.X_validation)
                        hf.create_dataset('X_test', data=self.X_test)
                        hf.create_dataset('Y_train', data=self.Y_train)
                        hf.create_dataset('Y_validation', data=self.Y_validation)
                        hf.create_dataset('Y_test', data=self.Y_test)
                        if self.usingSegmentationMasks:
                            hf.create_dataset('Y_segMasks_train', data=self.Y_segMasks_train)
                            hf.create_dataset('Y_segMasks_validation', data=self.Y_segMasks_validation)
                            hf.create_dataset('Y_segMasks_test', data=self.Y_segMasks_test)

            elif self.storeMode == 'STORE_PATCH_BASED':
                self.datasetOutputPath = outputFolderPath
                with open(outputFolderPath + os.sep + "labels.json", 'w') as fp:
                    json.dump(labelDict, fp)
        else:
            # no storage of patched datasets
            if self.patchingMode == 'PATCHING_2D':
                if not self.usingSegmentationMasks:
                    [self.X_train], [self.Y_train], [self.X_validation], [self.Y_validation], [self.X_test], [
                        self.Y_test] \
                        = fSplitDataset(dAllPatches, dAllLabels, allPats=self.selectedPatients,
                                        sSplitting=self.splittingMode,
                                        patchSize=[self.patchSizeX, self.patchSizeY],
                                        patchOverlap=self.patchOverlap,
                                        testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                        validationTrainRatio=self.trainValidationRatio,
                                        outPutPath=self.pathOutputPatching,
                                        nfolds=0, isRandomShuffle=self.isRandomShuffle)
                else:
                    # do segmentation mask split
                    [self.X_train], [self.Y_train], [self.Y_segMasks_train], \
                    [self.X_validation], [self.Y_validation], [self.Y_segMasks_validation], \
                    [self.X_test], [self.Y_test], [self.Y_segMasks_test] \
                        = fSplitSegmentationDataset(dAllPatches,
                                                    dAllLabels,
                                                    dAllSegmentationMaskPatches,
                                                    allPats=self.selectedPatients,
                                                    sSplitting=self.splittingMode,
                                                    patchSize=[self.patchSizeX, self.patchSizeY],
                                                    patchOverlap=self.patchOverlap,
                                                    testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                                    validationTrainRatio=self.trainValidationRatio,
                                                    outPutPath=self.pathOutputPatching,
                                                    nfolds=0, isRandomShuffle=self.isRandomShuffle)

            elif self.patchingMode == 'PATCHING_3D':
                if not self.usingSegmentationMasks:
                    [self.X_train], [self.Y_train], [self.X_validation], [self.Y_validation], [self.X_test], [
                        self.Y_test] \
                        = fSplitDataset(dAllPatches, dAllLabels, allPats=self.selectedPatients,
                                        sSplitting=self.splittingMode,
                                        patchSize=[self.patchSizeX, self.patchSizeY, self.patchSizeZ],
                                        patchOverlap=self.patchOverlap,
                                        testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                        validationTrainRatio=self.trainValidationRatio,
                                        outPutPath=self.pathOutputPatching,
                                        nfolds=0, isRandomShuffle=self.isRandomShuffle)
                else:
                    [self.X_train], [self.Y_train], [self.Y_segMasks_train], \
                    [self.X_validation], [self.Y_validation], [self.Y_segMasks_validation], \
                    [self.X_test], [self.Y_test], [self.Y_segMasks_test] \
                        = fSplitSegmentationDataset(dAllPatches,
                                                    dAllLabels,
                                                    dAllSegmentationMaskPatches,
                                                    allPats=self.selectedPatients,
                                                    sSplitting=self.splittingMode,
                                                    patchSize=[self.patchSizeX, self.patchSizeY, self.patchSizeZ],
                                                    patchOverlap=self.patchOverlap,
                                                    testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                                    validationTrainRatio=self.trainValidationRatio,
                                                    outPutPath=self.pathOutputPatching,
                                                    nfolds=0, isRandomShuffle=self.isRandomShuffle)

    def create_MASK_Array(self, pathMarking, proband, model, mrt_height, mrt_width, mrt_depth):
        mask = np.zeros((mrt_height, mrt_width, mrt_depth))

        # JSON file
        with open(pathMarking, 'r') as fp:
            loadMark = json.load(fp)

        # loadMark = shelve.open(Path_mark + proband + ".dumbdbm.slv")

        if model in loadMark['layer']:
            dataset_marks = loadMark['layer'][model]
            names = loadMark['names']['list']
            for key in dataset_marks:

                img_no = int(key[0:2]) - 1
                tool_no = int(key[2])
                artifact_num = int(key[3])
                artifact_str = names[artifact_num]
                artifact_region_num = int(key[4:6])

                if artifact_str == "motion":
                    artifact_num = 1
                elif artifact_str == "shim":
                    artifact_num = 2
                else:
                    # no known artifact
                    artifact_num = -1

                # print(mask[:, :, img_no].shape)
                # mask_lay = mask[:, :, img_no]
                mask_lay = np.zeros((mrt_height, mrt_width))
                p = dataset_marks[key]
                if tool_no == 1:
                    # p has to be an ndarray
                    p = np.asarray(p['points'])
                    mask_lay = self.mask_rectangle(p[0], p[1], p[2], p[3], mask_lay, artifact_num)
                elif tool_no == 2:
                    # p has to be an ndarray
                    p = np.asarray(p['points'])
                    mask_lay = self.mask_ellipse(p[0], p[1], p[2], p[3], mask_lay, artifact_num)
                elif tool_no == 3:
                    # p has to be a matplotlib path
                    p = path.Path(np.asarray(p['vertices']), p['codes'])
                    mask_lay = self.mask_lasso(p, mask_lay, artifact_num)
                mask[:, :, img_no] = mask[:, :, img_no] + mask_lay
        else:
            pass

        # loadMark.close()  # used for shelve
        # print(mask.dtype)

        return mask

    def mask_rectangle(self,x_coo1, y_coo1, x_coo2, y_coo2, layer_mask, art_no):
        x_coo1 = round(x_coo1)
        y_coo1 = round(y_coo1)
        x_coo2 = round(x_coo2)
        y_coo2 = round(y_coo2)
        layer_mask[int(min(y_coo1, y_coo2)):int(max(y_coo1, y_coo2)) + 1,
        int(min(x_coo1, x_coo2)):int(max(x_coo1, x_coo2)) + 1] = art_no

        return layer_mask

    def mask_ellipse(self,x_coo1, y_coo1, x_coo2, y_coo2, layer_mask, art_no):
        x_coo1 = round(x_coo1)
        y_coo1 = round(y_coo1)
        x_coo2 = round(x_coo2)
        y_coo2 = round(y_coo2)
        b_y, a_x = abs((y_coo2 - y_coo1) / 2), abs((x_coo2 - x_coo1) / 2)
        y_m, x_m = min(y_coo1, y_coo2) + b_y - 1, min(x_coo1, x_coo2) + a_x - 1
        y_height = layer_mask.shape[0]
        x_width = layer_mask.shape[1]
        y, x = np.ogrid[-y_m:y_height - y_m, -x_m:x_width - x_m]
        mask = b_y * b_y * x * x + a_x * a_x * y * y <= a_x * a_x * b_y * b_y
        layer_mask[mask] = art_no

        return layer_mask

    def mask_lasso(self, p, layer_mask, art_no):
        pix1 = np.arange(layer_mask.shape[1])
        pix2 = np.arange(layer_mask.shape[0])
        xv, yv = np.meshgrid(pix1, pix2)
        pix = np.vstack((xv.flatten(), yv.flatten())).T

        ind = p.contains_points(pix, radius=1)
        lin = np.arange(layer_mask.size)
        newArray = layer_mask.flatten()
        newArray[lin[ind]] = art_no
        mask_lay = newArray.reshape(layer_mask.shape)

        return mask_lay
