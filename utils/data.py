'''
Copyright: 2016-2019 Thomas Kuestner (thomas.kuestner@med.uni-tuebingen.de) under Apache2 license
@author: Thomas Kuestner
'''
import datetime
import json
import os

import dicom
import dicom_numpy as dicom_np
#import pydicom as dicom_np
import h5py
import nrrd
import pydicom
import scipy.io as sio
from matplotlib import path
import matplotlib.pyplot as plt

from utils.RigidPatching import *
from utils.RigidUnpatching import *
from utils.Training_Test_Split_FCN import *
from utils.Label import Label
from utils.tfrecord.medio import convert_tf, parse_tf_cnnart
from utils.patch_shaping import *


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
        self.patchOverlap = cfg['patchOverlap']
        self.batchSize = cfg['batchSize']

        self.labelingMode = 'MASK_LABELING'  # voxel-wise labeling
        self.usingSegmentationMasks = True  # voxel-wise classification
        self.splittingMode = 'PATIENT_CROSS_VALIDATION_SPLITTING'  # = "crossvalidation_patient"
        if self.splittingMode == 'PATIENT_CROSS_VALIDATION_SPLITTING':
            self.selectedTestPatients = [x-1 for x in cfg['trainTestDatasetRatio']]  # convert to python index
            self.trainTestDatasetRatio = 0
        else:
            self.selectedTestPatients = 0
            self.trainTestDatasetRatio = cfg['trainTestDatasetRatio'][0]  # either ratio for random splitting or selected test patients

        self.trainValidationRatio = cfg['trainValidationRatio']  # ratio between training and validation patches (percentage ratio)
        self.isRandomShuffle = cfg['randomShuffle']  # random shuffling in training
        self.nfolds = cfg['nfolds']  # number of cross-validation folds
        if 'storeMode' in cfg:
            self.storeMode = cfg['storeMode']
        else:
            self.storeMode = 'STORE_HDF5'  # 'STORE_DISABLED', 'STORE_HDF5', 'STORE_TFRECORD', 'STORE_PATCH_BASED'
        self.pathOutput = cfg['sOutputPath']
        self.pathOutputPatching = self.pathOutput  # same for now
        self.database = cfg['sDatabase']
        self.iGPU = cfg['iGPU']
        self.usingArtifacts = True
        self.usingBodyRegions = True
        self.usingTWeighting = True
        if self.splittingMode == 'PATIENT_CROSS_VALIDATION_SPLITTING':
            self.doUnpatching = True  # only unpatching possible for left out test subjects
        self.usingClassification = cfg['usingClassification']  # use classification output on deepest layer
        self.plotresults = cfg['plotting']['plotResults']
        self.plotFormat = cfg['plotting']['format']
        self.plotAlpha = cfg['plotting']['alpha']
        self.plotColormap = cfg['plotting']['colormap']
        self.datagenerator = []

        # selected database
        self.database = cfg['sDatabase']
        self.pathDatabase = cfg[self.database]['sPathIn']
        self.modelSubDir = cfg[self.database]['sSubDir']
        self.markingsPath = cfg[self.database]['sPathInLabel']
        # parse selected patients
        if cfg['sSelectedPatient'][0] == 'All':
            self.selectedPatients = sorted(os.listdir(self.pathDatabase))
        else:
            dpat = sorted(os.listdir(self.pathDatabase))
            self.selectedPatients = [dpat[i] for i in [x-1 for x in cfg['sSelectedPatient']]]

        # parse selected artifacts and datasets
        self.selectedDatasets = []
        for i, art in enumerate(cfg['sSelectedArtifact']):
            for j, dat in enumerate(cfg['sSelectedDataset']):
                #self.selectedDatasets.extend(cfg[self.database][art][dat])
                for k, selData in enumerate(cfg[self.database][art][dat]):
                    currdat = cfg[self.database]['MappingData'][selData]
                    if currdat[0] == 'None':
                        self.selectedDatasets.append(Dataset(selData,None,currdat[1],currdat[2],currdat[3]))
                    else:
                        self.selectedDatasets.append(Dataset(selData, currdat[0], currdat[1], currdat[2], currdat[3]))



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
        elif self.splittingMode == 'SIMPLE_RANDOM_SAMPLE_SPLITTING' or self.splittingMode == 'PATIENT_CROSS_VALIDATION_SPLITTING':
            outPutFolder = outPutFolder + "Split-patientCrossVal"

        outputFolderPath = self.pathOutput + os.sep + outPutFolder

        if not os.path.exists(outputFolderPath):
            os.makedirs(outputFolderPath)

        # create dataset summary
        # self.datasetName = outPutFolder
        # self.datasetForPrediction = outputFolderPath
        #self.createDatasetInfoSummary(outPutFolder, outputFolderPath)

        if self.storeMode == 'STORE_TFRECORD':
            # check if database was already converted
            pathtf = self.pathDatabase + '.tfrecord'
            if not os.path.exists(pathtf):
                # convert database to TFRecords
                self.convert2TFrecords(pathtf)
            # create TFRecord iterator (training data)
            self.datagenerator = self.create_dataset(pathtf)
            # load here test data and patch it
            selTestPat = self.selectedPatients[self.selectedTestPatients]
            self.X_test, _, self.Y_test, self.Y_segMasks_test = self.fload_and_patch(selTestPat, self.selectedDatasets)
            self.X_test, self.Y_test, self.Y_segMasks_test = freshape_numpy(self.X_test, self.Y_segMasks_test)  # only valid with segmentation masks!
            return 0

        if self.storeMode == 'STORE_PATCH_BASED':
            self.outPutFolderDataPath = outputFolderPath + os.sep + "data"
            if not os.path.exists(self.outPutFolderDataPath):
                os.makedirs(self.outPutFolderDataPath)

            labelDict = {}

        if self.storeMode == 'STORE_HDF5':  # shortcut if data already patched and splitted
            if os.path.exists(outputFolderPath + os.sep + 'datasets_' + ''.join(str(e) for e in self.selectedPatients) + '.hdf5'):
                print('Loading data from file: %s' % (outputFolderPath + os.sep + 'datasets_' + ''.join(str(e) for e in self.selectedPatients) + '.hdf5'))
                f = h5py.File(outputFolderPath + os.sep + 'datasets_' + ''.join(str(e) for e in self.selectedPatients) + '.hdf5', 'r')
                self.X_train = np.array(f.get('X_train'))
                self.Y_train = np.array(f.get('Y_train'))
                self.X_validation = np.array(f.get('X_validation'))
                self.Y_validation = np.array(f.get('Y_validation'))
                self.X_test = np.array(f.get('X_test'))
                self.Y_test = np.array(f.get('Y_test'))
                if self.usingSegmentationMasks:
                    self.Y_segMasks_train = np.array(f.get('Y_segMasks_train'))
                    self.Y_segMasks_validation = np.array(f.get('Y_segMasks_validation'))
                    self.Y_segMasks_test = np.array(f.get('Y_segMasks_test'))
                f.close()
                return 0

        # for storing patch based
        iPatchToDisk = 0

        # load and patch
        dAllPatches, dAllPats, dAllLabels, dAllSegmentationMaskPatches = self.fload_and_patch(self.selectedPatients, self.selectedDatasets)

        self.dAllPats = dAllPats  # save info of patients
        self.dAllLabels = dAllLabels  # save all label info
        # dataset splitting
        print('Dataset splitting')
        # store mode
        if self.storeMode != 'STORE_DISABLED':
            # H5py store mode
            if self.storeMode == 'STORE_HDF5':

                if self.patchingMode == 'PATCHING_2D':
                    if not self.usingSegmentationMasks:
                        [self.X_train], [self.Y_train], [self.X_validation], [self.Y_validation], [self.X_test], [
                            self.Y_test] \
                            = fSplitDataset(dAllPatches, dAllLabels, allPats=dAllPats,
                                            allTestPats=self.selectedTestPatients,
                                            sSplitting=self.splittingMode,
                                            patchSize=[self.patchSizeX, self.patchSizeY],
                                            patchOverlap=self.patchOverlap,
                                            testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                            validationTrainRatio=self.trainValidationRatio,
                                            outPutPath=self.pathOutputPatching,
                                            nfolds=self.nfolds, isRandomShuffle=self.isRandomShuffle)
                    else:
                        # do segmentation mask split
                        [self.X_train], [self.Y_train], [self.Y_segMasks_train], \
                        [self.X_validation], [self.Y_validation], [self.Y_segMasks_validation], \
                        [self.X_test], [self.Y_test], [self.Y_segMasks_test] \
                            = fSplitSegmentationDataset(dAllPatches, dAllLabels, dAllSegmentationMaskPatches,
                                                        allPats=dAllPats,
                                                        allTestPats=self.selectedTestPatients,
                                                        sSplitting=self.splittingMode,
                                                        patchSize=[self.patchSizeX, self.patchSizeY],
                                                        patchOverlap=self.patchOverlap,
                                                        testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                                        validationTrainRatio=self.trainValidationRatio,
                                                        outPutPath=self.pathOutputPatching,
                                                        nfolds=self.nfolds, isRandomShuffle=self.isRandomShuffle)

                    # store datasets with h5py
                    self.datasetOutputPath = outputFolderPath
                    with h5py.File(outputFolderPath + os.sep + 'datasets_' + ''.join(str(e) for e in self.selectedPatients) + '.hdf5', 'w') as hf:
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
                            = fSplitDataset(dAllPatches, dAllLabels, allPats=dAllPats,
                                            allTestPats=self.selectedTestPatients,
                                            sSplitting=self.splittingMode,
                                            patchSize=[self.patchSizeX, self.patchSizeY, self.patchSizeZ],
                                            patchOverlap=self.patchOverlap,
                                            testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                            validationTrainRatio=self.trainValidationRatio,
                                            outPutPath=self.pathOutputPatching,
                                            nfolds=self.nfolds, isRandomShuffle=self.isRandomShuffle)
                    else:
                        [self.X_train], [self.Y_train], [self.Y_segMasks_train], \
                        [self.X_validation], [self.Y_validation], [self.Y_segMasks_validation], \
                        [self.X_test], [self.Y_test], [self.Y_segMasks_test] \
                            = fSplitSegmentationDataset(dAllPatches,
                                                        dAllLabels,
                                                        dAllSegmentationMaskPatches,
                                                        allPats=dAllPats,
                                                        allTestPats=self.selectedTestPatients,
                                                        sSplitting=self.splittingMode,
                                                        patchSize=[self.patchSizeX, self.patchSizeY,
                                                                   self.patchSizeZ],
                                                        patchOverlap=self.patchOverlap,
                                                        testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                                        validationTrainRatio=self.trainValidationRatio,
                                                        outPutPath=self.pathOutputPatching,
                                                        nfolds=self.nfolds, isRandomShuffle=self.isRandomShuffle)

                    # store datasets with h5py
                    self.datasetOutputPath = outputFolderPath
                    with h5py.File(outputFolderPath + os.sep + 'datasets_' + ''.join(str(e) for e in self.selectedPatients) + 'hdf5', 'w') as hf:
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
                        = fSplitDataset(dAllPatches, dAllLabels, allPats=dAllPats,
                                            allTestPats=self.selectedTestPatients,
                                        sSplitting=self.splittingMode,
                                        patchSize=[self.patchSizeX, self.patchSizeY],
                                        patchOverlap=self.patchOverlap,
                                        testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                        validationTrainRatio=self.trainValidationRatio,
                                        outPutPath=self.pathOutputPatching,
                                        nfolds=self.nfolds, isRandomShuffle=self.isRandomShuffle)
                else:
                    # do segmentation mask split
                    [self.X_train], [self.Y_train], [self.Y_segMasks_train], \
                    [self.X_validation], [self.Y_validation], [self.Y_segMasks_validation], \
                    [self.X_test], [self.Y_test], [self.Y_segMasks_test] \
                        = fSplitSegmentationDataset(dAllPatches,
                                                    dAllLabels,
                                                    dAllSegmentationMaskPatches,
                                                    allPats=dAllPats,
                                                    allTestPats=self.selectedTestPatients,
                                                    sSplitting=self.splittingMode,
                                                    patchSize=[self.patchSizeX, self.patchSizeY],
                                                    patchOverlap=self.patchOverlap,
                                                    testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                                    validationTrainRatio=self.trainValidationRatio,
                                                    outPutPath=self.pathOutputPatching,
                                                    nfolds=self.nfolds, isRandomShuffle=self.isRandomShuffle)

            elif self.patchingMode == 'PATCHING_3D':
                if not self.usingSegmentationMasks:
                    [self.X_train], [self.Y_train], [self.X_validation], [self.Y_validation], [self.X_test], [
                        self.Y_test] \
                        = fSplitDataset(dAllPatches, dAllLabels, allPats=dAllPats,
                                            allTestPats=self.selectedTestPatients,
                                        sSplitting=self.splittingMode,
                                        patchSize=[self.patchSizeX, self.patchSizeY, self.patchSizeZ],
                                        patchOverlap=self.patchOverlap,
                                        testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                        validationTrainRatio=self.trainValidationRatio,
                                        outPutPath=self.pathOutputPatching,
                                        nfolds=self.nfolds, isRandomShuffle=self.isRandomShuffle)
                else:
                    [self.X_train], [self.Y_train], [self.Y_segMasks_train], \
                    [self.X_validation], [self.Y_validation], [self.Y_segMasks_validation], \
                    [self.X_test], [self.Y_test], [self.Y_segMasks_test] \
                        = fSplitSegmentationDataset(dAllPatches,
                                                    dAllLabels,
                                                    dAllSegmentationMaskPatches,
                                                    allPats=dAllPats,
                                                    allTestPats=self.selectedTestPatients,
                                                    sSplitting=self.splittingMode,
                                                    patchSize=[self.patchSizeX, self.patchSizeY, self.patchSizeZ],
                                                    patchOverlap=self.patchOverlap,
                                                    testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                                    validationTrainRatio=self.trainValidationRatio,
                                                    outPutPath=self.pathOutputPatching,
                                                    nfolds=self.nfolds, isRandomShuffle=self.isRandomShuffle)

    def fload_and_patch(self, selectedPatients, selectedDatasets):
        if self.patchingMode == 'PATCHING_2D':
            dAllPatches = np.zeros((self.patchSizeX, self.patchSizeY, 0))
            dAllLabels = np.zeros(0)
            dAllPats = np.zeros(0)
            if self.usingSegmentationMasks:
                dAllSegmentationMaskPatches = np.zeros((self.patchSizeX, self.patchSizeY, 0))
        elif self.patchingMode == 'PATCHING_3D':
            dAllPatches = np.zeros((self.patchSizeX, self.patchSizeY, self.patchSizeZ, 0))
            dAllLabels = np.zeros(0)
            dAllPats = np.zeros(0)
            if self.usingSegmentationMasks:
                dAllSegmentationMaskPatches = np.zeros((self.patchSizeX, self.patchSizeY, self.patchSizeZ, 0))
        else:
            raise IOError("We do not know your patching mode...")

        for ipat, patient in enumerate(selectedPatients):
            print('Loading patient %d\%d' % (ipat+1, len(selectedPatients)))
            for idat, dataset in enumerate(selectedDatasets):
                currentDataDir = self.pathDatabase + os.sep + patient + os.sep + self.modelSubDir + os.sep + dataset.pathdata

                if os.path.exists(currentDataDir):
                    # get list with all paths of dicoms for current patient and current dataset
                    fileNames = os.listdir(currentDataDir)
                    fileNames = [os.path.join(currentDataDir, f) for f in fileNames]

                    # read DICOMS
                    dicomDataset = [pydicom.read_file(f) for f in fileNames]
                    # TODO: add here reading in of phase images

                    # Combine DICOM Slices to a single 3D image (voxel)
                    try:
                        voxel_ndarray, _ = dicom_np.combine_slices(dicomDataset)
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
                    del newnparray

                    # 2D or 3D patching?
                    if self.patchingMode == 'PATCHING_2D':
                        # 2D patching
                        # mask labeling or path labeling
                        if self.labelingMode == 'MASK_LABELING':
                            # path to marking file
                            if self.database == 'MRPhysics':
                                currentMarkingsPath = self.markingsPath + os.sep + patient + ".json"
                                # get the markings mask
                                labelMask_ndarray = self.create_MASK_Array(currentMarkingsPath, patient, dataset.pathdata,
                                                                           voxel_ndarray.shape[0],
                                                                           voxel_ndarray.shape[1], voxel_ndarray.shape[2])
                            elif self.database == 'NAKO_IQA':
                                if dataset.pathdata == '3D_GRE_TRA_bh_F_COMPOSED_0014':  # reference --> all 0
                                    labelMask_ndarray = np.zeros((voxel_ndarray.shape[0],
                                                                  voxel_ndarray.shape[1], voxel_ndarray.shape[2]))
                                elif dataset.pathdata == '3D_GRE_TRA_fb_F_COMPOSED_0028':  # free-breathing mask
                                    currentMarkingsPath = self.markingsPath + os.sep + patient + os.sep + patient + '_fb.nrrd'
                                    labelMask_ndarray = nrrd.read(
                                        currentMarkingsPath)  # TODO: verify, should be 1 at positions with artifact
                                elif dataset.pathdata == '3D_GRE_TRA_fb_deep_F_COMPOSED_0042':  # free-breathing mask
                                    currentMarkingsPath = self.markingsPath + os.sep + patient + os.sep + patient + '_db.nrrd'
                                    labelMask_ndarray = nrrd.read(
                                        currentMarkingsPath)  # TODO: verify, should be 1 at positions with artifact

                            # compute 2D Mask labling patching
                            dPatches, dLabels = fRigidPatching_maskLabeling(norm_voxel_ndarray,
                                                                            [self.patchSizeX, self.patchSizeY],
                                                                            self.patchOverlap,
                                                                            labelMask_ndarray, 0.5,
                                                                            dataset)

                            # convert to float32
                            dPatches = np.asarray(dPatches, dtype=np.float32)
                            dLabels = np.asarray(dLabels, dtype=np.float32)
                            dPats = ipat * np.ones(dLabels.shape()[0], dtype=np.int16)

                            ############################################################################################
                            if self.usingSegmentationMasks:
                                dPatchesOfMask, dLabelsMask = fRigidPatching_maskLabeling(labelMask_ndarray,
                                                                                          [self.patchSizeX,
                                                                                           self.patchSizeY],
                                                                                          self.patchOverlap,
                                                                                          labelMask_ndarray, 0.5,
                                                                                          dataset)

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
                                labelMask_ndarray = self.create_MASK_Array(currentMarkingsPath, patient, dataset.pathdata,
                                                                           voxel_ndarray.shape[0],
                                                                           voxel_ndarray.shape[1], voxel_ndarray.shape[2])
                            elif self.database == 'NAKO_IQA':
                                if dataset.pathdata == '3D_GRE_TRA_bh_F_COMPOSED_0014':  # reference --> all 0
                                    labelMask_ndarray = np.zeros((voxel_ndarray.shape[0],
                                                                  voxel_ndarray.shape[1], voxel_ndarray.shape[2]))
                                elif dataset.pathdata == '3D_GRE_TRA_fb_F_COMPOSED_0028':  # free-breathing mask
                                    currentMarkingsPath = self.markingsPath + os.sep + patient + os.sep + patient + '_fb.nrrd'
                                    labelMask_ndarray = nrrd.read(
                                        currentMarkingsPath)  # TODO: verify, should be 1 at positions with artifact
                                elif dataset.pathdata == '3D_GRE_TRA_fb_deep_F_COMPOSED_0042':  # free-breathing mask
                                    currentMarkingsPath = self.markingsPath + os.sep + patient + os.sep + patient + '_db.nrrd'
                                    labelMask_ndarray = nrrd.read(
                                        currentMarkingsPath)  # TODO: verify, should be 1 at positions with artifact

                            # compute 3D Mask labling patching
                            dPatches, dLabels = fRigidPatching3D_maskLabeling(norm_voxel_ndarray,
                                                                              [self.patchSizeX, self.patchSizeY,
                                                                               self.patchSizeZ],
                                                                              self.patchOverlap,
                                                                              labelMask_ndarray,
                                                                              0.5,
                                                                              dataset)

                            # convert to float32
                            dPatches = np.asarray(dPatches, dtype=np.float32)
                            dLabels = np.asarray(dLabels, dtype=np.float32)
                            dPats = ipat * np.ones(dLabels.shape[0], dtype=np.int16)

                            ############################################################################################
                            if self.usingSegmentationMasks:
                                dPatchesOfMask, dLabelsMask = fRigidPatching3D_maskLabeling(labelMask_ndarray,
                                                                                            [self.patchSizeX,
                                                                                             self.patchSizeY,
                                                                                             self.patchSizeZ],
                                                                                            self.patchOverlap,
                                                                                            labelMask_ndarray, 0.5,
                                                                                            dataset)
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
                            dAllPats = np.concatenate((dAllPats, dPats), axis=0)
                            if self.usingSegmentationMasks:
                                dAllSegmentationMaskPatches = np.concatenate(
                                    (dAllSegmentationMaskPatches, dPatchesOfMask), axis=2)
                        elif self.patchingMode == 'PATCHING_3D':
                            dAllPatches = np.concatenate((dAllPatches, dPatches), axis=3)
                            dAllLabels = np.concatenate((dAllLabels, dLabels), axis=0)
                            dAllPats = np.concatenate((dAllPats, dPats), axis=0)
                            if self.usingSegmentationMasks:
                                dAllSegmentationMaskPatches = np.concatenate(
                                    (dAllSegmentationMaskPatches, dPatchesOfMask), axis=3)

        return dAllPatches, dAllPats, dAllLabels, dAllSegmentationMaskPatches

    def convert2TFrecords(self, pathtf):
        for ipat, patient in enumerate(self.selectedPatients):
            print('Loading patient %d\%d' % (ipat+1, len(self.selectedPatients)))
            for idat, dataset in enumerate(self.selectedDatasets):
                currentDataDir = self.pathDatabase + os.sep + patient + os.sep + self.modelSubDir + os.sep + dataset.pathdata

                if os.path.exists(currentDataDir):
                    # get list with all paths of dicoms for current patient and current dataset
                    fileNames = os.listdir(currentDataDir)
                    fileNames = [os.path.join(currentDataDir, f) for f in fileNames]

                    # read DICOMS
                    dicomDataset = [pydicom.read_file(f) for f in fileNames]
                    # TODO: add here reading in of phase images

                    # Combine DICOM Slices to a single 3D image (voxel)
                    try:
                        voxel_ndarray, _ = dicom_np.combine_slices(dicomDataset)
                        voxel_ndarray = voxel_ndarray.astype(int16)
                        voxel_ndarray = np.swapaxes(voxel_ndarray, 0, 1)
                    except dicom_np.DicomImportException as e:
                        # invalid DICOM data
                        raise

                    # normalization of DICOM voxel
                    #rangeNorm = [0, 1]
                    #norm_voxel_ndarray = (voxel_ndarray - np.min(voxel_ndarray)) * (rangeNorm[1] - rangeNorm[0]) / (
                    #        np.max(voxel_ndarray) - np.min(voxel_ndarray))

                    # sort array
                    #newnparray = np.zeros(shape=norm_voxel_ndarray.shape)
                    #for i in range(norm_voxel_ndarray.shape[-1]):
                    #    newnparray[:, :, norm_voxel_ndarray.shape[-1] - 1 - i] = norm_voxel_ndarray[:, :, i]
                    for i in range(voxel_ndarray.shape[-1]):
                        newnparray[:, :, voxel_ndarray.shape[-1] - 1 - i] = voxel_ndarray[:, :, i]

                    path_file = pathtf + os.sep + patient + os.sep + dataset.pathdata + '.tfrecord'
                    if not os.path.exists(pathtf + os.sep + patient):
                        os.makedirs(pathtf + os.sep + patient)

                    # body region
                    _, bodyRegionLabel = dataset.getBodyRegion()

                    # MRT weighting label (T1, T2)
                    _, weightingLabel = dataset.getMRTWeighting()
                    bodyRegionweightingLabel = bodyRegionLabel + weightingLabel
                    print('creating ', path_file)
                    convert_tf.im2tfrecord(image=newnparray, path=path_file, metadata=bodyRegionweightingLabel)
                    # save a numpy array with some image information
                    path_file = pathtf + os.sep + patient + os.sep + dataset.pathdata + '_info'
                    dataset.image_shape = newnparray.shape()
                    np.save(path_file, dataset)

                    # convert voxel masks as well
                    if self.labelingMode == 'MASK_LABELING':
                        # path to marking file
                        if self.database == 'MRPhysics':
                            currentMarkingsPath = self.markingsPath + os.sep + patient + ".json"
                            # get the markings mask
                            labelMask_ndarray = self.create_MASK_Array(currentMarkingsPath, patient, dataset.pathdata,
                                                                       voxel_ndarray.shape[0],
                                                                       voxel_ndarray.shape[1], voxel_ndarray.shape[2])
                        elif self.database == 'NAKO_IQA':
                            if dataset == '3D_GRE_TRA_bh_F_COMPOSED_0014':  # reference --> all 0
                                labelMask_ndarray = np.zeros((voxel_ndarray.shape[0],
                                                              voxel_ndarray.shape[1], voxel_ndarray.shape[2]))
                            elif dataset == '3D_GRE_TRA_fb_F_COMPOSED_0028':  # free-breathing mask
                                currentMarkingsPath = self.markingsPath + os.sep + patient + os.sep + patient + '_fb.nrrd'
                                labelMask_ndarray = nrrd.read(
                                    currentMarkingsPath)  # TODO: verify, should be 1 at positions with artifact
                            elif dataset == '3D_GRE_TRA_fb_deep_F_COMPOSED_0042':  # free-breathing mask
                                currentMarkingsPath = self.markingsPath + os.sep + patient + os.sep + patient + '_db.nrrd'
                                labelMask_ndarray = nrrd.read(
                                    currentMarkingsPath)  # TODO: verify, should be 1 at positions with artifact

                        path_file = pathtf + os.sep + patient + os.sep + dataset.pathdata + '_mask.tfrecord'
                        print('creating mask ', path_file)
                        convert_tf.im2tfrecord(image=labelMask_ndarray, path=path_file)

    def create_dataset(self, data_dir, num_parallel_calls=4, prefetched_buffer_size=8000):

        # choose & fetch all required training data / discard subjects missing crucial data
        # TODO: only fetch training data, test data is not used for training: only works for subject leave-out at the moment, no batch-based splitting!
        list_images = parse_tf_cnnart.fetch_paths(data_dir, self.selectedDatasets, self.selectedTestPatients)
        list_masks = parse_tf_cnnart.fetch_path(data_dir, self.selectedDatasets, self.selectedTestPatients, ismask = True)
        #print(list_images)
        #list_labels = [parse_label(x) for x in list_images]
        # print(list_labels)
        # TODO: patch-based labels not supported yet -> needs to be incorporated into dynamic loading in generator function
        # TODO: 2D patching not supported yet
        num_samples = len(list_images)
        print('num_samples ', num_samples)

        # define the patch you want to crop
        num_patches = []
        for files in list_images:
            # get corresponding image size
            dataInfo = np.load(files[:-9] + '_info')  # reload pickled dataset object
            num_patches += fcalculatepatches(dataInfo.image_shape, self.patchSize, self.patchOverlap)

        print('number of patches cropped per image: ', num_patches)

        #buffer_size = num_imgaes_loaded * num_patches  # shuffle patches from 34 different big images
        buffer_size = num_patches

        # This function splits a whole image into many splits and return
        get_patches_fn = lambda image, dataset, segMask: fRigidPatching3D_maskLabeling(image,  # dataset is here returned as int (bodRegion+weighting label) from TFRecord
                                                                              self.patchSize,
                                                                              self.patchOverlap,
                                                                              segMask,
                                                                              0.5,
                                                                              dataset)

        # TODO: this can be called later and no need to have image_shape as input as image is already reshaped

        # generate placeholders that receive paths of type str
        images = tf.placeholder(tf.string, shape=[None])
        masks = tf.placeholder(tf.string, shape=[None])
        #labels = tf.placeholder(tf.int32, shape=[None])

        # create dataset for your needs
        dataset_image = tf.data.TFRecordDataset(images)
        dataset_masks = tf.data.TFRecordDataset(masks)
        #dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((dataset_image, dataset_masks))

        # Note: this could be rewritten as one map call
        # map parse function to each zipped element
        # load data
        # input: a = string of stored image, b = string of stored segMask
        dataset = dataset.map(
            map_func=lambda a, b: (convert_tf.parse_withlabel_function(a), convert_tf.parse_function(b)),   # fist input zipped image list, 2nd zipped mask list
            num_parallel_calls=num_parallel_calls)  # returns: image, image_label, segMask

        # scale image to [0,1]
        # input: a = image, b = image_label, c = segMask
        dataset = dataset.map(
            map_func=lambda a, b, c: ( (a-tf.reduce_min(a))/(tf.reduce_max(a) - tf.reduce_min(a)), b, c),
            num_parallel_calls=num_parallel_calls
        )  # returns: scaledImage, image_label, segMask

        # patching
        # input: a = scaledImage, image_label, segMask
        dataset = dataset.map(
            map_func=lambda a, b, c: (get_patches_fn(a, b, c), get_patches_fn(c, b, c)),
            num_parallel_calls=num_parallel_calls
        )  # returns: imagePatch, segMaskPatch

        # reshaping of tensors
        # input: a = imagePatch, b = segMaskPatch
        dataset = dataset.map(
            map_func=lambda a, b: (freshape_tensor(a, b)),
            num_parallel_calls=num_parallel_calls)  # returns: X_train, Y_train, Y_segMasks_train

        dataset = dataset.apply(tf.contrib.data.unbatch())

        # here one could use shuffle, repeat, prefetch, ...
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset_batched = dataset.batch(batch_size=self.batchSize)
        dataset_batched = dataset_batched.prefetch(buffer_size=prefetched_buffer_size)
        dataset_batched = dataset_batched.repeat()

        iterator = dataset_batched.make_initializable_iterator()

        next_element = iterator.get_next()

        # dummy "model"
        result = next_element

        with tf.Session() as sess:

            sess.run(iterator.initializer,
                     feed_dict={images: list_images, masks: list_masks})

            while True:
                try:
                    yield sess.run(result)

                except tf.errors.OutOfRangeError:
                    print('finished')
                    break

    def handlepredictionssegmentation(self, predictions):
        # do unpatching if is enabled
        if self.doUnpatching:

            self.patchSizePrediction = [self.X_test.shape[1], self.X_test.shape[2], self.X_test.shape[3]]

            # find test patient in test set
            test_set_idx = []
            for ipat, patient in enumerate(self.selectedTestPatients):
                #for idat, dataset in enumerate(self.selectedDatasets):
                test_index = np.where(self.dAllPats == ipat)[0]
                test_set_tmp = self.dAllPats[test_index]
                test_set_idx.append(test_set_tmp)

            allUnpatchedTest = []
            # load corresponding original dataset
            for ipat, patient in enumerate(self.selectedTestPatients):
                for idat, dataset in enumerate(self.selectedDatasets):
                    currentDataDir = self.pathDatabase + os.sep + patient + os.sep + self.modelSubDir + os.sep + dataset.pathdata

                    if os.path.exists(currentDataDir):
                        # get list with all paths of dicoms for current patient and current dataset
                        fileNames = os.listdir(currentDataDir)
                        fileNames = [os.path.join(currentDataDir, f) for f in fileNames]

                        # read DICOMS
                        dicomDataset = [pydicom.read_file(f) for f in fileNames]
                        # reading in of phase images not really needed at this place, because this data is only required for displaying purposes

                        # Combine DICOM Slices to a single 3D image (voxel)
                        try:
                            voxel_ndarray, _ = dicom_np.combine_slices(dicomDataset)
                            voxel_ndarray = voxel_ndarray.astype(float)
                            voxel_ndarray = np.swapaxes(voxel_ndarray, 0, 1)
                        except dicom_np.DicomImportException as e:
                            # invalid DICOM data
                            raise

                        # sort array
                        newnparray = np.zeros(shape=voxel_ndarray.shape)
                        for i in range(voxel_ndarray.shape[-1]):
                            newnparray[:, :, voxel_ndarray.shape[-1] - 1 - i] = voxel_ndarray[:, :, i]

                        voxel_ndarray = newnparray
                        del newnparray

                    if self.labelingMode == 'MASK_LABELING':
                        # path to marking file
                        if self.database == 'MRPhysics':
                            currentMarkingsPath = self.markingsPath + os.sep + patient + ".json"
                            # get the markings mask
                            labelMask_ndarray = self.create_MASK_Array(currentMarkingsPath, patient, dataset.pathdata,
                                                                       voxel_ndarray.shape[0],
                                                                       voxel_ndarray.shape[1], voxel_ndarray.shape[2])
                        elif self.database == 'NAKO_IQA':
                            if dataset == '3D_GRE_TRA_bh_F_COMPOSED_0014':  # reference --> all 0
                                labelMask_ndarray = np.zeros((voxel_ndarray.shape[0],
                                                              voxel_ndarray.shape[1], voxel_ndarray.shape[2]))
                            elif dataset == '3D_GRE_TRA_fb_F_COMPOSED_0028':  # free-breathing mask
                                currentMarkingsPath = self.markingsPath + os.sep + patient + os.sep + patient + '_fb.nrrd'
                                labelMask_ndarray = nrrd.read(
                                    currentMarkingsPath)  # TODO: verify, should be 1 at positions with artifact
                            elif dataset == '3D_GRE_TRA_fb_deep_F_COMPOSED_0042':  # free-breathing mask
                                currentMarkingsPath = self.markingsPath + os.sep + patient + os.sep + patient + '_db.nrrd'
                                labelMask_ndarray = nrrd.read(
                                    currentMarkingsPath)  # TODO: verify, should be 1 at positions with artifact

                    dicom_size = [voxel_ndarray.shape[0], voxel_ndarray.shape[1], voxel_ndarray.shape[2]]

                    allPreds = predictions['prob_pre'][0][test_set_idx == ipat]

                    unpatched_img_foreground = fUnpatchSegmentation(allPreds,
                                                                    patchSize=self.patchSizePredictionSize,
                                                                    patchOverlap=self.patchOverlapPrediction,
                                                                    actualSize=dicom_size,
                                                                    iClass=1)
                    unpatched_img_background = fUnpatchSegmentation(allPreds,
                                                                    patchSize=self.patchSizePrediction,
                                                                    patchOverlap=self.patchOverlapPrediction,
                                                                    actualSize=dicom_size,
                                                                    iClass=0)

                    ones = np.ones((unpatched_img_background.shape[0], unpatched_img_background.shape[1],
                                    unpatched_img_background.shape[2]))
                    preds = np.divide(np.add(np.subtract(ones, unpatched_img_background), unpatched_img_foreground),
                                      2)

                    preds = preds > 0.5
                    unpatched_img_mask = np.zeros((unpatched_img_background.shape[0], unpatched_img_background.shape[1],
                                                   unpatched_img_background.shape[2]))
                    unpatched_img_mask[preds] = unpatched_img_mask[preds] + 1

                    unpatched_slices = {
                        'probability_mask_foreground': unpatched_img_foreground,
                        'probability_mask_background': unpatched_img_background,
                        'predicted_segmentation_mask': unpatched_img_mask,
                        'dicom_slices': voxel_ndarray,
                        'dicom_masks': labelMask_ndarray,
                    }

                    # plot output overlays
                    savepath = self.pathOutput + os.sep + patient + '_' + dataset.pathdata
                    plot_patient_mask(voxel_ndarray, unpatched_img_foreground, labelMask_ndarray, savepath, plot_overlay=True)

                    allUnpatchedTest.append(unpatched_slices)

        if self.usingClassification:
            # save prediction into .mat file
            modelSave = self.pathOutput + os.sep + 'model_predictions.mat'
            print('saving Model:{}'.format(modelSave))

            if not self.doUnpatching:
                allPreds = predictions['prob_pre']
                sio.savemat(modelSave, {'prob_pre': allPreds[0],
                                        'Y_test': self.Y_test,
                                        'classification_prob_pre': allPreds[1],
                                        'loss_test': predictions['loss_test'],
                                        'segmentation_output_loss_test': predictions[
                                            'segmentation_output_loss_test'],
                                        'classification_output_loss_test': predictions[
                                            'classification_output_loss_test'],
                                        'segmentation_output_dice_coef': predictions[
                                            'segmentation_output_dice_coef_test'],
                                        'classification_output_acc_test': predictions[
                                            'classification_output_acc_test']
                                        })
            else:
                sio.savemat(modelSave, {'prob_pre': allPreds[0],
                                        'Y_test': self.Y_test,
                                        'classification_prob_pre': allPreds[1],
                                        'loss_test': predictions['loss_test'],
                                        'segmentation_output_loss_test': predictions[
                                            'segmentation_output_loss_test'],
                                        'classification_output_loss_test': predictions[
                                            'classification_output_loss_test'],
                                        'segmentation_output_dice_coef_test': predictions[
                                            'segmentation_output_dice_coef_test'],
                                        'classification_output_acc_test': predictions[
                                            'classification_output_acc_test'],
                                        'unpatched_slices': allUnpatchedTest
                                        })
            #self.result_WorkSpace = modelSave

            # load training results
            #_, sPath = os.path.splitdrive(self.outPutFolderDataPath)
            #sPath, sFilename = os.path.split(sPath)
            #sFilename, sExt = os.path.splitext(sFilename)

            #training_results = sio.loadmat(sPath + os.sep + sFilename + ".mat")
            #self.acc_training = training_results['segmentation_output_dice_coef_training']
            #self.acc_validation = training_results['segmentation_output_dice_coef_val']
            #self.acc_test = training_results['segmentation_output_dice_coef_test']

        else:
            # save prediction into .mat file
            modelSave = self.pathOutput + os.sep + 'model_predictions.mat'
            print('saving Model:{}'.format(modelSave))
            if not self.doUnpatching:
                sio.savemat(modelSave, {'prob_pre': predictions['prob_pre'],
                                        'score_test': predictions['score_test'],
                                        'acc_test': predictions['acc_test'],
                                        })
            else:
                sio.savemat(modelSave, {'prob_pre': predictions['prob_pre'],
                                        'score_test': predictions['score_test'],
                                        'acc_test': predictions['acc_test'],
                                        'unpatched_slices': unpatched_slices
                                        })
            #self.result_WorkSpace = modelSave

            # load training results
            #_, sPath = os.path.splitdrive(self.outPutFolderDataPath)
            #sPath, sFilename = os.path.split(sPath)
            #sFilename, sExt = os.path.splitext(sFilename)

            #training_results = sio.loadmat(sPath + os.sep + sFilename + ".mat")
            #self.acc_training = training_results['dice_coef']
            #self.acc_validation = training_results['val_dice_coef']
            #self.acc_test = training_results['dice_coef_test']

    def handlepredictions(self, prediction):
        #self.predictions = prediction['predictions']
        #self.confusionMatrix = prediction['confusion_matrix']
        #self.classificationReport = prediction['classification_report']

        ############################################################

        # organize confusion matrix
        sum_all = np.array(np.sum(prediction['confusion_matrix'], axis=0))
        all = np.zeros((len(sum_all), len(sum_all)))
        for i in range(all.shape[0]):
            all[i, :] = sum_all
        prediction['confusion_matrix'] = np.divide(prediction['confusion_matrix'], all)

        # do unpatching if is enabled
        if self.doUnpatching:

            self.patchSizePrediction = [self.X_test.shape[1], self.X_test.shape[2], self.X_test.shape[3]]
            classVec = self.classMappingsForPrediction[self.classLabel]
            classVec = np.asarray(classVec, dtype=np.int32)
            iClass = np.where(classVec == 1)
            iClass = iClass[0]

            # find test patient in test set
            test_set_idx = []
            for ipat, patient in enumerate(self.selectedTestPatients):
                # for idat, dataset in enumerate(self.selectedDatasets):
                test_index = np.where(self.dAllPats == ipat)[0]
                test_set_tmp = self.dAllPats[test_index]
                test_set_idx.append(test_set_tmp)

            # load corresponding original dataset
            #for i in self.datasets:
            #    set = self.datasets[i]
            #    if set.getDatasetLabel() == self.classLabel:
            #        originalDatasetName = set.getPathdata()

            #pathToOriginalDataset = self.getPathToDatabase() + os.sep + str(
            #    patientsOfDataset[0]) + os.sep + 'dicom_sorted' + os.sep + originalDatasetName

            allUnpatchedTest = []
            # load corresponding original dataset
            for ipat, patient in enumerate(self.selectedTestPatients):
                for idat, dataset in enumerate(self.selectedDatasets):
                    currentDataDir = self.pathDatabase + os.sep + patient + os.sep + self.modelSubDir + os.sep + dataset.pathdata

                    if os.path.exists(currentDataDir):
                        # get list with all paths of dicoms for current patient and current dataset
                        fileNames = os.listdir(pathToOriginalDataset)
                        fileNames = [os.path.join(pathToOriginalDataset, f) for f in fileNames]

                        # read DICOMS
                        dicomDataset = [dicom.read_file(f) for f in fileNames]

                        # Combine DICOM Slices to a single 3D image (voxel)
                        try:
                            voxel_ndarray, ijk_to_xyz = dicom_np.combine_slices(dicomDataset)
                            voxel_ndarray = voxel_ndarray.astype(float)
                            voxel_ndarray = np.swapaxes(voxel_ndarray, 0, 1)
                        except dicom_np.DicomImportException as e:
                            # invalid DICOM data
                            raise

                        # sort array
                        newnparray = np.zeros(shape=voxel_ndarray.shape)
                        for i in range(voxel_ndarray.shape[-1]):
                            newnparray[:, :, voxel_ndarray.shape[-1] - 1 - i] = voxel_ndarray[:, :, i]

                        voxel_ndarray = newnparray

                        # load dicom mask
                        currentMarkingsPath = self.getMarkingsPath() + os.sep + str(patientsOfDataset[0]) + ".json"
                        # get the markings mask
                        labelMask_ndarray = create_MASK_Array(currentMarkingsPath,
                                                              patientsOfDataset[0],
                                                              originalDatasetName,
                                                              voxel_ndarray.shape[0],
                                                              voxel_ndarray.shape[1],
                                                              voxel_ndarray.shape[2])

                        dicom_size = [voxel_ndarray.shape[0], voxel_ndarray.shape[1], voxel_ndarray.shape[2]]

                        if len(self.patchOverlapPrediction) == 2:
                            multiclass_probability_masks = fMulticlassUnpatch2D(prediction['prob_pre'],
                                                                                self.patchSizePrediction,
                                                                                self.patchOverlapPrediction,
                                                                                dicom_size)

                        if len(self.patchSizePrediction) == 3:
                            multiclass_probability_masks = fUnpatch3D(prediction['prob_pre'],
                                                                      self.patchSizePrediction,
                                                                      self.patchOverlapPrediction,
                                                                      dicom_size)

                        ########################################################################################################
                        # Hatching and colors multicalss unpatching
                        prob_test = prediction['prob_pre']

                        IArte = []
                        IType = []

                        if prob_test.shape[1] == 11:
                            IndexType = np.argmax(prob_test, 1)
                            IndexType[IndexType == 0] = 1
                            IndexType[(IndexType > 1) & (IndexType < 4)] = 2
                            IndexType[(IndexType > 3) & (IndexType < 6)] = 3
                            IndexType[(IndexType > 5) & (IndexType < 8)] = 4
                            IndexType[IndexType > 7] = 5

                            a = Counter(IndexType).most_common(1)
                            domain = a[0][0]

                            PType = np.delete(prob_test, [1, 3, 5, 7, 9, 10],
                                              1)  # delete all artefact images,  only 5 region left
                            PArte = np.delete(prob_test, [0, 2, 4, 6, 8], 1)  # all artefacts
                            PArte[:, [3, 4]] = PArte[:, [4, 3]]
                            # PArte = np.reshape(PArte, (0, 1, 2, 3, 4, 5))
                            PNew = np.concatenate((PType, PArte), axis=1)
                            IndexArte = np.argmax(PNew, 1)

                            IType = UnpatchType(IndexType, domain, self.patchSizePrediction, self.patchOverlapPrediction,
                                                dicom_size)

                            IArte = UnpatchArte(IndexArte, self.patchSizePrediction, self.patchOverlapPrediction, dicom_size,
                                                11)

                        if prob_test.shape[1] == 3:
                            IndexType = np.argmax(prob_test, 1)
                            IndexType[IndexType == 0] = 1
                            IndexType[IndexType == 1] = 2
                            IndexType[IndexType == 2] = 3

                            a = Counter(IndexType).most_common(1)
                            domain = a[0][0]

                            PType = np.delete(prob_test, [1, 2], 1)  # delete all artefact images, only 5 region left
                            PArte = np.delete(prob_test, [0], 1)

                            # PArte = np.reshape(PArte, (0, 1, 2, 3, 4, 5))
                            PNew = np.concatenate((PType, PArte), axis=1)
                            IndexArte = np.argmax(PNew, 1)

                            IType = UnpatchType(IndexType, domain, self.patchSizePrediction, self.patchOverlapPrediction,
                                                dicom_size)

                            IArte = UnpatchArte(IndexArte, self.patchSizePrediction, self.patchOverlapPrediction, dicom_size, 3)

                        if prob_test.shape[1] == 8:
                            IndexType = np.argmax(prob_test, 1)
                            IndexType[IndexType == 0] = 1
                            IndexType[(IndexType > 1) & (IndexType < 5)] = 2
                            IndexType[(IndexType > 4) & (IndexType < 8)] = 3

                            a = Counter(IndexType).most_common(1)
                            domain = a[0][0]

                            PType = np.delete(prob_test, [1, 3, 4, 6, 7],
                                              1)  # delete all artefact images,  only 5 region left
                            PArte = np.delete(prob_test, [0, 2, 5], 1)  # all artefacts
                            # PArte[:, [3, 4]] = PArte[:, [4, 3]]
                            # PArte = np.reshape(PArte, (0, 1, 2, 3, 4, 5))
                            PNew = np.concatenate((PType, PArte), axis=1)
                            IndexArte = np.argmax(PNew, 1)

                            IType = UnpatchType(IndexType, domain, self.patchSizePrediction, self.patchOverlapPrediction,
                                                dicom_size)

                            IArte = UnpatchArte(IndexArte, self.patchSizePrediction, self.patchOverlapPrediction, dicom_size, 8)

                        ########################################################################################################

                        unpatched_slices = {
                            'multiclass_probability_masks': multiclass_probability_masks,
                            'dicom_slices': voxel_ndarray,
                            'dicom_masks': labelMask_ndarray,
                            'index_class': iClass,
                            'IType': IType,
                            'IArte': IArte
                        }
                        # plot output overlays
                        savepath = self.pathOutput + os.sep + patient + '_' + dataset.pathdata
                        plot_patient_mask(voxel_ndarray, unpatched_img_foreground, labelMask_ndarray, savepath,
                                          plot_overlay=True)

                        allUnpatchedTest.append(unpatched_slices)

        # save prediction into .mat file
        modelSave = self.pathOutput + os.sep + 'model_predictions.mat'
        print('saving Model:{}'.format(modelSave))
        if not self.doUnpatching:
            sio.savemat(modelSave, {'prob_pre': prediction['prob_pre'],
                                    'Y_test': self.Y_test,
                                    'score_test': prediction['score_test'],
                                    'acc_test': prediction['acc_test'],
                                    'classification_report': prediction['classification_report'],
                                    'confusion_matrix': prediction['confusion_matrix']
                                    })
        else:
            sio.savemat(modelSave, {'prob_pre': prediction['prob_pre'],
                                    'Y_test': self.Y_test,
                                    'score_test': prediction['score_test'],
                                    'acc_test': prediction['acc_test'],
                                    'classification_report': prediction['classification_report'],
                                    'confusion_matrix': prediction['confusion_matrix'],
                                    'unpatched_slices': allUnpatchedTest
                                    })
        print('Saved to: ' + modelSave)

        # load training results
        #_, sPath = os.path.splitdrive(self.outPutFolderDataPath)
        #sPath, sFilename = os.path.split(sPath)
        #sFilename, sExt = os.path.splitext(sFilename)

        #print(sPath + os.sep + sFilename + ".mat")

        #training_results = sio.loadmat(sPath + os.sep + sFilename + ".mat")
        #self.acc_training = training_results['acc']
        #self.acc_validation = training_results['val_acc']
        #self.acc_test = training_results['acc_test']

    def plot_patient_mask(self, images, prediction, labels, savepath, plot_overlay=False, rot_op=lambda x, dim_z: x[:,:,dim_z,:], ch='fat'):
        """Function for plotting patient with label."""
        x, y, z = images.shape[:3]
        images = (images).astype('float32')  # arrays for plotting
        prediction = (prediction).astype('float32')
        labels = (labels).astype('float32')

        chNr = 0  # channel: fat/T1w = 0, water = 1
        if ch == 'water':
            chNr = 1

        for dim_z in range(z):
            # prediction
            fig = plt.figure()
            plt.axis('off')
            plt.imshow(rot_op(images, dim_z), interpolation='none', cmap='gray')  # include channel for NAKO_IQA: rot_op(images, dim_z)[:, :, chNr]
            if plot_overlay:
                plt.imshow(rot_op(colorize(prediction, self.plotColormap), dim_z), interpolation='none', alpha=self.plotAlpha)

            print(' Save prediction at z ' + str(i))
            fig.savefig(savepath + '_pred_' + "{:03d}".format(i) + '.' + self.plotFormat,
                            format=self.plotFormat,
                            transparent=True,
                            bbox_inches='tight')
            fig.clf()

            # ground-truth/label
            fig = plt.figure()
            plt.axis('off')
            plt.imshow(rot_op(images, dim_z), interpolation='none', cmap='gray')
            if plot_overlay:
                plt.imshow(rot_op(colorize(labels, self.plotColormap), dim_z), interpolation='none', alpha=self.plotAlpha)

            print(' Save label at z ' + str(i))
            fig.savefig(savepath + '_label_' + "{:03d}".format(i) + '.' + self.plotFormat,
                        format=self.plotFormat,
                        transparent=True,
                        bbox_inches='tight')

            fig.clf()

    def colorize(prediction, colormap=np.array([[0, 0, 0], [1, 0, 0.2], [0, 1, 0.2]])):
        """Colorize for patient-plots."""
        pred_picture = colormap[prediction.astype(int)]
        return pred_picture

    def create_cnn_training_summary(self, name, outputFolderPath):
        dataDict = {}
        dataDict['Name'] = name
        dataDict['Date'] = datetime.datetime.today().strftime('%Y-%m-%d')
        dataDict['BatchSize'] = ''.join(str(e) for e in self.batchSize)
        dataDict['LearningRate'] = ''.join(str(e) for e in self.learningRates)
        dataDict['DataAugmentation'] = self.dataAugmentationEnabled
        dataDict['HorizontalFlip'] = self.horizontalFlip
        dataDict['VerticalFlip'] = self.verticalFlip
        dataDict['Rotation'] = self.rotation
        dataDict['Zoom'] = self.zoom
        dataDict['ZCA_Whitening'] = self.zcaWhitening
        dataDict['HeightShift'] = self.heightShift
        dataDict['WidthShift'] = self.widthShift
        dataDict['ContrastStretching'] = self.contrastStretching
        dataDict['HistogramEq'] = self.histogram_eq
        dataDict['AdaptiveEq'] = self.adaptive_eq

        dataDict['BodyRegions'] = self.usingBodyRegions
        dataDict['TWeightings'] = self.usingTWeighting
        dataDict['Artifacts'] = self.usingArtifacts

        dataDict['Optimizer'] = self.optimizer
        dataDict['Epochs'] = self.epochs
        dataDict['WeightDecay'] = self.weightDecay
        dataDict['Momentum'] = self.momentum
        dataDict['NesterovEnabled'] = self.nesterovEnabled
        dataDict['UsingClassification'] = self.usingClassification
        dataDict['SegmentationMaskUsed'] = self.usingSegmentationMasks

        dataDict['Dataset'] = self.datasetName
        # dataDict['ClassMappings'] = self.classMappings

        # dict integer keys to strings
        classMappingsDict = {}
        for i in self.classMappings:
            key = str(i)
            val = self.classMappings[i]
            classMappingsDict[key] = val.tolist()

        dataDict['ClassMappings'] = classMappingsDict

        with open((outputFolderPath + os.sep + 'cnn_training_info.json'), 'w') as fp:
            json.dump(dataDict, fp, indent=4)

    def createDatasetInfoSummary(self, name, outputFolderPath):
        '''
        creates a json info summary of the patched dataset
        :param outputFolderPath:
        :return:
        '''
        dataDict = {}
        dataDict['Name'] = name
        dataDict['Date'] = datetime.datetime.today().strftime('%Y-%m-%d')
        dataDict['Patients'] = self.selectedPatients
        dataDict['Datasets'] = self.selectedDatasets
        dataDict['PatchMode'] = self.patchingMode
        dataDict['PatchSizeX'] = self.patchSizeX
        dataDict['PatchSizeY'] = self.patchSizeY
        dataDict['PatchSizeZ'] = self.patchSizeZ
        dataDict['PatchOverlap'] = self.patchOverlap
        dataDict['LabelingMode'] = self.labelingMode
        dataDict['SplittingMode'] = self.splittingMode
        dataDict['NumFolds'] = self.numFolds
        dataDict['TrainTestRatio'] = self.trainTestDatasetRatio
        dataDict['TrainValidationRatio'] = self.trainValidationRatio
        dataDict['StoreMode'] = self.storeMode
        dataDict['SegmentationMaskUsed'] = self.usingSegmentationMasks

        with open((outputFolderPath + os.sep + 'dataset_info.json'), 'w') as fp:
            json.dump(dataDict, fp, indent=4)

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


class Dataset:
    def __init__(self, pathdata, pathlabel=None, artefact=None, bodyregion=None, tWeighting=None):
        self.pathdata = pathdata
        self.artefact = artefact
        self.bodyregion = bodyregion
        self.mrtWeighting = tWeighting
        self.image_shape = []

        if pathlabel==None:
            if tWeighting == None:
                if artefact == 'ref' and bodyregion =='head':
                    self.datasetLabel = Label.getLabel(Label.HEAD, Label.REFERENCE)
                elif artefact == 'motion' and bodyregion == 'head':
                    self.datasetLabel = Label.getLabel(Label.HEAD, Label.MOTION)
                elif artefact == 'ref' and bodyregion == 'abdomen':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.REFERENCE)
                elif artefact == 'motion' and bodyregion == 'abdomen':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.MOTION)
                elif artefact == 'shim' and bodyregion == 'abdomen':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.SHIM)
                elif artefact == 'ref' and bodyregion == 'pelvis':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.REFERENCE)
                elif artefact == 'motion' and bodyregion == 'pelvis':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.MOTION)
                elif artefact == 'shim' and bodyregion == 'pelvis':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.SHIM)
                else:
                    raise ValueError('Problem with dataset labeling!')
            else:
                if artefact == 'ref' and bodyregion =='head' and tWeighting == 't1':
                    self.datasetLabel = Label.getLabel(Label.HEAD, Label.REFERENCE, Label.T1)
                elif artefact == 'motion' and bodyregion == 'head' and tWeighting == 't1':
                    self.datasetLabel = Label.getLabel(Label.HEAD, Label.MOTION, Label.T1)
                elif artefact == 'ref' and bodyregion == 'abdomen' and tWeighting == 't1':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.REFERENCE, Label.T1)
                elif artefact == 'motion' and bodyregion == 'abdomen' and tWeighting == 't1':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.MOTION, Label.T1)
                elif artefact == 'ref' and bodyregion == 'abdomen' and tWeighting == 't2':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.REFERENCE, Label.T2)
                elif artefact == 'shim' and bodyregion == 'abdomen' and tWeighting == 't2':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.SHIM, Label.T2)
                elif artefact == 'ref' and bodyregion == 'pelvis' and tWeighting == 't1':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.REFERENCE, Label.T1)
                elif artefact == 'ref' and bodyregion == 'pelvis' and tWeighting == 't2':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.REFERENCE, Label.T2)
                elif artefact == 'motion' and bodyregion == 'pelvis' and tWeighting == 't1':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.MOTION, Label.T1)
                elif artefact == 'motion' and bodyregion == 'pelvis' and tWeighting == 't2':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.MOTION, Label.T2)
                elif artefact == 'shim' and bodyregion == 'pelvis' and tWeighting == 't2':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.SHIM, Label.T2)
                else:
                    raise ValueError('Problem with dataset labeling!')
        else:
            self.datasetLabel = pathlabel

    def getPathdata(self):
        return self.pathdata

    def getDatasetLabel(self):
        return self.datasetLabel

    def getArtefact(self):
        return self.artefact

    def getBodyRegion(self):
        labelBodyRegion = None
        if self.bodyregion == 'head':
            labelBodyRegion = Label.HEAD
        elif self.bodyregion == 'abdomen':
            labelBodyRegion = Label.ABDOMEN
        elif self.bodyregion == 'pelvis':
            labelBodyRegion = Label.PELVIS

        return self.bodyregion, labelBodyRegion

    def getMRTWeighting(self):
        tWeighting = None
        if self.mrtWeighting == 't1':
            tWeighting = Label.T1
        elif self.mrtWeighting == 't2':
            tWeighting = Label.T2
        return self.mrtWeighting, tWeighting