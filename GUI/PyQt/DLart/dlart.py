import datetime
from collections import Counter

import pandas
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from DLart.DataPreprocessing import create_MASK_Array
from DLart.Dataset import Dataset
import numpy as np
import h5py
import os
import json
import dicom
import dicom_numpy as dicom_np
import tensorflow as tf
import scipy.io as sio

from DLart.RigidPatching import fRigidPatching_maskLabeling, fRigidPatching_patchLabeling, fRigidPatching3D_maskLabeling
from config.PATH import PATH_OUT, LEARNING_OUT, LABEL_PATH, DATASETS
from utils.CNN_main import fRunCNN, RUN_CNN_TRAIN_TEST_VALIDATION, RUN_CNN_TRAIN_TEST
from utils.label import Label
from utils.Multiclass_Unpatching import UnpatchType, UnpatchArte
from utils.Prediction import predict_segmentation_model, predict_model
from utils.Training_Test_Split import fSplitSegmentationDataset, fSplitDataset, TransformDataset
from utils.Unpatching import fUnpatchSegmentation, fMulticlassUnpatch2D, fUnpatch3D
from DLart.Constants_DLart import *


class DeepLearningArtApp(QWidget):
    network_interface_update = pyqtSignal()
    _network_interface_update = pyqtSignal()  # used for activate network training
    datasetscsv = pandas.read_csv('config' + os.sep + 'database' + os.sep + DATASETS + os.sep + DATASETS + '.csv')
    datasets = {}
    for i in range(pandas.DataFrame.count(datasetscsv)['pathdata']):
        datasets[datasetscsv['pathdata'][i]] = Dataset(datasetscsv['pathdata'][i], datasetscsv['pathlabel'][i],
                                                       datasetscsv['artefact'][i], datasetscsv['bodyregion'][i],
                                                       datasetscsv['pathdata'][i].split('_')[0])

    deepNeuralNetworks = pandas.read_csv('DLart/networks.csv', index_col=0, squeeze=True).to_dict()

    # structure of the directory where the dicom files are located
    modelSubDir = "dicom_sorted"

    def __init__(self):
        super().__init__()
        # GUI handle
        self.dlart_GUI_handle = None
        self.network_canrun = False
        # GPU id
        self.gpu_id = 0
        self.gpu_prediction_id = 0
        self.result_WorkSpace = None

        # attributes for paths and database
        self.selectedPatients = ''
        self.selectedDatasets = ''

        self.pathDatabase, self.pathOutputPatching, self.markingsPath, self.learningOutputPath \
            = self.getOSPathes()  # for windows os=0, for linse server os=1. see method for pathes

        # attributes for patching
        self.patchSizeX = 40
        self.patchSizeY = 40
        self.patchSizeZ = 5
        self.patchOverlap = 0.6
        self.patchSizePrediction = [48, 48, 16]
        self.patchOverlapPrediction = 0.6

        self.usingSegmentationMasks = False

        self.usingClassification = False

        self.isRandomShuffle = True

        # attributes for labeling
        self.labelingMode = MASK_LABELING
        if self.labelingMode == MASK_LABELING:
            self.strlabelingMode = "Mask Labeling"
        else:
            self.strlabelingMode = "Patch Labeling"

        self.classMappings = None
        self.classMappingsForPrediction = None

        # attributes for patching
        self.patchingMode = PATCHING_2D
        if self.patchingMode == PATCHING_2D:
            self.strpatchingMode = "Patching 2D"
        else:
            self.strpatchingMode = "Patching 3D"
        self.storeMode = STORE_DISABLED
        if self.storeMode == STORE_PATCH_BASED:
            self.strstoreMode = "store patch based"
        elif self.storeMode == STORE_HDF5:
            self.strstoreMode = "store in HDF5"
        else:
            self.strstoreMode = "store disabled"

        # attributes for splitting
        self.datasetName = 'none'
        self.splittingMode = SIMPLE_RANDOM_SAMPLE_SPLITTING
        if self.splittingMode == SIMPLE_RANDOM_SAMPLE_SPLITTING:
            self.strsplittingMode = "simple random sample splitting"
        elif self.splittingMode == CROSS_VALIDATION_SPLITTING:
            self.strsplittingMode = "cross-validation splitting"
        elif self.splittingMode == PATIENT_CROSS_VALIDATION_SPLITTING:
            self.strsplittingMode = "patients cross validation splitting"
        elif self.splittingMode == DIY_SPLITTING:
            self.strsplittingMode = "splitting method self-defined"
        else:
            self.strsplittingMode = "no Splitting or Splitting mode not available"

        self.trainTestDatasetRatio = 0.2  # part of test data
        self.trainValidationRatio = 0.2  # part of Validation data in train data
        self.numFolds = 5

        ################################################################################################################
        # attributes for DNN and Training
        ################################################################################################################
        self.neuralNetworkModel = 'Multiclass DenseResNet'
        self.neuralnetworkPath = self.deepNeuralNetworks[self.neuralNetworkModel].replace(".", "/") + '.py'
        self.batchSizes = [32]
        self.batchSizePrediction = 32
        self.epochs = 10
        self.learningRates = np.array([0.01], dtype=np.float32)

        self.optimizer = SGD_OPTIMIZER
        if self.optimizer == RMS_PROP_OPTIMIZER:
            self.stroptimizer = "RMS_PROP_OPTIMIZER"
        elif self.optimizer == ADAGRAD_OPTIMIZER:
            self.stroptimizer = "ADAGRAD_OPTIMIZER"
        elif self.optimizer == ADADELTA_OPTIMIZER:
            self.stroptimizer = "ADADELTA_OPTIMIZER"
        elif self.optimizer == ADAM_OPTIMIZER:
            self.stroptimizer = "ADAM_OPTIMIZER"
        else:
            self.stroptimizer = "SGD_OPTIMIZER"
        self.weightDecay = 0.0001
        self.momentum = 0.9
        self.nesterovEnabled = False

        self.dataAugmentationEnabled = False
        self.horizontalFlip = True
        self.verticalFlip = False
        self.rotation = 0
        self.zcaWhitening = False
        self.heightShift = 0
        self.widthShift = 0
        self.zoom = 0
        self.contrastStretching = False
        self.adaptive_eq = False
        self.histogram_eq = False
        ################################################################################################################

        # Attributes for classes and labels
        self.usingArtifacts = True
        self.usingBodyRegions = True
        self.usingTWeighting = True
        self.usingArtifactsPrediction = True
        self.usingBodyRegionsPrediction = True
        self.usingTWeightingPrediction = True

        # train, validation, test dataset attributes
        self.X_train = None
        self.Y_train = None
        self.Y_segMasks_train = None

        self.X_validation = None
        self.Y_validation = None
        self.Y_segMasks_validation = None

        self.X_test = None
        self.Y_test = None
        self.Y_segMasks_test = None

        self.X_train_shape = (0, 0, 0)
        self.Y_train_shape = (0, 0, 0)
        self.Y_segMasks_train_shape = (0, 0, 0)

        self.X_validation_shape = (0, 0, 0)
        self.Y_validation_shape = (0, 0, 0)
        self.Y_segMasks_validation_shape = (0, 0, 0)

        self.X_test_shape = (0, 0, 0)
        self.Y_test_shape = (0, 0, 0)
        self.Y_segMasks_test_shape = (0, 0, 0)

        ################################################################################################################
        #### Stuff for prediction
        ################################################################################################################
        self.outPutFolderDataPath = self.pathOutputPatching
        self.datasetForPrediction = self.pathOutputPatching
        self.datasetOutputPath = self.pathOutputPatching
        self.modelPathPrediction = self.learningOutputPath
        self.modelPrediction = "FCN 3D-VResFCN-Unsampling"
        self.modelPredictionSource = self.deepNeuralNetworks[self.neuralNetworkModel].replace(".", "/") + '.py'
        self.doUnpatching = False

        self.usingSegmentationMasksForPrediction = False
        self.confusionMatrix = None
        self.classificationReport = None

        self.acc_training = None
        self.acc_validation = None
        self.acc_test = None

        self.predictions = None
        self.unpatched_slices = None

        ################################################################################################################
        self.params = [
            {'name': 'Dataset', 'type': 'group', 'children': [
                {'name': 'X_train', 'type': 'str', 'value': self.X_train_shape},
                {'name': 'y_train', 'type': 'str', 'value': self.Y_train_shape},
                {'name': 'X_validation', 'type': 'str', 'value': self.X_validation_shape},
                {'name': 'y_validation', 'type': 'str', 'value': self.Y_validation_shape},
                {'name': 'X_test', 'type': 'str', 'value': self.X_test_shape},
                {'name': 'y_test', 'type': 'str', 'value': self.Y_test_shape},
                {'name': 'Y_segMasks_train', 'type': 'str', 'value': self.Y_segMasks_train_shape},
                {'name': 'Y_segMasks_validation', 'type': 'str', 'value': self.Y_segMasks_validation_shape},
                {'name': 'Y_segMasks_test', 'type': 'str', 'value': self.Y_segMasks_test_shape},
            ]},
            {'name': 'Patching Options', 'type': 'group', 'children': [
                {'name': 'Labeling Mode', 'type': 'str', 'value': self.strlabelingMode},
                {'name': 'Patching Mode', 'type': 'str', 'value': self.strpatchingMode},
                {'name': 'Patch Size', 'type': 'str', 'value': [self.patchSizeX, self.patchSizeY, self.patchSizeZ]},
                {'name': 'Overlap', 'type': 'float', 'value': self.patchOverlap},
                {'name': 'Markings Path', 'type': 'str', 'value': self.markingsPath},
                {'name': 'Output Path', 'type': 'str', 'value': self.pathOutputPatching},
                {'name': 'Splitting Mode', 'type': 'str', 'value': self.strsplittingMode},
                {'name': 'Storage Mode', 'type': 'str', 'value': self.strstoreMode},
                {'name': 'Using Segmentation', 'type': 'bool', 'value': self.usingSegmentationMasks},
                {'name': 'Random Shuffle', 'type': 'bool', 'value': self.isRandomShuffle},
                {'name': 'Test/Train Ratio', 'type': 'float', 'value': self.trainTestDatasetRatio},
                {'name': 'Validation/Train Ratio', 'type': 'float', 'value': self.trainValidationRatio},
                {'name': 'Number of folds', 'type': 'int', 'value': self.numFolds}
            ]},
            {'name': 'Training Options', 'type': 'group', 'children': [
                {'name': 'neural Network', 'type': 'str', 'value': self.neuralNetworkModel},
                {'name': 'Learning Output Path', 'type': 'str', 'value': self.learningOutputPath},
                {'name': 'Learning Rate', 'type': 'float', 'value': self.learningRates[0]},
                {'name': 'Batch Size', 'type': 'int', 'value': self.batchSizes[0]},
                {'name': 'Optimizer', 'type': 'str', 'value': self.stroptimizer},
                {'name': 'Epochs', 'type': 'int', 'value': self.epochs},
                {'name': 'Weight Decay', 'type': 'float', 'value': self.weightDecay},
                {'name': 'Momentum', 'type': 'float', 'value': self.momentum},
                {'name': 'Using Nesterov', 'type': 'bool', 'value': self.nesterovEnabled},
                {'name': 'Using Data Augmentation', 'type': 'bool', 'value': 0, 'children': [
                    {'name': 'Horizontal Flip', 'type': 'bool', 'value': self.horizontalFlip},
                    {'name': 'Vertical Flip', 'type': 'bool', 'value': self.verticalFlip},
                    {'name': 'Rotation', 'type': 'bool', 'value': self.rotation},
                    {'name': 'Zoom', 'type': 'bool', 'value': self.zoom},
                    {'name': 'ZCA whitening', 'type': 'bool', 'value': self.zcaWhitening},
                    {'name': 'Height Shift', 'type': 'bool', 'value': self.heightShift},
                    {'name': 'Width Shift', 'type': 'bool', 'value': self.widthShift},
                    {'name': 'Contrast Stretching', 'type': 'bool', 'value': self.contrastStretching},
                    {'name': 'Histogram Equalization', 'type': 'bool', 'value': self.histogram_eq},
                    {'name': 'Adaptive Equalization', 'type': 'bool', 'value': self.adaptive_eq}]},
                {'name': 'Select Multiclasses', 'type': 'group', 'children': [
                    {'name': 'Artifacts', 'type': 'bool', 'value': self.usingArtifacts},
                    {'name': 'Body Region', 'type': 'bool', 'value': self.usingBodyRegions},
                    {'name': 'T Weighting', 'type': 'bool', 'value': self.usingTWeighting}
                ]},
                {'name': 'Current GPU', 'type': 'int', 'value': self.gpu_id},
            ]},
            {'name': 'Testing Options', 'type': 'group', 'children': [
                {'name': 'neural Network', 'type': 'str', 'value': self.modelPrediction},
                {'name': 'Batch Size', 'type': 'int', 'value': self.batchSizePrediction},
                {'name': 'Optimizer', 'type': 'str', 'value': self.stroptimizer},
                {'name': 'Select Multiclasses', 'type': 'group', 'children': [
                    {'name': 'Artifacts', 'type': 'bool', 'value': self.usingArtifactsPrediction},
                    {'name': 'Body Region', 'type': 'bool', 'value': self.usingBodyRegionsPrediction},
                    {'name': 'T Weighting', 'type': 'bool', 'value': self.usingTWeightingPrediction}
                ]},
                {'name': 'Patch Size', 'type': 'str', 'value': self.patchSizePrediction},
                {'name': 'Overlap', 'type': 'float', 'value': self.patchOverlapPrediction},
                {'name': 'Current GPU', 'type': 'int', 'value': self.gpu_prediction_id},
            ]},
            {'name': 'Learning Output', 'type': 'group', 'children': [
                {'name': 'dataset output for training', 'type': 'str', 'value': self.datasetOutputPath},
                {'name': 'network output for training', 'type': 'group', 'children': [
                    {'name': 'network source for training', 'type': 'str', 'value': self.neuralnetworkPath},
                    {'name': 'network output for training', 'type': 'str', 'value': self.outPutFolderDataPath},
                ]},
                {'name': 'dataset output for prediction', 'type': 'str', 'value': self.datasetForPrediction},
                {'name': 'network output for prediction', 'type': 'group', 'children': [
                    {'name': 'network source for prediction', 'type': 'str', 'value': self.modelPredictionSource},
                    {'name': 'network output for prediction', 'type': 'str', 'value': self.modelPathPrediction},
                ]},
            ]},
        ]

    def getParameters(self):
        self.updateParameters()
        return self.params

    def updateParameters(self):
        self.params = [
            {'name': 'Dataset', 'type': 'group', 'children': [
                {'name': 'X_train', 'type': 'str', 'value': self.X_train_shape},
                {'name': 'y_train', 'type': 'str', 'value': self.Y_train_shape},
                {'name': 'X_validation', 'type': 'str', 'value': self.X_validation_shape},
                {'name': 'y_validation', 'type': 'str', 'value': self.Y_validation_shape},
                {'name': 'X_test', 'type': 'str', 'value': self.X_test_shape},
                {'name': 'y_test', 'type': 'str', 'value': self.Y_test_shape},
                {'name': 'Y_segMasks_train', 'type': 'str', 'value': self.Y_segMasks_train_shape},
                {'name': 'Y_segMasks_validation', 'type': 'str', 'value': self.Y_segMasks_validation_shape},
                {'name': 'Y_segMasks_test', 'type': 'str', 'value': self.Y_segMasks_test_shape},
            ]},
            {'name': 'Patching Options', 'type': 'group', 'children': [
                {'name': 'Labeling Mode', 'type': 'str', 'value': self.strlabelingMode},
                {'name': 'Patching Mode', 'type': 'str', 'value': self.strpatchingMode},
                {'name': 'Patch Size', 'type': 'str', 'value': [self.patchSizeX, self.patchSizeY, self.patchSizeZ]},
                {'name': 'Overlap', 'type': 'float', 'value': self.patchOverlap},
                {'name': 'Markings Path', 'type': 'str', 'value': self.markingsPath},
                {'name': 'Output Path', 'type': 'str', 'value': self.pathOutputPatching},
                {'name': 'Splitting Mode', 'type': 'str', 'value': self.strsplittingMode},
                {'name': 'Storage Mode', 'type': 'str', 'value': self.strstoreMode},
                {'name': 'Using Segmentation', 'type': 'bool', 'value': self.usingSegmentationMasks},
                {'name': 'Random Shuffle', 'type': 'bool', 'value': self.isRandomShuffle},
                {'name': 'Test/Train Ratio', 'type': 'float', 'value': self.trainTestDatasetRatio},
                {'name': 'Validation/Train Ratio', 'type': 'float', 'value': self.trainValidationRatio},
                {'name': 'Number of folds', 'type': 'int', 'value': self.numFolds}
            ]},
            {'name': 'Training Options', 'type': 'group', 'children': [
                {'name': 'neural Network', 'type': 'str', 'value': self.neuralNetworkModel},
                {'name': 'Learning Output Path', 'type': 'str', 'value': self.learningOutputPath},
                {'name': 'Learning Rate', 'type': 'float', 'value': self.learningRates[0]},
                {'name': 'Batch Size', 'type': 'int', 'value': self.batchSizes[0]},
                {'name': 'Optimizer', 'type': 'str', 'value': self.stroptimizer},
                {'name': 'Epochs', 'type': 'int', 'value': self.epochs},
                {'name': 'Weight Decay', 'type': 'float', 'value': self.weightDecay},
                {'name': 'Momentum', 'type': 'float', 'value': self.momentum},
                {'name': 'Using Nesterov', 'type': 'bool', 'value': self.nesterovEnabled},
                {'name': 'Using Data Augmentation', 'type': 'bool', 'value': 0, 'children': [
                    {'name': 'Horizontal Flip', 'type': 'bool', 'value': self.horizontalFlip},
                    {'name': 'Vertical Flip', 'type': 'bool', 'value': self.verticalFlip},
                    {'name': 'Rotation', 'type': 'bool', 'value': self.rotation},
                    {'name': 'Zoom', 'type': 'bool', 'value': self.zoom},
                    {'name': 'ZCA whitening', 'type': 'bool', 'value': self.zcaWhitening},
                    {'name': 'Height Shift', 'type': 'bool', 'value': self.heightShift},
                    {'name': 'Width Shift', 'type': 'bool', 'value': self.widthShift},
                    {'name': 'Contrast Stretching', 'type': 'bool', 'value': self.contrastStretching},
                    {'name': 'Histogram Equalization', 'type': 'bool', 'value': self.histogram_eq},
                    {'name': 'Adaptive Equalization', 'type': 'bool', 'value': self.adaptive_eq}]},
                {'name': 'Select Multiclasses', 'type': 'group', 'children': [
                    {'name': 'Artifacts', 'type': 'bool', 'value': self.usingArtifacts},
                    {'name': 'Body Region', 'type': 'bool', 'value': self.usingBodyRegions},
                    {'name': 'T Weighting', 'type': 'bool', 'value': self.usingTWeighting}
                ]},
                {'name': 'Current GPU', 'type': 'int', 'value': self.gpu_id},
            ]},
            {'name': 'Testing Options', 'type': 'group', 'children': [
                {'name': 'neural Network', 'type': 'str', 'value': self.modelPrediction},
                {'name': 'Batch Size', 'type': 'int', 'value': self.batchSizePrediction},
                {'name': 'Optimizer', 'type': 'str', 'value': self.stroptimizer},
                {'name': 'Select Multiclasses', 'type': 'group', 'children': [
                    {'name': 'Artifacts', 'type': 'bool', 'value': self.usingArtifactsPrediction},
                    {'name': 'Body Region', 'type': 'bool', 'value': self.usingBodyRegionsPrediction},
                    {'name': 'T Weighting', 'type': 'bool', 'value': self.usingTWeightingPrediction}
                ]},
                {'name': 'Patch Size', 'type': 'str', 'value': self.patchSizePrediction},
                {'name': 'Overlap', 'type': 'float', 'value': self.patchOverlapPrediction},
                {'name': 'Current GPU', 'type': 'int', 'value': self.gpu_prediction_id},
            ]},
            {'name': 'Learning Output', 'type': 'group', 'children': [
                {'name': 'dataset output for training', 'type': 'str', 'value': self.datasetOutputPath},
                {'name': 'network output for training', 'type': 'group', 'children': [
                    {'name': 'network source for training', 'type': 'str', 'value': self.neuralnetworkPath},
                    {'name': 'network output for training', 'type': 'str', 'value': self.outPutFolderDataPath},
                ]},
                {'name': 'dataset output for prediction', 'type': 'str', 'value': self.datasetForPrediction},
                {'name': 'network output for prediction', 'type': 'group', 'children': [
                    {'name': 'network source for prediction', 'type': 'str', 'value': self.modelPredictionSource},
                    {'name': 'network output for prediction', 'type': 'str', 'value': self.modelPathPrediction},
                ]},
            ]},
        ]

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

        if self.patchingMode == PATCHING_2D:
            dAllPatches = np.zeros((self.patchSizeX, self.patchSizeY, 0))
            dAllLabels = np.zeros(0)
            if self.usingSegmentationMasks:
                dAllSegmentationMaskPatches = np.zeros((self.patchSizeX, self.patchSizeY, 0))
        elif self.patchingMode == PATCHING_3D:
            dAllPatches = np.zeros((self.patchSizeX, self.patchSizeY, self.patchSizeZ, 0))
            dAllLabels = np.zeros(0)
            if self.usingSegmentationMasks:
                dAllSegmentationMaskPatches = np.zeros((self.patchSizeX, self.patchSizeY, self.patchSizeZ, 0))
        else:
            raise IOError("We do not know your patching mode...")

        # stuff for storing
        if self.storeMode != STORE_DISABLED:
            # outPutFolder name:
            outPutFolder = "Patients-" + str(len(self.selectedPatients)) + "_" + \
                           "Datasets-" + str(len(self.selectedDatasets)) + "_" + \
                           ("2D" if self.patchingMode == PATCHING_2D else "3D") + \
                           ('_SegMask_' if self.usingSegmentationMasks else '_') + \
                           str(self.patchSizeX) + "x" + str(self.patchSizeY)
            if self.patchingMode == PATCHING_3D:
                outPutFolder = outPutFolder + "x" + str(self.patchSizeZ)

            outPutFolder = outPutFolder + "_Overlap-" + str(self.patchOverlap) + "_" + \
                           "Labeling-" + ("patch" if self.labelingMode == PATCH_LABELING else "mask")

            if self.splittingMode == SIMPLE_RANDOM_SAMPLE_SPLITTING:
                outPutFolder = outPutFolder + "_Split-simpleRand"
            elif self.splittingMode == CROSS_VALIDATION_SPLITTING:
                outPutFolder = outPutFolder + "_Split-crossVal"
            elif self.splittingMode == SIMPLE_RANDOM_SAMPLE_SPLITTING:
                outPutFolder = outPutFolder + "Split-patientCrossVal"

            outputFolderPath = self.pathOutputPatching + os.sep + outPutFolder

            if not os.path.exists(outputFolderPath):
                os.makedirs(outputFolderPath)

            # create dataset summary
            self.datasetName = outPutFolder
            self.createDatasetInfoSummary(outPutFolder, outputFolderPath)

            if self.storeMode == STORE_PATCH_BASED:
                self.outPutFolderDataPath = outputFolderPath + os.sep + "data"
                if not os.path.exists(self.outPutFolderDataPath):
                    os.makedirs(self.outPutFolderDataPath)

                labelDict = {}

        # for storing patch based
        iPatchToDisk = 0

        #### DIY splitting data set
        print(self.splittingMode)
        if self.splittingMode == DIY_SPLITTING:

            ### for training data

            for dataset in self.dataset_train:
                currentDataDir = self.pathDatabase + os.sep + dataset
                patient = dataset.split('/')[0]
                sequence = dataset.split('/')[-1]
                if os.path.exists(currentDataDir):
                    # get list with all paths of dicoms for current patient and current dataset
                    fileNames = os.listdir(currentDataDir)
                    fileNames = [os.path.join(currentDataDir, f) for f in fileNames]

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
                    if self.patchingMode == PATCHING_2D:
                        # 2D patching
                        # mask labeling or path labeling
                        if self.labelingMode == MASK_LABELING:
                            # path to marking file
                            currentMarkingsPath = self.getMarkingsPath() + os.sep + patient + ".json"
                            # get the markings mask
                            labelMask_ndarray = create_MASK_Array(currentMarkingsPath, patient, sequence,
                                                                  voxel_ndarray.shape[0],
                                                                  voxel_ndarray.shape[1], voxel_ndarray.shape[2])

                            # compute 2D Mask labling patching
                            dPatches, dLabels = fRigidPatching_maskLabeling(norm_voxel_ndarray,
                                                                            [self.patchSizeX, self.patchSizeY],
                                                                            self.patchOverlap,
                                                                            labelMask_ndarray, 0.5,
                                                                            self.datasets[sequence])

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
                                                                                              sequence])

                                dPatchesOfMask = np.asarray(dPatchesOfMask, dtype=np.float32)

                            ############################################################################################


                        elif self.labelingMode == PATCH_LABELING:
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
                    elif self.patchingMode == PATCHING_3D:
                        # 3D Patching
                        if self.labelingMode == MASK_LABELING:
                            # path to marking file
                            currentMarkingsPath = self.getMarkingsPath() + os.sep + patient + ".json"
                            # get the markings mask
                            labelMask_ndarray = create_MASK_Array(currentMarkingsPath, patient, sequence,
                                                                  voxel_ndarray.shape[0],
                                                                  voxel_ndarray.shape[1], voxel_ndarray.shape[2])

                            # compute 3D Mask labling patching
                            dPatches, dLabels = fRigidPatching3D_maskLabeling(norm_voxel_ndarray,
                                                                              [self.patchSizeX, self.patchSizeY,
                                                                               self.patchSizeZ],
                                                                              self.patchOverlap,
                                                                              labelMask_ndarray,
                                                                              0.5,
                                                                              self.datasets[sequence])

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
                                                                                                sequence])
                                dPatchesOfMask = np.asarray(dPatchesOfMask, dtype=np.byte)
                            ############################################################################################

                        elif self.labelingMode == PATCH_LABELING:
                            print("3D local patch labeling not available until now!")

                    else:
                        print("We do not know what labeling mode you want to use :p")

                    if self.storeMode == STORE_PATCH_BASED:
                        # patch based storage
                        if self.patchingMode == PATCHING_3D:
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
                        if self.patchingMode == PATCHING_2D:
                            dAllPatches = np.concatenate((dAllPatches, dPatches), axis=2)
                            dAllLabels = np.concatenate((dAllLabels, dLabels), axis=0)
                            if self.usingSegmentationMasks:
                                dAllSegmentationMaskPatches = np.concatenate(
                                    (dAllSegmentationMaskPatches, dPatchesOfMask), axis=2)
                        elif self.patchingMode == PATCHING_3D:
                            dAllPatches = np.concatenate((dAllPatches, dPatches), axis=3)
                            dAllLabels = np.concatenate((dAllLabels, dLabels), axis=0)
                            if self.usingSegmentationMasks:
                                dAllSegmentationMaskPatches = np.concatenate(
                                    (dAllSegmentationMaskPatches, dPatchesOfMask), axis=3)
            if self.storeMode != STORE_DISABLED:
                # H5py store mode
                if self.storeMode == STORE_HDF5:

                    if self.patchingMode == PATCHING_2D:
                        if not self.usingSegmentationMasks:
                            [self.X_train], [self.Y_train] = TransformDataset(dAllPatches, dAllLabels,
                                                                              patchSize=[self.patchSizeX,
                                                                                         self.patchSizeY],
                                                                              patchOverlap=self.patchOverlap,
                                                                              isRandomShuffle=self.isRandomShuffle,
                                                                              isUsingSegmentation=False,
                                                                              allSegmentationMasks=None)
                        else:
                            # do segmentation mask split
                            [self.X_train], [self.Y_train], [self.Y_segMasks_train] = \
                                TransformDataset(dAllPatches,
                                                 dAllLabels,
                                                 patchSize=[self.patchSizeX, self.patchSizeY],
                                                 patchOverlap=self.patchOverlap,
                                                 isRandomShuffle=self.isRandomShuffle,
                                                 isUsingSegmentation=True,
                                                 allSegmentationMasks=dAllSegmentationMaskPatches)

                        # store datasets with h5py
                        self.datasetOutputPath = outputFolderPath
                        with h5py.File(outputFolderPath + os.sep + 'datasets.hdf5', 'w') as hf:
                            hf.create_dataset('X_train', data=self.X_train)
                            hf.create_dataset('Y_train', data=self.Y_train)
                            if self.usingSegmentationMasks:
                                hf.create_dataset('Y_segMasks_train', data=self.Y_segMasks_train)

                    elif self.patchingMode == PATCHING_3D:
                        if not self.usingSegmentationMasks:
                            [self.X_train], [self.Y_train] = TransformDataset(dAllPatches, dAllLabels,
                                                                              patchSize=[self.patchSizeX,
                                                                                         self.patchSizeY,
                                                                                         self.patchSizeZ],
                                                                              patchOverlap=self.patchOverlap,
                                                                              isRandomShuffle=self.isRandomShuffle,
                                                                              isUsingSegmentation=False,
                                                                              allSegmentationMasks=None)
                        else:
                            [self.X_train], [self.Y_train], [self.Y_segMasks_train] = \
                                TransformDataset(dAllPatches,
                                                 dAllLabels,
                                                 patchSize=[self.patchSizeX,
                                                            self.patchSizeY,
                                                            self.patchSizeZ],
                                                 patchOverlap=self.patchOverlap,
                                                 isRandomShuffle=self.isRandomShuffle,
                                                 isUsingSegmentation=True,
                                                 allSegmentationMasks=dAllSegmentationMaskPatches)

                        # store datasets with h5py
                        self.datasetOutputPath = outputFolderPath
                        with h5py.File(outputFolderPath + os.sep + 'datasets.hdf5', 'w') as hf:
                            hf.create_dataset('X_train', data=self.X_train)
                            hf.create_dataset('Y_train', data=self.Y_train)
                            if self.usingSegmentationMasks:
                                hf.create_dataset('Y_segMasks_train', data=self.Y_segMasks_train)

                elif self.storeMode == STORE_PATCH_BASED:
                    self.datasetOutputPath = outputFolderPath
                    with open(outputFolderPath + os.sep + "labels.json", 'w') as fp:
                        json.dump(labelDict, fp)
            else:
                # no storage of patched datasets
                if self.patchingMode == PATCHING_2D:
                    if not self.usingSegmentationMasks:
                        [self.X_train], [self.Y_train] = TransformDataset(dAllPatches, dAllLabels,
                                                                          patchSize=[self.patchSizeX,
                                                                                     self.patchSizeY],
                                                                          patchOverlap=self.patchOverlap,
                                                                          isRandomShuffle=self.isRandomShuffle,
                                                                          isUsingSegmentation=False,
                                                                          allSegmentationMasks=None)
                    else:
                        # do segmentation mask split
                        [self.X_train], [self.Y_train], [self.Y_segMasks_train] = \
                            TransformDataset(dAllPatches,
                                             dAllLabels,
                                             patchSize=[self.patchSizeX, self.patchSizeY],
                                             patchOverlap=self.patchOverlap,
                                             isRandomShuffle=self.isRandomShuffle,
                                             isUsingSegmentation=True,
                                             allSegmentationMasks=dAllSegmentationMaskPatches)

                elif self.patchingMode == PATCHING_3D:
                    if not self.usingSegmentationMasks:
                        [self.X_train], [self.Y_train] = TransformDataset(dAllPatches, dAllLabels,
                                                                          patchSize=[self.patchSizeX,
                                                                                     self.patchSizeY,
                                                                                     self.patchSizeZ],
                                                                          patchOverlap=self.patchOverlap,
                                                                          isRandomShuffle=self.isRandomShuffle,
                                                                          isUsingSegmentation=False,
                                                                          allSegmentationMasks=None)
                    else:
                        [self.X_train], [self.Y_train], [self.Y_segMasks_train] = \
                            TransformDataset(dAllPatches,
                                             dAllLabels,
                                             patchSize=[self.patchSizeX,
                                                        self.patchSizeY,
                                                        self.patchSizeZ],
                                             patchOverlap=self.patchOverlap,
                                             isRandomShuffle=self.isRandomShuffle,
                                             isUsingSegmentation=True,
                                             allSegmentationMasks=dAllSegmentationMaskPatches)

            print('X_train', self.X_train.shape)
            print(self.dataset_train)

            ### for validation data

            for dataset in self.dataset_validation:
                currentDataDir = self.pathDatabase + os.sep + dataset
                patient = dataset.split('/')[0]
                sequence = dataset.split('/')[-1]
                if os.path.exists(currentDataDir):
                    # get list with all paths of dicoms for current patient and current dataset
                    fileNames = os.listdir(currentDataDir)
                    fileNames = [os.path.join(currentDataDir, f) for f in fileNames]

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
                    if self.patchingMode == PATCHING_2D:
                        # 2D patching
                        # mask labeling or path labeling
                        if self.labelingMode == MASK_LABELING:
                            # path to marking file
                            currentMarkingsPath = self.getMarkingsPath() + os.sep + patient + ".json"
                            # get the markings mask
                            labelMask_ndarray = create_MASK_Array(currentMarkingsPath, patient, sequence,
                                                                  voxel_ndarray.shape[0],
                                                                  voxel_ndarray.shape[1], voxel_ndarray.shape[2])

                            # compute 2D Mask labling patching
                            dPatches, dLabels = fRigidPatching_maskLabeling(norm_voxel_ndarray,
                                                                            [self.patchSizeX, self.patchSizeY],
                                                                            self.patchOverlap,
                                                                            labelMask_ndarray, 0.5,
                                                                            self.datasets[sequence])

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
                                                                                              sequence])

                                dPatchesOfMask = np.asarray(dPatchesOfMask, dtype=np.float32)

                            ############################################################################################


                        elif self.labelingMode == PATCH_LABELING:
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
                    elif self.patchingMode == PATCHING_3D:
                        # 3D Patching
                        if self.labelingMode == MASK_LABELING:
                            # path to marking file
                            currentMarkingsPath = self.getMarkingsPath() + os.sep + patient + ".json"
                            # get the markings mask
                            labelMask_ndarray = create_MASK_Array(currentMarkingsPath, patient, sequence,
                                                                  voxel_ndarray.shape[0],
                                                                  voxel_ndarray.shape[1], voxel_ndarray.shape[2])

                            # compute 3D Mask labling patching
                            dPatches, dLabels = fRigidPatching3D_maskLabeling(norm_voxel_ndarray,
                                                                              [self.patchSizeX, self.patchSizeY,
                                                                               self.patchSizeZ],
                                                                              self.patchOverlap,
                                                                              labelMask_ndarray,
                                                                              0.5,
                                                                              self.datasets[sequence])

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
                                                                                                sequence])
                                dPatchesOfMask = np.asarray(dPatchesOfMask, dtype=np.byte)
                            ############################################################################################

                        elif self.labelingMode == PATCH_LABELING:
                            print("3D local patch labeling not available until now!")

                    else:
                        print("We do not know what labeling mode you want to use :p")

                    if self.storeMode == STORE_PATCH_BASED:
                        # patch based storage
                        if self.patchingMode == PATCHING_3D:
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
                        if self.patchingMode == PATCHING_2D:
                            dAllPatches = np.concatenate((dAllPatches, dPatches), axis=2)
                            dAllLabels = np.concatenate((dAllLabels, dLabels), axis=0)
                            if self.usingSegmentationMasks:
                                dAllSegmentationMaskPatches = np.concatenate(
                                    (dAllSegmentationMaskPatches, dPatchesOfMask), axis=2)
                        elif self.patchingMode == PATCHING_3D:
                            dAllPatches = np.concatenate((dAllPatches, dPatches), axis=3)
                            dAllLabels = np.concatenate((dAllLabels, dLabels), axis=0)
                            if self.usingSegmentationMasks:
                                dAllSegmentationMaskPatches = np.concatenate(
                                    (dAllSegmentationMaskPatches, dPatchesOfMask), axis=3)
            if self.storeMode != STORE_DISABLED:
                # H5py store mode
                if self.storeMode == STORE_HDF5:

                    if self.patchingMode == PATCHING_2D:
                        if not self.usingSegmentationMasks:
                            [self.X_validation], [self.Y_validation] = TransformDataset(dAllPatches, dAllLabels,
                                                                                        patchSize=[self.patchSizeX,
                                                                                                   self.patchSizeY],
                                                                                        patchOverlap=self.patchOverlap,
                                                                                        isRandomShuffle=self.isRandomShuffle,
                                                                                        isUsingSegmentation=False,
                                                                                        allSegmentationMasks=None)
                        else:
                            # do segmentation mask split
                            [self.X_validation], [self.Y_validation], [self.Y_segMasks_validation] = \
                                TransformDataset(dAllPatches,
                                                 dAllLabels,
                                                 patchSize=[self.patchSizeX, self.patchSizeY],
                                                 patchOverlap=self.patchOverlap,
                                                 isRandomShuffle=self.isRandomShuffle,
                                                 isUsingSegmentation=True,
                                                 allSegmentationMasks=dAllSegmentationMaskPatches)

                        # store datasets with h5py
                        self.datasetOutputPath = outputFolderPath
                        with h5py.File(outputFolderPath + os.sep + 'datasets.hdf5', 'w') as hf:
                            hf.create_dataset('X_validation', data=self.X_validation)
                            hf.create_dataset('Y_validation', data=self.Y_validation)
                            if self.usingSegmentationMasks:
                                hf.create_dataset('Y_segMasks_validation', data=self.Y_segMasks_validation)

                    elif self.patchingMode == PATCHING_3D:
                        if not self.usingSegmentationMasks:
                            [self.X_validation], [self.Y_validation] = TransformDataset(dAllPatches, dAllLabels,
                                                                                        patchSize=[self.patchSizeX,
                                                                                                   self.patchSizeY,
                                                                                                   self.patchSizeZ],
                                                                                        patchOverlap=self.patchOverlap,
                                                                                        isRandomShuffle=self.isRandomShuffle,
                                                                                        isUsingSegmentation=False,
                                                                                        allSegmentationMasks=None)
                        else:
                            [self.X_validation], [self.Y_validation], [self.Y_segMasks_validation] = \
                                TransformDataset(dAllPatches,
                                                 dAllLabels,
                                                 patchSize=[self.patchSizeX,
                                                            self.patchSizeY,
                                                            self.patchSizeZ],
                                                 patchOverlap=self.patchOverlap,
                                                 isRandomShuffle=self.isRandomShuffle,
                                                 isUsingSegmentation=True,
                                                 allSegmentationMasks=dAllSegmentationMaskPatches)

                        # store datasets with h5py
                        self.datasetOutputPath = outputFolderPath
                        with h5py.File(outputFolderPath + os.sep + 'datasets.hdf5', 'w') as hf:
                            hf.create_dataset('X_validation', data=self.X_validation)
                            hf.create_dataset('Y_validation', data=self.Y_validation)
                            if self.usingSegmentationMasks:
                                hf.create_dataset('Y_segMasks_validation', data=self.Y_segMasks_validation)

                elif self.storeMode == STORE_PATCH_BASED:
                    self.datasetOutputPath = outputFolderPath
                    with open(outputFolderPath + os.sep + "labels.json", 'w') as fp:
                        json.dump(labelDict, fp)
            else:
                # no storage of patched datasets
                if self.patchingMode == PATCHING_2D:
                    if not self.usingSegmentationMasks:
                        [self.X_validation], [self.Y_validation] = TransformDataset(dAllPatches, dAllLabels,
                                                                                    patchSize=[self.patchSizeX,
                                                                                               self.patchSizeY],
                                                                                    patchOverlap=self.patchOverlap,
                                                                                    isRandomShuffle=self.isRandomShuffle,
                                                                                    isUsingSegmentation=False,
                                                                                    allSegmentationMasks=None)
                    else:
                        # do segmentation mask split
                        [self.X_validation], [self.Y_validation], [self.Y_segMasks_validation] = \
                            TransformDataset(dAllPatches,
                                             dAllLabels,
                                             patchSize=[self.patchSizeX, self.patchSizeY],
                                             patchOverlap=self.patchOverlap,
                                             isRandomShuffle=self.isRandomShuffle,
                                             isUsingSegmentation=True,
                                             allSegmentationMasks=dAllSegmentationMaskPatches)

                elif self.patchingMode == PATCHING_3D:
                    if not self.usingSegmentationMasks:
                        [self.X_validation], [self.Y_validation] = TransformDataset(dAllPatches, dAllLabels,
                                                                                    patchSize=[self.patchSizeX,
                                                                                               self.patchSizeY,
                                                                                               self.patchSizeZ],
                                                                                    patchOverlap=self.patchOverlap,
                                                                                    isRandomShuffle=self.isRandomShuffle,
                                                                                    isUsingSegmentation=False,
                                                                                    allSegmentationMasks=None)
                    else:
                        [self.X_validation], [self.Y_validation], [self.Y_segMasks_validation] = \
                            TransformDataset(dAllPatches,
                                             dAllLabels,
                                             patchSize=[self.patchSizeX,
                                                        self.patchSizeY,
                                                        self.patchSizeZ],
                                             patchOverlap=self.patchOverlap,
                                             isRandomShuffle=self.isRandomShuffle,
                                             isUsingSegmentation=True,
                                             allSegmentationMasks=dAllSegmentationMaskPatches)

            print('X_validation', self.X_validation.shape)
            print(self.dataset_validation)

            ### for test data

            for dataset in self.dataset_test:
                currentDataDir = self.pathDatabase + os.sep + dataset
                patient = dataset.split('/')[0]
                sequence = dataset.split('/')[-1]
                if os.path.exists(currentDataDir):
                    # get list with all paths of dicoms for current patient and current dataset
                    fileNames = os.listdir(currentDataDir)
                    fileNames = [os.path.join(currentDataDir, f) for f in fileNames]

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
                    if self.patchingMode == PATCHING_2D:
                        # 2D patching
                        # mask labeling or path labeling
                        if self.labelingMode == MASK_LABELING:
                            # path to marking file
                            currentMarkingsPath = self.getMarkingsPath() + os.sep + patient + ".json"
                            # get the markings mask
                            labelMask_ndarray = create_MASK_Array(currentMarkingsPath, patient, sequence,
                                                                  voxel_ndarray.shape[0],
                                                                  voxel_ndarray.shape[1], voxel_ndarray.shape[2])

                            # compute 2D Mask labling patching
                            dPatches, dLabels = fRigidPatching_maskLabeling(norm_voxel_ndarray,
                                                                            [self.patchSizeX, self.patchSizeY],
                                                                            self.patchOverlap,
                                                                            labelMask_ndarray, 0.5,
                                                                            self.datasets[sequence])

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
                                                                                              sequence])

                                dPatchesOfMask = np.asarray(dPatchesOfMask, dtype=np.float32)

                            ############################################################################################


                        elif self.labelingMode == PATCH_LABELING:
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
                    elif self.patchingMode == PATCHING_3D:
                        # 3D Patching
                        if self.labelingMode == MASK_LABELING:
                            # path to marking file
                            currentMarkingsPath = self.getMarkingsPath() + os.sep + patient + ".json"
                            # get the markings mask
                            labelMask_ndarray = create_MASK_Array(currentMarkingsPath, patient, sequence,
                                                                  voxel_ndarray.shape[0],
                                                                  voxel_ndarray.shape[1], voxel_ndarray.shape[2])

                            # compute 3D Mask labling patching
                            dPatches, dLabels = fRigidPatching3D_maskLabeling(norm_voxel_ndarray,
                                                                              [self.patchSizeX, self.patchSizeY,
                                                                               self.patchSizeZ],
                                                                              self.patchOverlap,
                                                                              labelMask_ndarray,
                                                                              0.5,
                                                                              self.datasets[sequence])

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
                                                                                                sequence])
                                dPatchesOfMask = np.asarray(dPatchesOfMask, dtype=np.byte)
                            ############################################################################################

                        elif self.labelingMode == PATCH_LABELING:
                            print("3D local patch labeling not available until now!")

                    else:
                        print("We do not know what labeling mode you want to use :p")

                    if self.storeMode == STORE_PATCH_BASED:
                        # patch based storage
                        if self.patchingMode == PATCHING_3D:
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
                        if self.patchingMode == PATCHING_2D:
                            dAllPatches = np.concatenate((dAllPatches, dPatches), axis=2)
                            dAllLabels = np.concatenate((dAllLabels, dLabels), axis=0)
                            if self.usingSegmentationMasks:
                                dAllSegmentationMaskPatches = np.concatenate(
                                    (dAllSegmentationMaskPatches, dPatchesOfMask), axis=2)
                        elif self.patchingMode == PATCHING_3D:
                            dAllPatches = np.concatenate((dAllPatches, dPatches), axis=3)
                            dAllLabels = np.concatenate((dAllLabels, dLabels), axis=0)
                            if self.usingSegmentationMasks:
                                dAllSegmentationMaskPatches = np.concatenate(
                                    (dAllSegmentationMaskPatches, dPatchesOfMask), axis=3)
            if self.storeMode != STORE_DISABLED:
                # H5py store mode
                if self.storeMode == STORE_HDF5:

                    if self.patchingMode == PATCHING_2D:
                        if not self.usingSegmentationMasks:
                            [self.X_test], [self.Y_test] = TransformDataset(dAllPatches, dAllLabels,
                                                                            patchSize=[self.patchSizeX,
                                                                                       self.patchSizeY],
                                                                            patchOverlap=self.patchOverlap,
                                                                            isRandomShuffle=self.isRandomShuffle,
                                                                            isUsingSegmentation=False,
                                                                            allSegmentationMasks=None)
                        else:
                            # do segmentation mask split
                            [self.X_test], [self.Y_test], [self.Y_segMasks_test] = \
                                TransformDataset(dAllPatches,
                                                 dAllLabels,
                                                 patchSize=[self.patchSizeX, self.patchSizeY],
                                                 patchOverlap=self.patchOverlap,
                                                 isRandomShuffle=self.isRandomShuffle,
                                                 isUsingSegmentation=True,
                                                 allSegmentationMasks=dAllSegmentationMaskPatches)

                        # store datasets with h5py
                        self.datasetOutputPath = outputFolderPath
                        with h5py.File(outputFolderPath + os.sep + 'datasets.hdf5', 'w') as hf:
                            hf.create_dataset('X_test', data=self.X_test)
                            hf.create_dataset('Y_test', data=self.Y_test)
                            if self.usingSegmentationMasks:
                                hf.create_dataset('Y_segMasks_test', data=self.Y_segMasks_test)

                    elif self.patchingMode == PATCHING_3D:
                        if not self.usingSegmentationMasks:
                            [self.X_test], [self.Y_test] = TransformDataset(dAllPatches, dAllLabels,
                                                                            patchSize=[self.patchSizeX,
                                                                                       self.patchSizeY,
                                                                                       self.patchSizeZ],
                                                                            patchOverlap=self.patchOverlap,
                                                                            isRandomShuffle=self.isRandomShuffle,
                                                                            isUsingSegmentation=False,
                                                                            allSegmentationMasks=None)
                        else:
                            [self.X_test], [self.Y_test], [self.Y_segMasks_test] = \
                                TransformDataset(dAllPatches,
                                                 dAllLabels,
                                                 patchSize=[self.patchSizeX,
                                                            self.patchSizeY,
                                                            self.patchSizeZ],
                                                 patchOverlap=self.patchOverlap,
                                                 isRandomShuffle=self.isRandomShuffle,
                                                 isUsingSegmentation=True,
                                                 allSegmentationMasks=dAllSegmentationMaskPatches)

                        # store datasets with h5py
                        self.datasetOutputPath = outputFolderPath
                        with h5py.File(outputFolderPath + os.sep + 'datasets.hdf5', 'w') as hf:
                            hf.create_dataset('X_test', data=self.X_test)
                            hf.create_dataset('Y_test', data=self.Y_test)
                            if self.usingSegmentationMasks:
                                hf.create_dataset('Y_segMasks_test', data=self.Y_segMasks_test)

                elif self.storeMode == STORE_PATCH_BASED:
                    self.datasetOutputPath = outputFolderPath
                    with open(outputFolderPath + os.sep + "labels.json", 'w') as fp:
                        json.dump(labelDict, fp)
            else:
                # no storage of patched datasets
                if self.patchingMode == PATCHING_2D:
                    if not self.usingSegmentationMasks:
                        [self.X_test], [self.Y_test] = TransformDataset(dAllPatches, dAllLabels,
                                                                        patchSize=[self.patchSizeX,
                                                                                   self.patchSizeY],
                                                                        patchOverlap=self.patchOverlap,
                                                                        isRandomShuffle=self.isRandomShuffle,
                                                                        isUsingSegmentation=False,
                                                                        allSegmentationMasks=None)
                    else:
                        # do segmentation mask split
                        [self.X_test], [self.Y_test], [self.Y_segMasks_test] = \
                            TransformDataset(dAllPatches,
                                             dAllLabels,
                                             patchSize=[self.patchSizeX, self.patchSizeY],
                                             patchOverlap=self.patchOverlap,
                                             isRandomShuffle=self.isRandomShuffle,
                                             isUsingSegmentation=True,
                                             allSegmentationMasks=dAllSegmentationMaskPatches)

                elif self.patchingMode == PATCHING_3D:
                    if not self.usingSegmentationMasks:
                        [self.X_test], [self.Y_test] = TransformDataset(dAllPatches, dAllLabels,
                                                                        patchSize=[self.patchSizeX,
                                                                                   self.patchSizeY,
                                                                                   self.patchSizeZ],
                                                                        patchOverlap=self.patchOverlap,
                                                                        isRandomShuffle=self.isRandomShuffle,
                                                                        isUsingSegmentation=False,
                                                                        allSegmentationMasks=None)
                    else:
                        [self.X_test], [self.Y_test], [self.Y_segMasks_test] = \
                            TransformDataset(dAllPatches,
                                             dAllLabels,
                                             patchSize=[self.patchSizeX,
                                                        self.patchSizeY,
                                                        self.patchSizeZ],
                                             patchOverlap=self.patchOverlap,
                                             isRandomShuffle=self.isRandomShuffle,
                                             isUsingSegmentation=True,
                                             allSegmentationMasks=dAllSegmentationMaskPatches)

            print('X_test', self.X_test.shape)
            print(self.dataset_test)

        else:
            for patient in self.selectedPatients:
                for dataset in self.selectedDatasets:
                    currentDataDir = self.pathDatabase + os.sep + patient + os.sep + self.modelSubDir + os.sep + dataset

                    if os.path.exists(currentDataDir):
                        # get list with all paths of dicoms for current patient and current dataset
                        fileNames = os.listdir(currentDataDir)
                        fileNames = [os.path.join(currentDataDir, f) for f in fileNames]

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
                        if self.patchingMode == PATCHING_2D:
                            # 2D patching
                            # mask labeling or path labeling
                            if self.labelingMode == MASK_LABELING:
                                # path to marking file
                                currentMarkingsPath = self.getMarkingsPath() + os.sep + patient + ".json"
                                # get the markings mask
                                labelMask_ndarray = create_MASK_Array(currentMarkingsPath, patient, dataset,
                                                                      voxel_ndarray.shape[0],
                                                                      voxel_ndarray.shape[1], voxel_ndarray.shape[2])

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


                            elif self.labelingMode == PATCH_LABELING:
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
                        elif self.patchingMode == PATCHING_3D:
                            # 3D Patching
                            if self.labelingMode == MASK_LABELING:
                                # path to marking file
                                currentMarkingsPath = self.getMarkingsPath() + os.sep + patient + ".json"
                                # get the markings mask
                                labelMask_ndarray = create_MASK_Array(currentMarkingsPath, patient, dataset,
                                                                      voxel_ndarray.shape[0],
                                                                      voxel_ndarray.shape[1], voxel_ndarray.shape[2])

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

                            elif self.labelingMode == PATCH_LABELING:
                                print("3D local patch labeling not available until now!")

                        else:
                            print("We do not know what labeling mode you want to use :p")

                        if self.storeMode == STORE_PATCH_BASED:
                            # patch based storage
                            if self.patchingMode == PATCHING_3D:
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
                            if self.patchingMode == PATCHING_2D:
                                dAllPatches = np.concatenate((dAllPatches, dPatches), axis=2)
                                dAllLabels = np.concatenate((dAllLabels, dLabels), axis=0)
                                if self.usingSegmentationMasks:
                                    dAllSegmentationMaskPatches = np.concatenate(
                                        (dAllSegmentationMaskPatches, dPatchesOfMask), axis=2)
                            elif self.patchingMode == PATCHING_3D:
                                dAllPatches = np.concatenate((dAllPatches, dPatches), axis=3)
                                dAllLabels = np.concatenate((dAllLabels, dLabels), axis=0)
                                if self.usingSegmentationMasks:
                                    dAllSegmentationMaskPatches = np.concatenate(
                                        (dAllSegmentationMaskPatches, dPatchesOfMask), axis=3)

            # dataset splitting
            # store mode
            if self.storeMode != STORE_DISABLED:
                # H5py store mode
                if self.storeMode == STORE_HDF5:

                    if self.patchingMode == PATCHING_2D:
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
                            if self.usingSegmentationMasks == True:
                                hf.create_dataset('Y_segMasks_train', data=self.Y_segMasks_train)
                                hf.create_dataset('Y_segMasks_validation', data=self.Y_segMasks_validation)
                                hf.create_dataset('Y_segMasks_test', data=self.Y_segMasks_test)

                    elif self.patchingMode == PATCHING_3D:
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

                elif self.storeMode == STORE_PATCH_BASED:
                    self.datasetOutputPath = outputFolderPath
                    with open(outputFolderPath + os.sep + "labels.json", 'w') as fp:
                        json.dump(labelDict, fp)
            else:
                # no storage of patched datasets
                if self.patchingMode == PATCHING_2D:
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

                elif self.patchingMode == PATCHING_3D:
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

    def setNetworkCanrun(self, run):
        self.network_canrun = run

    def performTraining(self):
        # set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # get output vector for different classes
        classes = np.asarray(np.unique(self.Y_train, ), dtype=int)
        self.classMappings = Label.mapClassesToOutputVector(classes=classes, usingArtefacts=self.usingArtifacts,
                                                            usingBodyRegion=self.usingBodyRegions,
                                                            usingTWeightings=self.usingTWeighting)

        ##### Dataset preparation ####################################################################

        if self.usingSegmentationMasks:
            iCase = 0
            self.usingClassification = True
        else:
            iCase = 1

        if iCase == 0:

            Y_train = []

            Y_validation = []

            Y_test = []

            self.Y_segMasks_train[self.Y_segMasks_train == 3] = 1
            self.Y_segMasks_train[self.Y_segMasks_train == 2] = 0

            self.Y_segMasks_test[self.Y_segMasks_test == 3] = 1
            self.Y_segMasks_test[self.Y_segMasks_test == 2] = 0

            ##########################
            ###########################
            # for generating patch labels

            y_labels_train = np.expand_dims(self.Y_segMasks_train, axis=-1)
            y_labels_train[y_labels_train == 0] = -1
            y_labels_train[y_labels_train == 1] = 1
            y_labels_train = np.sum(y_labels_train, axis=1)
            y_labels_train = np.sum(y_labels_train, axis=1)
            y_labels_train = np.sum(y_labels_train, axis=1)
            y_labels_train[y_labels_train >= 0] = 1
            y_labels_train[y_labels_train < 0] = 0
            for i in range(y_labels_train.shape[0]):
                Y_train.append([1, 0] if y_labels_train[i].all() == 0 else [0, 1])
            Y_train = np.asarray(Y_train)

            y_labels_test = np.expand_dims(self.Y_segMasks_test, axis=-1)
            y_labels_test[y_labels_test == 0] = -1
            y_labels_test[y_labels_test == 1] = 1
            y_labels_test = np.sum(y_labels_test, axis=1)
            y_labels_test = np.sum(y_labels_test, axis=1)
            y_labels_test = np.sum(y_labels_test, axis=1)
            y_labels_test[y_labels_test >= 0] = 1
            y_labels_test[y_labels_test < 0] = 0

            for i in range(y_labels_test.shape[0]):
                Y_test.append([1, 0] if y_labels_test[i].all() == 0 else [0, 1])
            Y_test = np.asarray(Y_test)

            # change the shape of the dataset -> at color channel -> here one for grey scale

            # Y_segMasks_train_foreground = np.expand_dims(self.Y_segMasks_train, axis=-1)
            # Y_segMasks_train_background = np.ones(Y_segMasks_train_foreground.shape) - Y_segMasks_train_foreground
            # self.Y_segMasks_train = np.concatenate((Y_segMasks_train_background, Y_segMasks_train_foreground),
            #                                        axis=-1)
            # self.Y_segMasks_train = np.sum(self.Y_segMasks_train, axis=-1)
            #
            # Y_segMasks_test_foreground = np.expand_dims(self.Y_segMasks_test, axis=-1)
            # Y_segMasks_test_background = np.ones(Y_segMasks_test_foreground.shape) - Y_segMasks_test_foreground
            # self.Y_segMasks_test = np.concatenate((Y_segMasks_test_background, Y_segMasks_test_foreground),
            #                                        axis=-1)
            # self.Y_segMasks_test = np.sum(self.Y_segMasks_test, axis=-1)

            if self.X_validation.size == 0 and self.Y_validation.size == 0:
                self.X_validation = 0
                self.Y_segMasks_validation = 0
                self.Y_validation = 0
                print("No Validation Dataset.")
            else:
                for i in range(self.Y_validation.shape[0]):
                    Y_validation.append(self.classMappings[self.Y_validation[i]])
                Y_validation = np.asarray(Y_validation)
                self.Y_segMasks_validation[self.Y_segMasks_validation == 3] = 1
                self.Y_segMasks_validation[self.Y_segMasks_validation == 2] = 0

                # Y_segMasks_valid_foreground = np.expand_dims(self.Y_segMasks_validation, axis=-1)
                # Y_segMasks_valid_background = np.ones(Y_segMasks_valid_foreground.shape) - Y_segMasks_valid_foreground
                # self.Y_segMasks_validation = np.concatenate((Y_segMasks_valid_background, Y_segMasks_valid_foreground),
                #                                       axis=-1)
                # self.Y_segMasks_validation = np.sum(self.Y_segMasks_validation, axis=-1)

            # everything - multi hot encoding
        elif iCase == 1:
            Y_train = []
            for i in range(self.Y_train.shape[0]):
                Y_train.append(self.classMappings[self.Y_train[i]])
            Y_train = np.asarray(Y_train)

            Y_validation = []
            for i in range(self.Y_validation.shape[0]):
                Y_validation.append(self.classMappings[self.Y_validation[i]])
            Y_validation = np.asarray(Y_validation)

            Y_test = []
            for i in range(self.Y_test.shape[0]):
                Y_test.append(self.classMappings[self.Y_test[i]])
            Y_test = np.asarray(Y_test)

            # validation check
            if self.X_validation.size == 0 and self.Y_validation.size == 0:
                self.X_validation = 0
                self.Y_validation = 0
                print("No Validation Dataset.")
            else:
                pass

        ################################################################################################################

        # for i in range(self.X_train.shape[0]):
        #
        #     plt.subplot(141)
        #     plt.imshow(self.X_train[i, :, :, 4, 0])
        #
        #     plt.subplot(142)
        #     plt.imshow(self.Y_segMasks_train[i, :, :, 4, 0])
        #
        #     plt.subplot(143)
        #     plt.imshow(self.Y_segMasks_train[i, :, :, 4, 1])
        #
        #     #plt.subplot(144)
        #     #plt.imshow(self.Y_segMasks_train[i, :, :, 4, 2])
        #
        #     plt.show()
        #
        #     print(i)

        ###################################################################################################################

        # output folder
        self.outPutFolderDataPath = self.learningOutputPath + os.sep + self.neuralNetworkModel + "_"
        if self.patchingMode == PATCHING_2D:
            self.outPutFolderDataPath += "2D" + "_" + str(self.patchSizeX) + "x" + str(self.patchSizeY)
        elif self.patchingMode == PATCHING_3D:
            self.outPutFolderDataPath += "3D" + "_" + str(self.patchSizeX) + "x" + str(self.patchSizeY) + \
                                         "x" + str(self.patchSizeZ)

        self.outPutFolderDataPath += "_" + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')

        if not os.path.exists(self.outPutFolderDataPath):
            os.makedirs(self.outPutFolderDataPath)

        if not os.path.exists(self.outPutFolderDataPath + os.sep + 'checkpoints'):
            os.makedirs(self.outPutFolderDataPath + os.sep + 'checkpoints')

        # summarize cnn and training
        self.create_cnn_training_summary(self.neuralNetworkModel, self.outPutFolderDataPath)

        if iCase == 1:
            self.X_train_shape = self.X_train.shape
            self.Y_train_shape = Y_train.shape
            self.X_validation_shape = self.X_validation.shape
            self.Y_validation_shape = Y_validation.shape
            self.X_test_shape = self.X_test.shape
            self.Y_test_shape = Y_test.shape
            self.neuralnetworkPath = self.deepNeuralNetworks[self.neuralNetworkModel].replace(".", "/") + '.py'
            self.updateParameters()
            self._network_interface_update.emit()

            if self.network_canrun:
                fRunCNN(dData={'X_train': self.X_train, 'y_train': Y_train, 'X_valid': self.X_validation,
                               'y_valid': Y_validation, 'X_test': self.X_test, 'y_test': Y_test,
                               'patchSize': [self.patchSizeX, self.patchSizeY, self.patchSizeZ]},
                        sModelIn=self.deepNeuralNetworks[self.neuralNetworkModel],
                        lTrain=RUN_CNN_TRAIN_TEST_VALIDATION,
                        sParaOptim='',
                        sOutPath=self.outPutFolderDataPath,
                        iBatchSize=self.batchSizes,
                        iLearningRate=self.learningRates,
                        iEpochs=self.epochs,
                        dlart_handle=self)
        else:
            # segmentation FCN training
            self.X_train_shape = self.X_train.shape
            self.Y_train_shape = Y_train.shape
            self.X_validation_shape = self.X_validation.shape
            self.Y_validation_shape = Y_validation.shape
            self.X_test_shape = self.X_test.shape
            self.Y_test_shape = Y_test.shape
            self.Y_segMasks_train_shape = self.Y_segMasks_train.shape
            self.Y_segMasks_validation_shape = self.Y_segMasks_validation.shape
            self.Y_segMasks_test_shape = self.Y_segMasks_test.shape
            self.neuralnetworkPath = self.deepNeuralNetworks[self.neuralNetworkModel].replace(".", "/") + '.py'
            self.updateParameters()
            self._network_interface_update.emit()

            if self.network_canrun:
                fRunCNN(dData={'X_train': self.X_train,
                               'y_train': Y_train,
                               'Y_segMasks_train': self.Y_segMasks_train,
                               'X_valid': self.X_validation,
                               'y_valid': Y_validation,
                               'Y_segMasks_validation': self.Y_segMasks_validation,
                               'X_test': self.X_test,
                               'y_test': Y_test,
                               'Y_segMasks_test': self.Y_segMasks_test,
                               'patchSize': [self.patchSizeX, self.patchSizeY, self.patchSizeZ]},
                        sModelIn=self.deepNeuralNetworks[self.neuralNetworkModel],
                        lTrain=RUN_CNN_TRAIN_TEST_VALIDATION,
                        sParaOptim='',
                        sOutPath=self.outPutFolderDataPath,
                        iBatchSize=self.batchSizes,
                        iLearningRate=self.learningRates,
                        iEpochs=self.epochs,
                        dlart_handle=self,
                        usingSegmentationMasks=self.usingSegmentationMasks)

        # exit()

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

    def create_cnn_training_summary(self, name, outputFolderPath):
        dataDict = {}
        dataDict['Name'] = name
        dataDict['Date'] = datetime.datetime.today().strftime('%Y-%m-%d')
        dataDict['BatchSize'] = ''.join(str(e) for e in self.batchSizes)
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

    def setLabelingMode(self, mode):
        if mode == MASK_LABELING or mode == PATCH_LABELING:
            self.labelingMode = mode
        self.updateParameters()
        self.network_interface_update.emit()

    def getLabelingMode(self):
        return self.labelingMode

    def setMarkingsPath(self, path):
        self.markingsPath = path
        self.updateParameters()
        self.network_interface_update.emit()

    def getMarkingsPath(self):
        return self.markingsPath

    def setPatchSizeX(self, s):
        self.patchSizeX = s
        self.updateParameters()
        self.network_interface_update.emit()

    def getPatchSizeX(self):
        return self.patchSizeX

    def setPatchSizeY(self, s):
        self.patchSizeY = s
        self.updateParameters()
        self.network_interface_update.emit()

    def getPatchSizeY(self):
        return self.patchSizeY

    def setPatchSizeZ(self, s):
        self.patchSizeZ = s
        self.updateParameters()
        self.network_interface_update.emit()

    def getPatchSizeZ(self):
        return self.patchSizeZ

    def setPatchOverlapp(self, o):
        self.patchOverlap = o
        self.updateParameters()
        self.network_interface_update.emit()

    def getPatchOverlapp(self):
        return self.patchOverlap

    def setPathToDatabase(self, pathToDB):
        self.pathDatabase = pathToDB
        self.updateParameters()
        self.network_interface_update.emit()

    def getPathToDatabase(self):
        return self.pathDatabase

    def setOutputPathForPatching(self, outPath):
        self.pathOutputPatching = outPath
        self.updateParameters()
        self.network_interface_update.emit()

    def getOutputPathForPatching(self):
        return self.pathOutputPatching

    def setSelectedPatients(self, pats):
        self.selectedPatients = pats
        self.updateParameters()
        self.network_interface_update.emit()

    def getSelectedPatients(self):
        return self.selectedPatients

    def setSelectedDatasets(self, sets):
        self.selectedDatasets = sets
        self.updateParameters()
        self.network_interface_update.emit()

    def getSelectedDatasets(self):
        return self.selectedDatasets

    def setDatasetSorted(self, train, val, test):
        self.dataset_train = train
        self.dataset_validation = val
        self.dataset_test = test
        self.updateParameters()
        self.network_interface_update.emit()

    def setPatchingMode(self, mode):
        if mode == PATCHING_2D or mode == PATCHING_3D:
            self.patchingMode = mode
            self.updateParameters()
            self.network_interface_update.emit()

    def getPatchingMode(self):
        return self.patchingMode

    def getLearningOutputPath(self):
        return self.learningOutputPath

    def setLearningOutputPath(self, path):
        self.learningOutputPath = path
        self.updateParameters()
        self.network_interface_update.emit()

    def getCurrentModelPath(self):
        model_name = os.path.split(self.outputFolderDataPath)[-1] + '_model.h5'
        modelPath = self.outPutFolderDataPath + os.sep + model_name
        return modelPath

    def setCurrentModelPath(self, model):
        self.outputFolderDataPath = os.path.split(model)[0]
        self.updateParameters()
        self.network_interface_update.emit()

    def getStoreMode(self):
        return self.storeMode

    def setStoreMode(self, mode):
        if mode == 0:
            self.storeMode = STORE_DISABLED
        elif mode == 1:
            self.storeMode = STORE_HDF5
        elif mode == 2:
            self.storeMode = STORE_PATCH_BASED
        else:
            raise ValueError('Unknown store mode!!!')
        self.updateParameters()
        self.network_interface_update.emit()

    def getResultWorkSpace(self):
        return self.result_WorkSpace

    def getTrainTestDatasetRatio(self):
        '''
        Function returns the splitting ratio of dataset into training set and test set
        :return: splitting ratio
        '''
        return self.trainTestDatasetRatio

    def setTrainTestDatasetRatio(self, ratio):
        if 0 <= ratio <= 1:
            self.trainTestDatasetRatio = ratio
        else:
            raise ValueError('Splitting ratio train set, test set too big or too small!')
        self.updateParameters()
        self.network_interface_update.emit()

    def getTrainValidationRatio(self):
        '''
        Function returns the splitting ratio of training set into sets used for training and validation
        :return:
        '''
        return self.trainValidationRatio

    def setTrainValidationRatio(self, ratio):
        if 0 <= ratio < 1:
            self.trainValidationRatio = ratio
        else:
            raise ValueError('Splitting ratio train, validation on training set is too big or too small!')
        self.updateParameters()
        self.network_interface_update.emit()

    def setSplittingMode(self, mode):
        self.splittingMode = mode
        self.updateParameters()
        self.network_interface_update.emit()

    def getSplittingMode(self):
        return self.splittingMode

    def getNumFolds(self):
        return self.numFolds

    def setNumFolds(self, folds):
        self.numFolds = folds
        self.updateParameters()
        self.network_interface_update.emit()

    def setNeuralNetworkModel(self, model):
        self.neuralNetworkModel = model
        self.network_interface_update.emit()

    def getNeuralNetworkModel(self):
        return self.neuralNetworkModel

    def setDeepNeuralNetworks(self, dnn):
        self.deepNeuralNetworks = dnn
        self.updateParameters()
        self.network_interface_update.emit()

    def getDeepNeuralNetworks(self):
        return self.deepNeuralNetworks

    def setBatchSizes(self, size):
        self.batchSizes = size
        self.updateParameters()
        self.network_interface_update.emit()

    def getBatchSizes(self):
        return self.batchSizes

    def setLearningRates(self, rates):
        self.learningRates = rates
        self.updateParameters()
        self.network_interface_update.emit()

    def getLearningRates(self):
        return self.learningRates

    def setEpochs(self, epochs):
        self.epochs = epochs
        self.updateParameters()
        self.network_interface_update.emit()

    def getEpochs(self):
        return self.epochs

    def getUsingArtifacts(self):
        return self.usingArtifacts

    def setUsingArtifacts(self, b):
        self.usingArtifacts = b
        self.updateParameters()
        self.network_interface_update.emit()

    def getUsingBodyRegions(self):
        return self.usingBodyRegions

    def setUsingBodyRegions(self, b):
        self.usingBodyRegions = b
        self.updateParameters()
        self.network_interface_update.emit()

    def getUsingTWeighting(self):
        return self.usingBodyRegions

    def setUsingTWeighting(self, b):
        self.usingTWeighting = b
        self.updateParameters()
        self.network_interface_update.emit()

    def setOptimizer(self, opt):
        self.optimizer = opt
        self.updateParameters()
        self.network_interface_update.emit()

    def getOptimizer(self):
        return self.optimizer

    def setWeightDecay(self, w):
        self.weightDecay = w
        self.updateParameters()
        self.network_interface_update.emit()

    def getWeightDecay(self):
        return self.weightDecay

    def setMomentum(self, m):
        self.momentum = m
        self.updateParameters()
        self.network_interface_update.emit()

    def getMomentum(self):
        return self.momentum

    def setNesterovEnabled(self, n):
        self.nesterovEnabled = n
        self.updateParameters()
        self.network_interface_update.emit()

    def getNesterovEnabled(self):
        return self.nesterovEnabled

    def setDataAugmentationEnabled(self, b):
        self.dataAugmentationEnabled = b
        self.updateParameters()
        self.network_interface_update.emit()

    def getDataAugmentationEnabled(self):
        return self.dataAugmentationEnabled

    def setHorizontalFlip(self, b):
        self.horizontalFlip = b

    def getHorizontalFlip(self):
        return self.horizontalFlip

    def setVerticalFlip(self, b):
        self.verticalFlip = b

    def getVerticalFlip(self):
        return self.verticalFlip

    def setRotation(self, b):
        if b:
            self.rotation = ROTATION_RANGE
        else:
            self.rotation = 0

    def getRotation(self):
        return self.rotation

    def setZCA_Whitening(self, b):
        self.zcaWhitening = b

    def getZCA_Whitening(self):
        return self.zcaWhitening

    def setHeightShift(self, b):
        if b:
            self.heightShift = HEIGHT_SHIFT_RANGE
        else:
            self.heightShift = 0

    def getHeightShift(self):
        return self.heightShift

    def setWidthShift(self, b):
        if b:
            self.widthShift = WIDTH_SHIFT_RANGE
        else:
            self.widthShift = 0

    def getWidthShift(self):
        return self.widthShift

    def setZoom(self, r):
        if r:
            self.zoom = ZOOM_RANGE
        else:
            self.zoom = 0

    def getZoom(self):
        return self.zoom

    def setContrastStretching(self, c):
        self.contrastStretching = c

    def getContrastStretching(self):
        return self.contrastStretching

    def setAdaptiveEqualization(self, e):
        self.adaptive_eq = e

    def getAdaptiveEqualization(self):
        return self.adaptive_eq

    def setHistogramEqualization(self, e):
        self.histogram_eq = e

    def getHistogramEqualization(self):
        return self.histogram_eq

    def setGUIHandle(self, handle):
        self.dlart_GUI_handle = handle

    def getGUIHandle(self):
        return self.dlart_GUI_handle

    def setIsRandomShuffle(self, b):
        self.isRandomShuffle = b
        self.updateParameters()
        self.network_interface_update.emit()

    def getIsRandomShuffle(self):
        return self.isRandomShuffle

    def getUsingSegmentationMasks(self):
        return self.usingSegmentationMasks

    def setUsingSegmentationMasks(self, b):
        self.usingSegmentationMasks = b
        self.updateParameters()
        self.network_interface_update.emit()

    def getUnpatchedSlices(self):
        return self.unpatched_slices

    def livePlotTrainingPerformance(self, train_acc, val_acc, train_loss, val_loss):
        curEpoch = len(train_acc)
        progress = np.around(curEpoch / self.epochs * 100, decimals=0)
        progress = int(progress)

        self.updateProgressBarTraining(progress)
        self.dlart_GUI_handle.plotTrainingLivePerformance(train_acc=train_acc, val_acc=val_acc, train_loss=train_loss,
                                                          val_loss=val_loss)

    def datasetAvailable(self):
        retbool = False
        if self.storeMode != STORE_PATCH_BASED:
            if self.X_train is not None and self.X_validation is not None \
                    and self.X_test is not None and self.Y_train is not None \
                    and self.Y_validation is not None and self.Y_test.all is not None:
                retbool = True
        return retbool

    def updateProgressBarTraining(self, val):
        self.dlart_GUI_handle.updateProgressBarTraining(val)

    def loadDataset(self, pathToDataset):
        '''
        Method loads an existing dataset out of hd5f files or handles the patch based datasets
        :param pathToDataset: path to dataset
        :return: boolean if loading was successful, and name of loaded dataset
        '''
        retbool = False
        # check for data info summary in json file
        try:
            with open(pathToDataset + os.sep + "dataset_info.json", 'r') as fp:
                dataset_info = json.load(fp)

            # hd5f or patch based?
            if dataset_info['StoreMode'] == STORE_HDF5:
                # loading hdf5
                self.datasetName = dataset_info['Name']
                self.patchSizeX = int(dataset_info['PatchSizeX'])
                self.patchSizeY = int(dataset_info['PatchSizeY'])
                self.patchSizeZ = int(dataset_info['PatchSizeZ'])
                self.patchOverlap = float(dataset_info['PatchOverlap'])
                self.patchingMode = int(dataset_info['PatchMode'])
                self.labelingMode = int(dataset_info['LabelingMode'])
                self.splittingMode = int(dataset_info['SplittingMode'])
                self.trainTestDatasetRatio = float(dataset_info['TrainTestRatio'])
                self.trainValidationRatio = float(dataset_info['TrainValidationRatio'])
                self.numFolds = int(dataset_info['NumFolds'])

                if 'ClassMappings' in dataset_info:
                    self.classMappings = dataset_info['ClassMappings']

                if 'SegmentationMaskUsed' in dataset_info:
                    self.usingSegmentationMasks = bool(dataset_info['SegmentationMaskUsed'])
                else:
                    self.usingSegmentationMasks = False

                # loading hdf5 dataset
                try:
                    with h5py.File(pathToDataset + os.sep + "datasets.hdf5", 'r') as hf:
                        self.X_train = hf['X_train'][:]
                        self.X_validation = hf['X_validation'][:]
                        self.X_test = hf['X_test'][:]
                        self.Y_train = hf['Y_train'][:]
                        self.Y_validation = hf['Y_validation'][:]
                        self.Y_test = hf['Y_test'][:]
                        if self.usingSegmentationMasks:
                            self.Y_segMasks_train = hf['Y_segMasks_train'][:]
                            self.Y_segMasks_validation = hf['Y_segMasks_validation'][:]
                            self.Y_segMasks_test = hf['Y_segMasks_test'][:]

                    retbool = True
                except:
                    raise TypeError("Can't read HDF5 dataset!")

            elif dataset_info['StoreMode'] == STORE_PATCH_BASED:
                # loading patchbased stuff
                self.datasetName = dataset_info['Name']

                print("still in progress")
            else:
                raise NameError("No such store Mode known!")

        except:
            raise FileNotFoundError("Error: Something went wrong at trying to load the dataset!!!")

        return retbool, self.datasetName

    def getDatasetForPrediction(self):
        return self.datasetForPrediction

    def setDatasetForPrediction(self, d):
        self.datasetForPrediction = d
        self.network_interface_update.emit()

    def getModelForPrediction(self):
        return self.modelPathPrediction

    def setModelForPrediction(self, m):
        self.modelPathPrediction = m
        self.network_interface_update.emit()

    def getClassificationReport(self):
        return self.classificationReport

    def getConfusionMatrix(self):
        return self.confusionMatrix

    def get_acc_training(self):
        return self.acc_training

    def get_acc_validation(self):
        return self.acc_validation

    def get_acc_test(self):
        return self.acc_test

    def getGPUId(self):
        return self.gpu_id

    def setGPUId(self, id):
        self.gpu_id = id
        self.updateParameters()
        self.network_interface_update.emit()

    def setGPUPredictionId(self, id):
        self.gpu_prediction_id = id

    def getGPUPredictionId(self):
        return self.gpu_id

    def getUsingSegmentationMasksForPredictions(self):
        return self.usingSegmentationMasksForPrediction

    def getClassMappingsForPrediction(self):
        return self.classMappingsForPrediction

    def setDoUnpatching(self, b):
        self.doUnpatching = b

    def getDoUnpatching(self):
        return self.doUnpatching

    def performPrediction(self):
        '''
            Method performs the prediction for cnn and fcn
            :param
            :return: returns boolean. If prediction was successfull then true is returned else false.
        '''
        # set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_prediction_id)

        # load dataset for prediction
        # check for data info summary in json file
        if self.datasetForPrediction is not None:
            with open(self.datasetForPrediction + os.sep + "dataset_info.json", 'r') as fp:
                dataset_info = json.load(fp)

            self.patchOverlapPrediction = dataset_info['PatchOverlap']
            patientsOfDataset = dataset_info['Patients']

            if self.doUnpatching:
                datasetForUnpatching = dataset_info['Datasets']
                datasetForUnpatching = datasetForUnpatching[0]

            # hd5f or patch based?
            self.storeModePrediction = dataset_info['StoreMode']
            if dataset_info['StoreMode'] == STORE_HDF5:
                # loading hdf5
                try:
                    self.usingSegmentationMasksForPrediction = bool(dataset_info['SegmentationMaskUsed'])
                except:
                    self.usingSegmentationMasksForPrediction = False

                # loading hdf5 dataset
                try:
                    with h5py.File(self.datasetForPrediction + os.sep + "datasets.hdf5", 'r') as hf:
                        self.X_train = hf['X_train'][:]
                        self.X_validation = hf['X_validation'][:]
                        self.X_test = hf['X_test'][:]
                        self.Y_train = hf['Y_train'][:]
                        self.Y_validation = hf['Y_validation'][:]
                        self.Y_test = hf['Y_test'][:]
                        if self.usingSegmentationMasksForPrediction:
                            self.Y_segMasks_train = hf['Y_segMasks_train'][:]
                            self.Y_segMasks_validation = hf['Y_segMasks_validation'][:]
                            self.Y_segMasks_test = hf['Y_segMasks_test'][:]
                except:
                    raise TypeError("Can't read HDF5 dataset!")
            else:
                raise NameError("No such store Mode known!")
        else:
            print('Using current data as test data for prediction')

        # dynamic loading of corresponding model
        with open(self.modelPathPrediction + os.sep + "cnn_training_info.json", 'r') as fp:
            cnn_info = json.load(fp)

        self.modelPrediction = cnn_info['Name']
        self.modelPredictionSource = self.deepNeuralNetworks[self.modelPrediction].replace(".", "/") + '.py'
        self.batchSizePrediction = int(cnn_info['BatchSize'])
        self.usingArtifactsPrediction = cnn_info['Artifacts']
        self.usingBodyRegionsPrediction = cnn_info['BodyRegions']
        self.usingTWeightingPrediction = cnn_info['TWeightings']
        self.usingClassificationPrediction = cnn_info['UsingClassification']

        self.network_interface_update.emit()

        # preprocess y
        if 'ClassMappings' in cnn_info:
            self.classMappingsForPrediction = cnn_info['ClassMappings']

            # convert string keys to int keys
            intKeysDict = {}
            for stringKey in self.classMappingsForPrediction:
                intKeysDict[int(stringKey)] = self.classMappingsForPrediction[stringKey]
            self.classMappingsForPrediction = intKeysDict

        else:
            # in old code version no class mappings were stored in cnn_info. so we have to recreate the class mappings
            # out of the original training dataset
            with h5py.File(self.pathOutputPatching + os.sep + cnn_info['Dataset'] + os.sep + "datasets.hdf5",
                           'r') as hf:
                Y_train_original = hf['Y_test'][:]

            classes = np.asarray(np.unique(Y_train_original, ), dtype=int)
            self.classMappingsForPrediction = Label.mapClassesToOutputVector(classes=classes,
                                                                             usingArtefacts=self.usingArtifactsPredictiongArtifacts,
                                                                             usingBodyRegion=self.usingBodyRegionsPrediction,
                                                                             usingTWeightings=self.usingTWeightingPrediction)

        if self.doUnpatching:
            classLabel = int(self.datasets[datasetForUnpatching].getDatasetLabel())

        if self.usingSegmentationMasksForPrediction:
            Y_train = []

            Y_validation = []

            Y_test = []

            self.Y_segMasks_test[self.Y_segMasks_test == 3] = 1
            self.Y_segMasks_test[self.Y_segMasks_test == 2] = 0

            y_labels_test = np.expand_dims(self.Y_segMasks_test, axis=-1)
            y_labels_test[y_labels_test == 0] = -1
            y_labels_test[y_labels_test == 1] = 1
            y_labels_test = np.sum(y_labels_test, axis=1)
            y_labels_test = np.sum(y_labels_test, axis=1)
            y_labels_test = np.sum(y_labels_test, axis=1)
            y_labels_test[y_labels_test >= 0] = 1
            y_labels_test[y_labels_test < 0] = 0

            for i in range(y_labels_test.shape[0]):
                Y_test.append([1, 0] if y_labels_test[i].all() == 0 else [0, 1])
            Y_test = np.asarray(Y_test)

            Y_test = []
            for i in range(y_labels_test.shape[0]):
                Y_test.append([1, 0] if y_labels_test[i].all() == 0 else [0, 1])
            Y_test = np.asarray(Y_test)

            self.X_test_shape = self.X_test.shape
            self.Y_test_shape = Y_test.shape
            self.Y_segMasks_test_shape = self.Y_segMasks_test.shape
            self.neuralnetworkPath = self.deepNeuralNetworks[self.neuralNetworkModel].replace(".", "/") + '.py'
            self.updateParameters()
            self._network_interface_update.emit()

            predictions = predict_segmentation_model(self.X_test,
                                                     Y_test,
                                                     self.Y_segMasks_test,
                                                     sModelPath=self.modelPathPrediction,
                                                     batch_size=self.batchSizePrediction,
                                                     usingClassification=self.usingClassificationPrediction,
                                                     dlart_handle=self)

            # do unpatching if is enabled
            if self.doUnpatching:

                self.patchSizePrediction = [self.X_test.shape[1], self.X_test.shape[2], self.X_test.shape[3]]

                # load corresponding original dataset
                for i in self.datasets:
                    set = self.datasets[i]
                    if set.getDatasetLabel() == classLabel:
                        originalDatasetName = set.getPathdata()

                pathToOriginalDataset = self.getPathToDatabase() + os.sep + str(
                    patientsOfDataset[0]) + os.sep + 'dicom_sorted' + os.sep + originalDatasetName
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

                allPreds = predictions['prob_pre']

                unpatched_img_foreground = fUnpatchSegmentation(allPreds[0],
                                                                patchSize=self.patchSizePredictionSize,
                                                                patchOverlap=self.patchOverlapPrediction,
                                                                actualSize=dicom_size,
                                                                iClass=1)
                unpatched_img_background = fUnpatchSegmentation(allPreds[0],
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

                self.unpatched_slices = {
                    'probability_mask_foreground': unpatched_img_foreground,
                    'probability_mask_background': unpatched_img_background,
                    'predicted_segmentation_mask': unpatched_img_mask,
                    'dicom_slices': voxel_ndarray,
                    'dicom_masks': labelMask_ndarray,

                }

            if self.usingClassificationPrediction:
                # save prediction into .mat file
                modelSave = self.modelPathPrediction + os.sep + 'model_predictions.mat'
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
                                            'unpatched_slices': self.unpatched_slices
                                            })
                self.result_WorkSpace = modelSave

                # load training results
                _, sPath = os.path.splitdrive(self.modelPathPrediction)
                sPath, sFilename = os.path.split(sPath)
                sFilename, sExt = os.path.splitext(sFilename)

                training_results = sio.loadmat(self.modelPathPrediction + os.sep + sFilename + ".mat")
                self.acc_training = training_results['segmentation_output_dice_coef_training']
                self.acc_validation = training_results['segmentation_output_dice_coef_val']
                self.acc_test = training_results['segmentation_output_dice_coef_test']

            else:
                # save prediction into .mat file
                modelSave = self.modelPathPrediction + os.sep + 'model_predictions.mat'
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
                                            'unpatched_slices': self.unpatched_slices
                                            })
                self.result_WorkSpace = modelSave

                # load training results
                _, sPath = os.path.splitdrive(self.modelPathPrediction)
                sPath, sFilename = os.path.split(sPath)
                sFilename, sExt = os.path.splitext(sFilename)

                training_results = sio.loadmat(self.modelPathPrediction + os.sep + sFilename + ".mat")
                self.acc_training = training_results['dice_coef']
                self.acc_validation = training_results['val_dice_coef']
                self.acc_test = training_results['dice_coef_test']

        else:

            Y_test = []
            for i in range(self.Y_test.shape[0]):
                Y_test.append(self.classMappingsForPrediction[self.Y_test[i]])
            Y_test = np.asarray(Y_test)

            self.X_test_shape = self.X_test.shape
            self.Y_test_shape = Y_test.shape
            self.neuralnetworkPath = self.deepNeuralNetworks[self.neuralNetworkModel].replace(".", "/") + '.py'
            self.updateParameters()
            self._network_interface_update.emit()

            prediction = predict_model(self.X_test,
                                       Y_test,
                                       sModelPath=self.modelPathPrediction,
                                       batch_size=self.batchSizePrediction,
                                       dlart_handle=self)

            self.predictions = prediction['predictions']
            self.confusionMatrix = prediction['confusion_matrix']
            self.classificationReport = prediction['classification_report']

            ############################################################

            # organize confusion matrix
            sum_all = np.array(np.sum(self.confusionMatrix, axis=0))
            all = np.zeros((len(sum_all), len(sum_all)))
            for i in range(all.shape[0]):
                all[i, :] = sum_all
            self.confusionMatrix = np.divide(self.confusionMatrix, all)

            # do unpatching if is enabled
            if self.doUnpatching:

                self.patchSizePrediction = [self.X_test.shape[1], self.X_test.shape[2], self.X_test.shape[3]]
                classVec = self.classMappingsForPrediction[classLabel]
                classVec = np.asarray(classVec, dtype=np.int32)
                iClass = np.where(classVec == 1)
                iClass = iClass[0]

                # load corresponding original dataset
                for i in self.datasets:
                    set = self.datasets[i]
                    if set.getDatasetLabel() == classLabel:
                        originalDatasetName = set.getPathdata()

                pathToOriginalDataset = self.getPathToDatabase() + os.sep + str(
                    patientsOfDataset[0]) + os.sep + 'dicom_sorted' + os.sep + originalDatasetName
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
                    multiclass_probability_masks = fMulticlassUnpatch2D(self.predictions,
                                                                        self.patchSizePrediction,
                                                                        self.patchOverlapPrediction,
                                                                        dicom_size)

                if len(self.patchSizePrediction) == 3:
                    multiclass_probability_masks = fUnpatch3D(self.predictions,
                                                              self.patchSizePrediction,
                                                              self.patchOverlapPrediction,
                                                              dicom_size)

                ########################################################################################################
                # Hatching and colors multicalss unpatching
                prob_test = self.predictions

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

                self.unpatched_slices = {
                    'multiclass_probability_masks': multiclass_probability_masks,
                    'dicom_slices': voxel_ndarray,
                    'dicom_masks': labelMask_ndarray,
                    'index_class': iClass,
                    'IType': IType,
                    'IArte': IArte
                }

            # save prediction into .mat file
            modelSave = self.modelPathPrediction + os.sep + 'model_predictions.mat'
            print('saving Model:{}'.format(modelSave))
            if not self.doUnpatching:
                sio.savemat(modelSave, {'prob_pre': prediction['predictions'],
                                        'Y_test': self.Y_test,
                                        'score_test': prediction['score_test'],
                                        'acc_test': prediction['acc_test'],
                                        'classification_report': prediction['classification_report'],
                                        'confusion_matrix': prediction['confusion_matrix']
                                        })
            else:
                sio.savemat(modelSave, {'prob_pre': prediction['predictions'],
                                        'Y_test': self.Y_test,
                                        'score_test': prediction['score_test'],
                                        'acc_test': prediction['acc_test'],
                                        'classification_report': prediction['classification_report'],
                                        'confusion_matrix': prediction['confusion_matrix'],
                                        'unpatched_slices': self.unpatched_slices
                                        })
            self.result_WorkSpace = modelSave

            # load training results
            _, sPath = os.path.splitdrive(self.modelPathPrediction)
            sPath, sFilename = os.path.split(sPath)
            sFilename, sExt = os.path.splitext(sFilename)

            print(self.modelPathPrediction + os.sep + sFilename + ".mat")

            training_results = sio.loadmat(self.modelPathPrediction + os.sep + sFilename + ".mat")
            self.acc_training = training_results['acc']
            self.acc_validation = training_results['val_acc']
            self.acc_test = training_results['acc_test']

        return True

    @staticmethod
    def getOSPathes():

        pathDatabase = PATH_OUT + os.sep + "MRPhysics" + os.sep + "newProtocol"

        pathOutputPatching = LEARNING_OUT + os.sep + "MRPhysics" + os.sep + "DeepLearningArt_Output" + \
                             os.sep + "Datasets"

        markingsPath = LABEL_PATH

        learningOutputPath = LEARNING_OUT + os.sep + "MRPhysics" + os.sep + "output" + \
                             os.sep + "Output_Learning"

        return pathDatabase, pathOutputPatching, markingsPath, learningOutputPath
