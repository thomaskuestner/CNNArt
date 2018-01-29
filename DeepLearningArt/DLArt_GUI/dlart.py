'''
@author: Yannick Wilhelm
@email: yannick.wilhelm@gmx.de
@date: January 2018
'''

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
import json
import datetime
import h5py

import cnn_main

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

    deepNeuralNetworks = {
        'Multiclass DenseResNet': 'networks.multiclass.DenseResNet.multiclass_DenseResNet',
        'Multiclass InceptionNet': 'networks.multiclass.InceptionNet.multiclass_InceptionNet'
    }

    modelSubDir = "dicom_sorted"

    # constants labeling modes
    MASK_LABELING = 0
    PATCH_LABELING = 1

    # constants patching modes
    PATCHING_2D = 0
    PATCHING_3D = 1

    # constants splitting modes
    NONE_SPLITTING = 0
    SIMPLE_RANDOM_SAMPLE_SPLITTING = 1
    CROSS_VALIDATION_SPLITTING = 2
    PATIENT_CROSS_VALIDATION_SPLITTING = 3

    # constants storage mode
    STORE_DISABLED = 0
    STORE_HDF5 = 1
    STORE_PATCH_BASED = 2

    def __init__(self):
        # attributes for paths and database
        self.selectedPatients = ''
        self.selectedDatasets = ''

        self.pathDatabase, self.pathOutputPatching, self.markingsPath, self.learningOutputPath \
            = DeepLearningArtApp.getOSPathes(operatingSystem=0)  # for windows os=0, for linse server os=1. see method for pathes

        # attributes for patching
        self.patchSizeX = 40
        self.patchSizeY = 40
        self.patchSizeZ = 5
        self.patchOverlapp = 0.6

        #attributes for labeling
        self.labelingMode = ''

        #attributes for patching
        self.patchingMode = DeepLearningArtApp.PATCHING_2D
        self.storeMode = ''

        # attributes for splitting
        self.splittingMode = DeepLearningArtApp.SIMPLE_RANDOM_SAMPLE_SPLITTING
        self.trainTestDatasetRatio = 0.2 #part of test data
        self.trainValidationRatio = 0.2 # part of Validation data in traindata
        self.numFolds = 5

        #attributes for DNN
        self.neuralNetworkModel = None
        self.batchSize = None
        self.epochs = None
        self.learningRates = None


        # train, validation, test dataset attributes
        self.X_train = None
        self.Y_train = None

        self.X_validation = None
        self.Y_validation = None

        self.X_test = None
        self.Y_test = None


    def generateDataset(self):
        '''
        method performs the splitting of the datasets to the learning datasets (training, validation, test)
        and handles the storage of datasets
        :return:
        '''
        self.X_test = []
        self.X_validation= []
        self.X_train = []
        self.Y_test = []
        self.Y_validation = []
        self.Y_train = []

        dAllPatches = np.zeros((self.patchSizeX, self.patchSizeY, 0))
        dAllLabels = np.zeros(0)

        # stuff for storing
        if self.storeMode != DeepLearningArtApp.STORE_DISABLED:
            # outPutFolder name:
            outPutFolder = "P" + str(len(self.selectedPatients)) + "_" + "D" + str(len(self.selectedDatasets)) + "_" + \
                           "PM" + str(self.patchingMode) + "_X" + str(self.patchSizeX) + "_Y" + str(self.patchSizeY) + "_O" + \
                           str(self.patchOverlapp) + "_L" + str(self.labelingMode) + "_S" + str(self.splittingMode) + \
                            "_STM" + str(self.storeMode)

            outputFolderPath = self.pathOutputPatching + os.sep + outPutFolder

            if not os.path.exists(outputFolderPath):
                os.makedirs(outputFolderPath)

            # create dataset summary
            self.createDatasetInfoSummary(outPutFolder, outputFolderPath)

            if self.storeMode == DeepLearningArtApp.STORE_PATCH_BASED:
                outPutFolderDataPath = outputFolderPath + os.sep + "data"
                if not os.path.exists(outPutFolderDataPath):
                    os.makedirs(outPutFolderDataPath)

                labelDict = {}

        #for storing patch based
        iPatchToDisk = 0

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
                                    [self.patchSizeX, self.patchSizeY], self.patchOverlapp, labelMask_ndarray, 0.5,
                                     DeepLearningArtApp.datasets[dataset])

                            #convert to float32
                            dPatches = np.asarray(dPatches, dtype=np.float32)
                            dLabels = np.asarray(dLabels, dtype=np.float32)

                        elif self.labelingMode == DeepLearningArtApp.PATCH_LABELING:
                            # get label
                            datasetLabel = DeepLearningArtApp.datasets[dataset].getDatasetLabel()

                            #compute 2D patch labeling patching
                            dPatches, dLabels = fRigidPatching_patchLabeling(norm_voxel_ndarray,
                                                                             [self.patchSizeX, self.patchSizeY],
                                                                             self.patchOverlapp, 1)
                            dLabels = dLabels*datasetLabel

                            # convert to float32
                            dPatches = np.asarray(dPatches, dtype=np.float32)
                            dLabels = np.asarray(dLabels, dtype=np.float32)
                    elif self.patchingMode == DeepLearningArtApp.PATCHING_3D:
                        # 3D Patching
                        print("Do 3D patching......")
                    else:
                            print("We do not know what labeling mode you want to use :p")

                if self.storeMode == DeepLearningArtApp.STORE_PATCH_BASED:
                    # patch based storage
                    for i in range(0, dPatches.shape[2]):
                        patchSlice = np.asarray(dPatches[:,:,i], dtype=np.float32)
                        np.save((outPutFolderDataPath + os.sep + "X"+str(iPatchToDisk)+".npy"), patchSlice, allow_pickle=False)
                        labelDict["Y"+str(iPatchToDisk)] = int(dLabels[i])
                        iPatchToDisk+=1

                else:
                    # concatenate all patches in one array
                    dAllPatches = np.concatenate((dAllPatches, dPatches), axis=2)
                    dAllLabels = np.concatenate((dAllLabels, dLabels), axis=0)


        # dataset splitting


        # store mode
        if self.storeMode != DeepLearningArtApp.STORE_DISABLED:
            # H5py store mode
            if self.storeMode == DeepLearningArtApp.STORE_HDF5:
                # train, validation, test datasets are computed by splitting all data
                [self.X_train], [self.Y_train], [self.X_validation], [self.Y_validation], [self.X_test], [self.Y_test] \
                    = fSplitDataset(dAllPatches, dAllLabels, allPats=self.selectedPatients,
                                    sSplitting=self.splittingMode,
                                    patchSize=[self.patchSizeX, self.patchSizeY], patchOverlap=self.patchOverlapp,
                                    testTrainingDatasetRatio=self.trainTestDatasetRatio,
                                    validationTrainRatio=self.trainValidationRatio,
                                    outPutPath=self.pathOutputPatching, nfolds=0)

                # store datasets with h5py
                with h5py.File(outputFolderPath+os.sep+'datasets.hdf5', 'w') as hf:
                    hf.create_dataset('X_train', data=self.X_train)
                    hf.create_dataset('X_validation', data=self.X_validation)
                    hf.create_dataset('X_test', data=self.X_test)
                    hf.create_dataset('Y_train', data=self.Y_train)
                    hf.create_dataset('Y_validation', data=self.Y_validation)
                    hf.create_dataset('Y_test', data=self.Y_test)

            elif self.storeMode == DeepLearningArtApp.STORE_PATCH_BASED:
                with open(outputFolderPath+os.sep+"labels.json", 'w') as fp:
                    json.dump(labelDict, fp)
        else:
            # no storage of patched datasets
            [self.X_train], [self.Y_train], [self.X_validation], [self.Y_validation], [self.X_test], [self.Y_test] = fSplitDataset(dAllPatches, dAllLabels, allPats=self.selectedPatients, sSplitting=self.splittingMode,
                                patchSize = [self.patchSizeX, self.patchSizeY], patchOverlap=self.patchOverlapp,
                                testTrainingDatasetRatio=self.trainTestDatasetRatio, validationTrainRatio=self.trainValidationRatio,
                                outPutPath=self.pathOutputPatching, nfolds=self.numFolds)

            print()

    def performTraining(self):
        # get output vector for different classes
        classes = np.asarray(np.unique(self.Y_train, ), dtype=int)
        classMappings = Label.mapClassesToOutputVector(classes=classes, usingArtefacts=True, usingBodyRegion=True, usingTWeightings=True)

        Y_train = []
        for i in range(self.Y_train.shape[0]):
            Y_train.append(classMappings[self.Y_train[i]])
        Y_train = np.asarray(Y_train)

        Y_validation = []
        for i in range(self.Y_validation.shape[0]):
            Y_validation.append(classMappings[self.Y_validation[i]])
        Y_validation = np.asarray(Y_validation)

        Y_test = []
        for i in range(self.Y_test.shape[0]):
            Y_test.append(classMappings[self.Y_test[i]])
        Y_test = np.asarray(Y_test)

        cnn_main.fRunCNN(
            dData={'X_train': self.X_train, 'y_train': Y_train, 'X_test': self.X_test, 'y_test': Y_test, 'patchSize': [self.patchSizeX, self.patchSizeY, self.patchSizeZ]},
            sModelIn=DeepLearningArtApp.deepNeuralNetworks[self.neuralNetworkModel],
            lTrain=True,
            sParaOptim='',
            sOutPath=self.learningOutputPath,
            iBatchSize=self.batchSize,
            iLearningRate=self.learningRates,
            iEpochs=self.epochs)

        print()


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
        dataDict['PatchOverlap'] = self.patchOverlapp
        dataDict['LabelingMode'] = self.labelingMode
        dataDict['SplittingMode'] = self.splittingMode
        dataDict['StoreMode'] = self.storeMode

        with open((outputFolderPath+os.sep+'dataset_info.json'), 'w') as fp:
            json.dump(dataDict, fp, indent=4)


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

    def getLearningOutputPath(self):
        return self.learningOutputPath

    def setLearningOutputPath(self, path):
        self.learningOutputPath = path

    def getStoreMode(self):
        return self.storeMode

    def setStoreMode(self, mode):
        if mode == 0:
            self.storeMode = DeepLearningArtApp.STORE_DISABLED
        elif mode == 1:
            self.storeMode = DeepLearningArtApp.STORE_HDF5
        elif mode == 2:
            self.storeMode = DeepLearningArtApp.STORE_PATCH_BASED
        else:
            raise ValueError('Unknown store mode!!!')

    def getTrainTestDatasetRatio(self):
        '''
        Function returns the splitting ratio of dataset into training set and test set
        :return: splitting ratio
        '''
        return self.trainTestDatasetRatio

    def setTrainTestDatasetRatio(self, ratio):
        if 0 < ratio < 1:
            self.trainTestDatasetRatio = ratio
        else:
            raise ValueError('Splitting ratio train set, test set too big or too small!')

    def getTrainValidationRatio(self):
        '''
        Function returns the splitting ratio of training set into sets used for training and validation
        :return:
        '''
        return self.trainValidationRatio

    def setTrainValidationRatio(self, ratio):
        if 0 < ratio < 1:
            self.trainValidationRatio = ratio
        else:
            raise ValueError('Splitting ratio train, validation on training set is too big or too small!')

    def setSplittingMode(self, mode):
        self.splittingMode = mode

    def getSplittingMode(self):
        return self.splittingMode

    def getNumFolds(self):
        return self.numFolds

    def setNumFolds(self, folds):
        self.numFolds = folds

    def setNeuralNetworkModel(self, model):
        self.neuralNetworkModel = model

    def getNeuralNetworkModel(self):
        return self.neuralNetworkModel

    def setBatchSize(self, size):
        self.batchSize = size

    def getBatchSize(self):
        return self.batchSize

    def setLearningRates(self, rates):
        self.learningRates = rates

    def getLearningRates(self):
        return self.learningRates

    def setEpochs(self, epochs):
        self.epochs = epochs

    def getEpochs(self):
        return self.epochs

    def datasetAvailable(self):
        retbool = False
        if  self.X_train.all and self.X_validation.all and self.X_test.all and self.Y_train.all and self.Y_validation.all and self.Y_test.all:
            retbool = True
        return retbool

    def loadDataset(self, pathToDataset):
        '''
        Method loads an existing dataset out of hd5f files or handles the patch based datasets
        :param pathToDataset: path to dataset
        :return: boolean if loading was successful, and name of loaded dataset
        '''
        retbool = False
        datasetName = ''
        #check for data info summary in json file
        try:
            with open(pathToDataset + os.sep + "dataset_info.json", 'r') as fp:
                dataset_info = json.load(fp)

            # hd5f or patch based?
            if dataset_info['StoreMode'] == DeepLearningArtApp.STORE_HDF5:
                # loading hdf5
                datasetName = dataset_info['Name']

                # loading hdf5 dataset
                try:
                    with h5py.File(pathToDataset + os.sep + "datasets.hdf5", 'r') as hf:
                        self.X_train = hf['X_train'][:]
                        self.X_validation = hf['X_validation'][:]
                        self.X_test = hf['X_test'][:]
                        self.Y_train = hf['Y_train'][:]
                        self.Y_validation = hf['Y_validation'][:]
                        self.Y_test = hf['Y_test'][:]

                    retbool = True
                except:
                    raise TypeError("Can't read HDF5 dataset!")

            elif dataset_info['StoreMode'] == DeepLearningArtApp.STORE_PATCH_BASED:
                #loading patchbased stuff
                datasetName = dataset_info['Name']

                print("still in progrss")
            else:
                raise NameError("No such store Mode known!")

        except:
            raise FileNotFoundError("Error: Something went wrong at trying to load the dataset!!!")

        return retbool, datasetName

    @staticmethod
    def getOSPathes(operatingSystem=0):
        if operatingSystem==0:
            # my windows PC
            pathDatabase = "D:" + os.sep + "med_data" + os.sep + "MRPhysics" + os.sep + "newProtocol"
            pathOutputPatching = "D:" + os.sep + "med_data" + os.sep + "MRPhysics" + os.sep + "DeepLearningArt_Output"
            markingsPath = "D:" + os.sep + "med_data" + os.sep + "MRPhysics" + os.sep + "Markings"
            learningOutputPath = "D:" + os.sep + "med_data" + os.sep + "MRPhysics" + os.sep + "DeepLearningArt_Output" + \
                                      os.sep + "Output_Learning"
        elif operatingSystem==1:
            pathDatabase = "/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol"
            pathOutputPatching = "/no_backup/d1237/DeepLearningArt_Output/"
            markingsPath = "/no_backup/d1237/Markings/"
            learningOutputPath = "/no_backup/d1237/DeepLearningArt_Output/"

        return pathDatabase, pathOutputPatching, markingsPath, learningOutputPath