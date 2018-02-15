'''
@author: Yannick Wilhelm
@email: yannick.wilhelm@gmx.de
@date: January 2018
'''

import sys
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import *

from DeepLearningArt.DLArt_GUI.dlart_gui import Ui_DLArt_GUI
from DeepLearningArt.DLArt_GUI.dlart import DeepLearningArtApp

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from DeepLearningArt.DLArt_GUI.PlotCanvas import PlotCanvas
from matplotlib.figure import Figure

import random

#from dlart_gui import Ui_DLArt_GUI
#from dlart import DeepLearningArtApp

#from DeepLearningArt.DLArt_GUI.dlart_gui import Ui_DLArt_GUI
#from DeepLearningArt.DLArt_GUI.dlart import DeepLearningArtApp

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Set up the user interface from Designer
        #loadUi("dlart_gui.ui", self)
        self.ui = Ui_DLArt_GUI();
        self.ui.setupUi(self)

        # initialize DeepLearningArt Application
        self.deepLearningArtApp = DeepLearningArtApp()
        self.deepLearningArtApp.setGUIHandle(self)

        # initialize TreeView Database
        self.manageTreeView()

        # intialize TreeView Datasets
        self.manageTreeViewDatasets()

        # initiliaze patch output path
        self.ui.Label_OutputPathPatching.setText(self.deepLearningArtApp.getOutputPathForPatching())

        # initialize markings path
        self.ui.Label_MarkingsPath.setText(self.deepLearningArtApp.getMarkingsPath())

        # initialize learning output path
        self.ui.Label_LearningOutputPath.setText(self.deepLearningArtApp.getLearningOutputPath())

        #initialize patching mode
        self.ui.ComboBox_Patching.setCurrentIndex(1)

        #initialize store mode
        self.ui.ComboBox_StoreOptions.setCurrentIndex(0)

        #initialize splitting mode
        self.ui.ComboBox_splittingMode.setCurrentIndex(DeepLearningArtApp.SIMPLE_RANDOM_SAMPLE_SPLITTING)
        self.ui.Label_SplittingParams.setText("using Test/Train="
                                              +str(self.deepLearningArtApp.getTrainTestDatasetRatio())
                                              +" and Valid/Train="+str(self.deepLearningArtApp.getTrainValidationRatio()))

        #initialize combox box for DNN selection
        self.ui.ComboBox_DNNs.addItem("Select Deep Neural Network Model...")
        self.ui.ComboBox_DNNs.addItems(DeepLearningArtApp.deepNeuralNetworks.keys())
        self.ui.ComboBox_DNNs.setCurrentIndex(1)
        self.deepLearningArtApp.setNeuralNetworkModel(self.ui.ComboBox_DNNs.currentText())

        #initialize check boxes for used classes
        self.ui.CheckBox_Artifacts.setChecked(self.deepLearningArtApp.getUsingArtifacts())
        self.ui.CheckBox_BodyRegion.setChecked(self.deepLearningArtApp.getUsingBodyRegions())
        self.ui.CheckBox_TWeighting.setChecked(self.deepLearningArtApp.getUsingTWeighting())

        #initilize training parameters
        self.ui.DoubleSpinBox_WeightDecay.setValue(self.deepLearningArtApp.getWeightDecay())
        self.ui.DoubleSpinBox_Momentum.setValue(self.deepLearningArtApp.getMomentum())
        self.ui.CheckBox_Nesterov.setChecked(self.deepLearningArtApp.getNesterovEnabled())
        self.ui.CheckBox_DataAugmentation.setChecked(self.deepLearningArtApp.getDataAugmentationEnabled())
        self.ui.CheckBox_DataAug_horizontalFlip.setChecked(self.deepLearningArtApp.getHorizontalFlip())
        self.ui.CheckBox_DataAug_verticalFlip.setChecked(self.deepLearningArtApp.getVerticalFlip())
        self.ui.CheckBox_DataAug_Rotation.setChecked(self.deepLearningArtApp.getRotation())
        self.ui.CheckBox_DataAug_zcaWeighting.setChecked(self.deepLearningArtApp.getZCA_Whitening())
        self.ui.CheckBox_DataAug_HeightShift.setChecked(self.deepLearningArtApp.getHeightShift())
        self.ui.CheckBox_DataAug_WidthShift.setChecked(self.deepLearningArtApp.getWidthShift())
        self.check_dataAugmentation_enabled()

        ################################################################################################################
        # ArtGAN Stuff
        self.manageTreeView_DB_ArtGAN()
        self.manageTreeViewDatasetsArtGAN()

        # Signals and Slots
        self.ui.Button_DB_ArtGAN.clicked.connect(self.button_DB_ArtGAN_clicked)
        self.ui.Button_Patching_ArtGAN.clicked.connect(self.button_patching_ArtGAN_clicked)
        self.ui.HorizontalSlider_ArtGAN.sliderMoved.connect(self.slider_ArtGAN_moved)
        self.ui.HorizontalSlider_ArtGAN.valueChanged.connect(self.slider_ArtGAN_changed)

        self.figureImg = Figure(figsize=(5, 5))
        self.canvas_figImg = FigureCanvas(self.figureImg)
        self.toolbar_figCanvas = NavigationToolbar(self.canvas_figImg, self)
        self.ui.verticalLayout_ArtGAN.addWidget(self.canvas_figImg)
        self.ui.verticalLayout_ArtGAN.addWidget(self.toolbar_figCanvas)

        ################################################################################################################

        ################################################################################################################
        # Signals and Slots
        ################################################################################################################

        #select database button clicked
        self.ui.Button_DB.clicked.connect(self.button_DB_clicked)
        # self.Button_DB.clicked.connect(self.button_DB_clicked)

        #output path button for patching clicked
        self.ui.Button_OutputPathPatching.clicked.connect(self.button_outputPatching_clicked)

        #TreeWidgets
        self.ui.TreeWidget_Patients.clicked.connect(self.getSelectedPatients)
        self.ui.TreeWidget_Datasets.clicked.connect(self.getSelectedDatasets)

        #Patching button
        self.ui.Button_Patching.clicked.connect(self.button_patching_clicked)

        #mask marking path button clicekd
        self.ui.Button_MarkingsPath.clicked.connect(self.button_markingsPath_clicked)

        # combo box splitting mode is changed
        self.ui.ComboBox_splittingMode.currentIndexChanged.connect(self.splittingMode_changed)

        # "use current data" button clicked
        self.ui.Button_useCurrentData.clicked.connect(self.button_useCurrentData_clicked)

        # select dataset is clicked
        self.ui.Button_selectDataset.clicked.connect(self.button_selectDataset_clicked)

        # learning output path button clicked
        self.ui.Button_LearningOutputPath.clicked.connect(self.button_learningOutputPath_clicked)

        # train button clicked
        self.ui.Button_train.clicked.connect(self.button_train_clicked)

        # combobox dnns
        self.ui.ComboBox_DNNs.currentIndexChanged.connect(self.selectedDNN_changed)

        # show Dataset for ArtGAN Button
        self.ui.Button_ShowDataset.clicked.connect(self.button_showDataset_clicked)

        # data augmentation enbaled changed
        self.ui.CheckBox_DataAugmentation.stateChanged.connect(self.check_dataAugmentation_enabled)

        ################################################################################################################

    def button_train_clicked(self):
        # set epochs
        self.deepLearningArtApp.setEpochs(self.ui.SpinBox_Epochs.value())

        # handle check states of check boxes for used classes
        self.deepLearningArtApp.setUsingArtifacts(self.ui.CheckBox_Artifacts.isChecked())
        self.deepLearningArtApp.setUsingBodyRegions(self.ui.CheckBox_BodyRegion.isChecked())
        self.deepLearningArtApp.setUsingTWeighting(self.ui.CheckBox_TWeighting.isChecked())

        # set learning rates and batch sizes
        try:
            batchSizes = np.fromstring(self.ui.LineEdit_BatchSizes.text(), dtype=np.int, sep=',')
            self.deepLearningArtApp.setBatchSizes(batchSizes)
            learningRates = np.fromstring(self.ui.LineEdit_LearningRates.text(), dtype=np.float32, sep=',')
            self.deepLearningArtApp.setLearningRates(learningRates)
        except:
            raise ValueError("Wrong input format of learning rates! Enter values seperated by ','. For example: 0.1,0.01,0.001")

        # set optimizer
        selectedOptimizer = self.ui.ComboBox_Optimizers.currentText()
        if selectedOptimizer == "SGD":
            self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.SGD_OPTIMIZER)
        elif selectedOptimizer == "RMSprop":
            self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.RMS_PROP_OPTIMIZER)
        elif selectedOptimizer == "Adagrad":
            self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.ADAGRAD_OPTIMIZER)
        elif selectedOptimizer == "Adadelta":
            self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.ADADELTA_OPTIMIZER)
        elif selectedOptimizer == "Adam":
            self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.ADAM_OPTIMIZER)
        else:
            raise ValueError("Unknown Optimizer!")

        # set weigth decay
        self.deepLearningArtApp.setWeightDecay(float(self.ui.DoubleSpinBox_WeightDecay.value()))
        # set momentum
        self.deepLearningArtApp.setMomentum(float(self.ui.DoubleSpinBox_Momentum.value()))
        # set nesterov enabled
        if self.ui.CheckBox_Nesterov.checkState() == Qt.Checked:
            self.deepLearningArtApp.setNesterovEnabled(True)
        else:
            self.deepLearningArtApp.setNesterovEnabled(False)

        # handle data augmentation
        if self.ui.CheckBox_DataAugmentation.checkState() == Qt.Checked:
            self.deepLearningArtApp.setDataAugmentationEnabled(True)
            # get all checked data augmentation options
            if self.ui.CheckBox_DataAug_horizontalFlip.checkState() == Qt.Checked:
                self.deepLearningArtApp.setHorizontalFlip(True)
            else:
                self.deepLearningArtApp.setHorizontalFlip(False)

            if self.ui.CheckBox_DataAug_verticalFlip.checkState() == Qt.Checked:
                self.deepLearningArtApp.setVerticalFlip(True)
            else:
                self.deepLearningArtApp.setVerticalFlip(False)

            if self.ui.CheckBox_DataAug_Rotation.checkState() == Qt.Checked:
                self.deepLearningArtApp.setRotation(True)
            else:
                self.deepLearningArtApp.setRotation(False)

            if self.ui.CheckBox_DataAug_zcaWeighting.checkState() == Qt.Checked:
                self.deepLearningArtApp.setZCA_Whitening(True)
            else:
                self.deepLearningArtApp.setZCA_Whitening(False)

            if self.ui.CheckBox_DataAug_HeightShift.checkState() == Qt.Checked:
                self.deepLearningArtApp.setHeightShift(True)
            else:
                self.deepLearningArtApp.setHeightShift(False)

            if self.ui.CheckBox_DataAug_WidthShift.checkState() == Qt.Checked:
                self.deepLearningArtApp.setWidthShift(True)
            else:
                self.deepLearningArtApp.setWidthShift(False)
        else:
            # disable data augmentation
            self.deepLearningArtApp.setDataAugmentationEnabled(False)


        # start training process
        self.deepLearningArtApp.performTraining()



    def button_markingsPath_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getMarkingsPath())
        self.ui.Label_MarkingsPath.setText(dir)
        self.deepLearningArtApp.setMarkingsPath(dir)



    def button_patching_clicked(self):
        if self.deepLearningArtApp.getSplittingMode() == DeepLearningArtApp.NONE_SPLITTING:
            QMessageBox.about(self, "My message box", "Select Splitting Mode!")
            return 0

        self.getSelectedDatasets()
        self.getSelectedPatients()

        # get patching parameters
        self.deepLearningArtApp.setPatchSizeX(self.ui.SpinBox_PatchX.value())
        self.deepLearningArtApp.setPatchSizeY(self.ui.SpinBox_PatchY.value())
        self.deepLearningArtApp.setPatchSizeZ(self.ui.SpinBox_PatchZ.value())
        self.deepLearningArtApp.setPatchOverlapp(self.ui.SpinBox_PatchOverlapp.value())

        # get labling parameters
        if self.ui.RadioButton_MaskLabeling.isChecked():
            self.deepLearningArtApp.setLabelingMode(DeepLearningArtApp.MASK_LABELING)
        elif self.ui.RadioButton_PatchLabeling.isChecked():
            self.deepLearningArtApp.setLabelingMode(DeepLearningArtApp.PATCH_LABELING)

        # get patching parameters
        if self.ui.ComboBox_Patching.currentIndex() == 1:
            # 2D patching selected
            self.deepLearningArtApp.setPatchingMode(DeepLearningArtApp.PATCHING_2D)
        elif self.ui.ComboBox_Patching.currentIndex() == 2:
            # 3D patching selected
            self.deepLearningArtApp.setPatchingMode(DeepLearningArtApp.PATCHING_3D)
        else:
            self.ui.ComboBox_Patching.setCurrentIndex(1)
            self.deepLearningArtApp.setPatchingMode(DeepLearningArtApp.PATCHING_2D)

        # handle store mode
        self.deepLearningArtApp.setStoreMode(self.ui.ComboBox_StoreOptions.currentIndex())

        print("Start Patching for ")
        print("the Patients:")
        for x in self.deepLearningArtApp.getSelectedPatients():
            print(x)
        print("and the Datasets:")
        for x in self.deepLearningArtApp.getSelectedDatasets():
            print(x)
        print("with the following Patch Parameters:")
        print("Patch Size X: " + str(self.deepLearningArtApp.getPatchSizeX()))
        print("Patch Size Y: " + str(self.deepLearningArtApp.getPatchSizeY()))
        print("Patch Overlapp: " + str(self.deepLearningArtApp.getPatchOverlapp()))

        #generate dataset
        self.deepLearningArtApp.generateDataset()

        #check if attributes in DeepLearningArtApp class contains dataset
        if self.deepLearningArtApp.datasetAvailable() == True:
            # if yes, make the use current data button available
            self.ui.Button_useCurrentData.setEnabled(True)



    def button_outputPatching_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
        self.ui.Label_OutputPathPatching.setText(dir)
        self.deepLearningArtApp.setOutputPathForPatching(dir)



    def getSelectedPatients(self):
        selectedPatients = []
        for i in range(self.ui.TreeWidget_Patients.topLevelItemCount()):
            if self.ui.TreeWidget_Patients.topLevelItem(i).checkState(0) == Qt.Checked:
                selectedPatients.append(self.ui.TreeWidget_Patients.topLevelItem(i).text(0))

        self.deepLearningArtApp.setSelectedPatients(selectedPatients)



    def button_DB_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getPathToDatabase())
        self.deepLearningArtApp.setPathToDatabase(dir)
        self.manageTreeView()



    def openFileNamesDialog(self, dir=None):
        if dir==None:
            dir = "D:" + os.sep + "med_data" + os.sep + "MRPhysics"

        options = QFileDialog.Options()
        options |=QFileDialog.DontUseNativeDialog

        ret = QFileDialog.getExistingDirectory(self, "Select Directory", dir)
        # path to database
        dir = str(ret)
        return dir



    def manageTreeView(self):
        # all patients in database
        if os.path.exists(self.deepLearningArtApp.getPathToDatabase()):
            subdirs = os.listdir(self.deepLearningArtApp.getPathToDatabase())
            self.ui.TreeWidget_Patients.setHeaderLabel("Patients:")

            for x in subdirs:
                item = QTreeWidgetItem()
                item.setText(0, str(x))
                item.setCheckState(0, Qt.Unchecked)
                self.ui.TreeWidget_Patients.addTopLevelItem(item)

            self.ui.Label_DB.setText(self.deepLearningArtApp.getPathToDatabase())



    def manageTreeViewDatasets(self):
        print(os.path.dirname(self.deepLearningArtApp.getPathToDatabase()))
        # manage datasets
        self.ui.TreeWidget_Datasets.setHeaderLabel("Datasets:")
        for ds in DeepLearningArtApp.datasets.keys():
            dataset = DeepLearningArtApp.datasets[ds].getPathdata()
            item = QTreeWidgetItem()
            item.setText(0, dataset)
            item.setCheckState(0, Qt.Unchecked)
            self.ui.TreeWidget_Datasets.addTopLevelItem(item)



    def getSelectedDatasets(self):
        selectedDatasets = []
        for i in range(self.ui.TreeWidget_Datasets.topLevelItemCount()):
            if self.ui.TreeWidget_Datasets.topLevelItem(i).checkState(0) == Qt.Checked:
                selectedDatasets.append(self.ui.TreeWidget_Datasets.topLevelItem(i).text(0))

        self.deepLearningArtApp.setSelectedDatasets(selectedDatasets)



    def selectedDNN_changed(self):
        self.deepLearningArtApp.setNeuralNetworkModel(self.ui.ComboBox_DNNs.currentText())



    def button_useCurrentData_clicked(self):
        if self.deepLearningArtApp.datasetAvailable() == True:
            self.ui.Label_currentDataset.setText("Current Dataset is used...")
            self.ui.GroupBox_TrainNN.setEnabled(True)
        else:
            self.ui.Button_useCurrentData.setEnabled(False)
            self.ui.Label_currentDataset.setText("No Dataset selected!")
            self.ui.GroupBox_TrainNN.setEnabled(False)



    def button_selectDataset_clicked(self):
        pathToDataset = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
        retbool, datasetName = self.deepLearningArtApp.loadDataset(pathToDataset)
        if retbool == True:
            self.ui.Label_currentDataset.setText(datasetName + " is used as dataset...")
        else:
            self.ui.Label_currentDataset.setText("No Dataset selected!")

        if self.deepLearningArtApp.datasetAvailable() == True:
            self.ui.GroupBox_TrainNN.setEnabled(True)
        else:
            self.ui.GroupBox_TrainNN.setEnabled(False)



    def button_learningOutputPath_clicked(self):
        path = self.openFileNamesDialog(self.deepLearningArtApp.getLearningOutputPath())
        self.deepLearningArtApp.setLearningOutputPath(path)
        self.ui.Label_LearningOutputPath.setText(path)



    def updateProgressBarTraining(self, val):
        self.ui.ProgressBar_training.setValue(val)



    def splittingMode_changed(self):

        if self.ui.ComboBox_splittingMode.currentIndex() == 0:
            self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
            self.ui.Label_SplittingParams.setText("Select splitting mode!")
        elif self.ui.ComboBox_splittingMode.currentIndex() == 1:
            # call input dialog for editting ratios
            testTrainingRatio, retBool = QInputDialog.getDouble(self, "Enter Test/Training Ratio:",
                                                             "Ratio Test/Training Set:", 0.2, 0, 1, decimals=2)
            if retBool == True:
                validationTrainingRatio, retBool = QInputDialog.getDouble(self, "Enter Validation/Training Ratio",
                                                                      "Ratio Validation/Training Set: ", 0.2, 0, 1, decimals=2)
                if retBool == True:
                    self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.SIMPLE_RANDOM_SAMPLE_SPLITTING)
                    self.deepLearningArtApp.setTrainTestDatasetRatio(testTrainingRatio)
                    self.deepLearningArtApp.setTrainValidationRatio(validationTrainingRatio)
                    txtStr = "using Test/Train=" + str(testTrainingRatio) + " and Valid/Train=" + str(validationTrainingRatio)
                    self.ui.Label_SplittingParams.setText(txtStr)
                else:
                    self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
                    self.ui.ComboBox_splittingMode.setCurrentIndex(0)
                    self.ui.Label_SplittingParams.setText("Select Splitting Mode!")
            else:
                self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
                self.ui.ComboBox_splittingMode.setCurrentIndex(0)
                self.ui.Label_SplittingParams.setText("Select Splitting Mode!")
        elif self.ui.ComboBox_splittingMode.currentIndex() == 2:
            # cross validation splitting
            testTrainingRatio, retBool = QInputDialog.getDouble(self, "Enter Test/Training Ratio:",
                                                             "Ratio Test/Training Set:", 0.2, 0, 1, decimals=2)

            if retBool == True:
                numFolds, retBool = QInputDialog.getInt(self, "Enter Number of Folds for Cross Validation",
                                                    "Number of Folds: ", 15, 0, 100000)
                if retBool == True:
                    self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.CROSS_VALIDATION_SPLITTING)
                    self.deepLearningArtApp.setTrainTestDatasetRatio(testTrainingRatio)
                    self.deepLearningArtApp.setNumFolds(numFolds)
                    self.ui.Label_SplittingParams.setText("Test/Train Ratio: " + str(testTrainingRatio) + \
                                                          ", and " + str(numFolds) + " Folds")
                else:
                    self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
                    self.ui.ComboBox_splittingMode.setCurrentIndex(0)
                    self.ui.Label_SplittingParams.setText("Select Splitting Mode!")
            else:
                self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
                self.ui.ComboBox_splittingMode.setCurrentIndex(0)
                self.ui.Label_SplittingParams.setText("Select Splitting Mode!")

        elif self.ui.ComboBox_splittingMode.currentIndex() == 3:
            self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.PATIENT_CROSS_VALIDATION_SPLITTING)



    def check_dataAugmentation_enabled(self):
        if self.ui.CheckBox_DataAugmentation.checkState() == Qt.Checked:
            self.ui.CheckBox_DataAug_horizontalFlip.setEnabled(True)
            self.ui.CheckBox_DataAug_verticalFlip.setEnabled(True)
            self.ui.CheckBox_DataAug_Rotation.setEnabled(True)
            self.ui.CheckBox_DataAug_zcaWeighting.setEnabled(True)
            self.ui.CheckBox_DataAug_HeightShift.setEnabled(True)
            self.ui.CheckBox_DataAug_WidthShift.setEnabled(True)
        else:
            self.ui.CheckBox_DataAug_horizontalFlip.setEnabled(False)
            self.ui.CheckBox_DataAug_verticalFlip.setEnabled(False)
            self.ui.CheckBox_DataAug_Rotation.setEnabled(False)
            self.ui.CheckBox_DataAug_zcaWeighting.setEnabled(False)
            self.ui.CheckBox_DataAug_HeightShift.setEnabled(False)
            self.ui.CheckBox_DataAug_WidthShift.setEnabled(False)



    ###################################################################################################################
    ##############################                 ArtGAN Stuff             ###########################################
    ###################################################################################################################
    def button_patching_ArtGAN_clicked(self):

        # handle store mode
        self.deepLearningArtApp.setStoreMode_ArtGAN(self.ui.ComboBox_StoreOptions.currentIndex())
        self.deepLearningArtApp.setPatchSizeX_ArtGAN(int(self.ui.SpinBox_PatchX_ArtGAN.value()))
        self.deepLearningArtApp.setPatchSizeY_ArtGAN(int(self.ui.SpinBox_PatchY_ArtGAN.value()))
        self.deepLearningArtApp.setPatchOverlap_ArtGAN(float(self.ui.SpinBox_PatchOverlapp_ArtGAN.value()))


        #get selected datasets
        selectedDatasets = []
        for i in range(self.ui.TreeWidget_Datasets_ArtGAN.topLevelItemCount()):
            if self.ui.TreeWidget_Datasets_ArtGAN.topLevelItem(i).checkState(0) == Qt.Checked:
                selectedDatasets.append(self.ui.TreeWidget_Datasets_ArtGAN.topLevelItem(i).text(0))

        #get selected patients
        selectedPatients = []
        for i in range(self.ui.TreeWidget_Patients_ArtGAN.topLevelItemCount()):
            if self.ui.TreeWidget_Patients_ArtGAN.topLevelItem(i).checkState(0) == Qt.Checked:
                selectedPatients.append(self.ui.TreeWidget_Patients_ArtGAN.topLevelItem(i).text(0))

        #get reference datasets for selected artifacts datasets
        artGAN_datapairs = {}
        for i in selectedDatasets:
            bodyRegionOfSelectedDataset, _ = DeepLearningArtApp.datasets[i].getBodyRegion()
            mrtWeightingOfSelectedDataset, _ = DeepLearningArtApp.datasets[i].getMRTWeighting()
            artGAN_datapairs[i] = self.getArtGANPairByBodyRegion(bodyRegionOfSelectedDataset,
                                                                 mrtWeightingOfSelectedDataset)

        self.deepLearningArtApp.setPatientsArtGAN(selectedPatients)
        self.deepLearningArtApp.setDatasetArtGAN(selectedDatasets)
        self.deepLearningArtApp.setDatasets_ArtGAN_Pairs(artGAN_datapairs)

        self.deepLearningArtApp.generateDataset_ArtGAN()



    def button_DB_ArtGAN_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getPathToDatabase())
        self.ui.Label_MarkingsPath.setText(dir)
        self.deepLearningArtApp.setPathToDatabase(dir)



    def getArtGANPairByBodyRegion(self, param_bodyregion, param_mrtWeighting):
        for item in DeepLearningArtApp.datasets:
            bodyregion, labelBodyRegion = DeepLearningArtApp.datasets[item].getBodyRegion()
            mrtWeighting, tWeighting = DeepLearningArtApp.datasets[item].getMRTWeighting()
            if param_bodyregion == bodyregion \
                    and DeepLearningArtApp.datasets[item].getArtefact() == 'ref' \
                    and mrtWeighting == param_mrtWeighting:
                datasetPair = item
                break
        return  datasetPair



    def manageTreeViewDatasetsArtGAN(self):
        print(os.path.dirname(self.deepLearningArtApp.getPathToDatabase()))
        # manage datasets
        self.ui.TreeWidget_Datasets_ArtGAN.setHeaderLabel("Artefact Datasets:")

        artefact_datasets = []

        for ds in DeepLearningArtApp.datasets.keys():
            dataset = DeepLearningArtApp.datasets[ds]
            if dataset.getArtefact() != 'ref':
                artefact_datasets.append(dataset.getPathdata())

        for i in artefact_datasets:
            item = QTreeWidgetItem()
            item.setText(0, i)
            item.setCheckState(0, Qt.Unchecked)
            self.ui.TreeWidget_Datasets_ArtGAN.addTopLevelItem(item)



    def manageTreeView_DB_ArtGAN(self):
        # all patients in database
        if os.path.exists(self.deepLearningArtApp.getPathToDatabase()):
            subdirs = os.listdir(self.deepLearningArtApp.getPathToDatabase())
            self.ui.TreeWidget_Patients_ArtGAN.setHeaderLabel("Patients:")

            for x in subdirs:
                item = QTreeWidgetItem()
                item.setText(0, str(x))
                item.setCheckState(0, Qt.Unchecked)
                self.ui.TreeWidget_Patients_ArtGAN.addTopLevelItem(item)

            self.ui.Label_DB_ArtGAN.setText(self.deepLearningArtApp.getPathToDatabase())



    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################



    def slider_ArtGAN_moved(self):
        val = int(self.ui.HorizontalSlider_ArtGAN.value())
        self.ui.Label_Slider_ArtGAN.setText(str(val))



    def button_showDataset_clicked(self):
        self.ui.HorizontalSlider_ArtGAN.setMaximum(int(self.deepLearningArtApp.getArtRefPairLength()-1))
        self.ui.HorizontalSlider_ArtGAN.setMinimum(0)
        self.ui.HorizontalSlider_ArtGAN.setValue(100)
        self.plotImg(figure=self.figureImg, canvas=self.canvas_figImg, index=100)



    def slider_ArtGAN_changed(self):
        val = int(self.ui.HorizontalSlider_ArtGAN.value())
        self.plotImg(figure=self.figureImg, canvas=self.canvas_figImg, index=val)
        self.ui.Label_Slider_ArtGAN.setText(str(val))



    def plotImg(self, figure, canvas, index):
        art, ref = self.deepLearningArtApp.getArtRefPair(index)

        data = [random.random() for i in range(25)]

        axes_art = figure.add_subplot(121)
        axes_art.imshow(art, cmap='gray')
        axes_art.set_title('Artefact')

        axes_ref = figure.add_subplot(122)
        axes_ref.imshow(ref, cmap='gray')
        axes_ref.set_title('Reference')

        canvas.draw()





sys._excepthook = sys.excepthook

def my_exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)

sys.excepthook = my_exception_hook

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_())