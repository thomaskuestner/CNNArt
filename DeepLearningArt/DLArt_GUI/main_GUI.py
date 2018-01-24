'''
@author: Yannick Wilhelm
@email: yannick.wilhelm@gmx.de
@date: January 2018
'''

import sys
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import *

from DeepLearningArt.DLArt_GUI.dlart_gui import Ui_DLArt_GUI
from DeepLearningArt.DLArt_GUI.dlart import DeepLearningArtApp
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

        # initialize TreeView Database
        self.manageTreeView()

        # intialize TreeView Datasets
        self.manageTreeViewDatasets()

        # initiliaze patch output path
        self.ui.Label_OutputPathPatching.setText(self.deepLearningArtApp.getOutputPathForPatching())

        # initialize markings path
        self.ui.Label_MarkingsPath.setText(self.deepLearningArtApp.getMarkingsPath())

        #initialize patching mode
        self.ui.ComboBox_Patching.setCurrentIndex(1)

        #initialize store mode
        self.ui.ComboBox_StoreOptions.setCurrentIndex(0)

        #initialize splitting mode
        self.ui.ComboBox_splittingMode.setCurrentIndex(DeepLearningArtApp.SIMPLE_RANDOM_SAMPLE_SPLITTING)
        self.ui.Label_SplittingParams.setText("using Test/Train="
                                              +str(self.deepLearningArtApp.getTrainTestDatasetRatio())
                                              +" and Valid/Train="+str(self.deepLearningArtApp.getTrainValidationRatio()))

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

        ################################################################################################################

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
            # 2D patching selected
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


        self.deepLearningArtApp.generateDataset()


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
            self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.CROSS_VALIDATION_SPLITTING)
        elif self.ui.ComboBox_splittingMode.currentIndex() == 3:
            self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.PATIENT_CROSS_VALIDATION_SPLITTING)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_())