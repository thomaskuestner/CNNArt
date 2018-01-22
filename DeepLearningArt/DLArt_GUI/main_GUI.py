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

        ################################################################################################################

    def button_markingsPath_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getMarkingsPath())
        self.ui.Label_MarkingsPath.setText(dir)
        self.deepLearningArtApp.setMarkingsPath(dir)

    def button_patching_clicked(self):
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
            item.setCheckState(0, Qt.Checked)
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_())