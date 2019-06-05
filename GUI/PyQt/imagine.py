#!/usr/bin/env python3
import os
os.environ['IMAGINEUSEGPU'] = 'False'

import codecs
import json
import os
import pickle
import subprocess
import sys
import h5py
if os.environ['IMAGINEUSEGPU'] == 'True':
    import keras.backend as K
    from keras.models import load_model
    from keras.utils.vis_utils import model_to_dot

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy.io as sio
import seaborn as sn
if os.environ['IMAGINEUSEGPU'] == 'True':
    import tensorflow as tf
    from utils.tftheanoFunction import TensorFlowTheanoFunction

import webbrowser
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot, QStringListModel
from PyQt5.QtWidgets import QAbstractItemView, QTableWidgetItem, QMdiSubWindow, QTreeWidgetItem, QFileDialog, \
    QMessageBox, QInputDialog, QSizePolicy, QComboBox, QGridLayout, QItemEditorCreatorBase, QItemEditorFactory


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib import colors
from matplotlib.patches import Rectangle, Ellipse, PathPatch


from config.PATH import DLART_OUT_PATH, PATH_OUT, MARKING_PATH, DATASETS, SUBDIRS
from configGUI.matplotlibwidget import MatplotlibWidget, MyMplCanvas
from configGUI.network_visualization import cnn2d_visual
from utils.Label import Label
from configGUI.canvas import Canvas
from configGUI.gridTable import TableWidget
from configGUI.labelDialog import LabelDialog
from configGUI.labelTable import LabelTable
from configGUI import network_visualization
from configGUI.Grey_window import grey_window
from configGUI.Patches_window import Patches_window
from configGUI.framework import Ui_MainWindow
from configGUI.loadf2 import *
from configGUI.Unpatch import UnpatchType, UnpatchArte
from configGUI.Unpatch_eight import UnpatchArte8
from configGUI.Unpatch_two import fUnpatch2D
from configGUI.activescene import Activescene
from configGUI.activeview import Activeview
from configGUI.loadf import loadImage
from DLart.network_interface import DataSetsWindow, NetworkInterface
from DLart.dlart import DeepLearningArtApp
from DLart.Constants_DLart import *

class imagine(QtWidgets.QMainWindow, Ui_MainWindow):
    update_data = QtCore.pyqtSignal(list)
    gray_data = QtCore.pyqtSignal(list)
    new_page = QtCore.pyqtSignal()
    update_data2 = QtCore.pyqtSignal(list)
    gray_data2 = QtCore.pyqtSignal(list)
    new_page2 = QtCore.pyqtSignal()
    update_data3 = QtCore.pyqtSignal(list)
    gray_data3 = QtCore.pyqtSignal(list)
    new_page3 = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    valueChanged = QtCore.pyqtSignal(list)

    def __init__(self, *args):
        super(imagine, self).__init__()
        self.setupUi(self)

        self.scrollAreaWidgetContents1 = QtWidgets.QWidget()
        self.maingrids1 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents1)
        self.scrollArea1.setWidget(self.scrollAreaWidgetContents1)
        self.scrollAreaWidgetContents2 = QtWidgets.QWidget()
        self.maingrids2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents2)
        self.scrollArea2.setWidget(self.scrollAreaWidgetContents2)
        self.stackedWidget.setCurrentIndex(0)
        self.tabWidget.setCurrentIndex(0)
        # Main widgets and related state.

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''
        self.listOfLabel = []

        self.vision = 2
        with open('configGUI/lastWorkspace.json', 'r') as json_data:
            lastState = json.load(json_data)
            lastState["Dim"][0] = self.vision
        with open('configGUI/lastWorkspace.json', 'w') as json_data:
            json_data.write(json.dumps(lastState))
        self.gridson = False
        self.gridsnr = 2
        self.linked = False
        self.viewAll = True

        with open('configGUI/editlabel.json', 'r') as json_data:
            self.infos = json.load(json_data)
            self.labelnames = self.infos['names']
            self.labelcolor = self.infos['colors']
            self.pathROI = self.infos['path'][0]

        global pathlist, list1, shapelist, pnamelist, empty1, problist, hatchlist, imagenum, resultnum, \
            cnrlist, indlist, ind2list, ind3list, classlist, colormaplist

        pathlist = []
        list1 = []
        shapelist = []
        pnamelist = []
        empty1 = []
        problist = []
        hatchlist = []
        imagenum = [0]
        resultnum = [0]
        cnrlist = []
        indlist = []
        ind2list = []
        ind3list = []
        classlist = []
        colormaplist = []

        global cmap1, cmap3, hmap1, hmap2, vtr1, vtr3, cmaps, vtrs
        with open('configGUI/colors0.json', 'r') as json_data:
            self.dcolors = json.load(json_data)
            cmap1 = self.dcolors['class2']['colors']
            cmap3 = self.dcolors['class11']['colors']
            hmap1 = self.dcolors['class8']['hatches']
            hmap2 = self.dcolors['class11']['hatches']
            vtr1 = self.dcolors['class2']['trans'][0]
            vtr3 = self.dcolors['class11']['trans'][0]
            cmaps = self.dcolors['classes']['colors']
            vtrs = self.dcolors['classes']['trans'][0]
            ## cmaps is not real colormaps, for detecting the class name, it should be
            ## integreted in the image array.

        self.newfig = plt.figure(figsize=(8, 6))  # 3
        self.newfig.set_facecolor("black")
        self.newax = self.newfig.add_subplot(111)
        self.newax.axis('off')
        self.pltc = None
        self.newcanvas = FigureCanvas(self.newfig)
        self.keylist = []
        self.mrinmain = None
        self.labelimage = False

        self.newfig2 = plt.figure(figsize=(8, 6))
        self.newfig2.set_facecolor("black")
        self.newax2 = self.newfig2.add_subplot(111)
        self.newax2.axis('off')
        self.pltc2 = None
        self.newcanvas2 = FigureCanvas(self.newfig2)  # must be defined because of selector next
        self.shownLabels2 = []
        self.keylist2 = []  # locate the key in combobox
        self.limage2 = None

        self.newfig3 = plt.figure(figsize=(8, 6))
        self.newfig3.set_facecolor("black")
        self.newax3 = self.newfig3.add_subplot(111)
        self.newax3.axis('off')
        self.pltc3 = None
        self.newcanvas3 = FigureCanvas(self.newfig3)  # must be defined because of selector next
        self.shownLabels3 = []
        self.keylist3 = []  # locate the key in combobox
        self.limage3 = None

        self.graylabel = QtWidgets.QLabel()
        self.slicelabel = QtWidgets.QLabel()
        self.zoomlabel = QtWidgets.QLabel()
        self.graylabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.graylabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slicelabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.slicelabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zoomlabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.zoomlabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.seditgray = QtWidgets.QPushButton()
        self.sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.sizePolicy.setHorizontalStretch(0)
        self.sizePolicy.setVerticalStretch(0)
        self.sizePolicy.setHeightForWidth(self.seditgray.sizePolicy().hasHeightForWidth())
        self.seditgray.setSizePolicy(self.sizePolicy)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/Icons/edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.seditgray.setText("")
        self.seditgray.setIcon(icon3)
        self.maingrids2.addWidget(self.slicelabel, 0, 0, 1, 1)
        self.maingrids2.addWidget(self.zoomlabel, 0, 1, 1, 1)
        self.maingrids2.addWidget(self.graylabel, 0, 2, 1, 1)
        self.maingrids2.addWidget(self.seditgray, 0, 3, 1, 1)
        self.viewLabel = Activeview()
        self.sceneLabel = Activescene()
        self.sceneLabel.addWidget(self.newcanvas)
        self.viewLabel.setScene(self.sceneLabel)
        self.maingrids2.addWidget(self.viewLabel, 1, 0, 1, 4)
        self.graylabel2 = QtWidgets.QLabel()
        self.slicelabel2 = QtWidgets.QLabel()
        self.zoomlabel2 = QtWidgets.QLabel()
        self.graylabel2.setFrameShape(QtWidgets.QFrame.Panel)
        self.graylabel2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slicelabel2.setFrameShape(QtWidgets.QFrame.Panel)
        self.slicelabel2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zoomlabel2.setFrameShape(QtWidgets.QFrame.Panel)
        self.zoomlabel2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.seditgray2 = QtWidgets.QPushButton()
        self.sizePolicy.setHeightForWidth(self.seditgray2.sizePolicy().hasHeightForWidth())
        self.seditgray2.setSizePolicy(self.sizePolicy)
        self.seditgray2.setText("")
        self.seditgray2.setIcon(icon3)
        self.maingrids2.addWidget(self.slicelabel2, 0, 4, 1, 1)
        self.maingrids2.addWidget(self.zoomlabel2, 0, 5, 1, 1)
        self.maingrids2.addWidget(self.graylabel2, 0, 6, 1, 1)
        self.maingrids2.addWidget(self.seditgray2, 0, 7, 1, 1)
        self.viewLabel2 = Activeview()
        self.sceneLabel2 = Activescene()
        self.sceneLabel2.addWidget(self.newcanvas2)
        self.viewLabel2.setScene(self.sceneLabel2)
        self.maingrids2.addWidget(self.viewLabel2, 1, 4, 1, 4)

        self.graylabel3 = QtWidgets.QLabel()
        self.slicelabel3 = QtWidgets.QLabel()
        self.zoomlabel3 = QtWidgets.QLabel()
        self.graylabel3.setFrameShape(QtWidgets.QFrame.Panel)
        self.graylabel3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slicelabel3.setFrameShape(QtWidgets.QFrame.Panel)
        self.slicelabel3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zoomlabel3.setFrameShape(QtWidgets.QFrame.Panel)
        self.zoomlabel3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.seditgray3 = QtWidgets.QPushButton()
        self.sizePolicy.setHeightForWidth(self.seditgray3.sizePolicy().hasHeightForWidth())
        self.seditgray3.setSizePolicy(self.sizePolicy)
        self.seditgray3.setText("")
        self.seditgray3.setIcon(icon3)
        self.maingrids2.addWidget(self.slicelabel3, 0, 8, 1, 1)
        self.maingrids2.addWidget(self.zoomlabel3, 0, 9, 1, 1)
        self.maingrids2.addWidget(self.graylabel3, 0, 10, 1, 1)
        self.maingrids2.addWidget(self.seditgray3, 0, 11, 1, 1)
        self.viewLabel3 = Activeview()
        self.sceneLabel3 = Activescene()
        self.sceneLabel3.addWidget(self.newcanvas3)
        self.viewLabel3.setScene(self.sceneLabel3)
        self.maingrids2.addWidget(self.viewLabel3, 1, 8, 1, 4)

        self.brectangle.setDisabled(True)
        self.bellipse.setDisabled(True)
        self.blasso.setDisabled(True)
        # self.bchoosemark.setDisabled(True)
        self.bnoselect.setDisabled(True)
        self.cursorCross.setDisabled(True)
        self.bnoselect.setChecked(True)
        self.deleteButton.setDisabled(True)
        self.openlabelButton.setDisabled(True)
        self.labelListShow.setDisabled(True)
        self.useDefaultLabelCheckbox.setDisabled(True)
        self.defaultLabelTextLine.setDisabled(True)
        self.labellistBox.setDisabled(True)
        self.viewallButton.setDisabled(True)
        self.horizontalSlider.setDisabled(True)

        self.view_group = None
        self.view_group_list = []
        self.view_line = None
        self.view_line_list = []

        self.selectoron = False
        self.x_clicked = None
        self.y_clicked = None
        self.mouse_second_clicked = False
        self.cursor_on = False
        self.labelon = False
        self.legendon = True
        self.label_mode = False

        self.selectedPatients = None
        self.selectedDatasets = None
        self.datasets_window = DataSetsWindow([], [])

        self.number = 0
        self.ind = 0
        self.ind2 = 0
        self.ind3 = 0
        self.ind4 = 0
        self.ind5 = 0
        self.slices = 0
        self.slices2 = 0
        self.slices3 = 0
        self.slices4 = 0
        self.slices5 = 0
        self.graylist = []
        self.graylist.append(None)
        self.graylist.append(None)
        self.emitlist = []
        self.emitlist.append(self.ind)
        self.emitlist.append(self.slices)

        self.graylist2 = []
        self.graylist2.append(None)
        self.graylist2.append(None)
        self.emitlist2 = []
        self.emitlist2.append(self.ind2)
        self.emitlist2.append(self.slices2)

        self.graylist3 = []
        self.graylist3.append(None)
        self.graylist3.append(None)
        self.emitlist3 = []
        self.emitlist3.append(self.ind3)
        self.emitlist3.append(self.slices3)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/Icons/folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.openImage.addItem(icon, '', 0)
        self.openImage.setIconSize(QtCore.QSize(35, 35))
        self.openImage.addItem('open a directory', 1)
        self.openImage.addItem('open a file', 2)
        self.openImage.addItem('load from workspace', 3)
        self.openImage.setCurrentIndex(0)
        self.openImage.activated.connect(self.load_MR)
        self.bdswitch.clicked.connect(self.switchview)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/Icons/results.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.openResult.addItem(icon, '', 0)
        self.openResult.setIconSize(QtCore.QSize(35, 35))
        self.openResult.addItem('from path', 1)
        self.openResult.addItem('from workspace', 2)
        self.openResult.setCurrentIndex(0)
        self.openResult.activated.connect(self.load_patch)

        self.horizontalSlider.setMaximum(80)
        self.horizontalSlider.setMinimum(20)
        self.horizontalSlider.valueChanged.connect(self.set_transparency)

        self.infoButton.clicked.connect(self.show_variables)
        self.gridButton.clicked.connect(self.show_grids_selection)
        self.fitButton.clicked.connect(self.fit_image)

        self.bselectoron.clicked.connect(self.selector_mode)
        self.roiButton.clicked.connect(self.select_roi_OnOff)
        self.inspectorButton.clicked.connect(self.mouse_tracking)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/Icons/label.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.labelButton.addItem(icon, '', 0)
        self.labelButton.setIconSize(QtCore.QSize(35, 35))
        self.labelButton.addItem('name around cursor', 1)
        self.labelButton.addItem('legend on/off', 2)
        self.labelButton.setCurrentIndex(0)
        self.labelButton.activated.connect(self.show_label_name)

        self.bnoselect.toggled.connect(lambda: self.marking_shape(0))
        self.brectangle.toggled.connect(lambda: self.marking_shape(1))
        self.bellipse.toggled.connect(lambda: self.marking_shape(2))
        self.blasso.toggled.connect(lambda: self.marking_shape(3))

        self.brectangle.toggled.connect(lambda: self.stop_view(1))
        self.bellipse.toggled.connect(lambda: self.stop_view(1))
        self.blasso.toggled.connect(lambda: self.stop_view(1))
        self.bnoselect.toggled.connect(lambda: self.stop_view(0))

        self.actionSave.triggered.connect(self.save_current)
        self.actionLoad.triggered.connect(self.load_old)
        self.actionColors.triggered.connect(self.set_color)
        self.actionLabels.triggered.connect(self.predefined_label)
        self.actionNetworks.triggered.connect(self.predefined_networks)
        self.actionData_Viewing.setChecked(True)
        self.actionNetwork_Training.setChecked(True)
        self.actionAbout_Network_Visualization.setChecked(True)
        self.actionNetwork_Interface.setChecked(True)
        self.actionData_Viewing.triggered.connect(lambda: self.handle_window_view(0))
        self.actionNetwork_Training.triggered.connect(lambda: self.handle_window_view(1))
        self.actionNetwork_Visualization.triggered.connect(lambda: self.handle_window_view(2))
        self.actionDatasets_Window.triggered.connect(lambda: self.handle_window_view(3))
        self.actionNetwork_Interface.triggered.connect(lambda: self.handle_window_view(4))
        self.actionImage_and_Result_Showing.triggered.connect(lambda: self.handle_help(0))
        self.actionAbout_Labeling.triggered.connect(lambda: self.handle_help(1))
        self.actionAbout_Network_Training.triggered.connect(lambda: self.handle_help(2))
        self.actionNetwork_Testing.triggered.connect(lambda: self.handle_help(3))
        self.actionAbout_Network_Visualization.triggered.connect(lambda: self.handle_help(4))

        self.cursorCross.setChecked(False)
        self.cursorCross.toggled.connect(self.handle_crossline)

        self.flipButton.clicked.connect(self.flip_image)

        self.newcanvas.mpl_connect('scroll_event', self.newonscroll)
        self.newcanvas.mpl_connect('button_press_event', self.mouse_clicked)
        self.newcanvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.newcanvas.mpl_connect('button_release_event', self.mouse_release)

        self.newcanvas2.mpl_connect('scroll_event', self.newonscrol2)
        self.newcanvas2.mpl_connect('button_press_event', self.mouse_clicked2)
        self.newcanvas2.mpl_connect('motion_notify_event', self.mouse_move2)
        self.newcanvas2.mpl_connect('button_release_event', self.mouse_release2)

        self.newcanvas3.mpl_connect('scroll_event', self.newonscrol3)
        self.newcanvas3.mpl_connect('button_press_event', self.mouse_clicked3)
        self.newcanvas3.mpl_connect('motion_notify_event', self.mouse_move3)
        self.newcanvas3.mpl_connect('button_release_event', self.mouse_release3)

        self.labelHist = []
        self.loadPredefinedClasses('configGUI/predefined_classes.txt')
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

        self.labelfile = 'Markings/marking_records.csv'
        self.deleteButton.clicked.connect(self.delete_labelItem)
        self.openlabelButton.clicked.connect(self.open_labelFile)
        self.labelListShow.clicked.connect(self.show_labelList)
        self.viewallButton.clicked.connect(self.viewAllLabel_on_off)

        self.isnotfinished = True
        self.layoutlines = 1
        self.layoutcolumns = 1
        self.num_image = 0
        self.PathDicom = ''
        self.loadPatch = False
        self.resultfile = None
        self.pathDatabase = PATH_OUT + os.sep + DATASETS + os.sep + SUBDIRS[0]

        #####directly load image in imagine(*args)

        if len(args) > 0:
            self.layoutcolumns = len(args)
            self.image_pathlist = list(args)
            while self.isnotfinished:
                if type(self.image_pathlist[self.num_image]) is str:
                    self.PathDicom = self.image_pathlist[self.num_image]
                elif type(self.image_pathlist[self.num_image]) is np.ndarray:
                    self.PathDicom = self.image_pathlist[self.num_image]
                self.load_MR(4)
                self.num_image = self.num_image + 1
                if self.num_image >= self.layoutcolumns:
                    self.isnotfinished = False

        ############ network training
        # initialize DeepLearningArt Application
        self.deepLearningArtApp = DeepLearningArtApp()
        self.deepLearningArtApp.setGUIHandle(self)
        self.network_interface = NetworkInterface(self.deepLearningArtApp.getParameters())
        self.deepLearningArtApp.network_interface_update.connect(self.update_network_interface)
        self.deepLearningArtApp._network_interface_update.connect(self._update_network_interface)

        # initiliaze patch output path
        self.Label_OutputPathPatching.setText(self.deepLearningArtApp.getOutputPathForPatching())

        # initialize markings path
        self.Label_MarkingsPath.setText(self.deepLearningArtApp.getMarkingsPath())

        # initialize learning output path
        self.Label_LearningOutputPath.setText(self.deepLearningArtApp.getLearningOutputPath())

        # initialize patching mode
        self.ComboBox_Patching.setCurrentIndex(1)

        # initialize store mode
        self.ComboBox_StoreOptions.setCurrentIndex(0)

        # initialize splitting mode
        self.ComboBox_splittingMode.setCurrentIndex(SIMPLE_RANDOM_SAMPLE_SPLITTING)
        self.Label_SplittingParams.setText("using Test/Train="
                                           + str(self.deepLearningArtApp.getTrainTestDatasetRatio())
                                           + " and Valid/Train=" + str(
            self.deepLearningArtApp.getTrainValidationRatio()))

        # initialize combox box for DNN selection
        self.ComboBox_DNNs.addItem("Select Deep Neural Network Model...")
        self.ComboBox_DNNs.addItems(DeepLearningArtApp.deepNeuralNetworks.keys())
        self.ComboBox_DNNs.addItem("add new architecture")
        self.ComboBox_DNNs.setCurrentIndex(1)
        self.deepLearningArtApp.setNeuralNetworkModel(self.ComboBox_DNNs.currentText())

        # initialize check boxes for used classes
        self.CheckBox_Artifacts.setChecked(self.deepLearningArtApp.getUsingArtifacts())
        self.CheckBox_BodyRegion.setChecked(self.deepLearningArtApp.getUsingBodyRegions())
        self.CheckBox_TWeighting.setChecked(self.deepLearningArtApp.getUsingTWeighting())

        # initilize training parameters
        self.DoubleSpinBox_WeightDecay.setValue(self.deepLearningArtApp.getWeightDecay())
        self.DoubleSpinBox_Momentum.setValue(self.deepLearningArtApp.getMomentum())
        self.CheckBox_Nesterov.setChecked(self.deepLearningArtApp.getNesterovEnabled())
        self.CheckBox_DataAugmentation.setChecked(self.deepLearningArtApp.getDataAugmentationEnabled())
        self.CheckBox_DataAug_horizontalFlip.setChecked(self.deepLearningArtApp.getHorizontalFlip())
        self.CheckBox_DataAug_verticalFlip.setChecked(self.deepLearningArtApp.getVerticalFlip())
        self.CheckBox_DataAug_Rotation.setChecked(False if self.deepLearningArtApp.getRotation() == 0 else True)
        self.CheckBox_DataAug_zcaWeighting.setChecked(self.deepLearningArtApp.getZCA_Whitening())
        self.CheckBox_DataAug_HeightShift.setChecked(False if self.deepLearningArtApp.getHeightShift() == 0 else True)
        self.CheckBox_DataAug_WidthShift.setChecked(False if self.deepLearningArtApp.getWidthShift() == 0 else True)
        self.CheckBox_DataAug_Zoom.setChecked(False if self.deepLearningArtApp.getZoom() == 0 else True)
        self.check_dataAugmentation_enabled()

        self.training_live_performance_figure = Figure(figsize=(10, 10))
        self.canvas_training_live_performance_figure = FigureCanvas(self.training_live_performance_figure)

        # network architecture
        self.network_architecture_figure = Figure(figsize=(10, 10))
        self.canvas_network_architecture_figure = FigureCanvas(self.network_architecture_figure)

        # confusion matrix
        self.confusion_matrix_figure = Figure(figsize=(10, 10))
        self.canvas_confusion_matrix_figure = FigureCanvas(self.confusion_matrix_figure)

        # artifact map plot
        self.artifact_map_figure = Figure(figsize=(10, 10))
        self.canvas_artifact_map_figure = FigureCanvas(self.artifact_map_figure)

        # segmentation masks plot
        self.segmentation_masks_figure = Figure(figsize=(10, 10))
        self.canvas_segmentation_masks_figure = FigureCanvas(self.segmentation_masks_figure)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/Icons/brush.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.wyPlot.addItem(icon, 'Showing Test Result', 0)
        self.wyPlot.setIconSize(QtCore.QSize(35, 35))
        self.wyPlot.addItem('plot segmentation predictions', 1)
        self.wyPlot.addItem('plot segmentation artifact maps', 2)
        self.wyPlot.addItem('plot confusion matrix', 3)
        self.wyPlot.setCurrentIndex(0)
        self.wyPlot.activated.connect(self.on_wyPlot_clicked)

        # select database button clicked
        self.Button_DB.clicked.connect(self.button_DB_clicked)

        # output path button for patching clicked
        self.Button_OutputPathPatching.clicked.connect(self.button_outputPatching_clicked)

        # TreeWidgets
        self.TreeWidget_Patients.setHeaderLabel("Patients:")
        self.TreeWidget_Patients.clicked.connect(self.getSelectedPatients)
        self.TreeWidget_Datasets.setHeaderLabel("Datasets:")
        self.TreeWidget_Datasets.clicked.connect(self.getSelectedDatasets)

        # Patching button
        self.Button_Patching.clicked.connect(self.button_patching_clicked)

        # mask marking path button clicekd
        self.Button_MarkingsPath.clicked.connect(self.button_markingsPath_clicked)

        # combo box splitting mode is changed
        self.ComboBox_splittingMode.currentIndexChanged.connect(self.splittingMode_changed)

        # "use current data" button clicked
        self.Button_useCurrentData.clicked.connect(self.button_useCurrentData_clicked)

        # select dataset is clicked
        self.Button_selectDataset.clicked.connect(self.button_selectDataset_clicked)

        # learning output path button clicked
        self.Button_LearningOutputPath.clicked.connect(self.button_learningOutputPath_clicked)

        # train button clicked
        self.Button_train.clicked.connect(self.button_train_clicked)

        # combobox dnns
        self.ComboBox_DNNs.currentIndexChanged.connect(self.selectedDNN_changed)

        # data augmentation enbaled changed
        self.CheckBox_DataAugmentation.stateChanged.connect(self.check_dataAugmentation_enabled)

        ##################################Network Visualizaiton
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/Icons/folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.wyChooseFile.addItem(icon, '', 0)
        self.wyChooseFile.setIconSize(QtCore.QSize(35, 35))
        self.wyChooseFile.addItem('from path', 1)
        self.wyChooseFile.addItem('from workspace', 2)
        self.wyChooseFile.setCurrentIndex(0)
        self.wyChooseFile.activated.connect(self.on_wyChooseFile_clicked)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/Icons/data.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.wyInputData.addItem(icon, '', 0)
        self.wyInputData.setIconSize(QtCore.QSize(35, 35))
        self.wyInputData.addItem('from path', 1)
        self.wyInputData.addItem('from workspace', 2)
        self.wyInputData.setCurrentIndex(0)
        self.wyInputData.activated.connect(self.on_wyInputData_clicked)

        self.horizontalSliderPatch.hide()
        self.horizontalSliderSlice.hide()

        self.labelPatch.hide()
        self.labelSlice.hide()

        self.lcdNumberPatch.hide()
        self.lcdNumberSlice.hide()
        # self.lcdNumberSS.hide()

        self.radioButton_3.hide()
        self.radioButton_4.hide()

        self.predictButton.setDisabled(True)

        self.resetW = False
        self.resetF = False
        self.resetS = False

        self.twoInput = False
        self.chosenLayerName = []
        # the slider's value is the chosen patch's number
        self.chosenWeightNumber = 1
        self.chosenWeightSliceNumber = 1
        self.chosenPatchNumber = 1
        self.chosenPatchSliceNumber = 1
        self.chosenSSNumber = 1
        self.openmodel_path = ''
        self.inputData_name = ''
        self.inputData = {}
        self.inputalpha = '0.19'
        self.inputGamma = '0.0000001'

        self.layer_index_name = {}
        self.model = {}
        self.qList = []
        self.totalWeights = 0
        self.totalWeightsSlices = 0
        self.totalPatches = 0
        self.totalPatchesSlices = 0
        self.totalSS = 0

        self.modelDimension = ''
        self.modelName = ''
        self.modelInput = {}
        self.modelInput2 = {}
        self.ssResult = {}
        self.activations = {}
        self.act = {}
        self.layers_by_depth = {}
        self.weights = {}
        self.w = {}
        self.LayerWeights = {}
        self.subset_selection = {}
        self.subset_selection_2 = {}
        self.radioButtonValue = []
        self.listView.clicked.connect(self.clickList)

        self.W_F = ''

        # slider of the weight and feature
        self.horizontalSliderPatch.sliderReleased.connect(self.sliderValue)
        self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)

        self.horizontalSliderSlice.sliderReleased.connect(self.sliderValue)
        self.horizontalSliderSlice.valueChanged.connect(self.lcdNumberSlice.display)

        self.lineEdit.textChanged[str].connect(self.textChangeAlpha)
        self.lineEdit_2.textChanged[str].connect(self.textChangeGamma)

        self.valueChanged.connect(self.update_network_interface)

    def handle_window_view(self, n):
        if n == 0:
            self.tabWidget.setTabEnabled(n, self.actionData_Viewing.isChecked())
        elif n == 1:
            self.tabWidget.setTabEnabled(n, self.actionNetwork_Training.isChecked())
        elif n == 2:
            self.tabWidget.setTabEnabled(n, self.actionNetwork_Visualization.isChecked())
        elif n == 3:
            self.datasets_window.show()
        elif n == 4:
            self.network_interface.window.show()

    def handle_help(self, n):
        help = 'help' + os.sep
        if n == 0:
            item = 'help_image_and_result_viewing'
        elif n == 1:
            item = 'help_about_labeling'
        elif n == 2:
            item = 'help_about_network_training'
        elif n == 3:
            item = 'help_about_network_testing'
        elif n == 4:
            item = 'help_about_network_visualization'
        help += item + os.sep + item + '.html'
        webbrowser.open_new(help)

    def switchview(self):
        if self.vision == 2:
            self.vision = 3
            self.visionlabel.setText('3D')
        elif self.vision == 3:
            self.vision = 2
            self.visionlabel.setText('2D')

        self.set_layout()
        with open('configGUI/lastWorkspace.json', 'r') as json_data:
            lastState = json.load(json_data)
            lastState["Dim"][0] = self.vision

        with open('configGUI/lastWorkspace.json', 'w') as json_data:
            json_data.write(json.dumps(lastState))

    def viewAllLabel_on_off(self):

        if self.viewAll == True:
            self.viewAll = False
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/icons/Icons/eye off.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.viewallButton.setIcon(icon)
            self.labelList.view_all(True)

        else:
            self.viewAll = True
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/icons/Icons/eye on.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.viewallButton.setIcon(icon)
            self.labelList.view_all(False)

    def clear_all(self):

        if self.gridsnr == 2:
            for i in reversed(range(self.maingrids1.count())):
                self.maingrids1.itemAt(i).clearWidgets()
                for j in reversed(range(self.maingrids1.itemAt(i).secondline.count())):
                    self.maingrids1.itemAt(i).secondline.itemAt(j).widget().setParent(None)
                self.maingrids1.removeItem(self.maingrids1.itemAt(i))  # invisible
        else:
            for i in reversed(range(self.maingrids1.count())):
                self.maingrids1.itemAt(i).clearWidgets()
                for j in reversed(range(self.maingrids1.itemAt(i).gridLayout_1.count())):
                    self.maingrids1.itemAt(i).gridLayout_1.itemAt(j).widget().setParent(None)
                for j in reversed(range(self.maingrids1.itemAt(i).gridLayout_2.count())):
                    self.maingrids1.itemAt(i).gridLayout_2.itemAt(j).widget().setParent(None)
                for j in reversed(range(self.maingrids1.itemAt(i).gridLayout_3.count())):
                    self.maingrids1.itemAt(i).gridLayout_3.itemAt(j).widget().setParent(None)
                for j in reversed(range(self.maingrids1.itemAt(i).gridLayout_4.count())):
                    self.maingrids1.itemAt(i).gridLayout_4.itemAt(j).widget().setParent(None)
                self.maingrids1.removeItem(self.maingrids1.itemAt(i))

    def save_current(self):
        self.save_file()
        if self.label_mode:
            self.save_label()

    def save_file(self):
        if self.gridson:
            with open('configGUI/lastWorkspace.json', 'r') as json_data:
                lastState = json.load(json_data)
                lastState['mode'] = self.gridsnr  ###
                if self.gridsnr == 3:
                    lastState["Dim"][0] = self.vision
                    lastState['layout'][0] = self.layout3D
                else:
                    lastState["Dim"][0] = self.vision
                    lastState['layout'][0] = self.layoutlines
                    lastState['layout'][1] = self.layoutcolumns

                global pathlist, list1, pnamelist, problist, hatchlist, cnrlist, shapelist, imagenum, resultnum, \
                    indlist, ind2list, ind3list, classlist, colormaplist

                # shapelist = (list(shapelist)).tolist()
                # shapelist = pd.Series(shapelist).to_json(orient='values')
                lastState['Shape'] = shapelist
                lastState['Pathes'] = pathlist
                lastState['NResults'] = pnamelist
                lastState['NrClass'] = cnrlist
                lastState['ImgNum'] = imagenum
                lastState['ResNum'] = resultnum
                lastState['Index'] = indlist
                lastState['Index2'] = ind2list
                lastState['Index3'] = ind3list
                lastState['Classes'] = classlist
                lastState['Colors'] = colormaplist

            with open('configGUI/lastWorkspace.json', 'w') as json_data:
                json_data.write(json.dumps(lastState))

            listA = open('config/dump1.txt', 'wb')
            pickle.dump(list1, listA)
            listA.close()
            listB = open('config/dump2.txt', 'wb')
            pickle.dump(problist, listB)
            listB.close()
            listC = open('config/dump3.txt', 'wb')
            pickle.dump(hatchlist, listC)
            listC.close()

    def load_old(self):
        if os.path.isfile('config/dump1.txt') & os.path.isfile('config/dump2.txt') & os.path.isfile('config/dump3.txt'):
            self.clear_all()
            global pathlist, list1, pnamelist, problist, hatchlist, imagenum, resultnum, cnrlist, shapelist, \
                indlist, ind2list, ind3list, classlist, colormaplist

            with open('configGUI/lastWorkspace.json', 'r') as json_data:
                lastState = json.load(json_data)
                # list1 = lastState['listA']
                # problist = lastState['Probs']
                # hatchlist = lastState['Hatches']
                gridsnr = lastState['mode']  ##
                shapelist = lastState['Shape']
                pathlist = lastState['Pathes']
                pnamelist = lastState['NResults']
                cnrlist = lastState['NrClass']
                imagenum = lastState['ImgNum']
                resultnum = lastState['ResNum']
                indlist = lastState['Index']
                ind2list = lastState['Index2']
                ind3list = lastState['Index3']
                classlist = lastState['Classes']
                colormaplist = lastState['Colors']

                if gridsnr == 2:
                    if self.vision == 3:
                        self.switchview()  # back to 2
                    self.layoutlines = lastState['layout'][0]
                    self.layoutcolumns = lastState['layout'][1]
                else:
                    if self.vision == 2:
                        self.switchview()
                    self.layout3D = lastState['layout'][0]

            listA = open('config/dump1.txt', 'rb')
            list1 = pickle.load(listA)
            listA.close()
            listB = open('config/dump2.txt', 'rb')
            problist = pickle.load(listB)
            listB.close()
            listC = open('config/dump3.txt', 'rb')
            hatchlist = pickle.load(listC)
            listC.close()

            self.set_layout()

    def show_grids_selection(self):

        self.screenGrids = QMdiSubWindow()
        self.screenGrids.setGeometry(200, 250, 180, 240)
        rows = 8
        columns = 6
        self.tableWidget = TableWidget(rows, columns)
        self.tableWidget.setMouseTracking(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setMouseTracking(True)
        self.tableWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tableWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tableWidget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(30)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setDefaultSectionSize(30)
        self.tableWidget.setMouseTracking(True)
        for column in range(columns):
            for row in range(rows):
                item = QTableWidgetItem('')
                self.tableWidget.setItem(row, column, item)
        self.screenGrids.setWidget(self.tableWidget)
        self.screenGrids.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)
        self.tableWidget.itemEntered.connect(self.handleItemEntered)
        self.tableWidget.itemExited.connect(self.handleItemExited)
        self.tableWidget.released.connect(self.handleItemCount)
        self.screenGrids.show()
        self.finished.connect(self.screenGrids.close)
        self.screenGrids.setMouseTracking(True)

    def set_layout(self):

        self.gridson = True
        if self.vision == 2:
            self.clear_all()
            self.gridsnr = 2
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    blocklayout = Viewgroup()
                    self.view_group = blocklayout
                    self.view_group.modechange.setDisabled(False)
                    self.view_group_list.append(self.view_group)
                    for dpath in pathlist:
                        blocklayout.addPathd(dpath, 0)
                    for cpath in pnamelist:
                        blocklayout.addPathre(cpath)
                    blocklayout.in_link.connect(self.link_mode)  # initial connection of in_link
                    self.maingrids1.addLayout(blocklayout, i, j)
            if pathlist:
                n = 0
                while n < len(pathlist):
                    for i in range(self.layoutlines):
                        for j in range(self.layoutcolumns):
                            self.maingrids1.itemAtPosition(i, j).pathbox.setCurrentIndex(n + 1)
                            n += 1
            if self.selectoron and self.labelon:
                self.load_markings(self.ind)
                self.show_labelList()
            else:
                pass
        elif self.vision == 3:
            self.clear_all()
            self.gridsnr = 3
            self.layout3D = self.layoutlines
            for i in range(self.layout3D):
                blockline = Viewline()
                self.view_line = blockline
                self.view_line_list.append(self.view_line)
                for dpath in pathlist:
                    blockline.addPathim(dpath)
                for cpath in pnamelist:
                    blockline.addPathre(cpath)
                blockline.in_link.connect(self.link_mode)  # 3d initial
                self.maingrids1.addLayout(blockline, i, 0)
            if pathlist:
                n = 0
                for i in range(self.layout3D):
                    if n < len(pathlist):
                        self.maingrids1.itemAtPosition(i, 0).imagebox.setCurrentIndex(n + 1)
                        n += 1
                    else:
                        break
            if self.selectoron and self.labelon:
                self.load_markings(self.ind)
                self.show_labelList()
            else:
                pass

        self.finished.emit()

    def link_mode(self):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids1.itemAtPosition(i, j).islinked and not self.maingrids1.itemAtPosition(i,
                                                                                                            j).skiplink:
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.zoom_link.connect(self.zoom_all)
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.move_link.connect(self.move_all)
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.grey_link.connect(self.grey_all)
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.slice_link.connect(self.slice_all)
                        self.maingrids1.itemAtPosition(i, j).skiplink = True  # avoid multi link
                        self.maingrids1.itemAtPosition(i, j).skipdis = False
                    elif not self.maingrids1.itemAtPosition(i, j).islinked and not self.maingrids1.itemAtPosition(i,
                                                                                                                  j).skipdis:
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.zoom_link.disconnect()
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.move_link.disconnect()
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.grey_link.disconnect()
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.slice_link.disconnect()
                        self.maingrids1.itemAtPosition(i, j).skipdis = True
                        self.maingrids1.itemAtPosition(i, j).skiplink = False
        else:
            for i in range(self.layout3D):
                if self.maingrids1.itemAtPosition(i, 0).islinked and not self.maingrids1.itemAtPosition(i,
                                                                                                        0).skiplink:
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.zoom_link.connect(self.zoom_all)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.move_link.connect(self.move_all)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas1.grey_link.connect(self.grey_all)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.zoom_link.connect(self.zoom_all)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.move_link.connect(self.move_all)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas2.grey_link.connect(self.grey_all)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.zoom_link.connect(self.zoom_all)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.move_link.connect(self.move_all)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas3.grey_link.connect(self.grey_all)
                    self.maingrids1.itemAtPosition(i, 0).skiplink = True
                    self.maingrids1.itemAtPosition(i, 0).skipdis = False
                elif not self.maingrids1.itemAtPosition(i, 0).islinked and not self.maingrids1.itemAtPosition(i,
                                                                                                              0).skipdis:
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.zoom_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.move_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).newcanvas1.grey_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.zoom_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.move_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).newcanvas2.grey_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.zoom_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.move_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).newcanvas3.grey_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).skipdis = True
                    self.maingrids1.itemAtPosition(i, 0).skiplink = False

    def zoom_all(self, factor):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids1.itemAtPosition(i, j).islinked:
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.linkedZoom(factor)
        else:
            for i in range(self.layout3D):
                if self.maingrids1.itemAtPosition(i, 0).islinked:
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.linkedZoom(factor)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.linkedZoom(factor)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.linkedZoom(factor)

    def move_all(self, movelist):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids1.itemAtPosition(i, j).islinked:
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.linkedMove(movelist)
        else:
            for i in range(self.layout3D):
                if self.maingrids1.itemAtPosition(i, 0).islinked:
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.linkedMove(movelist)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.linkedMove(movelist)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.linkedMove(movelist)

    def grey_all(self, glist):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids1.itemAtPosition(i, j).islinked:
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.linked_grey(glist)
        else:
            for i in range(self.layout3D):
                if self.maingrids1.itemAtPosition(i, 0).islinked:
                    self.maingrids1.itemAtPosition(i, 0).newcanvas1.linked_grey(glist)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas2.linked_grey(glist)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas3.linked_grey(glist)

    def slice_all(self, data):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids1.itemAtPosition(i, j).islinked:
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.linked_slice(data)

    def load_MR(self, index):
        nothing2show = False
        if not index == 4:

            MRPath = PATH_OUT + os.sep + DATASETS + os.sep + SUBDIRS[0] + os.sep
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.Directory
            options |= QtWidgets.QFileDialog.ExistingFile
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            self.file_dialog = QtWidgets.QFileDialog()
            self.file_dialog.setOptions(options)

            if index == 1:
                self.PathDicom = self.file_dialog.getExistingDirectory(self, 'open a directory', MRPath)
            elif index == 2:
                self.PathDicom = \
                    self.file_dialog.getOpenFileName(self, 'open an image', MRPath, 'All Files(*.*)')[
                        0]
            elif index == 3:
                imagefilelist = []
                if self.datasets_window.getCurrentRow() is not None:
                    imagefilelist.append(self.pathDatabase + os.sep + self.datasets_window.getCurrentRow())
                else:
                    if self.selectedDatasets is not None and self.selectedPatients is not None:
                        for patient in self.selectedPatients:
                            for dataset in self.selectedDatasets:
                                imagefilelist.append(
                                    self.pathDatabase + os.sep + patient + os.sep + SUBDIRS[1] + os.sep + dataset)
                    else:
                        nothing2show = True
                        QtWidgets.QMessageBox.information(self, 'Warning', 'No datasets to show!')
                if len(imagefilelist) > 0:
                    self.layoutlines = 1
                    self.layoutcolumns = len(imagefilelist)
                    self.image_pathlist = list(imagefilelist)
                    self.num_image = 0
                    while self.isnotfinished:
                        self.PathDicom = self.image_pathlist[self.num_image]
                        self.load_MR(4)
                        self.num_image = self.num_image + 1
                        if self.num_image >= self.layoutcolumns:
                            self.isnotfinished = False
            else:
                pass
        if type(self.PathDicom) is str:
            self.selectorPath = self.PathDicom
            self.newMR = loadImage(self.PathDicom)
        elif type(self.PathDicom) is np.ndarray:
            self.newMR = loadImage(self.PathDicom)
            self.PathDicom = ''
            self.selectorPath = self.PathDicom
        else:
            if not nothing2show:
                QtWidgets.QMessageBox.information(self, 'Warning', 'This format is not supported!')

        self.openImage.setDisabled(True)
        self.newMR.start()
        if not self.newMR.new_shape == []:
            self.NS = self.newMR.new_shape
            pathlist.append(self.PathDicom)
            shapelist.append(self.NS)
            self.mrinmain = self.newMR.voxel_ndarray
            list1.append(self.mrinmain)
            self.slices = self.mrinmain.shape[-1]
            self.ind = self.slices // 2
            # 40
            self.slices2 = self.mrinmain.shape[-3]
            self.ind2 = self.slices2 // 2
            self.slices3 = self.mrinmain.shape[-2]
            self.ind3 = self.slices3 // 2
            try:
                self.slices4 = self.mrinmain.shape[-5]
                self.slices5 = self.mrinmain.shape[-4]
            except:
                self.slices4 = 0
                self.slices5 = 0
            indlist.clear()
            indlist.append(self.ind)
            ind2list.clear()
            ind2list.append(self.ind2)
            ind3list.clear()
            ind3list.append(self.ind3)
            self.newMR.trigger.connect(self.load_end)
            self.set_layout()

            if not self.loadPatch:
                self.openImage.setCurrentIndex(0)
            else:
                self.load_patch_end()
        else:
            self.openImage.setDisabled(False)
            if not nothing2show:
                QtWidgets.QMessageBox.information(self, 'Warning', 'This format is not supported!')
        try:
            self.imgdialog.close()
        except:
            pass
        self.openImage.setCurrentIndex(0)

    def load_end(self):

        if not self.selectoron:
            self.openImage.setDisabled(False)
            if self.gridson:
                if self.gridsnr == 2:
                    for i in range(self.layoutlines):
                        for j in range(self.layoutcolumns):
                            self.maingrids1.itemAtPosition(i, j).addPathd(self.PathDicom, 0)
                    for i in range(self.layoutlines):
                        for j in range(self.layoutcolumns):
                            if self.maingrids1.itemAtPosition(i, j).mode == 1 and \
                                    self.maingrids1.itemAtPosition(i, j).pathbox.currentIndex() == 0:
                                self.maingrids1.itemAtPosition(i, j).pathbox.setCurrentIndex(len(pathlist))
                                break
                        else:
                            continue
                        break
                else:
                    for i in range(self.layout3D):
                        self.maingrids1.itemAtPosition(i, 0).addPathim(self.PathDicom)
                    for i in range(self.layout3D):
                        if self.maingrids1.itemAtPosition(i, 0).vmode == 1 and \
                                self.maingrids1.itemAtPosition(i, 0).imagebox.currentIndex() == 0:
                            self.maingrids1.itemAtPosition(i, 0).imagebox.setCurrentIndex(len(pathlist))
                            break
            else:
                pass
        else:
            self.load_select()

    def unpatching2(self, result, orig):
        PatchSize = np.array((40.0, 40.0))
        PatchOverlay = 0.5
        imglay = fUnpatch2D(result, PatchSize, PatchOverlay, orig.shape)
        return imglay

    def unpatching8(self, result, orig):
        PatchSize = np.array((40.0, 40.0))
        PatchOverlay = 0.5
        IndexArte = np.argmax(result, 1)
        Arte1, Arte2, Arte3 = UnpatchArte8(IndexArte, PatchSize, PatchOverlay, orig.shape)
        return Arte1, Arte2, Arte3

    def unpatching11(self, result, orig):
        PatchSize = np.array((40.0, 40.0))  ##
        PatchOverlay = 0.5

        IndexType = np.argmax(result, 1)
        IndexType[IndexType == 0] = 1
        IndexType[(IndexType > 1) & (IndexType < 4)] = 2
        IndexType[(IndexType > 6) & (IndexType < 9)] = 3
        IndexType[(IndexType > 3) & (IndexType < 7)] = 4
        IndexType[IndexType > 8] = 5

        from collections import Counter
        a = Counter(IndexType).most_common(1)
        domain = a[0][0]

        PType = np.delete(result, [1, 3, 5, 6, 8, 10], 1)  # only 5 region left
        PArte = np.delete(result, [0, 2, 4, 7, 9], 1)
        PArte[:, [4, 5]] = PArte[:, [5, 4]]
        PNew = np.concatenate((PType, PArte), axis=1)
        IndexArte = np.argmax(PNew, 1)

        Type = UnpatchType(IndexType, domain, PatchSize, PatchOverlay, orig.shape)
        Arte = UnpatchArte(IndexArte, PatchSize, PatchOverlay, orig.shape)
        return Type, Arte

    def load_patch(self, ind):

        self.loadPatch = True
        self.horizontalSlider.setDisabled(False)

        if ind == 1:
            dialog = QtWidgets.QFileDialog()
            options = dialog.Options()
            options |= QtWidgets.QFileDialog.Directory
            options |= QtWidgets.QFileDialog.ExistingFile
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            dialog.setOptions(options)
            self.resultfile = dialog.getOpenFileName(self, "choose the result file",
                                                     MARKING_PATH,
                                                     'mat files(*.mat);;h5 files(*.h5);;npy files(*.npy);;npz files('
                                                     '*.npz)')[0]
        elif ind == 2:
            self.resultfile = self.deepLearningArtApp.getResultWorkSpace()

        if self.resultfile is not None:
            if self.resultfile.upper().endswith('.MAT'):
                self.conten = sio.loadmat(self.resultfile)
            elif self.resultfile.upper().endswith(('.NPY', '.NPZ')):
                self.conten = np.load(self.resultfile)
            else:
                pass

            if self.conten is not None:
                if 'img' in self.conten:
                    self.PathDicom = self.resultfile
                    self.load_MR(4)
                elif 'img_path' in self.conten:
                    self.PathDicom = self.conten['img_path']
                    self.load_MR(4)
                else:
                    self.imgdialog = QtWidgets.QDialog()
                    self.imgdialog.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)
                    layout = QtWidgets.QVBoxLayout()
                    icon = QtGui.QIcon()
                    icon.addPixmap(QtGui.QPixmap(":/icons/Icons/folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                    self.openimg = QtWidgets.QComboBox()
                    self.openimg.addItem(icon, 'open image', 0)
                    self.openimg.setIconSize(QtCore.QSize(35, 35))
                    self.openimg.addItem('open a directory', 1)
                    self.openimg.addItem('open a file', 2)
                    self.openimg.addItem('load from workspace', 3)
                    self.openimg.setCurrentIndex(0)
                    self.openimg.activated.connect(self.load_MR)
                    layout.addWidget(self.openimg)
                    self.imgdialog.setLayout(layout)
                    self.imgdialog.setMinimumWidth(180)
                    self.imgdialog.show()
            else:
                QtWidgets.QMessageBox.information(self, 'Warning', 'This format is not supported!')
                return
        else:
            QtWidgets.QMessageBox.information(self, 'Warning', 'No result file!')
        self.openResult.setCurrentIndex(0)

    def load_patch_end(self):
        try:
            self.patch_color_df = pandas.read_csv('configGUI/patch_color.csv')
            if 'prob_pre' in self.conten:
                self.patch_color_df = self.patch_color_df.drop(self.patch_color_df.index[:])
                cnum = np.array(self.conten['prob_pre'])
                IType, IArte = self.unpatching11(self.conten['prob_pre'], list1[-1])
                # if IType[0] - list1[n][0] <= PatchSize/2 and IType[1] - list1[n][1] <= PatchSize/2:
                # else:
                #     QtWidgets.QMessageBox.information(self, 'Warning', 'Please choose the right file!')
                #     break
                problist.append([IType])
                hatchlist.append([IArte])
                count = 11
                cnrlist.append(count)
                default_patch_color = cmap3
                for c in range(len(cmap3)):
                    self.patch_color_df.loc[count, 'class'] = 'class' + ' %s' % c
                    self.patch_color_df.loc[count, 'color'] = default_patch_color[c]
                colormaplist.append(default_patch_color)

            elif 'prob_test' in self.conten:
                self.patch_color_df = self.patch_color_df.drop(self.patch_color_df.index[:])
                pred = self.conten['prob_test']
                pred = pred[0:4320, :]
                cnum = np.array(pred)
                if cnum.shape[1] == 2:
                    IType = self.unpatching2(pred, list1[-1])
                    problist.append([IType])
                    hatchlist.append(empty1)
                    count = 2
                    cnrlist.append(count)
                    default_patch_color = cmap1
                    for c in range(len(cmap1)):
                        self.patch_color_df.loc[count, 'class'] = 'class' + ' %s' % c
                        self.patch_color_df.loc[count, 'color'] = default_patch_color[c]
                    colormaplist.append(default_patch_color)
            else:
                self.patch_color_df = self.patch_color_df.drop(self.patch_color_df.index[:])
                mask_set = []
                class_set = []
                color_set = []
                if 'color' in self.conten:
                    default_patch_color = self.conten['color']
                    self.dcolors['classes']['colors'] = default_patch_color
                    with open('configGUI/colors0.json', 'w') as json_data:
                        json_data.write(json.dumps(self.dcolors))
                else:
                    default_patch_color = self.dcolors['classes']['colors']
                count = 0
                for item in self.conten:
                    if '__' not in item and 'readme' not in item and 'img' not in item and 'info' not in item:
                        if '_' or 'mask' in item:
                            mask = self.conten[item]
                            condition = np.sort(list(mask.shape)) == np.sort(self.NS)
                            if type(condition) is not bool:
                                condition = condition.all()
                            if condition:
                                mask_set.append(mask)
                                class_set.append(item)
                                color_set.append(default_patch_color[count])
                                self.patch_color_df.loc[count, 'class'] = item
                                self.patch_color_df.loc[count, 'color'] = default_patch_color[count]
                                count = count + 1
                hatchlist.append(mask_set)
                classlist.append(class_set)
                colormaplist.append(color_set)
                problist.append(empty1)
                cnrlist.append(count)
            self.patch_color_df.to_csv('configGUI/patch_color.csv', index=False)

            pnamelist.append(self.resultfile)
            if self.gridsnr == 3:  #############
                for i in range(self.layout3D):
                    view_lines = self.maingrids1.itemAtPosition(i, 0)
                    resultItems = [view_lines.refbox.itemText(item) for item in range(view_lines.refbox.count())]
                    if self.resultfile not in resultItems:
                        view_lines.addPathre(self.resultfile)
            else:
                for i in range(self.layoutlines):
                    for j in range(self.layoutcolumns):
                        view_group = self.maingrids1.itemAtPosition(i, j)
                        resultItems = [view_group.refbox.itemText(item) for item in range(view_group.refbox.count())]
                        if self.resultfile not in resultItems:
                            view_group.addPathre(self.resultfile)

            self.openResult.setCurrentIndex(0)
            self.loadPatch = False
        except:
            self.openResult.setCurrentIndex(0)
            self.loadPatch = False
            QMessageBox.information(self, 'Error', 'File is not supported!')
        try:
            self.openimg.close()
        except:
            pass

    def flip_image(self):
        # to avoid for confusion with rotate dimension of image, use flip here to rotate image clockwise 90 degree
        if self.vision == 2:
            for item in self.view_group_list:
                self.view_group = item
                self.canvasl = self.view_group.anewcanvas
                if self.canvasl is not None:
                    array = self.canvasl.get_image_array()
                    if array is not None:
                        array = np.rot90(array, axes=(-2, -1))
                        self.canvasl.set_image_array(array)
        elif self.vision == 3:
            for item in self.view_line_list:
                self.view_line = item
                self.canvas1 = self.view_line.newcanvas1
                if self.canvas1 is not None:
                    array = self.canvas1.get_image_array()
                    if array is not None:
                        array = np.rot90(array, axes=(-2, -1))
                        self.canvas1.set_image_array(array)
                self.canvas2 = self.view_line.newcanvas2
                if self.canvas2 is not None:
                    array = self.canvas2.get_image_array()
                    if array is not None:
                        array = np.rot90(array, axes=(-2, -1))
                        self.canvas2.set_image_array(array)
                self.canvas3 = self.view_line.newcanvas3
                if self.canvas3 is not None:
                    array = self.canvas3.get_image_array()
                    if array is not None:
                        array = np.rot90(array, axes=(-2, -1))
                        self.canvas3.set_image_array(array)

    def fit_image(self):
        if self.vision == 2:
            for item in self.view_group_list:
                self.view_group = item
                self.canvasl = self.view_group.anewcanvas
                if self.canvasl is not None:
                    aspect = self.canvasl.get_aspect()
                    aspect = 'auto' if aspect == 'equal' else 'equal'
                    self.canvasl.set_aspect(aspect)
        elif self.vision == 3:
            for item in self.view_line_list:
                self.view_line = item
                self.canvas1 = self.view_line.newcanvas1
                if self.canvas1 is not None:
                    aspect = self.canvas1.get_aspect()
                    aspect = 'auto' if aspect == 'equal' else 'equal'
                    self.canvas1.set_aspect(aspect)
                self.canvas2 = self.view_line.newcanvas2
                if self.canvas2 is not None:
                    aspect = self.canvas2.get_aspect()
                    aspect = 'auto' if aspect == 'equal' else 'equal'
                    self.canvas2.set_aspect(aspect)
                self.canvas3 = self.view_line.newcanvas3
                if self.canvas3 is not None:
                    aspect = self.canvas3.get_aspect()
                    aspect = 'auto' if aspect == 'equal' else 'equal'
                    self.canvas3.set_aspect(aspect)

    def set_color(self):

        if self.selectoron:
            self.predefined_label()
        else:
            c1, c3, h1, h2, v1, v3, cm, vs, ok = Patches_window.getData()
            if ok:
                global cmap1, cmap3, hmap1, hmap2, vtr1, vtr3, cmaps, vtrs, colormaplist
                cmap1 = c1
                cmap3 = c3
                hmap1 = h1
                hmap2 = h2
                vtr1 = v1
                vtr3 = v3
                cmaps = cm
                vtrs = vs
                if 'prob_pre' in self.conten:
                    colormaplist[self.view_group.refbox.currentIndex()-1] = cmap3
                elif 'prob_test' in self.conten:
                    colormaplist[self.view_group.refbox.currentIndex()-1] = cmap1
                else:
                    colormaplist[self.view_group.refbox.currentIndex()-1] = cmaps
                if self.vision == 2:
                    for item in self.view_group_list:
                        self.view_group = item
                        self.view_group.loadScene_result(self.view_group.refbox.currentIndex())
                        self.view_group.anewcanvas.set_color()
                elif self.vision == 3:
                    for item in self.view_line_list:
                        self.view_line = item
                        self.view_line.loadScene_result(self.view_line.refbox.currentIndex())


    def set_transparency(self, value):

        self.horizontalSlider.setToolTip('Transparency: %s' % value + '%')
        global vtrs, vtr1, vtr3
        vtrs = 1 - value / 100
        vtr1 = vtrs
        vtr3 = vtrs
        with open('configGUI/colors1.json', 'r') as json_data:
            self.colors = json.load(json_data)
            self.colors['class2']['trans'][0] = vtr1
            self.colors['class11']['trans'][0] = vtr3
            self.colors['classes']['trans'][0] = vtrs
        with open('configGUI/colors1.json', 'w') as json_data:
            json_data.write(json.dumps(self.colors))
        if self.vision == 2:
            for item in self.view_group_list:
                self.view_group = item
                if self.view_group.anewcanvas is not None:
                    self.view_group.anewcanvas.set_transparency(vtrs)
        elif self.vision == 3:
            for item in self.view_line_list:
                self.view_line = item
                if self.view_line.newcanvas1 is not None:
                    self.view_line.newcanvas1.set_transparency(vtrs)
                if self.view_line.newcanvas2 is not None:
                    self.view_line.newcanvas2.set_transparency(vtrs)
                if self.view_line.newcanvas3 is not None:
                    self.view_line.newcanvas3.set_transparency(vtrs)


    def newShape(self):
        if self.vision == 2:
            for item in self.view_group_list:
                self.view_group = item
                self.canvasl = self.view_group.anewcanvas
                if self.canvasl is not None:
                    if self.canvasl.get_open_dialog():

                        color = colors.to_hex('b', keep_alpha=True)

                        if not self.useDefaultLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
                            self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)
                            text = self.labelDialog.popUp(text=self.prevLabelText)
                            self.canvasl.set_open_dialog(False)
                            self.prevLabelText = text

                            if text is not None:
                                if text not in self.labelHist:
                                    self.labelHist.append(text)
                                else:
                                    pass

                                self.df = pandas.read_csv(self.labelfile)
                                df_size = pandas.DataFrame.count(self.df)
                                df_rows = df_size['labelshape'] - 1
                                self.number = df_rows
                                self.df.loc[df_rows, 'image'] = self.view_group.current_image()
                                self.df.loc[df_rows, 'labelname'] = self.prevLabelText
                                if self.labelDialog.getColor():
                                    color = self.labelDialog.getColor()
                                    self.df.loc[df_rows, 'labelcolor'] = color
                                else:
                                    self.df.loc[df_rows, 'labelcolor'] = color
                                self.df.to_csv(self.labelfile, index=False)

                        else:
                            text = self.defaultLabelTextLine.text()
                            self.prevLabelText = text
                            if text not in self.labelHist:
                                self.labelHist.append(text)
                            if text is not None:
                                self.df = pandas.read_csv(self.labelfile)
                                df_size = pandas.DataFrame.count(self.df)
                                df_rows = df_size['labelshape'] - 1
                                self.number = df_rows
                                self.df.loc[df_rows, 'image'] = self.view_group.current_image()
                                self.df.loc[df_rows, 'labelname'] = self.prevLabelText
                                dfcolor = pandas.read_csv('configGUI/predefined_label.csv')
                                if not dfcolor[dfcolor['label name'] == str(self.prevLabelText)].index.values.astype(
                                        int) == []:
                                    colorind = \
                                        dfcolor[dfcolor['label name'] == str(self.prevLabelText)].index.values.astype(int)[0]
                                    color = dfcolor.at[colorind, 'label color']
                                else:
                                    pass
                                self.df.loc[df_rows, 'labelcolor'] = color
                                self.df.to_csv(self.labelfile, index=False)

                        self.canvasl.set_facecolor(color)
                    self.updateList()

    def shapeSelectionChanged(self, selected):
        # selected shape changed --> selected item changes at the same time
        if self.vision == 2:
            for item in self.view_group_list:
                self.view_group = item
                self.canvasl = self.view_group.anewcanvas
                if self.canvasl is not None:
                    if selected:
                        for i in range(len(self.labelList.get_list())):
                            if self.labelList.get_list()[i][0] == str(self.canvasl.get_selected()):
                                selectedRow = i
                                if selectedRow is not None:
                                    self.selectedshape_name = self.labelList.get_list()[selectedRow][6]
                                    self.canvasl.set_labelon(True)
                                    self.canvasl.set_toolTip(self.selectedshape_name)
                                    self.labelList.selectRow(selectedRow)

    def selector_mode(self):

        if self.selectoron == False:
            self.selectoron = True
            icon2 = QtGui.QIcon()
            icon2.addPixmap(QtGui.QPixmap(":/icons/Icons/switchon.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
            self.bselectoron.setIcon(icon2)
            self.cursorCross.setDisabled(False)
            self.brectangle.setDisabled(False)
            self.bellipse.setDisabled(False)
            self.blasso.setDisabled(False)
            self.horizontalSlider.setDisabled(False)
            self.bnoselect.setDisabled(False)
            self.deleteButton.setDisabled(False)
            self.openlabelButton.setDisabled(False)
            self.labelListShow.setDisabled(False)
            self.useDefaultLabelCheckbox.setDisabled(False)
            self.defaultLabelTextLine.setDisabled(False)
            self.labellistBox.setDisabled(False)
            self.add_labelFile()
            self.bdswitch.setDisabled(True)
            self.openImage.setDisabled(True)
            self.openResult.setDisabled(True)
            self.view_group.setlinkoff(True)
            self.actionSave.setEnabled(True)
            self.actionLoad.setEnabled(False)
            self.save_file()
            self.load_markings(self.ind)
            self.show_labelList()

        else:
            self.selectoron = False
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/icons/Icons/switchoff.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.bselectoron.setIcon(icon1)
            self.cursorCross.setDisabled(True)
            self.brectangle.setDisabled(True)
            self.bellipse.setDisabled(True)
            self.blasso.setDisabled(True)
            self.horizontalSlider.setDisabled(True)
            self.bnoselect.setDisabled(True)
            self.deleteButton.setDisabled(True)
            self.openlabelButton.setDisabled(True)
            self.labelListShow.setDisabled(True)
            self.viewallButton.setDisabled(True)
            self.useDefaultLabelCheckbox.setDisabled(True)
            self.defaultLabelTextLine.setDisabled(True)
            self.labellistBox.setDisabled(True)
            self.bdswitch.setDisabled(False)
            self.openImage.setDisabled(False)
            self.openResult.setDisabled(False)
            self.view_group.setlinkoff(False)
            self.actionSave.setEnabled(True)
            self.actionLoad.setEnabled(True)
            self.save_file()
            self.clear_markings()
            self.labelListView.close()

    def stop_view(self, n):
        if self.vision == 2:
            for item in self.view_group_list:
                self.view_group = item
                self.view_group.Viewpanel.stopMove(n)

    def show_variables(self):
        dirlist = list(dir(self))
        globalslist = list(globals())
        localslist = list(locals())
        d = {"dir": dirlist, "globals": globalslist, "locals": localslist}
        with open('configGUI/variables.json', 'w') as json_data:
            json_data.write(json.dumps(d))
        self.info_popup()

    def info_popup(self):
        self.info_popup_window = QtWidgets.QMdiSubWindow()
        self.info_popup_window.setGeometry(500, 250, 300, 750)
        self.info_popup_window.setWindowTitle('show variables')
        self.info_popup_window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        infobox = QtWidgets.QComboBox()
        infobox.addItem('Please Select', 0)
        infobox.addItem('dir', 1)
        infobox.addItem('globals', 2)
        infobox.addItem('locals', 3)
        widget = QtWidgets.QWidget()
        widget.setLayout(QtWidgets.QVBoxLayout())
        widget.layout().addWidget(infobox)
        self.info_popup_window.setWidget(widget)
        infobox.activated.connect(self.show_infolist)
        infobox.setCurrentIndex(0)
        self.info_popup_window.show()

    def show_infolist(self, index):
        try:
            self.info_popup_window.widget().layout().removeItem(1)
            self.info_popup_window.update()
        except:
            pass

        with open('configGUI/variables.json', 'r') as json_data:
            d = json.load(json_data)
        lw = QtWidgets.QListWidget()
        if index == 1:
            lw.addItems(list(d["dir"]))
        elif index == 2:
            lw.addItems(list(d["globals"]))
        elif index == 3:
            lw.addItems(list(d["locals"]))
        old = self.info_popup_window.widget().layout().itemAt(1)
        if old is None:
            self.info_popup_window.widget().layout().addWidget(lw)
        else:
            self.info_popup_window.widget().layout().replaceWidget(old.widget(), lw)

    def edit_label(self, ind):

        if self.vision == 2:
            for item in self.view_group_list:
                self.view_group = item
                self.canvasl = self.view_group.anewcanvas
                if self.canvasl is not None:
                    self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)
                    text = self.labelDialog.popUp(text=self.prevLabelText)
                    self.canvasl.set_open_dialog(False)
                    self.prevLabelText = text

                    if text is not None:
                        if text not in self.labelHist:
                            self.labelHist.append(text)
                        else:
                            pass
                        self.selectind = ind.row()
                        self.number = self.selectind
                        self.df = pandas.read_csv(self.labelfile)
                        self.df.loc[self.selectind, 'labelname'] = self.prevLabelText
                        if self.labelDialog.getColor():
                            color = self.labelDialog.getColor()
                            self.df.loc[self.selectind, 'labelcolor'] = color
                        else:
                            color = self.df.loc[self.selectind, 'labelcolor']
                        self.df.to_csv(self.labelfile, index=False)
                        self.canvasl.set_facecolor(color)

                    self.updateList()

    def show_label_name(self, index):
        if self.vision == 2:
            if index == 1:
                for item in self.view_group_list:
                    self.view_group = item
                    self.canvasl = self.view_group.anewcanvas
                    if self.canvasl is not None:
                        if self.selectoron or self.view_group.mode == 2:
                            self.labelon = not self.labelon
                            self.canvasl.set_labelon(self.labelon)
            elif index == 2:
                for item in self.view_group_list:
                    self.view_group = item
                    self.canvasl = self.view_group.anewcanvas
                    if self.canvasl is not None:
                        if self.selectoron or self.view_group.mode == 2:
                            self.legendon = not self.legendon
                            self.canvasl.set_legendon(self.legendon)
            self.labelButton.setCurrentIndex(0)

    def show_labelList(self):

        self.label_mode = True
        self.labelList = LabelTable(self)
        self.labelList.set_table_model(labelfile=self.labelfile, imagefile=self.view_group.current_image())
        self.labelList.setSelectionMode(QAbstractItemView.SingleSelection)
        self.labelList.setSelectionBehavior(QAbstractItemView.SelectRows)
        # bind cell click to a method reference
        self.labelList.clicked.connect(self.labelList.get_selectRow)
        self.labelList.doubleClicked.connect(self.edit_label)
        self.labelList.setSortingEnabled(True)
        self.labelList.resizeRowsToContents()

        self.labelListView = QMdiSubWindow()
        self.labelListView.setWindowTitle('Label List')
        self.labelListView.setGeometry(300, 250, 400, 200)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.labelListView.setSizePolicy(sizePolicy)
        self.labelListView.setWidget(self.labelList)
        self.labelListView.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.labelListView.update()
        self.labelListView.show()
        self.labelList.statusChanged.connect(self.label_on_off)
        self.labelList.selectChanged.connect(self.labelItemChanged)

    def predefined_label(self):
        subprocess.Popen(["python", 'configGUI/predefined_label.py'])

    def select_roi_OnOff(self):

        if self.gridson:
            with open('configGUI/lastWorkspace.json', 'r') as json_data:
                lastState = json.load(json_data)
                lastState['mode'] = self.gridsnr  ###
                if self.gridsnr == 3:
                    lastState["Dim"][0] = self.vision
                    lastState['layout'][0] = self.layout3D
                else:
                    lastState["Dim"][0] = self.vision
                    lastState['layout'][0] = self.layoutlines
                    lastState['layout'][1] = self.layoutcolumns

                global pathlist, list1, pnamelist, problist, hatchlist, imagenum, resultnum, \
                    cnrlist, shapelist, indlist, ind2list, ind3list
                # shapelist = (list(shapelist)).tolist()
                # shapelist = pd.Series(shapelist).to_json(orient='values')
                lastState['Shape'] = shapelist
                lastState['Pathes'] = pathlist
                lastState['NResults'] = pnamelist
                lastState['NrClass'] = cnrlist
                lastState['ImgNum'] = imagenum
                lastState['ResNum'] = resultnum
                lastState['Index'] = indlist
                lastState['Index2'] = ind2list
                lastState['Index3'] = ind3list

            with open('configGUI/lastWorkspace.json', 'w') as json_data:
                json_data.write(json.dumps(lastState))

            listA = open('config/dump1.txt', 'wb')
            pickle.dump(list1, listA)
            listA.close()
            listB = open('config/dump2.txt', 'wb')
            pickle.dump(problist, listB)
            listB.close()
            listC = open('config/dump3.txt', 'wb')
            pickle.dump(hatchlist, listC)
            listC.close()

        subprocess.Popen(["python", 'configGUI/ROI_Selector.py'])

    def predefined_networks(self):
        subprocess.Popen(["python", 'configGUI/predefined_networks.py'])

    def save_label(self):
        list = np.array(self.labelList.get_list())
        column = ['artist', 'labelshape', 'slice', 'path', 'status', 'image', 'labelname', 'labelcolor']
        df = pandas.DataFrame(data=list, index=None, columns=column)
        try:
            save_dialog = QtWidgets.QFileDialog()
            save_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            file_path = save_dialog.getSaveFileName(self, 'Save as... File', './',
                                                    filter='All Files(*.*);; csv Files(*.csv)')

            if file_path[0]:
                self.file_path = file_path
                file_open = open(self.file_path[0], 'w')
                self.file_name = (self.file_path[0].split('/'))[-1] + '.csv'
                self.setWindowTitle(
                    "{} - Please save every label records in a individual folder".format(self.file_name))
                df.to_csv(self.file_path[0] + '.csv')

        except FileNotFoundError as why:
            self.error_box(why)
            pass

    def delete_labelItem(self):
        self.selectind = self.labelList.get_table_model().get_selectind()
        self.number = self.selectind
        self.df = pandas.read_csv(self.labelfile)
        self.df = self.df.drop(self.df.index[self.selectind])
        self.df.to_csv('Markings/marking_records.csv', index=False)
        self.updateList()
        self.delete_labelShape()

    def delete_labelShape(self):

        self.clear_markings()
        self.load_markings(self.view_group.current_slice())

    def open_labelFile(self):

        try:
            path = 'Markings'
            dialog = QtWidgets.QFileDialog()
            path = dialog.getExistingDirectory(self, "open file", path)

            self.labelfile = path + '/' + os.listdir(path)[0]
            if self.labelfile:
                self.add_labelFile()
                self.view_group.modechange.setDisabled(False)
                self.labellistBox.currentTextChanged.connect(self.update_labelFile)
            else:
                pass
        except IOError:
            QtWidgets.QMessageBox.information(self, 'Warning', 'Please open a label file!')

    def add_labelFile(self):

        imageItems = [self.labellistBox.itemText(i) for i in range(self.labellistBox.count())]
        region = os.path.split(self.labelfile)
        proband = os.path.split(os.path.split(region[0])[0])[1]
        region = region[1]
        newItem = '[' + (proband) + '][' + (region) + ']'
        if newItem not in imageItems:
            self.labellistBox.addItem(newItem)

    def update_labelFile(self):
        if not self.labellistBox.currentText() == "Label List":
            self.updateList()
            self.clear_markings()
            self.load_markings(self.view_group.current_slice())

    def load_select(self):  # from load_end
        self.vision = 2
        self.clear_all()
        self.gridsnr = 2
        self.view_group_list = []
        if not self.labelon:
            self.layoutlines = 1
            self.layoutcolumns = 1
        for i in range(self.layoutlines):
            for j in range(self.layoutcolumns):
                blocklayout = Viewgroup()
                self.view_group = blocklayout
                self.view_group_list.append(self.view_group)
                self.view_group.setlinkoff(True)
                for dpath in pathlist:
                    blocklayout.addPathd(dpath, 1)
                for cpath in pnamelist:
                    blocklayout.addPathre(cpath)
                self.maingrids1.addLayout(blocklayout, i, j)
        if pathlist:
            n = 0
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if n < len(pathlist):
                        self.maingrids1.itemAtPosition(i, j).pathbox.setCurrentIndex(n + 1)
                        n += 1
                    else:
                        break
        if self.vision == 2:
            for item in self.view_group_list:
                self.view_group = item
                if self.view_group.anewcanvas is not None:
                    self.updateList()
                    self.load_markings(self.view_group.current_slice())
                    self.view_group.rotateSignal.connect(self.load_markings)
                    self.view_group.scrollSignal.connect(self.load_markings)
                    self.view_group.rotateSignal.connect(self.changeSelector)
                    self.view_group.scrollSignal.connect(self.changeSelector)
                    self.view_group.zoomSignal.connect(self.changeSelector)

    def changeSelector(self, val):
        self.bnoselect.setChecked(True)
        self.brectangle.setChecked(False)
        self.bellipse.setChecked(False)
        self.blasso.setChecked(False)

    def load_markings(self, slice):

        df = pandas.read_csv(self.labelfile)
        num1 = df[df['image'] == self.view_group.current_image()].index.values.astype(int)
        num2 = df[df['slice'] == slice].index.values.astype(int)
        num = list(set(num1).intersection(num2))
        self.markings = []
        if self.vision == 2:
            for item in self.view_group_list:
                self.view_group = item
                if self.view_group.anewcanvas is not None:
                    for i in range(0, len(num)):
                        status = df['status'][num[i]]
                        df.select_dtypes(include='object')
                        if df['labelshape'][num[i]] == 'lasso':
                            path = Path(np.asarray(eval(df['path'][num[i]])))
                            newItem = PathPatch(path, fill=True, alpha=.2, edgecolor=None)
                        else:
                            newItem = eval(df['artist'][num[i]])
                        color = df['labelcolor'][num[i]]
                        self.markings.append(newItem)
                        if status == 0:
                            newItem.set_visible(True)
                            if type(newItem) is Rectangle or Ellipse:
                                newItem.set_picker(True)
                            else:
                                newItem.set_picker(False)
                            newItem.set_facecolor(color)
                            newItem.set_alpha(0.5)
                        else:
                            newItem.set_visible(False)
                        self.view_group.anewcanvas.ax1.add_artist(newItem)

                    self.view_group.anewcanvas.blit(self.view_group.anewcanvas.ax1.bbox)

    def label_on_off(self, status):

        for item in self.markings:
            item.remove()
        self.newcanvasview()
        if self.vision == 2:
            for item in self.view_group_list:
                self.view_group = item
                self.canvasl = self.view_group.anewcanvas
                if self.canvasl is not None:
                    self.canvasl.draw_idle()
                    df = pandas.read_csv(self.labelfile)
                    num1 = df[df['image'] == self.view_group.current_image()].index.values.astype(int)
                    num2 = df[df['slice'] == self.view_group.current_slice()].index.values.astype(int)
                    num = list(set(num1).intersection(num2))
                    self.markings = []

                    for i in range(0, len(num)):
                        status = df['status'][num[i]]
                        df.select_dtypes(include='object')
                        if df['labelshape'][num[i]] == 'lasso':
                            path = Path(np.asarray(eval(df['path'][num[i]])))
                            newItem = PathPatch(path, fill=True, alpha=.2, edgecolor=None)
                        else:
                            newItem = eval(df['artist'][num[i]])
                        color = df['labelcolor'][num[i]]
                        self.markings.append(newItem)

                        if not status:
                            newItem.set_visible(True)
                            if type(newItem) is Rectangle or Ellipse:
                                newItem.set_picker(True)
                            else:
                                newItem.set_picker(False)
                            newItem.set_facecolor(color)
                            newItem.set_alpha(0.5)
                        else:
                            newItem.set_visible(False)

                        self.canvasl.ax1.add_artist(newItem)

                    self.canvasl.blit(self.canvasl.ax1.bbox)

    def clear_markings(self):

        self.load_old()

    def marking_shape(self, n):
        # for labeling
        if self.vision == 2:
            for item in self.view_group_list:
                self.view_group = item
                self.canvasl = self.view_group.anewcanvas
                if self.canvasl is not None:
                    state = self.cursorCross.isChecked()

                    if self.selectoron:

                        if n == 0:
                            self.canvasl.set_state(2)
                            self.canvasl.rec_toggle_selector_off()
                            self.canvasl.ell_toggle_selector_off()
                            self.canvasl.lasso_toggle_selector_off()

                        elif n == 1:
                            self.canvasl.set_state(1)
                            self.canvasl.rec_toggle_selector_on()
                            self.canvasl.set_cursor(state)
                            self.canvasl.ell_toggle_selector_off()
                            self.canvasl.lasso_toggle_selector_off()

                        elif n == 2:
                            self.canvasl.set_state(1)
                            self.canvasl.rec_toggle_selector_off()
                            self.canvasl.ell_toggle_selector_on()
                            self.canvasl.set_cursor(state)
                            self.canvasl.lasso_toggle_selector_off()

                        elif n == 3:
                            self.canvasl.set_state(1)
                            self.canvasl.rec_toggle_selector_off()
                            self.canvasl.ell_toggle_selector_off()
                            self.canvasl.lasso_toggle_selector_on()
                            self.canvasl.set_cursor(state)

                    else:
                        pass

                    self.canvasl.newShape.connect(self.newShape)
                    self.canvasl.deleteEvent.connect(self.updateList)
                    self.canvasl.selectionChanged.connect(self.shapeSelectionChanged)

    def handle_crossline(self):
        # cursor inspector for labeling
        if self.vision == 2:
            for item in self.view_group_list:
                self.view_group = item
                self.canvasl = self.view_group.anewcanvas
                if self.canvasl is not None:
                    state = self.cursorCross.isChecked()
                    self.canvasl.set_cursor(state)

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)

    def labelItemChanged(self, item):
        # selected label item changed, selected shape changes at the same time
        for vg in self.view_group_list:
            self.view_group = vg
            self.canvasl = self.view_group.anewcanvas
            self.canvasl = self.view_group.anewcanvas
            shapeitem = self.labelList.get_list()[item]
            if not shapeitem[1] == 'lasso':
                shape = eval(shapeitem[0])
                self.canvasl.set_selected(shape)

    def updateList(self):
        self.labelList.set_table_model(self.labelfile, self.view_group.current_image())

    def newonscroll(self, event):

        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        if self.ind >= self.slices:
            self.ind = 0
        if self.ind <= -1:
            self.ind = self.slices - 1

        self.emitlist[0] = self.ind
        self.update_data.emit(self.emitlist)
        self.newcanvasview()

    def newonscrol2(self, event):

        if event.button == 'up':
            self.ind2 = (self.ind2 + 1) % self.slices2
        else:
            self.ind2 = (self.ind2 - 1) % self.slices2
        if self.ind2 >= self.slices2:
            self.ind2 = 0
        if self.ind2 <= -1:
            self.ind2 = self.slices2 - 1

        self.emitlist2[0] = self.ind2
        self.update_data2.emit(self.emitlist2)
        self.newcanvasview()

    def newonscrol3(self, event):

        if event.button == 'up':
            self.ind3 = (self.ind3 + 1) % self.slices3
        else:
            self.ind3 = (self.ind3 - 1) % self.slices3
        if self.ind3 >= self.slices3:
            self.ind3 = 0
        if self.ind3 <= -1:
            self.ind3 = self.slices3 - 1
        #
        self.emitlist3[0] = self.ind3
        self.update_data3.emit(self.emitlist3)
        self.newcanvasview()

    def newcanvasview(self):  # refreshing

        self.newax.clear()
        self.newax2.clear()
        self.newax3.clear()

        try:
            img = np.swapaxes(self.mrinmain[0, 0, :, :, self.ind], 0, 1)
            self.pltc = self.newax.imshow(img, cmap='gray',
                                          extent=[0, img.shape[1], img.shape[0], 0], interpolation='sinc')
            img = np.swapaxes(self.mrinmain[0, 0, self.ind2, :, :], 0, 1)
            self.pltc2 = self.newax2.imshow(img, cmap='gray',
                                            extent=[0, img.shape[1], img.shape[0], 0], interpolation='sinc')
            img = np.swapaxes(self.mrinmain[0, 0, :, self.ind3, :], 0, 1)
            self.pltc3 = self.newax3.imshow(img, cmap='gray',
                                            extent=[0, img.shape[1], img.shape[0], 0], interpolation='sinc')
        except:
            img = np.swapaxes(self.mrinmain[:, :, self.ind], 0, 1)
            self.pltc = self.newax.imshow(img, cmap='gray',
                                          extent=[0, img.shape[1], img.shape[0], 0], interpolation='sinc')
            img = np.swapaxes(self.mrinmain[self.ind2, :, :], 0, 1)
            self.pltc2 = self.newax2.imshow(img, cmap='gray',
                                            extent=[0, img.shape[1], img.shape[0], 0], interpolation='sinc')
            img = np.swapaxes(self.mrinmain[:, self.ind3, :], 0, 1)
            self.pltc3 = self.newax3.imshow(img, cmap='gray',
                                            extent=[0, img.shape[1], img.shape[0], 0], interpolation='sinc')

        sepkey = os.path.split(self.selectorPath)
        sepkey = sepkey[1]  # t1_tse_tra_Kopf_Motion_0003

        self.newcanvas.draw()  # not self.newcanvas.show()
        self.newcanvas2.draw()
        self.newcanvas3.draw()

        v_min, v_max = self.pltc.get_clim()
        self.graylist[0] = v_min
        self.graylist[1] = v_max
        self.new_page.emit()

        v_min, v_max = self.pltc2.get_clim()
        self.graylist2[0] = v_min
        self.graylist2[1] = v_max
        self.new_page2.emit()

        v_min, v_max = self.pltc3.get_clim()
        self.graylist3[0] = v_min
        self.graylist3[1] = v_max
        self.new_page3.emit()

    def deleteLabel(self):
        pass

    def mouse_clicked(self, event):

        if event.button == 2:
            self.x_clicked = event.x
            self.y_clicked = event.y
            self.mouse_second_clicked = True

    def mouse_clicked2(self, event):
        if event.button == 2:
            self.x_clicked = event.x
            self.y_clicked = event.y
            self.mouse_second_clicked = True

    def mouse_clicked3(self, event):
        if event.button == 2:
            self.x_clicked = event.x
            self.y_clicked = event.y
            self.mouse_second_clicked = True
            # todo

    def mouse_move(self, event):

        if self.mouse_second_clicked:
            factor = 10
            __x = event.x - self.x_clicked
            __y = event.y - self.y_clicked
            v_min, v_max = self.pltc.get_clim()
            if __x >= 0 and __y >= 0:
                __vmin = np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y >= 0:
                __vmin = -np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y < 0:
                __vmin = -np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor - np.abs(__y) * factor
            else:
                __vmin = np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = np.abs(__x) * factor - np.abs(__y) * factor

            if (float(__vmin - __vmax)) / (v_max - v_min + 0.001) > 1:
                nmb = (float(__vmin - __vmax)) / (v_max - v_min + 0.001) + 1
                __vmin = (float(__vmin - __vmax)) / nmb * (__vmin / (__vmin - __vmax))
                __vmax = (float(__vmin - __vmax)) / nmb * (__vmax / (__vmin - __vmax))

            v_min += __vmin
            v_max += __vmax
            if v_min < v_max:
                self.pltc.set_clim(vmin=v_min, vmax=v_max)
                self.graylist[0] = v_min.round(2)
                self.graylist[1] = v_max.round(2)
                self.gray_data.emit(self.graylist)

                self.newcanvas.draw_idle()
            else:
                v_min -= __vmin
                v_max -= __vmax

    def mouse_move2(self, event):

        if self.mouse_second_clicked:
            factor = 10
            __x = event.x - self.x_clicked
            __y = event.y - self.y_clicked
            v_min, v_max = self.pltc2.get_clim()
            if __x >= 0 and __y >= 0:
                __vmin = np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y >= 0:
                __vmin = -np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y < 0:
                __vmin = -np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor - np.abs(__y) * factor
            else:
                __vmin = np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = np.abs(__x) * factor - np.abs(__y) * factor

            if (float(__vmin - __vmax)) / (v_max - v_min + 0.001) > 1:
                nmb = (float(__vmin - __vmax)) / (v_max - v_min + 0.001) + 1
                __vmin = (float(__vmin - __vmax)) / nmb * (__vmin / (__vmin - __vmax))
                __vmax = (float(__vmin - __vmax)) / nmb * (__vmax / (__vmin - __vmax))

            v_min += __vmin
            v_max += __vmax
            if v_min < v_max:
                self.pltc2.set_clim(vmin=v_min, vmax=v_max)
                self.graylist2[0] = v_min.round(2)
                self.graylist2[1] = v_max.round(2)
                self.gray_data2.emit(self.graylist2)

                self.newcanvas2.draw_idle()
            else:
                v_min -= __vmin
                v_max -= __vmax

    def mouse_move3(self, event):

        if self.mouse_second_clicked:
            factor = 10
            __x = event.x - self.x_clicked
            __y = event.y - self.y_clicked
            v_min, v_max = self.pltc3.get_clim()
            if __x >= 0 and __y >= 0:
                __vmin = np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y >= 0:
                __vmin = -np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y < 0:
                __vmin = -np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor - np.abs(__y) * factor
            else:
                __vmin = np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = np.abs(__x) * factor - np.abs(__y) * factor

            if (float(__vmin - __vmax)) / (v_max - v_min + 0.001) > 1:
                nmb = (float(__vmin - __vmax)) / (v_max - v_min + 0.001) + 1
                __vmin = (float(__vmin - __vmax)) / nmb * (__vmin / (__vmin - __vmax))
                __vmax = (float(__vmin - __vmax)) / nmb * (__vmax / (__vmin - __vmax))

            v_min += __vmin
            v_max += __vmax
            if v_min < v_max:
                self.pltc3.set_clim(vmin=v_min, vmax=v_max)
                self.graylist3[0] = v_min.round(2)
                self.graylist3[1] = v_max.round(2)
                self.gray_data3.emit(self.graylist3)

                self.newcanvas3.draw_idle()
            else:
                v_min -= __vmin
                v_max -= __vmax

    def mouse_release(self, event):
        if event.button == 2:
            self.mouse_second_clicked = False

    def mouse_release2(self, event):
        if event.button == 2:
            self.mouse_second_clicked = False

    def mouse_release3(self, event):
        if event.button == 2:
            self.mouse_second_clicked = False

    def mouse_tracking(self):
        self.cursor_on = self.inspectorButton.isChecked()
        if self.vision == 2:
            for item in self.view_group_list:
                self.view_group = item
                if self.view_group.anewcanvas is not None:
                    self.view_group.mouse_tracking(self.cursor_on)
        elif self.vision == 3:
            for item in self.view_line_list:
                if self.view_line.newcanvas1 is not None:
                    self.view_line = item
                    self.view_line.mouse_tracking(self.cursor_on)

    def newSliceview(self):
        self.graylabel.setText('G %s' % (self.graylist))

    def updateSlices(self, elist):
        self.slicelabel.setText(
            'S %s' % (elist[0] + 1) + '/ %s' % (elist[1]) + "       " + 'T %s' % (elist[2] + 1) + '/ %s' % (
                elist[3]) + "       " + 'D %s' % (elist[4] + 1) + '/ %s' % (elist[5]))
        indlist.append(elist[0])

    def updateGray(self, elist):
        self.graylabel.setText('G %s' % (elist))

    def updateZoom(self, factor):
        self.zoomlabel.setText('XY %s' % (factor))

    def setGreymain(self):
        maxv, minv, ok = grey_window.getData()
        if ok:
            self.pltc.set_clim(vmin=minv, vmax=maxv)
            self.pltc2.set_clim(vmin=minv, vmax=maxv)
            self.pltc3.set_clim(vmin=minv, vmax=maxv)
            self.graylist[0] = minv
            self.graylist[1] = maxv
            self.gray_data.emit(self.graylist)
            self.newcanvas.draw_idle()
            self.gray_data2.emit(self.graylist)
            self.newcanvas2.draw_idle()
            self.gray_data3.emit(self.graylist)
            self.newcanvas3.draw_idle()

    def newSliceview2(self):

        self.graylabel2.setText('G %s' % (self.graylist2))

    def updateSlices2(self, elist):
        self.slicelabel2.setText(
            'S %s' % (elist[0] + 1) + '/ %s' % (elist[1]) + "       " + 'T %s' % (elist[2] + 1) + '/ %s' % (
                elist[3]) + "       " + 'D %s' % (elist[4] + 1) + '/ %s' % (elist[5]))

    def updateGray2(self, elist):
        self.graylabel2.setText('G %s' % (elist))

    def updateZoom2(self, factor):
        self.zoomlabel2.setText('YZ %s' % (factor))

    def setGreymain2(self):
        maxv, minv, ok = grey_window.getData()
        if ok:
            self.pltc.set_clim(vmin=minv, vmax=maxv)
            self.pltc2.set_clim(vmin=minv, vmax=maxv)
            self.pltc3.set_clim(vmin=minv, vmax=maxv)
            self.graylist2[0] = minv
            self.graylist2[1] = maxv
            self.gray_data.emit(self.graylist)
            self.newcanvas.draw_idle()
            self.gray_data2.emit(self.graylist)
            self.newcanvas2.draw_idle()
            self.gray_data3.emit(self.graylist)
            self.newcanvas3.draw_idle()

    def newSliceview3(self):

        self.graylabel3.setText('G %s' % (self.graylist3))

    def updateSlices3(self, elist):
        self.slicelabel3.setText(
            'S %s' % (elist[0] + 1) + '/ %s' % (elist[1]) + "       " + 'T %s' % (elist[2] + 1) + '/ %s' % (
                elist[3]) + "       " + 'D %s' % (elist[4] + 1) + '/ %s' % (elist[5]))

    def updateGray3(self, elist):
        self.graylabel3.setText('G %s' % (elist))

    def updateZoom3(self, factor):
        self.zoomlabel3.setText('XZ %s' % (factor))

    def setGreymain3(self):
        maxv, minv, ok = grey_window.getData()
        if ok:
            self.pltc.set_clim(vmin=minv, vmax=maxv)
            self.pltc2.set_clim(vmin=minv, vmax=maxv)
            self.pltc3.set_clim(vmin=minv, vmax=maxv)
            self.graylist3[0] = minv
            self.graylist3[1] = maxv
            self.gray_data.emit(self.graylist)
            self.newcanvas.draw_idle()
            self.gray_data2.emit(self.graylist)
            self.newcanvas2.draw_idle()
            self.gray_data3.emit(self.graylist)
            self.newcanvas3.draw_idle()

    def closeEvent(self, QCloseEvent):
        reply = QtWidgets.QMessageBox.question(self, 'Warning', 'Are you sure to exit?',
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                self.screenGrids.close()
                self.labelListView.close()
                self.info_popup_window.close()
                self.datasets_window.close()
                self.network_interface.window.close()
                self.imgdialog.close()
            except:
                pass
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()

    def handleItemEntered(self, item):
        item.setBackground(QColor('moccasin'))
        try:
            self.tableWidget.setToolTip(str(self.row) + 'x' + str(self.col))
        except:
            self.tableWidget.setToolTip(str(1) + 'x' + str(1))

    def handleItemExited(self, item):
        item.setBackground(QTableWidgetItem().background())
        self.row = self.tableWidget.currentRow() + 1
        self.col = self.tableWidget.currentColumn() + 1

    def handleItemCount(self, e):
        self.tableWidget.setToolTip(str(self.row) + 'x' + str(self.col))
        count = 0
        pos = [[-1, -1]]
        pos.append([0, 0])
        for i in range(8):
            for j in range(6):
                elementitem = self.tableWidget.item(i, j)
                if elementitem.isSelected():
                    pos.append([i, j])
                    count = count + 1
        self.layoutlines = pos[-1][0] - pos[0][0]
        self.layoutcolumns = pos[-1][1] - pos[0][1]
        if count > 1 or count == 1 and pos[-1] == [0, 0]:
            self.set_layout()

            ###############################################################################################################################################################################

    ####### network training

    def updateProgressBarTraining(self, val):
        if val >= 0 and val <= 100:
            self.progressBar_Training.setValue(val)

    def plotTrainingLivePerformance(self, train_acc, val_acc, train_loss, val_loss):
        epochs = np.arange(1, len(train_acc) + 1)

        self.training_live_performance_figure.clear()
        ax1 = self.training_live_performance_figure.add_subplot(211)
        ax1.clear()
        ax1.plot(epochs, train_acc, 'r')
        ax1.plot(epochs, val_acc, 'b')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Dice Coefficient')
        ax1.grid(b=True, which='both')
        ax1.legend(['Training Accuracy', 'Validation Accuracy'])
        ax1.set_xlim(1, self.deepLearningArtApp.getEpochs())

        ax2 = self.training_live_performance_figure.add_subplot(212)
        ax2.clear()
        ax2.plot(epochs, train_loss, 'r')
        ax2.plot(epochs, val_loss, 'b')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.grid(b=True, which='both')
        ax2.legend(['Training loss', 'Validation loss'])
        ax2.set_xlim(1, self.deepLearningArtApp.getEpochs())

        self.canvas_training_live_performance_figure.draw()

        QtWidgets.QApplication.processEvents()
        # QTimer.singleShot(0, lambda: self.update)

    def button_train_clicked(self):
        # set gpu
        gpuId = self.ComboBox_GPU.currentIndex()
        self.deepLearningArtApp.setGPUId(gpuId)
        self.deepLearningArtApp.setGPUPredictionId(gpuId)

        # set epochs
        self.deepLearningArtApp.setEpochs(self.SpinBox_Epochs.value())

        # handle check states of check boxes for used classes
        self.deepLearningArtApp.setUsingArtifacts(self.CheckBox_Artifacts.isChecked())
        self.deepLearningArtApp.setUsingBodyRegions(self.CheckBox_BodyRegion.isChecked())
        self.deepLearningArtApp.setUsingTWeighting(self.CheckBox_TWeighting.isChecked())

        # set learning rates and batch sizes
        try:
            batchSizes = np.fromstring(self.LineEdit_BatchSizes.text(), dtype=np.int, sep=',')
            self.deepLearningArtApp.setBatchSizes(batchSizes)
            learningRates = np.fromstring(self.LineEdit_LearningRates.text(), dtype=np.float32, sep=',')
            self.deepLearningArtApp.setLearningRates(learningRates)
        except:
            raise ValueError(
                "Wrong input format of learning rates! Enter values separated by ','. For example: 0.1,0.01,0.001")

        # set optimizer
        selectedOptimizer = self.ComboBox_Optimizers.currentText()
        if selectedOptimizer == "SGD":
            self.deepLearningArtApp.setOptimizer(SGD_OPTIMIZER)
        elif selectedOptimizer == "RMSprop":
            self.deepLearningArtApp.setOptimizer(RMS_PROP_OPTIMIZER)
        elif selectedOptimizer == "Adagrad":
            self.deepLearningArtApp.setOptimizer(ADAGRAD_OPTIMIZER)
        elif selectedOptimizer == "Adadelta":
            self.deepLearningArtApp.setOptimizer(ADADELTA_OPTIMIZER)
        elif selectedOptimizer == "Adam":
            self.deepLearningArtApp.setOptimizer(ADAM_OPTIMIZER)
        else:
            raise ValueError("Unknown Optimizer!")

        # set weigth decay
        self.deepLearningArtApp.setWeightDecay(float(self.DoubleSpinBox_WeightDecay.value()))
        # set momentum
        self.deepLearningArtApp.setMomentum(float(self.DoubleSpinBox_Momentum.value()))
        # set nesterov enabled
        if self.CheckBox_Nesterov.checkState() == QtCore.Qt.Checked:
            self.deepLearningArtApp.setNesterovEnabled(True)
        else:
            self.deepLearningArtApp.setNesterovEnabled(False)

        # handle data augmentation
        if self.CheckBox_DataAugmentation.checkState() == QtCore.Qt.Checked:
            self.deepLearningArtApp.setDataAugmentationEnabled(True)
            # get all checked data augmentation options
            if self.CheckBox_DataAug_horizontalFlip.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setHorizontalFlip(True)
            else:
                self.deepLearningArtApp.setHorizontalFlip(False)

            if self.CheckBox_DataAug_verticalFlip.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setVerticalFlip(True)
            else:
                self.deepLearningArtApp.setVerticalFlip(False)

            if self.CheckBox_DataAug_Rotation.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setRotation(True)
            else:
                self.deepLearningArtApp.setRotation(False)

            if self.CheckBox_DataAug_zcaWeighting.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setZCA_Whitening(True)
            else:
                self.deepLearningArtApp.setZCA_Whitening(False)

            if self.CheckBox_DataAug_HeightShift.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setHeightShift(True)
            else:
                self.deepLearningArtApp.setHeightShift(False)

            if self.CheckBox_DataAug_WidthShift.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setWidthShift(True)
            else:
                self.deepLearningArtApp.setWidthShift(False)

            if self.CheckBox_DataAug_Zoom.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setZoom(True)
            else:
                self.deepLearningArtApp.setZoom(False)

            # contrast improvement (contrast stretching, adaptive equalization, histogram equalization)
            # it is not recommended to set more than one of them to true
            if self.RadioButton_DataAug_contrastStretching.isChecked():
                self.deepLearningArtApp.setContrastStretching(True)
            else:
                self.deepLearningArtApp.setContrastStretching(False)

            if self.RadioButton_DataAug_histogramEq.isChecked():
                self.deepLearningArtApp.setHistogramEqualization(True)
            else:
                self.deepLearningArtApp.setHistogramEqualization(False)

            if self.RadioButton_DataAug_adaptiveEq.isChecked():
                self.deepLearningArtApp.setAdaptiveEqualization(True)
            else:
                self.deepLearningArtApp.setAdaptiveEqualization(False)
        else:
            # disable data augmentation
            self.deepLearningArtApp.setDataAugmentationEnabled(False)
        self.valueChanged.emit(self.deepLearningArtApp.params)
        # start training process
        self.show_network_interface()
        self.verticalLayout_training_performance.addWidget(self.canvas_training_live_performance_figure)
        self.deepLearningArtApp.performTraining()

    def show_network_interface(self):
        self.network_interface = NetworkInterface(self.deepLearningArtApp.getParameters())
        self.network_interface.window.show()

    def update_network_interface(self):
        self.deepLearningArtApp.setNetworkCanrun(False)
        self.network_interface.updataParameter(self.deepLearningArtApp.getParameters(), 0)

    def _update_network_interface(self):
        self.network_interface.updataParameter(self.deepLearningArtApp.getParameters(), 1)
        self.deepLearningArtApp.setNetworkCanrun(True)

    def button_markingsPath_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getMarkingsPath())
        self.Label_MarkingsPath.setText(dir)
        self.deepLearningArtApp.setMarkingsPath(dir)
        self.valueChanged.emit(self.deepLearningArtApp.params)

    def button_patching_clicked(self):

        df = pandas.read_csv('DLart/network_interface_datasets.csv')
        self.datasets_train = list(df[df['usingfor'] == 'Training']['image'])
        self.datasets_validation = list(df[df['usingfor'] == 'Validation']['image'])
        self.datasets_test = list(df[df['usingfor'] == 'Test']['image'])
        print(self.datasets_test)

        if self.deepLearningArtApp.getSplittingMode() == NONE_SPLITTING:
            QMessageBox.about(self, "My message box", "Select Splitting Mode!")
            return 0

        if not (self.datasets_train == [] or self.datasets_validation == [] or self.datasets_test == []):
            self.deepLearningArtApp.setSplittingMode(DIY_SPLITTING)
            self.deepLearningArtApp.setDatasetSorted(self.datasets_train, self.datasets_validation, self.datasets_test)

        # random shuffle
        if self.CheckBox_randomShuffle.isChecked():
            self.deepLearningArtApp.setIsRandomShuffle(True)
        else:
            self.deepLearningArtApp.setIsRandomShuffle(False)

        # get patching parameters
        self.deepLearningArtApp.setPatchSizeX(self.SpinBox_PatchX.value())
        self.deepLearningArtApp.setPatchSizeY(self.SpinBox_PatchY.value())
        self.deepLearningArtApp.setPatchSizeZ(self.SpinBox_PatchZ.value())
        self.deepLearningArtApp.setPatchOverlapp(self.SpinBox_PatchOverlapp.value())

        # get labling parameters
        if self.RadioButton_MaskLabeling.isChecked():
            self.deepLearningArtApp.setLabelingMode(MASK_LABELING)
        elif self.RadioButton_PatchLabeling.isChecked():
            self.deepLearningArtApp.setLabelingMode(PATCH_LABELING)

        # get patching parameters
        if self.ComboBox_Patching.currentIndex() == 1:
            # 2D patching selected
            self.deepLearningArtApp.setPatchingMode(PATCHING_2D)
        elif self.ComboBox_Patching.currentIndex() == 2:
            # 3D patching selected
            self.deepLearningArtApp.setPatchingMode(PATCHING_3D)
        else:
            self.ComboBox_Patching.setCurrentIndex(1)
            self.deepLearningArtApp.setPatchingMode(PATCHING_2D)

        # using segmentation mask
        self.deepLearningArtApp.setUsingSegmentationMasks(self.CheckBox_SegmentationMask.isChecked())

        # handle store mode
        self.deepLearningArtApp.setStoreMode(self.ComboBox_StoreOptions.currentIndex())

        # generate dataset
        self.deepLearningArtApp.generateDataset()

        # check if attributes in DeepLearningArtApp class contains dataset
        if self.deepLearningArtApp.datasetAvailable() == True:
            # if yes, make the use current data button available
            self.Button_useCurrentData.setEnabled(True)
        self.valueChanged.emit(self.deepLearningArtApp.params)

    def button_outputPatching_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
        self.Label_OutputPathPatching.setText(dir)
        self.deepLearningArtApp.setOutputPathForPatching(dir)
        self.valueChanged.emit(self.deepLearningArtApp.params)

    def getSelectedPatients(self):
        selectedPatients = []
        for i in range(self.TreeWidget_Patients.topLevelItemCount()):
            if self.TreeWidget_Patients.topLevelItem(i).checkState(0) == QtCore.Qt.Checked:
                selectedPatients.append(self.TreeWidget_Patients.topLevelItem(i).text(0))

        self.deepLearningArtApp.setSelectedPatients(selectedPatients)
        self.selectedPatients = self.deepLearningArtApp.getSelectedPatients()
        self.valueChanged.emit(self.deepLearningArtApp.params)

    def button_DB_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getPathToDatabase())
        self.deepLearningArtApp.setPathToDatabase(dir)
        self.TreeWidget_Patients.clear()
        self.manageTreeView()
        self.manageTreeViewDatasets()
        self.datasets_train = []
        self.datasets_validation = []
        self.datasets_test = []
        if self.selectedPatients is not None:
            self.datasets_window = DataSetsWindow(self.selectedPatients, self.selectedDatasets)
            self.datasets_window.show()
        self.valueChanged.emit(self.deepLearningArtApp.params)

    def openFileNamesDialog(self, dir=None):
        if dir == None:
            dir = PATH_OUT + os.sep + "MRPhysics"

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        ret = QFileDialog.getExistingDirectory(self, "Select Directory", dir)
        # path to database
        dir = str(ret)
        return dir

    def manageTreeView(self):
        # all patients in database
        if os.path.exists(self.deepLearningArtApp.getPathToDatabase()):
            subdirs = os.listdir(self.deepLearningArtApp.getPathToDatabase())

            for x in sorted(subdirs):
                item = QTreeWidgetItem()
                item.setText(0, str(x))
                item.setCheckState(0, QtCore.Qt.Unchecked)
                self.TreeWidget_Patients.addTopLevelItem(item)

            self.Label_DB.setText(self.deepLearningArtApp.getPathToDatabase())

    def manageTreeViewDatasets(self):
        print(os.path.dirname(self.deepLearningArtApp.getPathToDatabase()))
        # manage datasets
        for ds in sorted(DeepLearningArtApp.datasets.keys()):
            dataset = DeepLearningArtApp.datasets[ds].getPathdata()
            item = QTreeWidgetItem()
            item.setText(0, dataset)
            item.setCheckState(0, QtCore.Qt.Unchecked)
            self.TreeWidget_Datasets.addTopLevelItem(item)

    def getSelectedDatasets(self):
        selectedDatasets = []
        for i in range(self.TreeWidget_Datasets.topLevelItemCount()):
            if self.TreeWidget_Datasets.topLevelItem(i).checkState(0) == QtCore.Qt.Checked:
                selectedDatasets.append(self.TreeWidget_Datasets.topLevelItem(i).text(0))

        self.deepLearningArtApp.setSelectedDatasets(selectedDatasets)
        self.selectedDatasets = self.deepLearningArtApp.getSelectedDatasets()
        if self.selectedPatients is not None:
            self.datasets_window = DataSetsWindow(self.selectedPatients, self.selectedDatasets)
            self.datasets_window.show()
        self.valueChanged.emit(self.deepLearningArtApp.params)

    def selectedDNN_changed(self):
        if self.ComboBox_DNNs.currentText() == "add new architecture":
            self.addNeuralNetworkModel()
        else:
            self.deepLearningArtApp.setNeuralNetworkModel(self.ComboBox_DNNs.currentText())
        self.valueChanged.emit(self.deepLearningArtApp.params)

    def addNeuralNetworkModel(self):
        file_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose the file', '.', 'python files(*.py)')[0]
        list = file_path.split('/')
        model = os.path.splitext(list[-1])[0]
        ind = list.index('networks')
        model_path = ''
        for item in list[ind:-1]:
            model_path = model_path + item + '.'
        model_path += model
        dnn = self.deepLearningArtApp.getDeepNeuralNetworks()
        dnn[model] = model_path
        self.deepLearningArtApp.setDeepNeuralNetworks(dnn)
        self.deepLearningArtApp.setNeuralNetworkModel(model)
        self.ComboBox_DNNs.addItem(model, -2)
        self.ComboBox_DNNs.setCurrentText(model)
        networkcsv = pandas.read_csv('DLart/networks.csv')
        networkcsv_size = pandas.DataFrame.count(networkcsv)['name']
        networkcsv.loc[networkcsv_size, 'name'] = model
        networkcsv.loc[networkcsv_size, 'path'] = model_path
        networkcsv.to_csv('DLart/networks.csv', index=False)
        self.valueChanged.emit(self.deepLearningArtApp.params)

    def button_useCurrentData_clicked(self):
        if self.deepLearningArtApp.datasetAvailable() == True:
            self.Label_currentDataset.setText("Current Dataset is used...")
            self.GroupBox_TrainNN.setEnabled(True)
        else:
            self.Button_useCurrentData.setEnabled(False)
            self.Label_currentDataset.setText("No Dataset selected!")
            self.GroupBox_TrainNN.setEnabled(False)
        self.valueChanged.emit(self.deepLearningArtApp.params)

    def button_selectDataset_clicked(self):
        pathToDataset = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
        retbool, datasetName = self.deepLearningArtApp.loadDataset(pathToDataset)
        self.deepLearningArtApp.setDatasetForPrediction(pathToDataset)
        if retbool == True:
            self.Label_currentDataset.setText(datasetName + " is used as dataset...")
        else:
            self.Label_currentDataset.setText("No Dataset selected!")

        if self.deepLearningArtApp.datasetAvailable() == True:
            self.GroupBox_TrainNN.setEnabled(True)
        else:
            self.GroupBox_TrainNN.setEnabled(False)
        self.valueChanged.emit(self.deepLearningArtApp.params)

    def button_learningOutputPath_clicked(self):
        path = self.openFileNamesDialog(self.deepLearningArtApp.getLearningOutputPath())
        self.deepLearningArtApp.setLearningOutputPath(path)
        self.Label_LearningOutputPath.setText(path)
        self.valueChanged.emit(self.deepLearningArtApp.params)

    def splittingMode_changed(self):

        if self.ComboBox_splittingMode.currentIndex() == 0:
            self.deepLearningArtApp.setSplittingMode(NONE_SPLITTING)
            self.Label_SplittingParams.setText("Select splitting mode!")
        elif self.ComboBox_splittingMode.currentIndex() == 1:
            # call input dialog for editting ratios
            testTrainingRatio, retBool = QInputDialog.getDouble(self, "Enter Test/Training Ratio:",
                                                                "Ratio Test/Training Set:", 0.2, 0, 1, decimals=2)
            if retBool == True:
                validationTrainingRatio, retBool = QInputDialog.getDouble(self, "Enter Validation/Training Ratio",
                                                                          "Ratio Validation/Training Set: ", 0.2, 0, 1,
                                                                          decimals=2)
                if retBool == True:
                    self.deepLearningArtApp.setSplittingMode(SIMPLE_RANDOM_SAMPLE_SPLITTING)
                    self.deepLearningArtApp.setTrainTestDatasetRatio(testTrainingRatio)
                    self.deepLearningArtApp.setTrainValidationRatio(validationTrainingRatio)
                    txtStr = "using Test/Train=" + str(testTrainingRatio) + " and Valid/Train=" + str(
                        validationTrainingRatio)
                    self.Label_SplittingParams.setText(txtStr)
                else:
                    self.deepLearningArtApp.setSplittingMode(NONE_SPLITTING)
                    self.ComboBox_splittingMode.setCurrentIndex(0)
                    self.Label_SplittingParams.setText("Select Splitting Mode!")
            else:
                self.deepLearningArtApp.setSplittingMode(NONE_SPLITTING)
                self.ComboBox_splittingMode.setCurrentIndex(0)
                self.Label_SplittingParams.setText("Select Splitting Mode!")
        elif self.ComboBox_splittingMode.currentIndex() == 2:
            # cross validation splitting
            testTrainingRatio, retBool = QInputDialog.getDouble(self, "Enter Test/Training Ratio:",
                                                                "Ratio Test/Training Set:", 0.2, 0, 1, decimals=2)

            if retBool == True:
                numFolds, retBool = QInputDialog.getInt(self, "Enter Number of Folds for Cross Validation",
                                                        "Number of Folds: ", 15, 0, 100000)
                if retBool == True:
                    self.deepLearningArtApp.setSplittingMode(CROSS_VALIDATION_SPLITTING)
                    self.deepLearningArtApp.setTrainTestDatasetRatio(testTrainingRatio)
                    self.deepLearningArtApp.setNumFolds(numFolds)
                    self.Label_SplittingParams.setText("Test/Train Ratio: " + str(testTrainingRatio) + \
                                                       ", and " + str(numFolds) + " Folds")
                else:
                    self.deepLearningArtApp.setSplittingMode(NONE_SPLITTING)
                    self.ComboBox_splittingMode.setCurrentIndex(0)
                    self.Label_SplittingParams.setText("Select Splitting Mode!")
            else:
                self.deepLearningArtApp.setSplittingMode(NONE_SPLITTING)
                self.ComboBox_splittingMode.setCurrentIndex(0)
                self.Label_SplittingParams.setText("Select Splitting Mode!")

        elif self.ComboBox_splittingMode.currentIndex() == 3:
            self.deepLearningArtApp.setSplittingMode(PATIENT_CROSS_VALIDATION_SPLITTING)
        self.valueChanged.emit(self.deepLearningArtApp.params)

    def check_dataAugmentation_enabled(self):
        if self.CheckBox_DataAugmentation.checkState() == QtCore.Qt.Checked:
            self.CheckBox_DataAug_horizontalFlip.setEnabled(True)
            self.CheckBox_DataAug_verticalFlip.setEnabled(True)
            self.CheckBox_DataAug_Rotation.setEnabled(True)
            self.CheckBox_DataAug_zcaWeighting.setEnabled(True)
            self.CheckBox_DataAug_HeightShift.setEnabled(True)
            self.CheckBox_DataAug_WidthShift.setEnabled(True)
            self.CheckBox_DataAug_Zoom.setEnabled(True)
            self.RadioButton_DataAug_contrastStretching.setEnabled(True)
            self.RadioButton_DataAug_histogramEq.setEnabled(True)
            self.RadioButton_DataAug_adaptiveEq.setEnabled(True)
        else:
            self.CheckBox_DataAug_horizontalFlip.setEnabled(False)
            self.CheckBox_DataAug_verticalFlip.setEnabled(False)
            self.CheckBox_DataAug_Rotation.setEnabled(False)
            self.CheckBox_DataAug_zcaWeighting.setEnabled(False)
            self.CheckBox_DataAug_HeightShift.setEnabled(False)
            self.CheckBox_DataAug_WidthShift.setEnabled(False)
            self.CheckBox_DataAug_Zoom.setEnabled(False)

            self.RadioButton_DataAug_contrastStretching.setEnabled(False)
            self.RadioButton_DataAug_contrastStretching.setAutoExclusive(False)
            self.RadioButton_DataAug_contrastStretching.setChecked(False)
            self.RadioButton_DataAug_contrastStretching.setAutoExclusive(True)

            self.RadioButton_DataAug_histogramEq.setEnabled(False)
            self.RadioButton_DataAug_histogramEq.setAutoExclusive(False)
            self.RadioButton_DataAug_histogramEq.setChecked(False)
            self.RadioButton_DataAug_histogramEq.setAutoExclusive(True)

            self.RadioButton_DataAug_adaptiveEq.setEnabled(False)
            self.RadioButton_DataAug_adaptiveEq.setAutoExclusive(False)
            self.RadioButton_DataAug_adaptiveEq.setChecked(False)
            self.RadioButton_DataAug_adaptiveEq.setAutoExclusive(True)

    ########## third tab
    def plotSegmentationPredictions(self):
        unpatched_slices = self.deepLearningArtApp.getUnpatchedSlices()

        if unpatched_slices is not None:
            predicted_segmentation_mask = unpatched_slices['predicted_segmentation_mask']
            dicom_slices = unpatched_slices['dicom_slices']
            dicom_masks = unpatched_slices['dicom_masks']

            index = int(self.horizontalSliderSlice.value())
            self.horizontalSliderSlice.setMaximum(int(dicom_slices.shape[-1]))

            if index >= 0 and index < dicom_slices.shape[-1]:
                pred_seg_mask_slice = np.squeeze(predicted_segmentation_mask[:, :, index])
                dicom_slice = np.squeeze(dicom_slices[:, :, index])
                dicom_mask_slice = np.squeeze(dicom_masks[:, :, index])

                self.segmentation_masks_figure.clear()
                ax1 = self.segmentation_masks_figure.add_subplot(121)
                ax1.clear()
                ax1.imshow(dicom_slice, cmap='gray')
                ax1.imshow(dicom_mask_slice, cmap=self.ground_truth_colormap, interpolation='nearest', alpha=1.)
                ax1.set_title('Ground Truth')

                ax2 = self.segmentation_masks_figure.add_subplot(122)
                ax2.clear()
                ax2.imshow(dicom_slice, cmap='gray')
                ax2.imshow(pred_seg_mask_slice, cmap=self.artifact_colormap, interpolation='nearest', alpha=.4)
                ax2.set_title('Predicted Segmentation Mask')

                self.segmentation_masks_figure.tight_layout()
                self.segmentation_masks_figure.savefig(self.deepLearningArtApp.getModelForPrediction()
                                                       + os.sep + str(index) + "_segmentation_masks_figure" + '.png')
                self.canvas_segmentation_masks_figure.draw()
                self.graphicscene_segmentation_masks = QtWidgets.QGraphicsScene()
                self.graphicscene_segmentation_masks.addWidget(self.canvas_segmentation_masks_figure)
                self.graphicview_segmentation_masks = Activeview()
                scrollAreaWidgetContents = QtWidgets.QWidget()
                maingrids = QtWidgets.QGridLayout(scrollAreaWidgetContents)
                self.showResultArea.setWidget(scrollAreaWidgetContents)
                maingrids.addWidget(self.graphicview_segmentation_masks)
                self.graphicview_segmentation_masks.setScene(self.graphicscene_segmentation_masks)


    def plotSegmentationArtifactMaps(self):
        unpatched_slices = self.deepLearningArtApp.getUnpatchedSlices()

        if unpatched_slices is not None:
            probability_mask_background = unpatched_slices['probability_mask_background']
            probability_mask_foreground = unpatched_slices['probability_mask_foreground']
            dicom_slices = unpatched_slices['dicom_slices']
            dicom_masks = unpatched_slices['dicom_masks']

            index = int(self.horizontalSliderSlice.value())
            self.horizontalSliderSlice.setMaximum(int(dicom_slices.shape[-1]))

            if index >= 0 and index < dicom_slices.shape[-1]:
                dicom_slice = np.squeeze(dicom_slices[:, :, index])
                prob_mask_back_slice = np.squeeze(probability_mask_background[:, :, index])
                prob_mask_fore_slice = np.squeeze(probability_mask_foreground[:, :, index])
                dicom_mask_slice = np.squeeze(dicom_masks[:, :, index])

                # plot artifact map
                self.artifact_map_figure.clear()
                ax1 = self.artifact_map_figure.add_subplot(131)
                ax1.clear()
                ax1.imshow(dicom_slice, cmap='gray')
                ax1.imshow(dicom_mask_slice, cmap=self.ground_truth_colormap, interpolation='nearest', alpha=1.)
                ax1.set_title('Ground Truth')

                ax2 = self.artifact_map_figure.add_subplot(132)
                ax2.clear()
                ax2.imshow(dicom_slice, cmap='gray')
                ax2.imshow(prob_mask_fore_slice, cmap=self.artifact_colormap, interpolation='nearest', alpha=.4,
                           vmin=0, vmax=1)
                ax2.set_title('Predicted Foreground')

                ax3 = self.artifact_map_figure.add_subplot(133)
                ax3.clear()
                ax3.imshow(dicom_slice, cmap='gray')
                map = ax3.imshow(prob_mask_back_slice, cmap=self.artifact_colormap, interpolation='nearest', alpha=.4,
                                 vmin=0, vmax=1)
                ax3.set_title('Predicted Background')

                self.artifact_map_figure.colorbar(mappable=map, ax=ax3)
                self.artifact_map_figure.tight_layout()

                self.artifact_map_figure.savefig(self.deepLearningArtApp.getModelForPrediction()
                                                 + os.sep + str(index) + "_artifact_map_figure" + '.png')

                self.canvas_artifact_map_figure.draw()
                self.graphicscene_artifact_map = QtWidgets.QGraphicsScene()
                self.graphicscene_artifact_map.addWidget(self.canvas_artifact_map_figure)
                self.graphicview_artifact_map = Activeview()
                scrollAreaWidgetContents = QtWidgets.QWidget()
                maingrids = QtWidgets.QGridLayout(scrollAreaWidgetContents)
                self.showResultArea.setWidget(scrollAreaWidgetContents)
                maingrids.addWidget(self.graphicview_artifact_map)
                self.graphicview_artifact_map.setScene(self.graphicscene_artifact_map)

    def plotConfusionMatrix(self):
        confusion_matrix = self.deepLearningArtApp.getConfusionMatrix()
        if confusion_matrix is not None:
            target_names = []
            classMappings = self.deepLearningArtApp.getClassMappingsForPrediction()
            if len(classMappings[list(classMappings.keys())[0]]) == 3:
                for i in sorted(classMappings):
                    i = int(i) % 100
                    i = int(i) % 10
                    if Label.LABEL_STRINGS[i] not in target_names:
                        target_names.append(Label.LABEL_STRINGS[i])
            elif len(classMappings[list(classMappings.keys())[0]]) == 8:
                for i in sorted(classMappings):
                    i = int(i) % 100
                    if Label.LABEL_STRINGS[i] not in target_names:
                        target_names.append(Label.LABEL_STRINGS[i])
            else:
                for i in sorted(self.deepLearningArtApp.getClassMappingsForPrediction()):
                    target_names.append(Label.LABEL_STRINGS[int(i)])
            if confusion_matrix.shape == (len(target_names), len(target_names)):
                df_cm = pandas.DataFrame(confusion_matrix,
                                         index=[i for i in target_names],
                                         columns=[i for i in target_names], )

                axes_confmat = self.confusion_matrix_figure.add_subplot(111)
                axes_confmat.clear()

                sn.heatmap(df_cm, annot=True, fmt='.3f', annot_kws={"size": 8}, ax=axes_confmat, linewidths=.2)
                axes_confmat.set_xlabel('Predicted Label')
                axes_confmat.set_ylabel('True Label')
                self.confusion_matrix_figure.tight_layout()
                self.confusion_matrix_figure.savefig(self.deepLearningArtApp.getModelForPrediction()
                                                     + os.sep + str(
                    len(target_names)) + "_canvas_confusion_matrix_figure" + '.png')

                self.canvas_confusion_matrix_figure.draw()
                self.graphicscene_confusion_matrix = QtWidgets.QGraphicsScene()
                self.graphicscene_confusion_matrix.addWidget(self.canvas_confusion_matrix_figure)
                self.graphicview_confusion_matrix = Activeview()
                scrollAreaWidgetContents = QtWidgets.QWidget()
                maingrids = QtWidgets.QGridLayout(scrollAreaWidgetContents)
                self.showResultArea.setWidget(scrollAreaWidgetContents)
                maingrids.addWidget(self.graphicview_confusion_matrix)
                self.graphicview_confusion_matrix.setScene(self.graphicscene_confusion_matrix)
            else:
                raise ValueError('Confusion matrix shape does not match to target names')

        else:
            raise ValueError('There is no confusion matrix for segmentation')

    def plotNetworkArchitecture(self, path=None):

        if path is not None:
            filename = os.path.splitext(path)[0]
        else:
            modelpath = os.path.split(self.openmodel_path)[0]
            filename = modelpath + os.sep + 'architecture'
            cnn2d_visual(self.model, title='', filename=filename)
        self.network_architecture_figure.clear()
        ax1 = self.network_architecture_figure.add_subplot(111)
        ax1.clear()
        image = plt.imread(filename + '.png')
        ax1.imshow(image)
        ax1.set_title('architecture')

        self.canvas_network_architecture_figure.draw()
        self.graphicscene_network_architecture = QtWidgets.QGraphicsScene()
        self.graphicscene_network_architecture.addWidget(self.canvas_network_architecture_figure)
        self.graphicview_network_architecture = Activeview()
        scrollAreaWidgetContents = QtWidgets.QWidget()
        maingrids = QtWidgets.QGridLayout(scrollAreaWidgetContents)
        self.showResultArea.setWidget(scrollAreaWidgetContents)
        maingrids.addWidget(self.graphicview_network_architecture)
        self.graphicview_network_architecture.setScene(self.graphicscene_network_architecture)

    def textChangeAlpha(self, text):
        self.inputalpha = text
        # if text.isdigit():
        #     self.inputalpha=text
        # else:
        #     self.alphaShouldBeNumber()

    def textChangeGamma(self, text):
        self.inputGamma = text
        # if text.isdigit():
        #     self.inputGamma=text
        # else:
        #     self.GammaShouldBeNumber()

    def wheelScroll(self, ind, oncrollStatus):
        if oncrollStatus == 'on_scroll':
            self.horizontalSliderPatch.setValue(ind)
            self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)
        elif oncrollStatus == 'onscrollW' or oncrollStatus == 'onscroll_3D':
            self.wheelScrollW(ind)
        elif oncrollStatus == 'onscrollSS':
            self.wheelScrollSS(ind)
        else:
            pass

    def wheelScrollW(self, ind):
        self.horizontalSliderPatch.setValue(ind)
        self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)

    def wheelScrollSS(self, indSS):
        self.horizontalSliderPatch.setValue(indSS)
        self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)

    def clickList(self, qModelIndex):

        self.chosenLayerName = self.qList[qModelIndex.row()]

        if self.radioButton.isChecked() == True:
            if len(self.chosenLayerName) != 0:

                self.W_F = 'w'
                # show the weights
                try:
                    if self.modelDimension == '2D':
                        if hasattr(self.LayerWeights[self.chosenLayerName], "ndim"):

                            if self.LayerWeights[self.chosenLayerName].ndim == 4:
                                self.lcdNumberPatch.hide()
                                self.lcdNumberSlice.hide()
                                self.horizontalSliderPatch.hide()
                                self.horizontalSliderSlice.hide()
                                self.labelPatch.hide()
                                self.labelSlice.hide()

                                # self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                                # self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                                # self.overlay.show()

                                self.matplotlibwidget_static.mpl.getLayersWeights(self.LayerWeights)
                                self.wyPlot.setDisabled(True)
                                self.newW2D = loadImage_weights_plot_2D(self.matplotlibwidget_static,
                                                                        self.chosenLayerName)
                                self.newW2D.trigger.connect(self.loadEnd2)
                                self.newW2D.start()

                                # self.matplotlibwidget_static.mpl.weights_plot_2D(self.chosenLayerName)
                                self.matplotlibwidget_static.show()
                            # elif self.LayerWeights[self.chosenLayerName].ndim==0:
                            #     self.showNoWeights()
                            else:
                                self.showWeightsDimensionError()

                        elif self.LayerWeights[self.chosenLayerName] == 0:
                            self.showNoWeights()

                    elif self.modelDimension == '3D':
                        if hasattr(self.LayerWeights[self.chosenLayerName], "ndim"):

                            if self.LayerWeights[self.chosenLayerName].ndim == 5:

                                self.w = self.LayerWeights[self.chosenLayerName]
                                self.totalWeights = self.w.shape[0]
                                # self.totalWeightsSlices=self.w.shape[2]
                                self.horizontalSliderPatch.setMinimum(1)
                                self.horizontalSliderPatch.setMaximum(self.totalWeights)
                                # self.horizontalSliderSlice.setMinimum(1)
                                # self.horizontalSliderSlice.setMaximum(self.totalWeightsSlices)
                                self.chosenWeightNumber = 1
                                self.horizontalSliderPatch.setValue(self.chosenWeightNumber)

                                # self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                                # self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                                # self.overlay.show()

                                self.wyPlot.setDisabled(True)
                                self.newW3D = loadImage_weights_plot_3D(self.matplotlibwidget_static, self.w,
                                                                        self.chosenWeightNumber, self.totalWeights,
                                                                        self.totalWeightsSlices)
                                self.newW3D.trigger.connect(self.loadEnd2)
                                self.newW3D.start()

                                # self.matplotlibwidget_static.mpl.weights_plot_3D(self.w,self.chosenWeightNumber,self.totalWeights,self.totalWeightsSlices)

                                self.matplotlibwidget_static.show()
                                self.horizontalSliderSlice.hide()
                                self.horizontalSliderPatch.show()
                                self.labelPatch.show()
                                self.labelSlice.hide()
                                self.lcdNumberSlice.hide()
                                self.lcdNumberPatch.show()
                            # elif self.LayerWeights[self.chosenLayerName].ndim==0:
                            #     self.showNoWeights()
                            else:
                                self.showWeightsDimensionError3D()

                        elif self.LayerWeights[self.chosenLayerName] == 0:
                            self.showNoWeights()

                    else:
                        print('the dimesnion should be 2D or 3D')
                except:
                    QMessageBox.information(self, "warning", "This layer is not supported to show")

            else:
                self.showChooseLayerDialog()

        elif self.radioButton_2.isChecked() == True:
            if len(self.chosenLayerName) != 0:
                self.W_F = 'f'
                try:
                    if self.modelDimension == '2D':
                        if self.act[self.chosenLayerName].ndim == 4:
                            self.activations = self.act[self.chosenLayerName]
                            self.totalPatches = self.activations.shape[0]

                            self.matplotlibwidget_static.mpl.getLayersFeatures(self.activations, self.totalPatches)

                            # show the features
                            self.chosenPatchNumber = 1
                            self.horizontalSliderPatch.setMinimum(1)
                            self.horizontalSliderPatch.setMaximum(self.totalPatches)
                            self.horizontalSliderPatch.setValue(self.chosenPatchNumber)

                            # self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                            # self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                            # self.overlay.show()
                            self.wyPlot.setDisabled(True)
                            self.newf = loadImage_features_plot(self.matplotlibwidget_static,
                                                                self.chosenPatchNumber)
                            self.newf.trigger.connect(self.loadEnd2)
                            self.newf.start()

                            # self.matplotlibwidget_static.mpl.features_plot(self.chosenPatchNumber)
                            self.matplotlibwidget_static.show()
                            self.horizontalSliderSlice.hide()
                            self.horizontalSliderPatch.show()
                            self.labelPatch.show()
                            self.labelSlice.hide()
                            self.lcdNumberPatch.show()
                            self.lcdNumberSlice.hide()
                        else:
                            self.showNoFeatures()

                    elif self.modelDimension == '3D':
                        a = self.act[self.chosenLayerName]
                        if self.act[self.chosenLayerName].ndim == 5:
                            self.activations = self.act[self.chosenLayerName]
                            self.totalPatches = self.activations.shape[0]
                            self.totalPatchesSlices = self.activations.shape[1]

                            self.matplotlibwidget_static.mpl.getLayersFeatures_3D(self.activations,
                                                                                  self.totalPatches,
                                                                                  self.totalPatchesSlices)

                            self.chosenPatchNumber = 1
                            self.chosenPatchSliceNumber = 1
                            self.horizontalSliderPatch.setMinimum(1)
                            self.horizontalSliderPatch.setMaximum(self.totalPatches)
                            self.horizontalSliderPatch.setValue(self.chosenPatchNumber)
                            self.horizontalSliderSlice.setMinimum(1)
                            self.horizontalSliderSlice.setMaximum(self.totalPatchesSlices)
                            self.horizontalSliderSlice.setValue(self.chosenPatchSliceNumber)

                            # self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                            # self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                            # self.overlay.show()
                            self.wyPlot.setDisabled(True)
                            self.newf = loadImage_features_plot_3D(self.matplotlibwidget_static,
                                                                   self.chosenPatchNumber,
                                                                   self.chosenPatchSliceNumber)
                            self.newf.trigger.connect(self.loadEnd2)
                            self.newf.start()

                            # self.matplotlibwidget_static.mpl.features_plot_3D(self.chosenPatchNumber,self.chosenPatchSliceNumber)
                            self.horizontalSliderSlice.show()
                            self.horizontalSliderPatch.show()
                            self.labelPatch.show()
                            self.labelSlice.show()
                            self.lcdNumberPatch.show()
                            self.lcdNumberSlice.show()
                            self.matplotlibwidget_static.show()
                        else:
                            self.showNoFeatures()

                    else:
                        print('the dimesnion should be 2D or 3D')
                except:
                    QMessageBox.information(self, "warning", "This layer is not supported to show")

            else:
                self.showChooseLayerDialog()

    def simpleName(self, inpName):
        if "/" in inpName:
            inpName = inpName.split("/")[0]
            if ":" in inpName:
                inpName = inpName.split(':')[0]
        else:
            if ":" in inpName:
                inpName = inpName.split(":")[0]
                if "/" in inpName:
                    inpName = inpName.split('/')[0]

        return inpName

    def show_layer_name(self):

        self.matplotlibwidget_static = MatplotlibWidget(self.NetworkVisualization)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.matplotlibwidget_static.sizePolicy().hasHeightForWidth())
        self.matplotlibwidget_static.setSizePolicy(sizePolicy)
        self.matplotlibwidget_static.setMouseTracking(True)
        self.matplotlibwidget_static.setObjectName("matplotlibwidget_static")
        self.showResultArea.setWidget(self.matplotlibwidget_static)
        self.matplotlibwidget_static.show()
        qList = []

        for i in self.act:
            qList.append(i)

            # if self.act[i].ndim==5 and self.modelDimension=='3D':
            #     self.act[i]=np.transpose(self.act[i],(0,4,1,2,3))
        self.qList = qList

    def sliderValue(self):
        if self.W_F == 'w':

            self.chosenWeightNumber = self.horizontalSliderPatch.value()
            # self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
            # self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
            # self.overlay.show()
            self.wyPlot.setDisabled(True)
            self.newW3D = loadImage_weights_plot_3D(self.matplotlibwidget_static, self.w, self.chosenWeightNumber,
                                                    self.totalWeights, self.totalWeightsSlices)
            self.newW3D.trigger.connect(self.loadEnd2)
            self.newW3D.start()

            # self.matplotlibwidget_static.mpl.weights_plot_3D(self.w, self.chosenWeightNumber, self.totalWeights,self.totalWeightsSlices)
        elif self.W_F == 'f':

            if self.modelDimension == '2D':
                self.chosenPatchNumber = self.horizontalSliderPatch.value()
                # self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                # self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
                # self.overlay.show()
                self.wyPlot.setDisabled(True)
                self.newf = loadImage_features_plot(self.matplotlibwidget_static, self.chosenPatchNumber)
                self.newf.trigger.connect(self.loadEnd2)
                self.newf.start()
                # self.matplotlibwidget_static.mpl.features_plot(self.chosenPatchNumber)
            elif self.modelDimension == '3D':

                self.chosenPatchNumber = self.horizontalSliderPatch.value()
                self.chosenPatchSliceNumber = self.horizontalSliderSlice.value()
                # self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                # self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
                # self.overlay.show()
                self.wyPlot.setDisabled(True)
                self.newf = loadImage_features_plot_3D(self.matplotlibwidget_static, self.chosenPatchNumber,
                                                       self.chosenPatchSliceNumber)
                self.newf.trigger.connect(self.loadEnd2)
                self.newf.start()
                # self.matplotlibwidget_static.mpl.features_plot_3D(self.chosenPatchNumber,self.chosenPatchSliceNumber)
        elif self.W_F == 's':

            self.chosenSSNumber = self.horizontalSliderPatch.value()
            # self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
            # self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
            # self.overlay.show()
            self.wyPlot.setDisabled(True)
            self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
            self.newf.trigger.connect(self.loadEnd2)
            self.newf.start()
            # self.matplotlibwidget_static.mpl.subset_selection_plot(self.chosenSSNumber)

        else:
            pass

    def on_wyChooseFile_clicked(self, index):
        if index == 1:
            self.openmodel_path = \
                QtWidgets.QFileDialog.getOpenFileName(self, 'Choose model', DLART_OUT_PATH, 'H5 files(*.h5)')[0]
        elif index == 2:
            self.openmodel_path = self.deepLearningArtApp.getCurrentModelPath()
            print('Using current model')
        if self.openmodel_path is not None:
            self.horizontalSliderPatch.hide()
            self.horizontalSliderSlice.hide()
            self.labelPatch.hide()
            self.labelSlice.hide()
            self.lcdNumberSlice.hide()
            self.lcdNumberPatch.hide()

            pathToModel = os.path.split(self.openmodel_path)[0]
            self.deepLearningArtApp.setModelForPrediction(pathToModel)
            QMessageBox.information(self, "Working in Progress", "Loading model")
            print("Loading model")
            if index == 1:
                try:
                    self.model = load_model(self.openmodel_path)
                except:
                    try:
                        def dice_coef(y_true, y_pred, epsilon=1e-5):
                            dice_numerator = 2.0 * K.sum(y_true * y_pred, axis=[1, 2, 3, 4])
                            dice_denominator = K.sum(K.square(y_true), axis=[1, 2, 3, 4]) + K.sum(K.square(y_pred),
                                                                                                  axis=[1, 2, 3, 4])

                            dice_score = dice_numerator / (dice_denominator + epsilon)
                            return K.mean(dice_score, axis=0)

                        def dice_coef_loss(y_true, y_pred):
                            return 1 - dice_coef(y_true, y_pred)

                        self.model = load_model(self.openmodel_path,
                                                custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
                    except:
                        self.wyChooseFile.setCurrentIndex(0)
                        QMessageBox.information(self, "Warning", "This model is not supported, please load another model")
                        return
                print(len(self.model.layers), 'layers')
            elif index == 2:
                # # using current model
                self.model = self.deepLearningArtApp.getModel()

            QMessageBox.information(self, "Finish Progress", "Finish loading model")
            print("Finish loading model")
            self.model_png_dir = os.path.split(self.openmodel_path)[0] + os.sep + "model.png"

        self.wyChooseFile.setCurrentIndex(0)

    def on_wyInputData_clicked(self, index):
        if index == 1:
            pathToDataset = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
            FL = os.listdir(pathToDataset)
            for fl in FL:
                if fl == 'datasets.hdf5':
                    self.inputData_name = pathToDataset + os.sep + fl
                    print(self.inputData_name)
            self.deepLearningArtApp.setDatasetForPrediction(pathToDataset)
        elif index == 2:
            self.inputData_name = None
            print('using current datasets for prediction')

        if len(self.openmodel_path) != 0:
            self.horizontalSliderPatch.hide()
            self.horizontalSliderSlice.hide()
            self.labelPatch.hide()
            self.labelSlice.hide()
            self.lcdNumberSlice.hide()
            self.lcdNumberPatch.hide()

            if self.inputData_name is not None:

                self.inputData = h5py.File(self.inputData_name, 'r')
                # the number of the input
                for i in self.inputData:
                    if i == 'X_test_p2' or i == 'y_test_p2':
                        self.twoInput = True
                        break

                if self.inputData['X_test'].ndim == 3:
                    self.modelDimension = '2D'
                    X_test = self.inputData['X_test']
                    X_test = np.expand_dims(np.array(X_test), axis=-1)
                    self.subset_selection = X_test

                    if self.twoInput:
                        X_test_p2 = self.inputData['X_test_p2']
                        X_test_p2 = np.expand_dims(np.array(X_test_p2), axis=-1)
                        self.subset_selection_2 = X_test_p2

                elif self.inputData['X_test'].ndim == 4:
                    self.modelDimension = '3D'
                    X_test = self.inputData['X_test']
                    X_test = np.expand_dims(np.array(X_test), axis=-1)
                    self.subset_selection = X_test

                    if self.twoInput:
                        X_test_p2 = self.inputData['X_test_p2']
                        X_test_p2 = np.expand_dims(np.array(X_test_p2), axis=-1)
                        self.subset_selection_2 = X_test_p2

                else:
                    print('the dimension of X_test should be 4 or 5')

                if self.twoInput:
                    self.radioButton_3.show()
                    self.radioButton_4.show()
                    self.modelInput = self.model.input[0]
                    self.modelInput2 = self.model.input[1]
                else:
                    self.modelInput = self.model.input
                QMessageBox.information(self, "Working in Progress", "Loading datasets")
                self.show_layer_list()
                QMessageBox.information(self, "Finish Progress", "Finish loading datasets")
            else:
                QMessageBox.information(self, "Finish Loading", "using current datasets for prediction")
            self.predictButton.setDisabled(False)
        else:
            pass
        self.wyInputData.setCurrentIndex(0)

    def show_layer_list(self):

        self.layer_index_name = {}

        for i, layer in enumerate(self.model.input_layers):
            get_activations = K.function([layer.input, K.learning_phase()],
                                         [layer.output, ])
            if i == 0:
                self.act[self.simpleName(layer.name)] = get_activations([self.subset_selection, 0])[0]
                self.num_inputlayer = 1
            elif i == 1:
                self.act[self.simpleName(layer.name)] = get_activations([self.subset_selection_2, 0])[0]
                self.num_inputlayer = 2
            else:
                print('no output of the input layer is created')
        for i, layer in enumerate(self.model.layers):
            self.layer_index_name[i] = self.simpleName(layer.name)
            if i < self.num_inputlayer:
                pass
            else:
                if not type(layer.input) == list:
                    try:
                        inputLayerNameList = [self.simpleName(layer.input.name)[0:-2]]
                        get_activations = K.function([layer.input, K.learning_phase()],
                                                     [layer.output, ])
                        self.act[self.simpleName(layer.name)] = \
                            get_activations([self.act[inputLayerNameList[0]], 0])[0]
                    except:
                        inputLayerNameList = [self.simpleName(layer.input.name)]
                        get_activations = K.function([layer.input, K.learning_phase()],
                                                     [layer.output, ])
                        self.act[self.simpleName(layer.name)] = \
                            get_activations([self.act[inputLayerNameList[0]], 0])[0]
                else:
                    if len(layer.input) == 2:

                        try:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name)[0:-2])
                            get_activations = K.function([layer.input[0], layer.input[1], K.learning_phase()],
                                                         [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 0])[0]
                        except:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))
                            get_activations = K.function([layer.input[0], layer.input[1], K.learning_phase()],
                                                         [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 0])[0]
                    elif len(layer.input) == 3:

                        try:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name)[0:-2])

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 0])[0]
                        except:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 0])[0]

                    elif len(layer.input) == 4:

                        try:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name)[0:-2])

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 self.act[inputLayerNameList[3]],
                                                 0])[0]
                        except:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 self.act[inputLayerNameList[3]],
                                                 0])[0]

                    elif len(layer.input) == 5:

                        try:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name)[0:-2])

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 self.act[inputLayerNameList[3]],
                                                 self.act[inputLayerNameList[4]],
                                                 0])[0]
                        except:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 self.act[inputLayerNameList[3]],
                                                 self.act[inputLayerNameList[4]],
                                                 0])[0]
                    elif len(layer.input) == 6:

                        try:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name)[0:-2])

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],
                                 layer.input[5],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 self.act[inputLayerNameList[3]],
                                                 self.act[inputLayerNameList[4]],
                                                 self.act[inputLayerNameList[5]],
                                                 0])[0]
                        except:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],
                                 layer.input[5],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 self.act[inputLayerNameList[3]],
                                                 self.act[inputLayerNameList[4]],
                                                 self.act[inputLayerNameList[5]],
                                                 0])[0]

                    elif len(layer.input) == 7:

                        try:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name)[0:-2])

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],
                                 layer.input[5], layer.input[6],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 self.act[inputLayerNameList[3]],
                                                 self.act[inputLayerNameList[4]],
                                                 self.act[inputLayerNameList[5]],
                                                 self.act[inputLayerNameList[6]],
                                                 0])[0]
                        except:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],
                                 layer.input[5], layer.input[6],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 self.act[inputLayerNameList[3]],
                                                 self.act[inputLayerNameList[4]],
                                                 self.act[inputLayerNameList[5]],
                                                 self.act[inputLayerNameList[6]],
                                                 0])[0]
                    elif len(layer.input) == 8:

                        try:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name)[0:-2])

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],
                                 layer.input[5], layer.input[6], layer.input[7],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 self.act[inputLayerNameList[3]],
                                                 self.act[inputLayerNameList[4]],
                                                 self.act[inputLayerNameList[5]],
                                                 self.act[inputLayerNameList[6]],
                                                 self.act[inputLayerNameList[7]],
                                                 0])[0]
                        except:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],
                                 layer.input[5], layer.input[6], layer.input[7],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 self.act[inputLayerNameList[3]],
                                                 self.act[inputLayerNameList[4]],
                                                 self.act[inputLayerNameList[5]],
                                                 self.act[inputLayerNameList[6]],
                                                 self.act[inputLayerNameList[7]],
                                                 0])[0]

                    elif len(layer.input) == 9:

                        try:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name)[0:-2])

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],
                                 layer.input[5], layer.input[6], layer.input[7], layer.input[8],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 self.act[inputLayerNameList[3]],
                                                 self.act[inputLayerNameList[4]],
                                                 self.act[inputLayerNameList[5]],
                                                 self.act[inputLayerNameList[6]],
                                                 self.act[inputLayerNameList[7]],
                                                 self.act[inputLayerNameList[8]],
                                                 0])[0]
                        except:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],
                                 layer.input[5], layer.input[6], layer.input[7], layer.input[8],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[self.simpleName(layer.name)] = \
                                get_activations([self.act[inputLayerNameList[0]],
                                                 self.act[inputLayerNameList[1]],
                                                 self.act[inputLayerNameList[2]],
                                                 self.act[inputLayerNameList[3]],
                                                 self.act[inputLayerNameList[4]],
                                                 self.act[inputLayerNameList[5]],
                                                 self.act[inputLayerNameList[6]],
                                                 self.act[inputLayerNameList[7]],
                                                 self.act[inputLayerNameList[8]],
                                                 0])[0]
                    else:
                        print('the number of input is more than 9')

        dot = model_to_dot(self.model, show_shapes=False, show_layer_names=True, rankdir='TB')
        if hasattr(self.model, "layers_by_depth"):
            self.layers_by_depth = self.model.layers_by_depth
        elif hasattr(self.model.model, "layers_by_depth"):
            self.layers_by_depth = self.model.model.layers_by_depth
        else:
            print('the model or model.model should contain parameter layers_by_depth')

        maxCol = 0

        for i in range(len(self.layers_by_depth)):

            for ind, layer in enumerate(self.layers_by_depth[i]):  # the layers in No i layer in the model
                if maxCol < ind:
                    maxCow = ind

                if len(layer.weights) == 0:
                    w = 0
                else:

                    w = layer.weights[0]
                    init = tf.global_variables_initializer()
                    with tf.Session() as sess_i:
                        sess_i.run(init)
                        # print(sess_i.run(w))
                        w = sess_i.run(w)

                self.weights[layer.name] = w

        if self.modelDimension == '3D':
            for i in self.weights:
                # a=self.weights[i]
                # b=a.ndim
                if hasattr(self.weights[i], "ndim"):
                    if self.weights[i].ndim == 5:
                        self.LayerWeights[i] = np.transpose(self.weights[i], (4, 3, 2, 0, 1))
                else:
                    self.LayerWeights[i] = self.weights[i]
        elif self.modelDimension == '2D':
            for i in self.weights:
                if hasattr(self.weights[i], "ndim"):

                    if self.weights[i].ndim == 4:
                        self.LayerWeights[i] = np.transpose(self.weights[i], (3, 2, 0, 1))
                else:
                    self.LayerWeights[i] = self.weights[i]
        else:
            print('the dimesnion of the weights should be 2D or 3D')

        self.show_layer_name()

        self.totalSS = len(self.subset_selection)

        # show the activations' name in the List
        slm = QStringListModel()
        slm.setStringList(self.qList)
        self.listView.setModel(slm)
        self.listView.update()

    @pyqtSlot()
    def on_predictButton_clicked(self):

        # preform preddiciton
        self.deepLearningArtApp.performPrediction()

    @pyqtSlot()
    def on_wyShowArchitecture_clicked(self):
        
        # Show the structure of the model
        if len(self.openmodel_path) != 0:

            self.canvasStructure = MyMplCanvas()
            try:
                self.canvasStructure.loadImage(self.model_png_dir)
            except:
                raise ValueError('No model image exits')
            self.graphicscene_network_architecture = QtWidgets.QGraphicsScene()
            self.graphicscene_network_architecture.addWidget(self.canvasStructure)
            self.graphicview_network_architecture = Activeview()
            self.graphicview_network_architecture.doubleClicked.connect(self.openArchitectureImage)
            scrollAreaWidgetContents = QtWidgets.QWidget()
            maingrids = QtWidgets.QGridLayout(scrollAreaWidgetContents)
            self.showArchitectureArea.setWidget(scrollAreaWidgetContents)
            maingrids.addWidget(self.graphicview_network_architecture)
            self.graphicview_network_architecture.setScene(self.graphicscene_network_architecture)
            try:
                # # for a simple 2DCNN network
                self.plotNetworkArchitecture()

            except:
                QMessageBox.information(self, "warning", "Network contains layer not supported to visualize")
                return

        else:
            self.showChooseFileDialog()

    def openArchitectureImage(self):
        webbrowser.open_new(self.model_png_dir)

    def on_wyPlot_clicked(self, index):
        # self.matplotlibwidget_static_2.hide(

        # Show the structure of the model and plot the weights
        if len(self.openmodel_path) != 0:

            if index == 1:
                self.plotSegmentationPredictions()
            elif index == 2:
                self.plotSegmentationArtifactMaps()
            elif index == 3:
                self.plotConfusionMatrix()
        else:
            self.showChooseFileDialog()
        self.wyPlot.setCurrentIndex(0)

    @pyqtSlot()
    def on_wySubsetSelection_clicked(self):
        # Show the Subset Selection
        if len(self.openmodel_path) != 0:
            self.W_F = 's'
            self.chosenSSNumber = 1
            self.horizontalSliderPatch.setMinimum(1)
            self.horizontalSliderPatch.setMaximum(self.totalSS)
            self.horizontalSliderPatch.setValue(self.chosenSSNumber)
            self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)
            self.lcdNumberPatch.show()
            self.lcdNumberSlice.hide()
            self.horizontalSliderPatch.show()
            self.horizontalSliderSlice.hide()
            self.labelPatch.show()
            self.labelSlice.hide()

            # create input patch
            if self.twoInput == False:
                self.matplotlibwidget_static.mpl.getSubsetSelections(self.subset_selection, self.totalSS)

                self.createSubset(self.modelInput, self.subset_selection)
                self.matplotlibwidget_static.mpl.getSSResult(self.ssResult)

                # self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                # self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                # self.overlay.show()
                self.wyPlot.setDisabled(True)
                self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
                self.newf.trigger.connect(self.loadEnd2)
                self.newf.start()

                # self.matplotlibwidget_static.mpl.subset_selection_plot(self.chosenSSNumber)
            elif self.twoInput:
                if self.radioButton_3.isChecked():  # the 1st input
                    self.matplotlibwidget_static.mpl.getSubsetSelections(self.subset_selection, self.totalSS)
                    self.createSubset(self.modelInput, self.subset_selection)
                    self.matplotlibwidget_static.mpl.getSSResult(self.ssResult)

                    # self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                    # self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                    # self.overlay.show()
                    self.wyPlot.setDisabled(True)
                    self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
                    self.newf.trigger.connect(self.loadEnd2)
                    self.newf.start()

                elif self.radioButton_4.isChecked():  # the 2nd input
                    self.matplotlibwidget_static.mpl.getSubsetSelections(self.subset_selection_2, self.totalSS)
                    self.createSubset(self.modelInput2, self.subset_selection_2)
                    self.matplotlibwidget_static.mpl.getSSResult(self.ssResult)

                    # self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                    # self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                    # self.overlay.show()
                    self.wyPlot.setDisabled(True)
                    self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
                    self.newf.trigger.connect(self.loadEnd2)
                    self.newf.start()

                else:
                    self.showChooseInput()
            else:
                print('the number of input should be 1 or 2')

        else:
            self.showChooseFileDialog()

    def showChooseFileDialog(self):
        reply = QtWidgets.QMessageBox.information(self,
                                                  "Warning",
                                                  "Please select one H5 File at first",
                                                  QtWidgets.QMessageBox.Ok)

    def showChooseLayerDialog(self):
        reply = QtWidgets.QMessageBox.information(self,
                                                  "Warning",
                                                  "Please select one Layer at first",
                                                  QtWidgets.QMessageBox.Ok)

    def showChooseButtonDialog(self):
        reply = QtWidgets.QMessageBox.information(self,
                                                  "Warning",
                                                  "Please select to plot the weights or the features",
                                                  QtWidgets.QMessageBox.Ok)

    def showNoWeights(self):
        reply = QtWidgets.QMessageBox.information(self,
                                                  "Warning",
                                                  "This layer does not have weighst,please select other layers",
                                                  QtWidgets.QMessageBox.Ok)

    def showWeightsDimensionError(self):
        reply = QtWidgets.QMessageBox.information(self,
                                                  "Warning",
                                                  "The diemnsion of the weights should be 0 or 4",
                                                  QtWidgets.QMessageBox.Ok)

    def showWeightsDimensionError3D(self):
        reply = QtWidgets.QMessageBox.information(self,
                                                  "Warning",
                                                  "The diemnsion of the weights should be 0 or 5",
                                                  QtWidgets.QMessageBox.Ok)

    def showNoFeatures(self):
        reply = QtWidgets.QMessageBox.information(self,
                                                  "Warning",
                                                  "This layer does not have feature maps, please select other layers",
                                                  QtWidgets.QMessageBox.Ok)

    def loadEnd2(self):
        # self.overlay.killTimer(self.overlay.timer)
        # self.overlay.hide()
        self.wyPlot.setDisabled(False)

    def alphaShouldBeNumber(self):
        reply = QtWidgets.QMessageBox.information(self,
                                                  "Warning",
                                                  "Alpha should be a number!!!",
                                                  QtWidgets.QMessageBox.Ok)

    def GammaShouldBeNumber(self):
        reply = QtWidgets.QMessageBox.information(self,
                                                  "Warning",
                                                  "Gamma should be a number!!!",
                                                  QtWidgets.QMessageBox.Ok)

    def createSubset(self, modelInput, subset_selection):
        class_idx = 0
        reg_param = 1 / (2e-4)

        if lusegpu:
            input = modelInput  # tensor
            cost = -K.sum(K.log(input[:, class_idx] + 1e-8))  # tensor
            gradient = K.gradients(cost, input)  # list

            sess = tf.InteractiveSession()
            calcCost = TensorFlowTheanoFunction([input], cost)
            calcGrad = TensorFlowTheanoFunction([input], gradient)

            step_size = float(self.inputalpha)
            reg_param = float(self.inputGamma)

            test = subset_selection
            data_c = test
            oss_v = network_visualization.SubsetSelection(calcGrad, calcCost, data_c, alpha=reg_param, gamma=step_size)
            result = oss_v.optimize(np.random.uniform(0, 1.0, size=data_c.shape))
            result = result * test
            result[result > 0] = 1
            self.ssResult = result
        else:
            self.ssResult = None

    def showChooseInput(self):
        reply = QtWidgets.QMessageBox.information(self,
                                                  "Warning",
                                                  "Please select to plot the input 1 or 2",
                                                  QtWidgets.QMessageBox.Ok)


############################################################ class of grids
class Viewgroup(QtWidgets.QGridLayout):
    in_link = QtCore.pyqtSignal()
    rotateSignal = QtCore.pyqtSignal(str)
    scrollSignal = QtCore.pyqtSignal(str)
    zoomSignal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(Viewgroup, self).__init__(parent)

        self.mode = 1
        self.viewnr1 = 1
        self.viewnr2 = 1
        self.anewcanvas = None
        self.islinked = False
        self.skiplink = False
        self.skipdis = True
        self.labeled = False

        self.pathbox = QtWidgets.QComboBox()
        self.pathbox.addItem('closed')
        self.addWidget(self.pathbox, 0, 0, 1, 1)
        self.refbox = QtWidgets.QComboBox()
        self.refbox.addItem('closed')
        self.addWidget(self.refbox, 0, 1, 1, 1)
        self.pathbox.setDisabled(False)
        self.refbox.setDisabled(True)

        self.graylabel = QtWidgets.QLabel()
        self.zoomlabel = QtWidgets.QLabel()
        self.slicelabel = QtWidgets.QLabel()
        self.graylabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.graylabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zoomlabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.zoomlabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slicelabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.slicelabel.setFrameShadow(QtWidgets.QFrame.Raised)

        self.secondline = QtWidgets.QGridLayout()
        self.secondline.addWidget(self.slicelabel, 0, 0, 1, 1)
        self.secondline.addWidget(self.zoomlabel, 0, 1, 1, 1)
        self.secondline.addWidget(self.graylabel, 0, 2, 1, 1)
        self.addLayout(self.secondline, 1, 0, 1, 2)

        self.modechange = QtWidgets.QPushButton()
        self.addWidget(self.modechange, 0, 2, 1, 1)
        self.imrotate = QtWidgets.QPushButton()
        self.addWidget(self.imrotate, 1, 3, 1, 1)
        self.imageedit = QtWidgets.QPushButton()
        self.addWidget(self.imageedit, 1, 2, 1, 1)
        self.linkon = QtWidgets.QPushButton()
        self.addWidget(self.linkon, 0, 3, 1, 1)

        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/Icons/switch2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.icon2 = QtGui.QIcon()
        self.icon2.addPixmap(QtGui.QPixmap(":/icons/Icons/blink.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/Icons/edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/Icons/rotate.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.icon5 = QtGui.QIcon()
        self.icon5.addPixmap(QtGui.QPixmap(":/icons/Icons/link.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.sizePolicy.setHorizontalStretch(0)
        self.sizePolicy.setVerticalStretch(0)
        self.sizePolicy.setHeightForWidth(self.modechange.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.linkon.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.imageedit.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.imrotate.sizePolicy().hasHeightForWidth())
        self.modechange.setSizePolicy(self.sizePolicy)
        self.modechange.setText("")
        self.modechange.setIcon(icon1)
        self.modechange.setToolTip("change source between path and reference")
        self.modechange.setDisabled(True)
        self.linkon.setSizePolicy(self.sizePolicy)
        self.linkon.setText("")
        self.linkon.setIcon(self.icon2)
        self.linkon.setToolTip("link on between 3 windows")
        self.imageedit.setSizePolicy(self.sizePolicy)
        self.imageedit.setText("")
        self.imageedit.setIcon(icon3)
        self.imageedit.setToolTip("edit gray scale")
        self.imrotate.setSizePolicy(self.sizePolicy)
        self.imrotate.setText("")
        self.imrotate.setIcon(icon4)
        self.imrotate.setToolTip(
            "change view_image XY/YZ/XZ\nduring labeling\nafter change view_image\nplease click_event "
            "button\nnoselection on")
        self.Viewpanel = Activeview()
        self.addWidget(self.Viewpanel, 2, 0, 1, 4)

        self.anewscene = Activescene()
        self.Viewpanel.setScene(self.anewscene)
        self.Viewpanel.zooming_data.connect(self.updateZoom)

        self.modechange.clicked.connect(self.switchMode)
        self.imrotate.clicked.connect(self.rotateView)
        self.pathbox.currentIndexChanged.connect(self.loadScene_image)
        self.refbox.currentIndexChanged.connect(self.loadScene_result)
        self.imageedit.clicked.connect(self.setGrey)
        self.linkon.clicked.connect(self.linkPanel)
        self.oldindex = 0

        self.zoomscale = 1.0
        self.currentImage = None

    def switchMode(self):
        if self.mode == 1:
            self.mode = 2
            self.pathbox.setDisabled(True)
            self.refbox.setDisabled(False)
            self.pathbox.setCurrentIndex(0)
        else:
            self.mode = 1
            self.pathbox.setDisabled(False)
            self.refbox.setDisabled(True)
            self.refbox.setCurrentIndex(0)

    def rotateView(self):
        if self.pathbox.currentIndex() != 0 or self.refbox.currentIndex() != 0:
            if self.mode == 1:
                if self.viewnr1 == 1:
                    self.viewnr1 = 2
                    param2 = {'image': list1[imagenum[0]], 'mode': 2, 'shape': shapelist[imagenum[0]]}
                    self.anewcanvas = Canvas(param2)
                elif self.viewnr1 == 2:
                    self.viewnr1 = 3
                    param3 = {'image': list1[imagenum[0]], 'mode': 3, 'shape': shapelist[imagenum[0]]}
                    self.anewcanvas = Canvas(param3)
                elif self.viewnr1 == 3:
                    self.viewnr1 = 1
                    param1 = {'image': list1[imagenum[0]], 'mode': 1, 'shape': shapelist[imagenum[0]]}
                    self.anewcanvas = Canvas(param1)
                self.loadImage()

            else:
                if cnrlist[resultnum[0]] == 11:
                    cmap3 = colormaplist[resultnum[0]]
                    if self.viewnr2 == 1:
                        self.viewnr2 = 2
                        param2 = {'image': list1[imagenum[0]], 'mode': 5,
                                  'color': problist[resultnum[0]],
                                  'hatch': hatchlist[resultnum[0]], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                                  'shape': shapelist[imagenum[0]]}
                        self.anewcanvas = Canvas(param2)
                    elif self.viewnr2 == 2:
                        self.viewnr2 = 3
                        param3 = {'image': list1[imagenum[0]], 'mode': 6,
                                  'color': problist[resultnum[0]],
                                  'hatch': hatchlist[resultnum[0]], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                                  'shape': shapelist[imagenum[0]]}
                        self.anewcanvas = Canvas(param3)
                    elif self.viewnr2 == 3:
                        self.viewnr2 = 1
                        param1 = {'image': list1[imagenum[0]], 'mode': 4,
                                  'color': problist[resultnum[0]],
                                  'hatch': hatchlist[resultnum[0]], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                                  'shape': shapelist[imagenum[0]]}
                        self.anewcanvas = Canvas(param1)

                elif cnrlist[resultnum[0]] == 2:
                    cmap1 = colormaplist[resultnum[0]]
                    if self.viewnr2 == 1:
                        self.viewnr2 = 2
                        param2 = {'image': list1[imagenum[0]], 'mode': 8,
                                  'color': problist[resultnum[0]],
                                  'cmap': cmap1, 'trans': vtr1, 'shape': shapelist[imagenum[0]]}
                        self.anewcanvas = Canvas(param2)
                    elif self.viewnr2 == 2:
                        self.viewnr2 = 3
                        param3 = {'image': list1[imagenum[0]], 'mode': 9,
                                  'color': problist[resultnum[0]],
                                  'cmap': cmap1, 'trans': vtr1, 'shape': shapelist[imagenum[0]]}
                        self.anewcanvas = Canvas(param3)
                    elif self.viewnr2 == 3:
                        self.viewnr2 = 1
                        param1 = {'image': list1[imagenum[0]], 'mode': 7,
                                  'color': problist[resultnum[0]],
                                  'cmap': cmap1, 'trans': vtr1, 'shape': shapelist[imagenum[0]]}
                        self.anewcanvas = Canvas(param1)
                else:
                    cmaps = colormaplist[resultnum[0]]
                    if self.viewnr2 == 1:
                        self.viewnr2 = 2
                        param2 = {'image': list1[imagenum[0]], 'mode': 5,
                                  'color': problist[resultnum[0]],
                                  'hatch': hatchlist[resultnum[0]], 'cmap': cmaps, 'trans': vtrs,
                                  'shape': shapelist[imagenum[0]]}
                        self.anewcanvas = Canvas(param2)
                    elif self.viewnr2 == 2:
                        self.viewnr2 = 3
                        param3 = {'image': list1[imagenum[0]], 'mode': 6,
                                  'color': problist[resultnum[0]],
                                  'hatch': hatchlist[resultnum[0]], 'cmap': cmaps, 'trans': vtrs,
                                  'shape': shapelist[imagenum[0]]}
                        self.anewcanvas = Canvas(param3)
                    elif self.viewnr2 == 3:
                        self.viewnr2 = 1
                        param1 = {'image': list1[imagenum[0]], 'mode': 4,
                                  'color': problist[resultnum[0]],
                                  'hatch': hatchlist[resultnum[0]], 'cmap': cmaps, 'trans': vtrs,
                                  'shape': shapelist[imagenum[0]]}
                        self.anewcanvas = Canvas(param1)
                self.loadImage()
            self.rotateSignal.emit(self.current_slice())

        else:
            pass

    def linkPanel(self):
        if self.pathbox.currentIndex() == 0 and self.refbox.currentIndex() == 0:  # trigger condition
            pass
        else:
            if self.islinked == False:
                self.linkon.setIcon(self.icon5)
                self.islinked = True
                self.in_link.emit()
            else:
                self.linkon.setIcon(self.icon2)
                self.islinked = False
                self.in_link.emit()

    def loadImage(self):
        self.anewscene.clear()
        self.anewscene.addWidget(self.anewcanvas)
        self.Viewpanel.zoomback()
        self.slicelabel.setText('S %s' % (self.anewcanvas.ind + 1) + '/ %s' % (self.anewcanvas.slices))
        try:
            self.graylabel.setText('G %s' % (self.anewcanvas.graylist))
        except:
            pass
        if self.viewnr1 == 2:
            self.zoomlabel.setText('YZ')
        elif self.viewnr1 == 3:
            self.zoomlabel.setText('XZ')
        elif self.viewnr1 == 1:
            self.zoomlabel.setText('XY')

        self.anewcanvas.update_data.connect(self.updateSlices)
        self.anewcanvas.gray_data.connect(self.updateGray)
        self.anewcanvas.new_page.connect(self.newSliceview)
        self.anewcanvas.mpl_connect('motion_notify_event', self.mouse_move)

        self.currentImage = self.pathbox.currentText()

        if self.islinked == True:
            self.Viewpanel.zoom_link.disconnect()
            self.Viewpanel.move_link.disconnect()
            self.skiplink = False
            self.skipdis = True
            self.in_link.emit()

    def current_image(self):
        if self.currentImage is not None:
            if not self.currentImage.find('_Labeling') == -1:
                self.currentImage = self.currentImage.replace('_Labeling', '')
        return self.currentImage

    def current_slice(self):
        if self.viewnr1 == 2:
            slice = 'X %s' % (self.anewcanvas.ind + 1)
        elif self.viewnr1 == 3:
            slice = 'Y %s' % (self.anewcanvas.ind + 1)
        elif self.viewnr1 == 1:
            slice = 'Z %s' % (self.anewcanvas.ind + 1)
        return slice

    def setlinkoff(self, value):
        self.linkon.setDisabled(value)

    def addPathd(self, pathDicom, value):
        imageItems = [self.pathbox.itemText(i) for i in range(self.pathbox.count())]
        region = os.path.split(pathDicom)
        proband = os.path.split(os.path.split(region[0])[0])[1]
        region = region[1]
        if value == 0:
            newItem = '[' + (proband) + '][' + (region) + ']'
        else:
            newItem = '[' + (proband) + '][' + (region) + ']_Labeling'
        if newItem not in imageItems:
            self.pathbox.addItem(newItem)

    def addPathre(self, pathColor):
        self.refbox.addItem(pathColor)

    def loadScene_image(self, i):
        imagenum.clear()
        imagenum.append(i - 1)  # position in combobox
        self.viewnr1 = 1
        self.viewnr2 = 1
        if i != 0:
            param = {'image': list1[imagenum[0]], 'mode': 1, 'shape': shapelist[imagenum[0]]}
            self.anewcanvas = Canvas(param)
            self.oldindex = i
            self.loadImage()
        else:
            self.anewscene.clear()
            self.slicelabel.setText('')
            self.graylabel.setText('')
            self.zoomlabel.setText('')
            if self.islinked == True:
                self.linkon.setIcon(self.icon2)
                self.islinked = False
                self.skiplink = False
                self.skipdis = True
                self.Viewpanel.zoom_link.disconnect()
                self.Viewpanel.move_link.disconnect()
                self.in_link.emit()

    def loadScene_result(self, i):
        resultnum.clear()
        resultnum.append(i - 1)  # position in combobox
        imagenum.clear()
        imagenum.append(self.oldindex - 1)
        self.viewnr1 = 1
        self.viewnr2 = 1
        if i != 0:
            if cnrlist[i - 1] == 11:
                cmap3 = colormaplist[resultnum[0]]
                param1 = {'image': list1[imagenum[0]], 'mode': 4, 'color': problist[resultnum[0]],
                          'hatch': hatchlist[resultnum[0]], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                          'shape': shapelist[imagenum[0]]}
                self.anewcanvas = Canvas(param1)
            elif cnrlist[i - 1] == 2:
                cmap1 = colormaplist[resultnum[0]]
                param1 = {'image': list1[imagenum[0]], 'mode': 7, 'color': problist[resultnum[0]],
                          'cmap': cmap1, 'trans': vtr1, 'shape': shapelist[imagenum[0]]}
                self.anewcanvas = Canvas(param1)
            else:
                cmaps = colormaplist[resultnum[0]]
                self.patch_color_df = pandas.read_csv('configGUI/patch_color.csv')
                count = len(cmaps)
                for i in range(count):
                    self.patch_color_df.loc[i, 'color'] = matplotlib.colors.to_hex(cmaps[i])
                    self.patch_color_df.loc[i, 'class'] = classlist[resultnum[0]][i]
                self.patch_color_df.to_csv('configGUI/patch_color.csv', index=False)
                param1 = {'image': list1[imagenum[0]], 'mode': 4,
                          'color': problist[resultnum[0]],
                          'hatch': hatchlist[resultnum[0]], 'cmap': cmaps, 'trans': vtrs,
                          'shape': shapelist[imagenum[0]]}
                self.anewcanvas = Canvas(param1)
            self.loadImage()
        else:
            self.anewscene.clear()
            self.slicelabel.setText('')
            self.graylabel.setText('')
            self.zoomlabel.setText('')
            if self.islinked == True:
                self.linkon.setIcon(self.icon2)
                self.islinked = False
                self.skiplink = False
                self.skipdis = True
                self.Viewpanel.zoom_link.disconnect()
                self.Viewpanel.move_link.disconnect()
                self.in_link.emit()

    def newSliceview(self):
        self.graylabel.setText('G %s' % (self.anewcanvas.graylist))

    def updateSlices(self, elist):
        self.slicelabel.setText(
            'S %s' % (elist[0] + 1) + '/ %s' % (elist[1]) + "       " + 'T %s' % (elist[2] + 1) + '/ %s' % (
                elist[3]) + "       " + 'D %s' % (elist[4] + 1) + '/ %s' % (elist[5]))
        indlist.append(elist[0])
        self.scrollSignal.emit(self.current_slice())

    def updateZoom(self, data):
        self.zoomscale = data
        if self.viewnr1 == 2:
            self.zoomlabel.setText('YZ %s' % (data))
        elif self.viewnr1 == 3:
            self.zoomlabel.setText('XZ %s' % (data))
        elif self.viewnr1 == 1:
            self.zoomlabel.setText('XY %s' % (data))
        self.zoomSignal.emit(self.zoomlabel.text())

    def updateGray(self, elist):
        self.graylabel.setText('G %s' % (elist))

    def clearWidgets(self):
        for i in reversed(range(self.count())):
            widget = self.takeAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def setGrey(self):
        maxv, minv, ok = grey_window.getData()
        if ok and self.anewcanvas:
            greylist = []
            greylist.append(minv)
            greylist.append(maxv)
            self.anewcanvas.set_greyscale(greylist)

    def mouse_move(self, event):

        x = event.xdata
        y = event.ydata
        if y:
            z = y / 3.3
            if x:
                x = "%.2f" % x
                y = "%.2f" % y
                z = "%.2f" % z
                if self.viewnr1 == 1:
                    self.zoomlabel.setText(
                        'XY %s' % self.zoomscale + "             Pos     " + 'X %s' % x + "      " + 'Y %s' % y)
                elif self.viewnr1 == 2:
                    y = x
                    self.zoomlabel.setText(
                        'YZ %s' % self.zoomscale + "             Pos     " + "Y %s" % y + "      " + 'Z %s' % z)
                elif self.viewnr1 == 3:
                    self.zoomlabel.setText(
                        'XZ %s' % self.zoomscale + "             Pos     " + "X %s" % x + "      " + 'Z %s' % z)
                else:
                    pass
        else:
            pass

    def mouse_tracking(self, cursor):
        if self.anewcanvas is not None:
            self.anewcanvas.set_cursor2D(cursor)
            self.anewcanvas.blit(self.anewcanvas.ax1.bbox)


class Viewline(QtWidgets.QGridLayout):
    in_link = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(Viewline, self).__init__(parent)

        self.vmode = 1
        self.oldindex = 0
        self.newcanvas1 = None
        self.newcanvas2 = None
        self.newcanvas3 = None
        self.islinked = False
        self.skiplink = False
        self.skipdis = True
        self.cursor_on = False
        self.background1 = None
        self.background2 = None
        self.background3 = None

        self.gridLayout_1 = QtWidgets.QGridLayout()
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_4 = QtWidgets.QGridLayout()

        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/Icons/switch2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.icon2 = QtGui.QIcon()
        self.icon2.addPixmap(QtGui.QPixmap(":/icons/Icons/blink.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/Icons/edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.icon4 = QtGui.QIcon()
        self.icon4.addPixmap(QtGui.QPixmap(":/icons/Icons/link.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.sizePolicy.setHorizontalStretch(0)
        self.sizePolicy.setVerticalStretch(0)

        self.imagebox = QtWidgets.QComboBox()
        self.imagebox.addItem('closed')
        self.refbox = QtWidgets.QComboBox()
        self.refbox.addItem('closed')
        self.refbox.setDisabled(True)
        self.bimre = QtWidgets.QPushButton()
        self.bimre.setText('')
        self.bimre.setSizePolicy(self.sizePolicy)
        self.bimre.setIcon(icon1)
        self.blinkon = QtWidgets.QPushButton()
        self.blinkon.setText('')
        self.blinkon.setSizePolicy(self.sizePolicy)
        self.blinkon.setIcon(self.icon2)
        self.gridLayout_1.addWidget(self.imagebox, 0, 0, 1, 1)
        self.gridLayout_1.addWidget(self.refbox, 0, 1, 1, 1)
        self.gridLayout_1.addWidget(self.bimre, 0, 2, 1, 1)
        self.gridLayout_1.addWidget(self.blinkon, 0, 3, 1, 1)

        self.ed31 = QtWidgets.QPushButton()
        self.ed31.setText('')
        self.ed31.setSizePolicy(self.sizePolicy)
        self.ed31.setIcon(icon3)
        self.ed32 = QtWidgets.QPushButton()
        self.ed32.setText('')
        self.ed32.setSizePolicy(self.sizePolicy)
        self.ed32.setIcon(icon3)
        self.ed33 = QtWidgets.QPushButton()
        self.ed33.setText('')
        self.ed33.setSizePolicy(self.sizePolicy)
        self.ed33.setIcon(icon3)
        self.sizePolicy.setHeightForWidth(self.bimre.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.blinkon.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.ed31.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.ed32.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.ed33.sizePolicy().hasHeightForWidth())

        self.grt1 = QtWidgets.QLabel()
        self.zot1 = QtWidgets.QLabel()
        self.grt2 = QtWidgets.QLabel()
        self.zot2 = QtWidgets.QLabel()
        self.grt3 = QtWidgets.QLabel()
        self.zot3 = QtWidgets.QLabel()
        self.grt1.setFrameShape(QtWidgets.QFrame.Panel)
        self.grt1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zot1.setFrameShape(QtWidgets.QFrame.Panel)
        self.zot1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.grt2.setFrameShape(QtWidgets.QFrame.Panel)
        self.grt2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zot2.setFrameShape(QtWidgets.QFrame.Panel)
        self.zot2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.grt3.setFrameShape(QtWidgets.QFrame.Panel)
        self.grt3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zot3.setFrameShape(QtWidgets.QFrame.Panel)
        self.zot3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slt1 = QtWidgets.QLabel()
        self.slt2 = QtWidgets.QLabel()
        self.slt3 = QtWidgets.QLabel()
        self.slt1.setFrameShape(QtWidgets.QFrame.Panel)
        self.slt1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slt2.setFrameShape(QtWidgets.QFrame.Panel)
        self.slt2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slt3.setFrameShape(QtWidgets.QFrame.Panel)
        self.slt3.setFrameShadow(QtWidgets.QFrame.Raised)

        self.gridLayout_2.addWidget(self.slt1, 1, 0, 1, 1)
        self.gridLayout_2.addWidget(self.zot1, 1, 1, 1, 1)
        self.gridLayout_2.addWidget(self.grt1, 1, 2, 1, 1)
        self.gridLayout_2.addWidget(self.ed31, 1, 3, 1, 1)

        self.gridLayout_3.addWidget(self.slt2, 1, 0, 1, 1)
        self.gridLayout_3.addWidget(self.zot2, 1, 1, 1, 1)
        self.gridLayout_3.addWidget(self.grt2, 1, 2, 1, 1)
        self.gridLayout_3.addWidget(self.ed32, 1, 3, 1, 1)

        self.gridLayout_4.addWidget(self.slt3, 1, 0, 1, 1)
        self.gridLayout_4.addWidget(self.zot3, 1, 1, 1, 1)
        self.gridLayout_4.addWidget(self.grt3, 1, 2, 1, 1)
        self.gridLayout_4.addWidget(self.ed33, 1, 3, 1, 1)

        self.addLayout(self.gridLayout_1, 0, 0, 1, 3)
        self.addLayout(self.gridLayout_2, 1, 0, 1, 1)
        self.addLayout(self.gridLayout_3, 1, 1, 1, 1)
        self.addLayout(self.gridLayout_4, 1, 2, 1, 1)

        self.Viewpanel1 = Activeview()
        self.Viewpanel2 = Activeview()
        self.Viewpanel3 = Activeview()
        self.scene1 = Activescene()
        self.scene2 = Activescene()
        self.scene3 = Activescene()

        self.Viewpanel1.zooming_data.connect(self.updateZoom1)
        self.Viewpanel2.zooming_data.connect(self.updateZoom2)
        self.Viewpanel3.zooming_data.connect(self.updateZoom3)

        self.addWidget(self.Viewpanel1, 2, 0, 1, 1)
        self.addWidget(self.Viewpanel2, 2, 1, 1, 1)
        self.addWidget(self.Viewpanel3, 2, 2, 1, 1)
        self.itemAtPosition(2, 0).widget().setScene(self.scene1)
        self.itemAtPosition(2, 1).widget().setScene(self.scene2)
        self.itemAtPosition(2, 2).widget().setScene(self.scene3)

        self.bimre.clicked.connect(self.switchMode)
        self.imagebox.currentIndexChanged.connect(self.loadScene_image)
        self.ed31.clicked.connect(self.setGrey1)
        self.ed32.clicked.connect(self.setGrey2)
        self.ed33.clicked.connect(self.setGrey3)
        self.refbox.currentIndexChanged.connect(self.loadScene_result)
        self.blinkon.clicked.connect(self.linkPanel)

        self.zoomscale1 = 1
        self.zoomscale2 = 1
        self.zoomscale3 = 1

    def switchMode(self):
        if self.vmode == 1:
            self.vmode = 2
            # self.im_re.setText('Result')
            self.refbox.setDisabled(False)
            self.imagebox.setDisabled(True)
            self.imagebox.setCurrentIndex(0)
            self.refbox.setCurrentIndex(0)
        else:
            self.vmode = 1
            # self.im_re.setText('Image')
            self.refbox.setDisabled(True)
            self.imagebox.setDisabled(False)
            self.refbox.setCurrentIndex(0)
            self.imagebox.setCurrentIndex(0)

    def addPathim(self, pathDicom):
        imageItems = [self.imagebox.itemText(i) for i in range(self.imagebox.count())]
        region = os.path.split(pathDicom)
        proband = os.path.split(os.path.split(region[0])[0])[1]
        region = region[1]
        newItem = 'Proband: %s' % (proband) + '   Image: %s' % (region)
        if newItem not in imageItems:
            self.imagebox.addItem(newItem)

    def addPathre(self, pathColor):
        self.refbox.addItem(pathColor)

    def linkPanel(self):
        if self.imagebox.currentIndex() == 0 and self.refbox.currentIndex() == 0:  # trigger condition
            pass
        else:
            if self.islinked == False:
                self.blinkon.setIcon(self.icon4)
                self.islinked = True
                self.in_link.emit()
            else:
                self.blinkon.setIcon(self.icon2)
                self.islinked = False
                self.in_link.emit()

    def loadScene_image(self, i):
        imagenum.clear()
        imagenum.append(i - 1)
        if i != 0:
            param1 = {'image': list1[imagenum[0]], 'mode': 1, 'shape': shapelist[imagenum[0]]}
            param2 = {'image': list1[imagenum[0]], 'mode': 2, 'shape': shapelist[imagenum[0]]}
            param3 = {'image': list1[imagenum[0]], 'mode': 3, 'shape': shapelist[imagenum[0]]}
            self.newcanvas1 = Canvas(param1)
            self.newcanvas2 = Canvas(param2)
            self.newcanvas3 = Canvas(param3)
            self.oldindex = i
            self.loadImage()
        else:
            self.scene1.clear()
            self.scene2.clear()
            self.scene3.clear()
            self.slt1.setText('')
            self.grt1.setText('')
            self.zot1.setText('')
            self.slt2.setText('')
            self.grt2.setText('')
            self.zot2.setText('')
            self.slt3.setText('')
            self.grt3.setText('')
            self.zot3.setText('')
            if self.islinked == True:
                self.islinked = False
                self.skiplink = False
                self.skipdis = True
                self.Viewpanel1.zoom_link.disconnect()
                self.Viewpanel1.move_link.disconnect()
                self.Viewpanel2.zoom_link.disconnect()
                self.Viewpanel2.move_link.disconnect()
                self.Viewpanel3.zoom_link.disconnect()
                self.Viewpanel3.move_link.disconnect()
                self.in_link.emit()

    def loadScene_result(self, i):
        resultnum.clear()
        resultnum.append(i - 1)
        imagenum.clear()
        imagenum.append(self.oldindex - 1)
        if i != 0:
            if cnrlist[i - 1] == 11:
                cmap3 = colormaplist[resultnum[0]]
                param1 = {'image': list1[imagenum[0]], 'mode': 4, 'color': problist[resultnum[0]],
                          'hatch': hatchlist[resultnum[0]], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                          'shape': shapelist[imagenum[0]]}
                self.newcanvas1 = Canvas(param1)
                param2 = {'image': list1[imagenum[0]], 'mode': 5, 'color': problist[resultnum[0]],
                          'hatch': hatchlist[resultnum[0]], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                          'shape': shapelist[imagenum[0]]}
                self.newcanvas2 = Canvas(param2)
                param3 = {'image': list1[imagenum[0]], 'mode': 6, 'color': problist[resultnum[0]],
                          'hatch': hatchlist[resultnum[0]], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                          'shape': shapelist[imagenum[0]]}
                self.newcanvas3 = Canvas(param3)
            elif cnrlist[i - 1] == 2:
                cmap1 = colormaplist[resultnum[0]]
                param1 = {'image': list1[imagenum[0]], 'mode': 7, 'color': problist[resultnum[0]],
                          'cmap': cmap1, 'trans': vtr1, 'shape': shapelist[imagenum[0]]}
                self.newcanvas1 = Canvas(param1)
                param2 = {'image': list1[imagenum[0]], 'mode': 8, 'color': problist[resultnum[0]],
                          'cmap': cmap1, 'trans': vtr1, 'shape': shapelist[imagenum[0]]}
                self.newcanvas2 = Canvas(param2)
                param3 = {'image': list1[imagenum[0]], 'mode': 9, 'color': problist[resultnum[0]],
                          'cmap': cmap1, 'trans': vtr1, 'shape': shapelist[imagenum[0]]}
                self.newcanvas3 = Canvas(param3)
            else:
                cmaps = colormaplist[resultnum[0]]
                self.patch_color_df = pandas.read_csv('configGUI/patch_color.csv')
                count = len(cmaps)
                for i in range(count):
                    self.patch_color_df.loc[i, 'color'] = matplotlib.colors.to_hex(cmaps[i])
                    self.patch_color_df.loc[i, 'class'] = classlist[resultnum[0]][i]
                self.patch_color_df.to_csv('configGUI/patch_color.csv', index=False)

                param1 = {'image': list1[imagenum[0]], 'mode': 4,
                          'color': problist[resultnum[0]],
                          'hatch': hatchlist[resultnum[0]], 'cmap': cmaps, 'trans': vtrs,
                          'shape': shapelist[imagenum[0]]}
                self.newcanvas1 = Canvas(param1)

                param2 = {'image': list1[imagenum[0]], 'mode': 5,
                          'color': problist[resultnum[0]],
                          'hatch': hatchlist[resultnum[0]], 'cmap': cmaps, 'trans': vtrs,
                          'shape': shapelist[imagenum[0]]}
                self.newcanvas2 = Canvas(param2)

                param3 = {'image': list1[imagenum[0]], 'mode': 6,
                          'color': problist[resultnum[0]],
                          'hatch': hatchlist[resultnum[0]], 'cmap': cmaps, 'trans': vtrs,
                          'shape': shapelist[imagenum[0]]}
                self.newcanvas3 = Canvas(param3)
            self.loadImage()
        else:
            self.scene1.clear()
            self.scene2.clear()
            self.scene3.clear()
            self.slt1.setText('')
            self.grt1.setText('')
            self.zot1.setText('')
            self.slt2.setText('')
            self.grt2.setText('')
            self.zot2.setText('')
            self.slt3.setText('')
            self.grt3.setText('')
            self.zot3.setText('')
            if self.islinked == True:
                self.islinked = False
                self.skiplink = False
                self.skipdis = True
                self.Viewpanel1.zoom_link.disconnect()
                self.Viewpanel1.move_link.disconnect()
                self.Viewpanel2.zoom_link.disconnect()
                self.Viewpanel2.move_link.disconnect()
                self.Viewpanel3.zoom_link.disconnect()
                self.Viewpanel3.move_link.disconnect()
                self.in_link.emit()

    def loadImage(self):

        self.scene1.clear()
        self.scene2.clear()
        self.scene3.clear()
        self.scene1.addWidget(self.newcanvas1)
        self.scene2.addWidget(self.newcanvas2)
        self.scene3.addWidget(self.newcanvas3)
        self.Viewpanel1.zoomback()
        self.Viewpanel2.zoomback()
        self.Viewpanel3.zoomback()
        self.slt1.setText('S %s' % (self.newcanvas1.ind + 1) + '/ %s' % (self.newcanvas1.slices))
        self.ind1 = self.newcanvas1.ind
        self.grt1.setText('G %s' % (self.newcanvas1.graylist))
        self.zot1.setText('XY')
        self.newcanvas1.update_data.connect(self.updateSlices1)
        self.newcanvas1.gray_data.connect(self.updateGray1)
        self.newcanvas1.new_page.connect(self.newSliceview1)
        self.newcanvas1.mpl_connect('motion_notify_event', self.mouse_move1)
        self.newcanvas1.mpl_connect('button_press_event', self.mouse_clicked1)
        self.slt2.setText('S %s' % (self.newcanvas2.ind + 1) + '/ %s' % (self.newcanvas2.slices))
        self.ind2 = self.newcanvas2.ind
        self.grt2.setText('G %s' % (self.newcanvas2.graylist))
        self.zot2.setText('YZ')
        self.newcanvas2.update_data.connect(self.updateSlices2)
        self.newcanvas2.gray_data.connect(self.updateGray2)
        self.newcanvas2.new_page.connect(self.newSliceview2)
        self.newcanvas2.mpl_connect('motion_notify_event', self.mouse_move2)
        self.newcanvas2.mpl_connect('button_press_event', self.mouse_clicked2)
        self.slt3.setText('S %s' % (self.newcanvas3.ind + 1) + '/ %s' % (self.newcanvas3.slices))
        self.ind3 = self.newcanvas3.ind
        self.grt3.setText('G %s' % (self.newcanvas3.graylist))
        self.zot3.setText('XZ')
        self.newcanvas3.update_data.connect(self.updateSlices3)
        self.newcanvas3.gray_data.connect(self.updateGray3)
        self.newcanvas3.new_page.connect(self.newSliceview3)
        self.newcanvas3.mpl_connect('motion_notify_event', self.mouse_move3)
        self.newcanvas3.mpl_connect('button_press_event', self.mouse_clicked3)
        if self.islinked == True:
            self.Viewpanel1.zoom_link.disconnect()
            self.Viewpanel1.move_link.disconnect()
            self.Viewpanel2.zoom_link.disconnect()
            self.Viewpanel2.move_link.disconnect()
            self.Viewpanel3.zoom_link.disconnect()
            self.Viewpanel3.move_link.disconnect()
            self.skiplink = False
            self.skipdis = True
            self.in_link.emit()

        self.axes = [self.newcanvas1.ax1, self.newcanvas2.ax1, self.newcanvas3.ax1]
        self.lines = []

    def newSliceview1(self):
        self.grt1.setText('G %s' % (self.newcanvas1.graylist))

    def newSliceview2(self):
        self.grt2.setText('G %s' % (self.newcanvas2.graylist))

    def newSliceview3(self):
        self.grt3.setText('G %s' % (self.newcanvas3.graylist))

    def updateSlices1(self, elist):
        self.ind1 = elist[0]
        self.slt1.setText(
            'S %s' % (elist[0] + 1) + '/ %s' % (elist[1]) + "       " + 'T %s' % (elist[2] + 1) + '/ %s' % (
                elist[3]) + "       " + 'D %s' % (elist[4] + 1) + '/ %s' % (elist[5]))
        indlist.append(elist[0])

    def updateZoom1(self, data):
        self.zoomscale1 = data
        self.zot1.setText('XY %s' % (data))

    def updateGray1(self, elist):
        self.grt1.setText('G %s' % (elist))

    def updateSlices2(self, elist):
        self.ind2 = elist[0]
        self.slt2.setText(
            'S %s' % (elist[0] + 1) + '/ %s' % (elist[1]) + "       " + 'T %s' % (elist[2] + 1) + '/ %s' % (
                elist[3]) + "       " + 'D %s' % (elist[4] + 1) + '/ %s' % (elist[5]))
        ind2list.append(elist[0])

    def updateZoom2(self, data):
        self.zoomscale2 = data
        self.zot2.setText('YZ %s' % (data))

    def updateGray2(self, elist):
        self.grt2.setText('G %s' % (elist))

    def updateSlices3(self, elist):
        self.ind3 = elist[0]
        self.slt3.setText(
            'S %s' % (elist[0] + 1) + '/ %s' % (elist[1]) + "       " + 'T %s' % (elist[2] + 1) + '/ %s' % (
                elist[3]) + "       " + 'D %s' % (elist[4] + 1) + '/ %s' % (elist[5]))
        ind3list.append(elist[0])

    def updateZoom3(self, data):
        self.zoomscale3 = data
        self.zot3.setText('XZ %s' % (data))

    def updateGray3(self, elist):
        self.grt3.setText('G %s' % (elist))

    def clearWidgets(self):
        for i in reversed(range(self.count())):
            widget = self.takeAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def setGrey1(self):
        maxv, minv, ok = grey_window.getData()
        if ok and self.newcanvas1:
            greylist = []
            greylist.append(minv)
            greylist.append(maxv)
            self.newcanvas1.set_greyscale(greylist)

    def setGrey2(self):
        maxv, minv, ok = grey_window.getData()
        if ok and self.newcanvas2:
            greylist = []
            greylist.append(minv)
            greylist.append(maxv)
            self.newcanvas2.set_greyscale(greylist)

    def setGrey3(self):
        maxv, minv, ok = grey_window.getData()
        if ok and self.newcanvas3:
            greylist = []
            greylist.append(minv)
            greylist.append(maxv)
            self.newcanvas3.set_greyscale(greylist)

    def mouse_move1(self, event):

        x = event.xdata
        if x:
            x = "%.2f" % x
        y = event.ydata
        if y:
            z = y / 3.3
            y = "%.2f" % y
            z = "%.2f" % z
            self.zot1.setText(
                'XY %s' % self.zoomscale1 + "             Pos     " + 'X %s' % x + "      " + 'Y %s' % y)

    def mouse_move2(self, event):

        x = event.xdata
        if x:
            x = "%.2f" % x
        y = event.ydata
        if y:
            z = y / 3.3
            y = "%.2f" % y
            z = "%.2f" % z
            self.zot2.setText(
                'YZ %s' % self.zoomscale2 + "             Pos     " + 'Y %s' % y + "      " + 'Z %s' % z)

    def mouse_move3(self, event):

        x = event.xdata
        if x:
            x = "%.2f" % x
        y = event.ydata
        if y:
            z = y / 3.3
            y = "%.2f" % y
            z = "%.2f" % z
            self.zot3.setText(
                'XZ %s' % self.zoomscale3 + "             Pos     " + 'X %s' % x + "      " + 'Z %s' % z)

    def mouse_clicked1(self, event):

        self.ind2 = int(event.xdata)
        self.ind3 = int(event.ydata)
        if imagenum == []:
            voxel = list1[0]
            shape = shapelist[0]
        else:
            voxel = list1[imagenum[0]]
            shape = shapelist[imagenum[0]]
        img = np.swapaxes(voxel[self.ind2, :, :], 0, 1)
        self.axes[1].imshow(img, cmap='gray',
                            extent=[0, img.shape[1], img.shape[0], 0], interpolation='sinc')
        img = np.swapaxes(voxel[:, self.ind3, :], 0, 1)
        self.axes[2].imshow(img, cmap='gray',
                            extent=[0, img.shape[1], img.shape[0], 0], interpolation='sinc')

        # for line in self.newcanvas1.ax1.lines:
        #     self.newcanvas1.ax1.draw_artist(line)
        self.newcanvas1.blit(self.newcanvas1.figure.bbox)
        self.newcanvas1.draw()
        # for line in self.newcanvas2.ax1.lines:
        #     self.newcanvas2.ax1.draw_artist(line)
        self.newcanvas2.blit(self.newcanvas2.figure.bbox)
        self.newcanvas2.draw()
        # for line in self.newcanvas3.ax1.lines:
        #     self.newcanvas3.ax1.draw_artist(line)
        self.newcanvas3.blit(self.newcanvas3.figure.bbox)
        self.newcanvas3.draw()
        if self.cursor_on:
            self.newcanvas1.set_cursor_position(event.xdata, event.ydata)
            self.newcanvas2.set_cursor_position(event.ydata, self.ind1 * 3.3)
            self.newcanvas3.set_cursor_position(event.xdata, self.ind1 * 3.3)

    def mouse_clicked2(self, event):

        self.ind1 = int(event.ydata // 3.3)
        self.ind3 = int(event.xdata)

        if imagenum == []:
            voxel = list1[0]
            shape = shapelist[0]
        else:
            voxel = list1[imagenum[0]]
            shape = shapelist[imagenum[0]]
        img = np.swapaxes(voxel[:, :, self.ind1], 0, 1)
        self.axes[0].imshow(img, cmap='gray',
                            extent=[0, img.shape[1], img.shape[0], 0], interpolation='sinc')
        img = np.swapaxes(voxel[:, self.ind3, :], 0, 1)
        self.axes[2].imshow(img, cmap='gray',
                            extent=[0, img.shape[1], img.shape[0], 0], interpolation='sinc')

        # for line in self.newcanvas1.ax1.lines:
        #     self.newcanvas1.ax1.draw_artist(line)
        self.newcanvas1.blit(self.newcanvas1.figure.bbox)
        self.newcanvas1.draw()
        # for line in self.newcanvas2.ax1.lines:
        #     self.newcanvas2.ax1.draw_artist(line)
        self.newcanvas2.blit(self.newcanvas2.figure.bbox)
        self.newcanvas2.draw()
        # for line in self.newcanvas3.ax1.lines:
        #     self.newcanvas3.ax1.draw_artist(line)
        self.newcanvas3.blit(self.newcanvas3.figure.bbox)
        self.newcanvas3.draw()
        if self.cursor_on:
            self.newcanvas1.set_cursor_position(self.ind2, event.xdata)
            self.newcanvas2.set_cursor_position(event.xdata, event.ydata)
            self.newcanvas3.set_cursor_position(self.ind2, event.ydata)

    def mouse_clicked3(self, event):

        self.ind1 = int(event.ydata // 3.3)
        self.ind2 = int(event.xdata)

        if imagenum == []:
            voxel = list1[0]
            shape = shapelist[0]
        else:
            voxel = list1[imagenum[0]]
            shape = shapelist[imagenum[0]]
        img = np.swapaxes(voxel[:, :, self.ind1], 0, 1)
        self.axes[0].imshow(img, cmap='gray', extent=[0, img.shape[1], img.shape[0], 0])
        img = np.swapaxes(voxel[self.ind2, :, :], 0, 1)
        self.axes[1].imshow(img, cmap='gray',
                            extent=[0, img.shape[1], img.shape[0], 0], interpolation='sinc')

        # for line in self.newcanvas1.ax1.lines:
        #     self.newcanvas1.ax1.draw_artist(line)
        self.newcanvas1.blit(self.newcanvas1.figure.bbox)
        self.newcanvas1.draw()
        # for line in self.newcanvas2.ax1.lines:
        #     self.newcanvas2.ax1.draw_artist(line)
        self.newcanvas2.blit(self.newcanvas2.figure.bbox)
        self.newcanvas2.draw()
        # for line in self.newcanvas3.ax1.lines:
        #     self.newcanvas3.ax1.draw_artist(line)
        self.newcanvas3.blit(self.newcanvas3.figure.bbox)
        self.newcanvas3.draw()
        if self.cursor_on:
            self.newcanvas1.set_cursor_position(event.xdata, self.ind3)
            self.newcanvas2.set_cursor_position(self.ind3, event.ydata)
            self.newcanvas3.set_cursor_position(event.xdata, event.ydata)

    def mouse_tracking(self, cursor):
        self.cursor_on = cursor
        if self.newcanvas1 is not None:
            self.newcanvas1.set_cursor2D(self.cursor_on)
            self.newcanvas1.blit(self.newcanvas1.ax1.bbox)
        if self.newcanvas2 is not None:
            self.newcanvas2.set_cursor2D(self.cursor_on)
            self.newcanvas2.blit(self.newcanvas2.ax1.bbox)
        if self.newcanvas3 is not None:
            self.newcanvas3.set_cursor2D(self.cursor_on)
            self.newcanvas3.blit(self.newcanvas3.ax1.bbox)


##################
sys._excepthook = sys.excepthook


def my_exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


def start(*args):
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = imagine(*args)
    mainWindow.showMaximized()
    mainWindow.show()
    # exceptionHandler.errorSignal.connect(something)
    sys.exit(app.exec_())


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = imagine()
    mainWindow.showMaximized()
    mainWindow.show()
    sys.exit(app.exec_())
