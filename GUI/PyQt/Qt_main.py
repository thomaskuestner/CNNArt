import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os
import dicom
import dicom_numpy
from matplotlib import pyplot as plt
from matplotlib.widgets import LassoSelector, RectangleSelector, EllipseSelector
from matplotlib import path
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

from framework1 import Ui_MainWindow
from Patches_window import*
from cPre_window import*
from label_window import*

from activeview import Activeview
from activescene import Activescene
from canvas import Canvas
from utilsGUI.Unpatch import*
from utilsGUI.Unpatch_eight import*
from utilsGUI.Unpatch_two import*

import sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from DatabaseInfo import*
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import json

from pathlib import Path

from loadingIcon import Overlay
import scipy.io as sio
from Grey_window import*
import sys
import h5py
from dlart import* # << resolve dependency from dlart
from matplotlib.figure import Figure

import pickle
import numpy as np
import pandas as pd
import codecs

from keras.utils.vis_utils import plot_model, model_to_dot
from keras.models import load_model
from loadf2 import *

import tensorflow as tf
import keras.backend as K
#import network_visualization
from deepvis.network_visualization import *

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    update_data = QtCore.pyqtSignal(list)
    gray_data = QtCore.pyqtSignal(list)
    new_page = QtCore.pyqtSignal()

    def __init__(self):
        super(MyApp, self).__init__()
        self.setupUi(self)

        self.scrollAreaWidgetContents1 = QtWidgets.QWidget()
        self.maingrids1 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents1)
        self.scrollArea1.setWidget(self.scrollAreaWidgetContents1)
        # self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollAreaWidgetContents2 = QtWidgets.QWidget()
        self.maingrids2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents2)
        self.scrollArea2.setWidget(self.scrollAreaWidgetContents2)

        self.vision = 2
        self.gridson = False
        self.gridsnr = 2
        self.voxel_ndarray = []
        self.i = -1
        self.linked = False

        with open('editlabel.json', 'r') as json_data:
            self.infos = json.load(json_data)
            self.labelnames = self.infos['names']
            self.labelcolor = self.infos['colors']
            self.pathROI = self.infos['path'][0]

        global pathlist, list1, shapelist, pnamelist, empty1, cmap1, cmap3, hmap1, hmap2, vtr1,  \
            vtr3, problist, hatchlist, correslist, cnrlist
        pathlist = []
        list1 = []
        shapelist = []
        pnamelist = []
        empty1 = []

        with open(os.path.join('configGUI','colors0.json'), 'r') as json_data:
            self.dcolors = json.load(json_data)
            cmap1 = self.dcolors['class2']['colors']
            cmap1 = mpl.colors.ListedColormap(cmap1)
            cmap3 = self.dcolors['class11']['colors']
            cmap3 = mpl.colors.ListedColormap(cmap3)
            hmap1 = self.dcolors['class8']['hatches']
            hmap2 = self.dcolors['class11']['hatches']
            vtr1 = self.dcolors['class2']['trans'][0]
            vtr3 = self.dcolors['class11']['trans'][0]

        problist = []
        hatchlist = []
        correslist = []
        cnrlist = []

        self.newfig = plt.figure(50) # 3
        self.newfig.set_facecolor("black")
        self.newax = self.newfig.add_subplot(111)
        self.newax.axis('off')
        self.pltc = None
        self.newcanvas = FigureCanvas(self.newfig)
        self.keylist = []
        self.mrinmain = None
        self.labelimage = False

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
        self.maingrids2.addWidget(self.graylabel, 0, 0, 1, 1)
        self.maingrids2.addWidget(self.slicelabel, 0, 1, 1, 1)
        self.maingrids2.addWidget(self.zoomlabel, 0, 2, 1, 1)
        self.maingrids2.addWidget(self.seditgray, 0, 3, 1, 1)
        self.viewLabel = Activeview()
        self.sceneLabel = Activescene()
        self.sceneLabel.addWidget(self.newcanvas)
        self.viewLabel.setScene(self.sceneLabel)
        self.maingrids2.addWidget(self.viewLabel, 1, 0, 1, 4)

        def formsclick(n):
            if n==0:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(False)
            elif n==1:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(True)
                toggle_selector.LS.set_active(False)
            elif n==2:
                toggle_selector.ES.set_active(True)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(False)
            elif n==3:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(True)

        def toggle_selector(event):
            if self.brectangle.isChecked() and not toggle_selector.RS.active and (
                        toggle_selector.LS.active or toggle_selector.ES.active):
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(True)
                toggle_selector.LS.set_active(False)
            elif self.bellipse.isChecked() and not toggle_selector.ES.active and (
                        toggle_selector.LS.active or toggle_selector.RS.active):
                toggle_selector.ES.set_active(True)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(False)
            elif self.blasso.isChecked() and (toggle_selector.ES.active or toggle_selector.RS.active
                ) and not toggle_selector.LS.active:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(True)
            elif self.bnoselect.isChecked() and (toggle_selector.ES.active or toggle_selector.RS.active
                                                 or toggle_selector.LS.active):
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(False)

        def lasso_onselect(verts):
            p = path.Path(verts)
            n = self.labelbox.currentIndex()
            col_str = '3' + str(n)
            patch = patches.PathPatch(p, fill=True, alpha=.2, edgecolor= None, facecolor=self.labelcolor[n])
            self.newax.add_patch(patch)

            sepkey = os.path.split(self.selectorPath)
            sepkey = sepkey[1]
            layer_name = sepkey  # bodyregion:t1_tse...

            with open(self.markfile, 'r') as json_data:
                saveFile = json.load(json_data)
                p = np.ndarray.tolist(p.vertices)
                if self.ind < 9:
                    ind = '0' + str(self.ind+1)
                else:
                    ind = str(self.ind+1)

                num = 0  # num of some specific roi
                if layer_name in saveFile['layer']:
                    for key in saveFile['layer'][layer_name].keys():
                        if key[3]==str(n):     # str!！
                            num+=1
                    if num<9:
                        numinkey='0' + str(num + 1)
                    else:
                        numinkey=str(num + 1)
                    number_str = ind + col_str + numinkey
                    saveFile['layer'][layer_name][number_str] = {'vertices': p, 'codes': None}
                else:
                    numinkey ='01'
                    number_str = ind + col_str + numinkey
                    saveFile['layer'][layer_name] = {number_str: {'vertices': p, 'codes': None}}
                saveFile['names']['list'] = self.labelnames
                saveFile['colors']['list'] = self.labelcolor
            with open(self.markfile, 'w') as json_data:
                json_data.write(json.dumps(saveFile))

            self.orderROIs()

        def ronselect(eclick, erelease):
            col_str = None
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            p = np.array(([x1, y1, x2, y2]))
            sepkey = os.path.split(self.selectorPath)
            sepkey = sepkey[1]
            layer_name = sepkey

            n = self.labelbox.currentIndex()
            if toggle_selector.RS.active and not toggle_selector.ES.active:
                col_str = '1' + str(n)
                rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=True,
                                     alpha=.2, edgecolor= None, facecolor=self.labelcolor[n])
                self.newax.add_patch(rect)
            elif toggle_selector.ES.active and not toggle_selector.RS.active:
                col_str = '2' + str(n)
                ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                              width=np.abs(x1 - x2), height=np.abs(y1 - y2), alpha=.2, edgecolor= None,
                              facecolor=self.labelcolor[n])
                self.newax.add_patch(ell)

            with open(self.markfile, 'r') as json_data:
                saveFile = json.load(json_data)
                p = np.ndarray.tolist(p)  # format from datapre

                if self.ind < 9:
                    ind = '0' + str(self.ind+1)
                else:
                    ind = str(self.ind+1)
                num = 0
                if layer_name in saveFile['layer']:
                    for key in saveFile['layer'][layer_name].keys():
                        if key[3]==str(n):
                            num+=1
                    if num<9:
                        numinkey='0' + str(num + 1)
                    else:
                        numinkey=str(num + 1)
                    number_str = ind +  col_str + numinkey
                    saveFile['layer'][layer_name][number_str] = {'points': p}
                else:
                    numinkey = '01'
                    number_str = ind + col_str + numinkey
                    saveFile['layer'][layer_name] = {number_str: {'points': p}}
                saveFile['names']['list'] = self.labelnames
                saveFile['colors']['list'] = self.labelcolor
            with open(self.markfile, 'w') as json_data:
                json_data.write(json.dumps(saveFile))

            self.orderROIs()

        toggle_selector.RS = RectangleSelector(self.newax, ronselect, button=[1], drawtype='box', useblit=True,
                                       minspanx=5, minspany=5,spancoords='pixels',interactive=False)

        toggle_selector.ES = EllipseSelector(self.newax, ronselect, drawtype='box', button=[1], minspanx=5,
                                             useblit=True, minspany=5, spancoords='pixels',interactive=False)

        toggle_selector.LS = LassoSelector(self.newax, lasso_onselect, button=[1])

        toggle_selector.ES.set_active(False)
        toggle_selector.RS.set_active(False)
        toggle_selector.LS.set_active(False)

        self.openfile.clicked.connect(self.loadMR)
        self.bdswitch.clicked.connect(self.switchview)
        self.bgrids.clicked.connect(self.setlayout)
        self.bpatch.clicked.connect(self.loadPatch)
        self.bsetcolor.clicked.connect(self.setColor)

        self.bselectoron.clicked.connect(self.selectormode)
        self.bchoosemark.clicked.connect(self.chooseMark)
        self.bdelete.clicked.connect(self.deleteLabel)
        self.labelbox.setDisabled(True)
        self.brectangle.setDisabled(True)
        self.bellipse.setDisabled(True)
        self.blasso.setDisabled(True)
        self.bchoosemark.setDisabled(True)
        self.bnoselect.setDisabled(True)
        self.labelWidget.setDisabled(True)
        self.bdelete.setDisabled(True)
        self.bnoselect.toggled.connect(lambda:formsclick(0))
        self.brectangle.toggled.connect(lambda:formsclick(1))
        self.bellipse.toggled.connect(lambda:formsclick(2))
        self.blasso.toggled.connect(lambda:formsclick(3))
 #
        self.bnoselect.setChecked(True)
        self.brectangle.toggled.connect(lambda:self.stopView(1))
        self.bellipse.toggled.connect(lambda:self.stopView(1))
        self.blasso.toggled.connect(lambda: self.stopView(1))
        self.bnoselect.toggled.connect(lambda: self.stopView(0))

        self.selectoron = False
        self.x_clicked = None
        self.y_clicked = None
        self.mouse_second_clicked = False

        self.actionOpen_file.triggered.connect(self.loadMR)
        self.actionSave.triggered.connect(self.saveCurrent)
        self.actionLoad.triggered.connect(self.loadOld)
        self.actionColor.triggered.connect(self.defaultColor)
        self.actionLabels.triggered.connect(self.editLabel)

        # plt.connect('key_press_event', toggle_selector)
        self.ind = 0
        self.slice = 0
        self.newcanvas.mpl_connect('scroll_event', self.newonscroll)
        self.newcanvas.mpl_connect('button_press_event', self.mouse_clicked)
        self.newcanvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.newcanvas.mpl_connect('button_release_event', self.mouse_release)

        ############ second tab from Yannick
        # initialize DeepLearningArt Application
        self.deepLearningArtApp = DeepLearningArtApp()
        self.deepLearningArtApp.setGUIHandle(self)

        # initialize TreeView Database
        self.manageTreeView()

        # intialize TreeView Datasets
        self.manageTreeViewDatasets()

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
        self.ComboBox_splittingMode.setCurrentIndex(DeepLearningArtApp.SIMPLE_RANDOM_SAMPLE_SPLITTING)
        self.Label_SplittingParams.setText("using Test/Train="
                                           + str(self.deepLearningArtApp.getTrainTestDatasetRatio())
                                           + " and Valid/Train=" + str(
            self.deepLearningArtApp.getTrainValidationRatio()))

        # initialize combox box for DNN selection
        self.ComboBox_DNNs.addItem("Select Deep Neural Network Model...")
        self.ComboBox_DNNs.addItems(DeepLearningArtApp.deepNeuralNetworks.keys())
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

        # Signals and Slots

        # select database button clicked
        self.Button_DB.clicked.connect(self.button_DB_clicked)
        # self.Button_DB.clicked.connect(self.button_DB_clicked)

        # output path button for patching clicked
        self.Button_OutputPathPatching.clicked.connect(self.button_outputPatching_clicked)

        # TreeWidgets
        self.TreeWidget_Patients.clicked.connect(self.getSelectedPatients)
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

##################################3
        self.matplotlibwidget_static.show()
        # self.matplotlibwidget_static_2.hide()
        self.scrollArea.show()
        self.horizontalSliderPatch.hide()
        self.horizontalSliderSlice.hide()

        self.labelPatch.hide()
        self.labelSlice.hide()
        # self.horizontalSliderSS.hide()

        self.lcdNumberPatch.hide()
        self.lcdNumberSlice.hide()
        # self.lcdNumberSS.hide()

        self.radioButton_3.hide()
        self.radioButton_4.hide()

        self.wyChooseFile.setToolTip('Choose .H5 File')
        self.wyShowArchitecture.setToolTip('Show the Architecture')
        self.radioButton.setToolTip('Weights of Filters')
        self.radioButton_2.setToolTip('Feature Maps')
        self.wyPlot.setToolTip('Plot Weights or Filters')
        self.labelPatch.setToolTip('Number of Input Patch')
        self.labelSlice.setToolTip('Number of Feature Maps')
        self.wySubsetSelection.setToolTip('Created Input Patches With Subset Selection')
        self.radioButton_3.setToolTip('Plot the 1st input')
        self.radioButton_4.setToolTip('Plot the 2nd input')

        self.resetW=False
        self.resetF = False
        self.resetS = False

        self.twoInput=False
        self.chosenLayerName = []
        # the slider's value is the chosen patch's number
        self.chosenWeightNumber =1
        self.chosenWeightSliceNumber=1
        self.chosenPatchNumber = 1
        self.chosenPatchSliceNumber =1
        self.chosenSSNumber = 1
        self.openfile_name=''
        self.inputData_name=''
        self.inputData={}
        self.inputalpha = '0.19'
        self.inputGamma = '0.0000001'

        self.layer_index_name = {}
        self.model={}
        self.qList=[]
        self.totalWeights=0
        self.totalWeightsSlices =0
        self.totalPatches=0
        self.totalPatchesSlices =0
        self.totalSS=0

        self.modelDimension= ''
        self.modelName=''
        self.modelInput={}
        self.modelInput2={}
        self.ssResult={}
        self.activations = {}
        self.act = {}
        self.layers_by_depth={}
        self.weights ={}
        self.w={}
        self.LayerWeights = {}
        self.subset_selection = {}
        self.subset_selection_2 = {}
        self.radioButtonValue=[]
        self.listView.clicked.connect(self.clickList)

        self.W_F=''

        # slider of the weight and feature
        self.horizontalSliderPatch.sliderReleased.connect(self.sliderValue)
        self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)

        self.horizontalSliderSlice.sliderReleased.connect(self.sliderValue)
        self.horizontalSliderSlice.valueChanged.connect(self.lcdNumberSlice.display)

        # self.matplotlibwidget_static.mpl.wheel_scroll_W_signal.connect(self.wheelScrollW)
        self.matplotlibwidget_static.mpl.wheel_scroll_signal.connect(self.wheelScroll)
        # self.matplotlibwidget_static.mpl.wheel_scroll_3D_signal.connect(self.wheelScroll)
        # self.matplotlibwidget_static.mpl.wheel_scroll_SS_signal.connect(self.wheelScrollSS)

        self.lineEdit.textChanged[str].connect(self.textChangeAlpha)
        self.lineEdit_2.textChanged[str].connect(self.textChangeGamma)

    def switchview(self):
        if self.vision == 2:
            self.vision = 3
            self.visionlabel.setText('3D')
            self.columnbox.setCurrentIndex(2)
            self.columnlabel.setDisabled(True)
            self.columnbox.setDisabled(True)
            self.linebox.addItem("4")
            self.linebox.addItem("5")
            self.linebox.addItem("6")
            self.linebox.addItem("7")
            self.linebox.addItem("8")
            self.linebox.addItem("9")
        else:
            self.vision = 2
            self.visionlabel.setText('2D')
            self.columnbox.setCurrentIndex(0)
            self.linebox.removeItem(8)
            self.linebox.removeItem(7)
            self.linebox.removeItem(6)
            self.linebox.removeItem(5)
            self.linebox.removeItem(4)
            self.linebox.removeItem(3)
            self.columnlabel.setDisabled(False)
            self.columnbox.setDisabled(False)

    def clearall(self):
        if self.gridsnr == 2:
            for i in reversed(range(self.maingrids1.count())):
                self.maingrids1.itemAt(i).clearWidgets()
                for j in reversed(range(self.maingrids1.itemAt(i).secondline.count())):
                    self.maingrids1.itemAt(i).secondline.itemAt(j).widget().setParent(None)
                self.maingrids1.removeItem(self.maingrids1.itemAt(i)) # invisible
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

    def saveCurrent(self):
        if self.gridson:
            with open('lastWorkspace.json', 'r') as json_data:
                lastState = json.load(json_data)
                lastState['mode'] = self.gridsnr ###
                if self.gridsnr == 2:
                    lastState['layout'][0] = self.layoutlines
                    lastState['layout'][1] = self.layoutcolumns
                else:
                    lastState['layout'][0] = self.layout3D

                global pathlist, list1, pnamelist, problist, hatchlist, correslist, cnrlist, shapelist
                # shapelist = (list(shapelist)).tolist()
                # shapelist = pd.Series(shapelist).to_json(orient='values')
                lastState['Shape'] = shapelist
                lastState['Pathes'] = pathlist
                lastState['NResults'] = pnamelist
                lastState['NrClass'] = cnrlist
                lastState['Corres'] = correslist

            with open('lastWorkspace.json', 'w') as json_data:
                json_data.write(json.dumps(lastState))

            listA = open('dump1.txt', 'wb')
            pickle.dump(list1, listA)
            listA.close()
            listB = open('dump2.txt', 'wb')
            pickle.dump(problist, listB)
            listB.close()
            listC = open('dump3.txt', 'wb')
            pickle.dump(hatchlist, listC)
            listC.close()

    def loadOld(self):
        self.clearall()
        global pathlist, list1, pnamelist, problist, hatchlist, correslist, cnrlist, shapelist

        with open('lastWorkspace.json', 'r') as json_data:
            lastState = json.load(json_data)
            # list1 = lastState['listA']
            # problist = lastState['Probs']
            # hatchlist = lastState['Hatches']
            gridsnr = lastState['mode']  ##
            shapelist = lastState['Shape']
            pathlist = lastState['Pathes']
            pnamelist = lastState['NResults']
            cnrlist = lastState['NrClass']
            correslist = lastState['Corres']

            if gridsnr == 2:
                if self.vision == 3:
                    self.switchview() # back to 2
                self.layoutlines = lastState['layout'][0]
                self.layoutcolumns = lastState['layout'][1]
                self.linebox.setCurrentIndex(self.layoutlines - 1)
                self.columnbox.setCurrentIndex(self.layoutcolumns - 1)
            else:
                if self.vision == 2:
                    self.switchview()
                self.layout3D = lastState['layout'][0]
                self.linebox.setCurrentIndex(self.layout3D - 1)

        listA = open('dump1.txt', 'rb')
        list1 = pickle.load(listA)
        listA.close()
        listB = open('dump2.txt', 'rb')
        problist = pickle.load(listB)
        listB.close()
        listC = open('dump3.txt', 'rb')
        hatchlist = pickle.load(listC)
        listC.close()

        self.setlayout()

    def setlayout(self):
        self.gridson = True
        if self.vision == 2:
            self.clearall()
            self.gridsnr = 2
            self.layoutlines = self.linebox.currentIndex() + 1
            self.layoutcolumns = self.columnbox.currentIndex() + 1
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    blocklayout = Viewgroup()
                    for dpath in pathlist:
                        blocklayout.addPathd(dpath)
                    for cpath in pnamelist:
                        blocklayout.addPathre(cpath)
                    blocklayout.in_link.connect(self.linkMode)     # initial connection of in_link

                    self.maingrids1.addLayout(blocklayout, i, j)
            if pathlist:
                n = 0
                for i in range(self.layoutlines):
                    for j in range(self.layoutcolumns):
                        if n < len(pathlist):
                            self.maingrids1.itemAtPosition(i, j).pathbox.setCurrentIndex(n+1)
                            n+=1
                        else:
                            break
        else:
            self.clearall()
            self.gridsnr = 3
            self.layout3D = self.linebox.currentIndex() + 1
            for i in range(self.layout3D):
                blockline = Viewline()
                for dpath in pathlist:
                    blockline.addPathim(dpath)
                for cpath in pnamelist:
                    blockline.addPathre(cpath)
                blockline.in_link.connect(self.linkMode) # 3d initial
                self.maingrids1.addLayout(blockline, i, 0)
            if pathlist:
                n = 0
                for i in range(self.layout3D):
                    if n < len(pathlist):
                        self.maingrids1.itemAtPosition(i, 0).imagelist.setCurrentIndex(n+1)
                        n+=1
                    else:
                        break

    def linkMode(self):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids1.itemAtPosition(i, j).islinked and not self.maingrids1.itemAtPosition(i, j).skiplink:
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.zoom_link.connect(self.zoomAll)
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.move_link.connect(self.moveAll)
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.grey_link.connect(self.greyAll)
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.slice_link.connect(self.sliceAll)
                        self.maingrids1.itemAtPosition(i, j).skiplink = True # avoid multi link
                        self.maingrids1.itemAtPosition(i, j).skipdis = False
                    elif not self.maingrids1.itemAtPosition(i, j).islinked and not self.maingrids1.itemAtPosition(i, j).skipdis:
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.zoom_link.disconnect()
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.move_link.disconnect()
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.grey_link.disconnect()
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.slice_link.disconnect()
                        self.maingrids1.itemAtPosition(i, j).skipdis = True
                        self.maingrids1.itemAtPosition(i, j).skiplink = False
        else:
            for i in range(self.layout3D):
                if self.maingrids1.itemAtPosition(i, 0).islinked and not self.maingrids1.itemAtPosition(i, 0).skiplink:
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.zoom_link.connect(self.zoomAll)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.move_link.connect(self.moveAll)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas1.grey_link.connect(self.greyAll)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.zoom_link.connect(self.zoomAll)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.move_link.connect(self.moveAll)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas2.grey_link.connect(self.greyAll)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.zoom_link.connect(self.zoomAll)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.move_link.connect(self.moveAll)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas3.grey_link.connect(self.greyAll)
                    self.maingrids1.itemAtPosition(i, 0).skiplink = True
                    self.maingrids1.itemAtPosition(i, 0).skipdis = False
                elif not self.maingrids1.itemAtPosition(i, 0).islinked and not self.maingrids1.itemAtPosition(i, 0).skipdis:
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

    def zoomAll(self, factor):
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

    def moveAll(self, movelist):
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

    def greyAll(self, glist):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids1.itemAtPosition(i, j).islinked:
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.linkedGrey(glist)
        else:
            for i in range(self.layout3D):
                if self.maingrids1.itemAtPosition(i, 0).islinked:
                    self.maingrids1.itemAtPosition(i, 0).newcanvas1.linkedGrey(glist)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas2.linkedGrey(glist)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas3.linkedGrey(glist)

    def sliceAll(self, data):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids1.itemAtPosition(i, j).islinked:
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.linkedSlice(data)

    def loadMR(self):
        current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        cfgFile = Path(parent_dir + os.sep + 'config' + os.sep + 'param.yml')

        if(cfgFile.exists()):
            with open(parent_dir + os.sep + 'config' + os.sep + 'param.yml', 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
                if(Path(cfg['MRdatabase']).exists()):
                    dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])
                    sDICOMPath = dbinfo.sPathIn
                else:
                    sDICOMPath = current_dir
        else:
            sDICOMPath = current_dir

        if( not Path(sDICOMPath).exists()): # safety check
            sDICOMPath = current_dir

        self.PathDicom = QtWidgets.QFileDialog.getExistingDirectory(self, "open file", sDICOMPath)
        # self.PathDicom = QtWidgets.QFileDialog.getExistingDirectory(self, "open file", "C:/Users/hansw/Videos/artefacts")
        if self.PathDicom:
            self.i = self.i + 1
            self.openfile.setDisabled(True)
            self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
            self.overlay.setGeometry(QtCore.QRect(950, 400, 171, 141))
            self.overlay.show()
            from loadf import loadImage
            self.newMR = loadImage(self.PathDicom)
            self.newMR.trigger0.connect(self.loadEnd)
            self.newMR.start()
        else:
            pass

    def loadEnd(self):
        self.overlay.killTimer(self.overlay.timer)
        self.overlay.hide()

        if self.selectoron == False:
            self.openfile.setDisabled(False)

            # self.plot_3d(self.newMR.svoxel, 200)
            pathlist.append(self.PathDicom)
            list1.append(self.newMR.voxel_ndarray)
            shapelist.append(self.newMR.new_shape)
            if self.gridson == True:
                if self.gridsnr == 2:
                    for i in range(self.layoutlines):
                        for j in range(self.layoutcolumns):
                            self.maingrids1.itemAtPosition(i, j).addPathd(self.PathDicom)
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
                                self.maingrids1.itemAtPosition(i, 0).imagelist.currentIndex() == 0:
                            self.maingrids1.itemAtPosition(i, 0).imagelist.setCurrentIndex(len(pathlist))
                            break
            else:
                pass
        else:
            self.loadSelect()

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


    def loadPatch(self):
        resultfile = QtWidgets.QFileDialog.getOpenFileName(self, 'choose the result file', '',
                'mat files(*.mat);;h5 files(*.h5)', None, QtWidgets.QFileDialog.DontUseNativeDialog)[0]

        if resultfile:
            with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
            dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])
            PathDicom = QtWidgets.QFileDialog.getExistingDirectory(self, "choose the corresponding image", dbinfo.sPathIn)
            if PathDicom in pathlist:
                n = pathlist.index(PathDicom)
                correslist.append(n)
                conten = sio.loadmat(resultfile)
                if 'prob_pre' in conten:
                    cnum = np.array(conten['prob_pre'])
                    IType, IArte = self.unpatching11(conten['prob_pre'], list1[n])
                    # if IType[0] - list1[n][0] <= PatchSize/2 and IType[1] - list1[n][1] <= PatchSize/2:
                    # else:
                    #     QtWidgets.QMessageBox.information(self, 'Warning', 'Please choose the right file!')
                    #     break
                    problist.append(IType)
                    hatchlist.append(IArte)
                    cnrlist.append(11)
                else:
                    pred = conten['prob_test']
                    pred = pred[0:4320, :]
                    cnum = np.array(pred)
                    if cnum.shape[1] == 2:
                        IType = self.unpatching2(pred, list1[n])

                        problist.append(IType)
                        hatchlist.append(empty1)
                        cnrlist.append(2)
                    # elif cnum.shape[1] == 8:

                nameofCfile = os.path.split(resultfile)[1]
                nameofCfile = nameofCfile + '   class:' + str(cnum.shape[1])
                pnamelist.append(nameofCfile)
                if self.gridsnr == 3:  #############
                    for i in range(self.layout3D):
                        self.maingrids1.itemAtPosition(i, 0).addPathre(nameofCfile)
                else:
                    for i in range(self.layoutlines):
                        for j in range(self.layoutcolumns):
                            self.maingrids1.itemAtPosition(i, j).addPathre(nameofCfile)

            else:
                QtWidgets.QMessageBox.information(self, 'Warning', 'Please load the original file first!')
        else:
            pass

    def setColor(self):
        c1, c3, h1, h2, v1, v3, ok = Patches_window.getData()
        if ok:
            global cmap1, cmap3, hmap1, hmap2, vtr1, vtr3
            cmap1 = c1
            cmap3 = c3
            hmap1 = h1
            hmap2 = h2
            vtr1 = v1
            vtr3 = v3

    def defaultColor(self):
        c1, c3, h1, h2, v1, v3, ok = cPre_window.getData()
        if ok:
            global cmap1, cmap3, hmap1, hmap2, vtr1, vtr3
            cmap1 = c1
            cmap3 = c3
            hmap1 = h1
            hmap2 = h2
            vtr1 = v1
            vtr3 = v3

    def editLabel(self):
        labelnames, labelcolors, pathROI, ok= Label_window.getData()

        self.labelnames = labelnames
        self.labelcolor = labelcolors
        self.pathROI = pathROI
        self.labelbox.clear()
        self.labelbox.addItems(self.labelnames)

        with open('editlabel.json', 'r') as json_data:
            self.infos = json.load(json_data)
        with open('editlabel.json', 'w') as json_data:
            self.infos['names'] = self.labelnames
            self.infos['colors'] = self.labelcolor
            self.infos['path'][0] = self.pathROI
            json_data.write(json.dumps(self.infos))

        # if type(self.mrinmain)== 'numpy.ndarray': #
        if self.labelimage == True:
            with open(self.markfile, 'r') as json_data:
                saveFile = json.load(json_data)
            with open(self.markfile, 'w') as json_data:
                saveFile['names']['list'] = self.labelnames
                saveFile['colors']['list'] = self.labelcolor
                json_data.write(json.dumps(saveFile))
            self.updateList()
            # self.orderROIs()
##################################################
    def selectormode(self):
        if self.selectoron == False:
            self.selectoron = True
            self.stackedWidget.setCurrentIndex(1)
            icon2 = QtGui.QIcon()
            icon2.addPixmap(QtGui.QPixmap(":/icons/Icons/switchoff.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.bselectoron.setIcon(icon2)

            self.labelbox.setDisabled(False)
            self.brectangle.setDisabled(False)
            self.bellipse.setDisabled(False)
            self.blasso.setDisabled(False)
            self.bchoosemark.setDisabled(False)
            self.bnoselect.setDisabled(False)
            self.labelWidget.setDisabled(False)
            self.bdelete.setDisabled(False)

            self.bdswitch.setDisabled(True)
            self.linebox.setDisabled(True)
            self.columnbox.setDisabled(True)
            self.bgrids.setDisabled(True)
            self.openfile.setDisabled(True)
            self.bpatch.setDisabled(True)
            self.bsetcolor.setDisabled(True)

            self.actionOpen_file.setEnabled(False)
            self.actionSave.setEnabled(False)
            self.actionLoad.setEnabled(False)
        else:
            self.selectoron = False
            self.stackedWidget.setCurrentIndex(0)
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/icons/Icons/switchon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.bselectoron.setIcon(icon1)
            self.labelbox.setDisabled(True)
            self.brectangle.setDisabled(True)
            self.bellipse.setDisabled(True)
            self.blasso.setDisabled(True)
            self.bchoosemark.setDisabled(True)
            self.bnoselect.setDisabled(True)
            self.labelWidget.setDisabled(True)
            self.labelWidget.clear()
            self.bdelete.setDisabled(True)

            self.bdswitch.setDisabled(False)
            self.linebox.setDisabled(False)
            self.bgrids.setDisabled(False)
            self.openfile.setDisabled(False)
            self.bpatch.setDisabled(False)
            self.bsetcolor.setDisabled(False)

            self.actionOpen_file.setEnabled(True)
            self.actionSave.setEnabled(True)
            self.actionLoad.setEnabled(True)

            # if self.vision == 2:
            #     self.columnbox.setDisabled(False)
            # for i in reversed(range(self.maingrids1.count())):
            #     self.maingrids1.itemAt(i).widget().setParent(None)

    def stopView(self, n):
        self.viewLabel.stopMove(n)

    def chooseMark(self):
        with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])
        self.selectorPath = QtWidgets.QFileDialog.getExistingDirectory(self, "choose the image to view", dbinfo.sPathIn)
        if self.selectorPath:
            probandname = os.path.split(os.path.split(os.path.split(self.selectorPath)[0])[0])[1]
            self.markfile = str(self.pathROI) + '/' + str(probandname) + '.json'

            self.overlay = Overlay(self.centralWidget())
            self.overlay.setGeometry(QtCore.QRect(950, 400, 171, 141))
            self.overlay.show()
            from loadf import loadImage
            self.newMR = loadImage(self.selectorPath)
            self.newMR.trigger0.connect(self.loadEnd)
            self.newMR.start()
        else:
            pass

    def loadSelect(self): # from loadEnd
        self.mrinmain = self.newMR.voxel_ndarray
        self.labelimage = True
        self.ind = 0
        self.slices = self.mrinmain.shape[2]

        self.graylist = []
        self.graylist.append(None)
        self.graylist.append(None)
        self.emitlist = []
        self.emitlist.append(self.ind)
        self.emitlist.append(self.slices)

        self.slicelabel.setText('Slice %s' % (self.ind + 1) + '/ %s' % (self.slices))
        # self.graylabel.setText('Grayscale Range %s' % (self.graylist))
        self.zoomlabel.setText('Current Zooming %s' % (1))
        self.update_data.connect(self.updateSlices)
        self.gray_data.connect(self.updateGray)
        self.new_page.connect(self.newSliceview)
        self.viewLabel.zooming_data.connect(self.updateZoom)
        self.seditgray.clicked.connect(self.setGreymain)

        # replace reference with old values
        with open(self.markfile, 'r') as json_data:
            saveFile = json.load(json_data)
            self.labelnames=saveFile['names']['list']
            self.labelcolor = saveFile['colors']['list']
        with open('editlabel.json', 'r') as json_data:
            self.infos = json.load(json_data)
            self.infos['names'] = self.labelnames
            self.infos['colors'] = self.labelcolor
        with open('editlabel.json', 'w') as json_data:
            json_data.write(json.dumps(self.infos))

        self.labelbox.clear()
        self.labelbox.addItems(self.labelnames)

        self.orderROIs()

    def orderROIs(self):
        with open(self.markfile, 'r') as json_data:
            saveFile = json.load(json_data)
        sepkey = os.path.split(self.selectorPath)[1]
        self.newkeylist=[]
        if sepkey in saveFile['layer']:
            for key in saveFile['layer'][sepkey].keys():
                newkey = (key[0]+key[1], key[2], key[3], key[4]+key[5])
                self.newkeylist.append(newkey)

            first = sorted(self.newkeylist, key=lambda e: e.__getitem__(2))
            second = sorted(first, key=lambda e: (e.__getitem__(2), e.__getitem__(3)))
            self.third = sorted(second, key=lambda e: (e.__getitem__(2), e.__getitem__(3), e.__getitem__(0)))
            self.updateList()

    def updateList(self):
        self.keylist = []
        self.labelWidget.clear()
        for element in self.third:
            oldkey =element[0]+element[1]+element[2]+element[3]
            self.keylist.append(oldkey)
            item=self.labelnames[int(oldkey[3])] + '-'+ oldkey[4]+oldkey[5] + '-'+ oldkey[0]+oldkey[1] + '/' + str(self.slices)
            self.labelWidget.addItem(item)

        self.newslicesview()

    def newonscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1)  % self.slices
        else:
            self.ind = (self.ind - 1)  % self.slices
        if self.ind >= self.slices:
            self.ind = 0
        if self.ind <= -1:
            self.ind = self.slices - 1

        self.emitlist[0] = self.ind
        self.update_data.emit(self.emitlist)
        self.newslicesview()

    def newslicesview(self): # refreshing
        self.newax.clear()
        self.pltc = self.newax.imshow(np.swapaxes(self.mrinmain[:, :, self.ind], 0, 1), cmap='gray', vmin=0, vmax=2094)

        sepkey = os.path.split(self.selectorPath)
        sepkey = sepkey[1]  # t1_tse_tra_Kopf_Motion_0003

        with open(self.markfile, 'r') as json_data:
            loadFile = json.load(json_data)

            if sepkey in loadFile['layer']:
                layer = loadFile['layer'][sepkey]
                for key in layer.keys():
                    if key[0]+key[1]== str(self.ind+1):
                        p = layer[key]
                        if key[2] == '1':
                            p = np.asarray(p['points'])
                            patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]),
                                                  np.abs(p[1] - p[3]), fill=True,
                                                  alpha=.2, edgecolor= None, facecolor=self.labelcolor[int(key[3])])
                        elif key[2] == '2':
                            p = np.asarray(p['points'])
                            patch = Ellipse(
                                xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                                width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]),
                                alpha=.2, edgecolor= None, facecolor=self.labelcolor[int(key[3])])
                        else:
                            p = path.Path(np.asarray(p['vertices']), p['codes'])
                            patch = patches.PathPatch(p, fill=True, alpha=.2, edgecolor= None, facecolor=self.labelcolor[int(key[3])])

                        self.newax.add_patch(patch)
        self.newcanvas.draw()

        v_min, v_max = self.pltc.get_clim()
        self.graylist[0] = v_min
        self.graylist[1] = v_max
        self.new_page.emit()

    def deleteLabel(self):
        pos = self.labelWidget.currentRow()
        item = self.labelWidget.takeItem(pos)
        item = None
        with open(self.markfile) as json_data:
            deleteMark = json.load(json_data)
            sepkey = os.path.split(self.selectorPath)
            sepkey = sepkey[1]
            layer = deleteMark['layer'][sepkey]
            # if pos != -1: # none left
            if len(layer.keys())!=0:
                layer.pop(self.keylist[pos], None)
                self.keylist.pop(pos)   ### !
                # for key in layer.keys():
                #     print(key)
            if pos != 0:
                self.labelWidget.setCurrentRow(pos-1)
        with open(self.markfile, 'w') as json_data:
            json_data.write(json.dumps(deleteMark))

        self.newslicesview()

    def mouse_clicked(self, event):
        if event.button == 2:
            self.x_clicked = event.x
            self.y_clicked = event.y
            self.mouse_second_clicked = True

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

    def mouse_release(self, event):
        if event.button == 2:
            self.mouse_second_clicked = False

    def newSliceview(self):
        self.graylabel.setText('Grayscale Range %s' % (self.graylist))

    def updateSlices(self, elist):
        self.slicelabel.setText('Slice %s' % (elist[0] + 1) + '/ %s' % (elist[1]))

    def updateGray(self, elist):
        self.graylabel.setText('Grayscale Range %s' % (elist))

    def updateZoom(self, factor):
        self.zoomlabel.setText('Current Zooming %s' % (factor))

    def setGreymain(self):
        maxv, minv, ok = grey_window.getData()
        if ok:
            self.pltc.set_clim(vmin=minv, vmax=maxv)
            self.graylist[0] = minv
            self.graylist[1] = maxv
            self.gray_data.emit(self.graylist)
            self.newcanvas.draw_idle()

    def closeEvent(self, QCloseEvent):
        reply = QMessageBox.question(self, 'Warning', 'Are you sure to exit?', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()


####### second tab
    def button_train_clicked(self):
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
            raise ValueError("Wrong input format of learning rates! Enter values seperated by ','. For example: 0.1,0.01,0.001")

        # set optimizer
        selectedOptimizer = self.ComboBox_Optimizers.currentText()
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
        self.deepLearningArtApp.setWeightDecay(float(self.DoubleSpinBox_WeightDecay.value()))
        # set momentum
        self.deepLearningArtApp.setMomentum(float(self.DoubleSpinBox_Momentum.value()))
        # set nesterov enabled
        if self.CheckBox_Nesterov.checkState() == Qt.Checked:
            self.deepLearningArtApp.setNesterovEnabled(True)
        else:
            self.deepLearningArtApp.setNesterovEnabled(False)

        # handle data augmentation
        if self.CheckBox_DataAugmentation.checkState() == Qt.Checked:
            self.deepLearningArtApp.setDataAugmentationEnabled(True)
            # get all checked data augmentation options
            if self.CheckBox_DataAug_horizontalFlip.checkState() == Qt.Checked:
                self.deepLearningArtApp.setHorizontalFlip(True)
            else:
                self.deepLearningArtApp.setHorizontalFlip(False)

            if self.CheckBox_DataAug_verticalFlip.checkState() == Qt.Checked:
                self.deepLearningArtApp.setVerticalFlip(True)
            else:
                self.deepLearningArtApp.setVerticalFlip(False)

            if self.CheckBox_DataAug_Rotation.checkState() == Qt.Checked:
                self.deepLearningArtApp.setRotation(True)
            else:
                self.deepLearningArtApp.setRotation(False)

            if self.CheckBox_DataAug_zcaWeighting.checkState() == Qt.Checked:
                self.deepLearningArtApp.setZCA_Whitening(True)
            else:
                self.deepLearningArtApp.setZCA_Whitening(False)

            if self.CheckBox_DataAug_HeightShift.checkState() == Qt.Checked:
                self.deepLearningArtApp.setHeightShift(True)
            else:
                self.deepLearningArtApp.setHeightShift(False)

            if self.CheckBox_DataAug_WidthShift.checkState() == Qt.Checked:
                self.deepLearningArtApp.setWidthShift(True)
            else:
                self.deepLearningArtApp.setWidthShift(False)

            if self.CheckBox_DataAug_Zoom.checkState() == Qt.Checked:
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


        # start training process
        self.deepLearningArtApp.performTraining()



    def button_markingsPath_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getMarkingsPath())
        self.Label_MarkingsPath.setText(dir)
        self.deepLearningArtApp.setMarkingsPath(dir)



    def button_patching_clicked(self):
        if self.deepLearningArtApp.getSplittingMode() == DeepLearningArtApp.NONE_SPLITTING:
            QMessageBox.about(self, "My message box", "Select Splitting Mode!")
            return 0

        self.getSelectedDatasets()
        self.getSelectedPatients()

        # get patching parameters
        self.deepLearningArtApp.setPatchSizeX(self.SpinBox_PatchX.value())
        self.deepLearningArtApp.setPatchSizeY(self.SpinBox_PatchY.value())
        self.deepLearningArtApp.setPatchSizeZ(self.SpinBox_PatchZ.value())
        self.deepLearningArtApp.setPatchOverlapp(self.SpinBox_PatchOverlapp.value())

        # get labling parameters
        if self.RadioButton_MaskLabeling.isChecked():
            self.deepLearningArtApp.setLabelingMode(DeepLearningArtApp.MASK_LABELING)
        elif self.RadioButton_PatchLabeling.isChecked():
            self.deepLearningArtApp.setLabelingMode(DeepLearningArtApp.PATCH_LABELING)

        # get patching parameters
        if self.ComboBox_Patching.currentIndex() == 1:
            # 2D patching selected
            self.deepLearningArtApp.setPatchingMode(DeepLearningArtApp.PATCHING_2D)
        elif self.ComboBox_Patching.currentIndex() == 2:
            # 3D patching selected
            self.deepLearningArtApp.setPatchingMode(DeepLearningArtApp.PATCHING_3D)
        else:
            self.ComboBox_Patching.setCurrentIndex(1)
            self.deepLearningArtApp.setPatchingMode(DeepLearningArtApp.PATCHING_2D)

        #using segmentation mask
        self.deepLearningArtApp.setUsingSegmentationMasks(self.CheckBox_SegmentationMask.isChecked())

        # handle store mode
        self.deepLearningArtApp.setStoreMode(self.ComboBox_StoreOptions.currentIndex())

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
            self.Button_useCurrentData.setEnabled(True)


    def button_outputPatching_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
        self.Label_OutputPathPatching.setText(dir)
        self.deepLearningArtApp.setOutputPathForPatching(dir)

    def getSelectedPatients(self):
        selectedPatients = []
        for i in range(self.TreeWidget_Patients.topLevelItemCount()):
            if self.TreeWidget_Patients.topLevelItem(i).checkState(0) == Qt.Checked:
                selectedPatients.append(self.TreeWidget_Patients.topLevelItem(i).text(0))

        self.deepLearningArtApp.setSelectedPatients(selectedPatients)

    def button_DB_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getPathToDatabase())
        self.deepLearningArtApp.setPathToDatabase(dir)
        self.manageTreeView()

    def openFileNamesDialog(self, dir=None):
        if dir==None:
            with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
            dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])
            dir = dbinfo.sPathIn + os.sep + 'MRPhysics'  + os.sep + 'newProtocol'

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
            self.TreeWidget_Patients.setHeaderLabel("Patients:")

            for x in subdirs:
                item = QTreeWidgetItem()
                item.setText(0, str(x))
                item.setCheckState(0, Qt.Unchecked)
                self.TreeWidget_Patients.addTopLevelItem(item)

            self.Label_DB.setText(self.deepLearningArtApp.getPathToDatabase())

    def manageTreeViewDatasets(self):
        # print(os.path.dirname(self.deepLearningArtApp.getPathToDatabase()))
        # manage datasets
        self.TreeWidget_Datasets.setHeaderLabel("Datasets:")
        for ds in DeepLearningArtApp.datasets.keys():
            dataset = DeepLearningArtApp.datasets[ds].getPathdata()
            item = QTreeWidgetItem()
            item.setText(0, dataset)
            item.setCheckState(0, Qt.Unchecked)
            self.TreeWidget_Datasets.addTopLevelItem(item)

    def getSelectedDatasets(self):
        selectedDatasets = []
        for i in range(self.TreeWidget_Datasets.topLevelItemCount()):
            if self.TreeWidget_Datasets.topLevelItem(i).checkState(0) == Qt.Checked:
                selectedDatasets.append(self.TreeWidget_Datasets.topLevelItem(i).text(0))

        self.deepLearningArtApp.setSelectedDatasets(selectedDatasets)

    def selectedDNN_changed(self):
        self.deepLearningArtApp.setNeuralNetworkModel(self.ComboBox_DNNs.currentText())

    def button_useCurrentData_clicked(self):
        if self.deepLearningArtApp.datasetAvailable() == True:
            self.Label_currentDataset.setText("Current Dataset is used...")
            self.GroupBox_TrainNN.setEnabled(True)
        else:
            self.Button_useCurrentData.setEnabled(False)
            self.Label_currentDataset.setText("No Dataset selected!")
            self.GroupBox_TrainNN.setEnabled(False)

    def button_selectDataset_clicked(self):
        pathToDataset = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
        retbool, datasetName = self.deepLearningArtApp.loadDataset(pathToDataset)
        if retbool == True:
            self.Label_currentDataset.setText(datasetName + " is used as dataset...")
        else:
            self.Label_currentDataset.setText("No Dataset selected!")

        if self.deepLearningArtApp.datasetAvailable() == True:
            self.GroupBox_TrainNN.setEnabled(True)
        else:
            self.GroupBox_TrainNN.setEnabled(False)

    def button_learningOutputPath_clicked(self):
        path = self.openFileNamesDialog(self.deepLearningArtApp.getLearningOutputPath())
        self.deepLearningArtApp.setLearningOutputPath(path)
        self.Label_LearningOutputPath.setText(path)

    def updateProgressBarTraining(self, val):
        self.ProgressBar_training.setValue(val)

    def splittingMode_changed(self):

        if self.ComboBox_splittingMode.currentIndex() == 0:
            self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
            self.Label_SplittingParams.setText("Select splitting mode!")
        elif self.ComboBox_splittingMode.currentIndex() == 1:
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
                    self.Label_SplittingParams.setText(txtStr)
                else:
                    self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
                    self.ComboBox_splittingMode.setCurrentIndex(0)
                    self.Label_SplittingParams.setText("Select Splitting Mode!")
            else:
                self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
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
                    self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.CROSS_VALIDATION_SPLITTING)
                    self.deepLearningArtApp.setTrainTestDatasetRatio(testTrainingRatio)
                    self.deepLearningArtApp.setNumFolds(numFolds)
                    self.Label_SplittingParams.setText("Test/Train Ratio: " + str(testTrainingRatio) + \
                                                          ", and " + str(numFolds) + " Folds")
                else:
                    self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
                    self.ComboBox_splittingMode.setCurrentIndex(0)
                    self.Label_SplittingParams.setText("Select Splitting Mode!")
            else:
                self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
                self.ComboBox_splittingMode.setCurrentIndex(0)
                self.Label_SplittingParams.setText("Select Splitting Mode!")

        elif self.ComboBox_splittingMode.currentIndex() == 3:
            self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.PATIENT_CROSS_VALIDATION_SPLITTING)

    def check_dataAugmentation_enabled(self):
        if self.CheckBox_DataAugmentation.checkState() == Qt.Checked:
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
    def textChangeAlpha(self,text):
        self.inputalpha = text
        # if text.isdigit():
        #     self.inputalpha=text
        # else:
        #     self.alphaShouldBeNumber()


    def textChangeGamma(self,text):
        self.inputGamma = text
        # if text.isdigit():
        #     self.inputGamma=text
        # else:
        #     self.GammaShouldBeNumber()

    def wheelScroll(self,ind,oncrollStatus):
        if oncrollStatus=='onscroll':
            self.horizontalSliderPatch.setValue(ind)
            self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)
        elif oncrollStatus=='onscrollW' or oncrollStatus=='onscroll_3D':
            self.wheelScrollW(ind)
        elif oncrollStatus=='onscrollSS':
            self.wheelScrollSS(ind)
        else:
            pass

    def wheelScrollW(self,ind):
        self.horizontalSliderPatch.setValue(ind)
        self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)

    def wheelScrollSS(self,indSS):
        self.horizontalSliderPatch.setValue(indSS)
        self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)

    def clickList(self,qModelIndex):

        self.chosenLayerName = self.qList[qModelIndex.row()]

    def simpleName(self,inpName):
        if "/" in inpName:
            inpName = inpName.split("/")[0]
            if ":" in inpName:
                inpName = inpName.split(':')[0]
        elif ":" in inpName:
            inpName = inpName.split(":")[0]
            if "/" in inpName:
                inpName = inpName.split('/')[0]

        return inpName

    def show_layer_name(self):
        qList = []

        for i in self.act:
            qList.append(i)

            # if self.act[i].ndim==5 and self.modelDimension=='3D':
            #     self.act[i]=np.transpose(self.act[i],(0,4,1,2,3))
        self.qList = qList

    def sliderValue(self):
        if self.W_F=='w':

            self.chosenWeightNumber=self.horizontalSliderPatch.value()
            self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
            self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
            self.overlay.show()

            from loadf2 import loadImage_weights_plot_3D
            self.wyPlot.setDisabled(True)
            self.newW3D = loadImage_weights_plot_3D(self.matplotlibwidget_static, self.w, self.chosenWeightNumber,
                                                    self.totalWeights, self.totalWeightsSlices)
            self.newW3D.trigger.connect(self.loadEnd2)
            self.newW3D.start()

            # self.matplotlibwidget_static.mpl.weights_plot_3D(self.w, self.chosenWeightNumber, self.totalWeights,self.totalWeightsSlices)
        elif self.W_F=='f':

            if self.modelDimension=='2D':
                self.chosenPatchNumber=self.horizontalSliderPatch.value()
                self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
                self.overlay.show()

                from loadf2 import loadImage_features_plot
                self.wyPlot.setDisabled(True)
                self.newf = loadImage_features_plot(self.matplotlibwidget_static, self.chosenPatchNumber)
                self.newf.trigger.connect(self.loadEnd2)
                self.newf.start()
                # self.matplotlibwidget_static.mpl.features_plot(self.chosenPatchNumber)
            elif self.modelDimension == '3D':

                self.chosenPatchNumber = self.horizontalSliderPatch.value()
                self.chosenPatchSliceNumber =self.horizontalSliderSlice.value()
                self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
                self.overlay.show()

                from loadf2 import loadImage_features_plot_3D
                self.wyPlot.setDisabled(True)
                self.newf = loadImage_features_plot_3D(self.matplotlibwidget_static, self.chosenPatchNumber,
                                                       self.chosenPatchSliceNumber)
                self.newf.trigger.connect(self.loadEnd2)
                self.newf.start()
                # self.matplotlibwidget_static.mpl.features_plot_3D(self.chosenPatchNumber,self.chosenPatchSliceNumber)
        elif self.W_F=='s':

            self.chosenSSNumber = self.horizontalSliderPatch.value()
            self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
            self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
            self.overlay.show()

            from loadf2 import loadImage_subset_selection_plot
            self.wyPlot.setDisabled(True)
            self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
            self.newf.trigger.connect(self.loadEnd2)
            self.newf.start()
            # self.matplotlibwidget_static.mpl.subset_selection_plot(self.chosenSSNumber)

        else:
            pass

    def sliderValueSS(self):
        self.chosenSSNumber=self.horizontalSliderSS.value()
        # self.matplotlibwidget_static_2.mpl.subset_selection_plot(self.chosenSSNumber)
        self.matplotlibwidget_static.mpl.subset_selection_plot(self.chosenSSNumber)

    @pyqtSlot()
    def on_wyChooseFile_clicked(self):
        self.openfile_name = QFileDialog.getOpenFileName(self,'Choose the file','.','H5 files(*.h5)')[0]
        if len(self.openfile_name)==0:
            pass
        else:
            self.horizontalSliderPatch.hide()
            self.horizontalSliderSlice.hide()
            self.labelPatch.hide()
            self.labelSlice.hide()
            self.lcdNumberSlice.hide()
            self.lcdNumberPatch.hide()
            self.matplotlibwidget_static.mpl.fig.clf()

            self.model=load_model(self.openfile_name)
            print()



    @pyqtSlot()
    def on_wyInputData_clicked(self):
        self.inputData_name = QFileDialog.getOpenFileName(self, 'Choose the file', '.', 'H5 files(*.h5)')[0]
        if len(self.inputData_name)==0:
            pass
        else:
            if len(self.openfile_name) != 0:
                self.horizontalSliderPatch.hide()
                self.horizontalSliderSlice.hide()
                self.labelPatch.hide()
                self.labelSlice.hide()
                self.lcdNumberSlice.hide()
                self.lcdNumberPatch.hide()
                self.matplotlibwidget_static.mpl.fig.clf()

                self.inputData=h5py.File(self.inputData_name,'r')
                # the number of the input
                for i in self.inputData:
                    if i == 'X_test_p2' or i == 'y_test_p2':
                        self.twoInput = True
                        break

                if self.inputData['X_test'].ndim == 4:
                    self.modelDimension = '2D'
                    X_test = self.inputData['X_test'][:, 2052:2160, :, :]
                    X_test = np.transpose(np.array(X_test), (1, 0, 2, 3))
                    self.subset_selection = X_test

                    if self.twoInput:
                        X_test_p2 = self.inputData['X_test_p2'][:, 2052:2160, :, :]
                        X_test_p2 = np.transpose(np.array(X_test_p2), (1, 0, 2, 3))
                        self.subset_selection_2 = X_test_p2


                elif self.inputData['X_test'].ndim == 5:
                    self.modelDimension = '3D'
                    X_test = self.inputData['X_test'][:, 0:20, :, :, :]
                    X_test = np.transpose(np.array(X_test), (1, 0, 2, 3, 4))
                    self.subset_selection = X_test

                    if self.twoInput:
                        X_test_p2 = self.inputData['X_test_p2'][:, 0:20, :, :, :]
                        X_test_p2 = np.transpose(np.array(X_test_p2), (1, 0, 2, 3, 4))
                        self.subset_selection_2 = X_test_p2

                else:
                    print('the dimension of X_test should be 4 or 5')

                if self.twoInput:
                    self.radioButton_3.show()
                    self.radioButton_4.show()


                plot_model(self.model, 'model.png')
                if self.twoInput:
                    self.modelInput = self.model.input[0]
                    self.modelInput2 = self.model.input[1]
                else:
                    self.modelInput = self.model.input

                self.layer_index_name = {}
                for i, layer in enumerate(self.model.layers):
                    self.layer_index_name[layer.name] = i


                for i, layer in enumerate(self.model.input_layers):

                    get_activations = K.function([layer.input, K.learning_phase()],
                                                 [layer.output, ])

                    if i == 0:
                        self.act[layer.name] = get_activations([self.subset_selection, 0])[0]
                    elif i == 1:
                        self.act[layer.name] = get_activations([self.subset_selection_2, 0])[0]
                    else:
                        print('no output of the input layer is created')

                for i, layer in enumerate(self.model.layers):
                    # input_len=layer.input.len()
                    if hasattr(layer.input, "__len__"):
                        if len(layer.input) == 2:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function([layer.input[0], layer.input[1], K.learning_phase()],
                                                         [layer.output, ])
                            self.act[layer.name] = get_activations([self.act[inputLayerNameList[0]],
                                                                    self.act[inputLayerNameList[1]],
                                                                             0])[0]

                        elif len(layer.input) == 3:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], K.learning_phase()], [layer.output, ])
                            self.act[layer.name] = get_activations([self.act[inputLayerNameList[0]],
                                                                    self.act[inputLayerNameList[1]],
                                                                    self.act[inputLayerNameList[2]],
                                                                             0])[0]

                        elif len(layer.input) == 4:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], K.learning_phase()],
                                [layer.output, ])
                            self.act[layer.name] = get_activations([self.act[inputLayerNameList[0]],
                                                                    self.act[inputLayerNameList[1]],
                                                                    self.act[inputLayerNameList[2]],
                                                                    self.act[inputLayerNameList[3]],
                                                                             0])[0]

                        elif len(layer.input) == 5:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[layer.name] = get_activations([self.act[inputLayerNameList[0]],
                                                                    self.act[inputLayerNameList[1]],
                                                                    self.act[inputLayerNameList[2]],
                                                                    self.act[inputLayerNameList[3]],
                                                                    self.act[inputLayerNameList[4]],
                                                                             0])[0]

                        else:
                            print('the number of input is more than 5')

                    else:
                        get_activations = K.function([layer.input, K.learning_phase()], [layer.output, ])
                        inputLayerName = self.simpleName(layer.input.name)
                        self.act[layer.name] = get_activations([self.act[inputLayerName], 0])[0]

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
                        if hasattr(self.weights[i],"ndim"):
                            if self.weights[i].ndim==5:
                                self.LayerWeights[i] = np.transpose(self.weights[i], (4, 3, 2, 0, 1))
                        else:
                            self.LayerWeights[i] =self.weights[i]
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
                slm = QStringListModel();
                slm.setStringList(self.qList)
                self.listView.setModel(slm)

            else:
                self.showChooseFileDialog()


    @pyqtSlot()
    def on_wyShowArchitecture_clicked(self):
        # Show the structure of the model and plot the weights
        if len(self.openfile_name) != 0:

            self.canvasStructure = MyMplCanvas()

            self.canvasStructure.loadImage()
            self.graphicscene = QtWidgets.QGraphicsScene()
            self.graphicscene.addWidget(self.canvasStructure)
            self.graphicview = Activeview()
            self.scrollAreaWidgetContents = QtWidgets.QWidget()
            self.maingrids = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
            self.scrollArea.setWidget(self.scrollAreaWidgetContents)
            self.maingrids.addWidget(self.graphicview)
            self.graphicview.setScene(self.graphicscene)
            # self.graphicsView.setScene(self.graphicscene)

        else:
            self.showChooseFileDialog()

    @pyqtSlot()
    def on_wyPlot_clicked(self):
        # self.matplotlibwidget_static_2.hide()

        # Show the structure of the model and plot the weights
        if len(self.openfile_name) != 0:
            if self.radioButton.isChecked()== True :
                if len(self.chosenLayerName) != 0:

                    self.W_F='w'
                    # show the weights
                    if self.modelDimension == '2D':
                        if hasattr(self.LayerWeights[self.chosenLayerName], "ndim"):

                            if self.LayerWeights[self.chosenLayerName].ndim==4:
                                self.lcdNumberPatch.hide()
                                self.lcdNumberSlice.hide()
                                self.horizontalSliderPatch.hide()
                                self.horizontalSliderSlice.hide()
                                self.labelPatch.hide()
                                self.labelSlice.hide()

                                self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                                self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                                self.overlay.show()

                                self.matplotlibwidget_static.mpl.getLayersWeights(self.LayerWeights)
                                from loadf2 import loadImage_weights_plot_2D
                                self.wyPlot.setDisabled(True)
                                self.newW2D = loadImage_weights_plot_2D(self.matplotlibwidget_static,self.chosenLayerName)
                                self.newW2D.trigger.connect(self.loadEnd2)
                                self.newW2D.start()

                                # self.matplotlibwidget_static.mpl.weights_plot_2D(self.chosenLayerName)
                                self.matplotlibwidget_static.show()
                            # elif self.LayerWeights[self.chosenLayerName].ndim==0:
                            #     self.showNoWeights()
                            else:
                                self.showWeightsDimensionError()

                        elif self.LayerWeights[self.chosenLayerName]==0:
                            self.showNoWeights()


                    elif self.modelDimension == '3D':
                        if hasattr(self.LayerWeights[self.chosenLayerName],"ndim"):

                            if self.LayerWeights[self.chosenLayerName].ndim == 5:

                                self.w=self.LayerWeights[self.chosenLayerName]
                                self.totalWeights=self.w.shape[0]
                                # self.totalWeightsSlices=self.w.shape[2]
                                self.horizontalSliderPatch.setMinimum(1)
                                self.horizontalSliderPatch.setMaximum(self.totalWeights)
                                # self.horizontalSliderSlice.setMinimum(1)
                                # self.horizontalSliderSlice.setMaximum(self.totalWeightsSlices)
                                self.chosenWeightNumber=1
                                self.horizontalSliderPatch.setValue(self.chosenWeightNumber)

                                self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                                self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                                self.overlay.show()

                                from loadf2 import loadImage_weights_plot_3D
                                self.wyPlot.setDisabled(True)
                                self.newW3D = loadImage_weights_plot_3D(self.matplotlibwidget_static, self.w,self.chosenWeightNumber,self.totalWeights,self.totalWeightsSlices)
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

                        elif self.LayerWeights[self.chosenLayerName]==0:
                            self.showNoWeights()

                    else:
                        print('the dimesnion should be 2D or 3D')

                else:
                    self.showChooseLayerDialog()

            elif self.radioButton_2.isChecked()== True :
                if len(self.chosenLayerName) != 0:
                    self.W_F = 'f'
                    if self.modelDimension == '2D':
                        if self.act[self.chosenLayerName].ndim==4:
                            self.activations=self.act[self.chosenLayerName]
                            self.totalPatches=self.activations.shape[0]

                            self.matplotlibwidget_static.mpl.getLayersFeatures(self.activations, self.totalPatches)

                            # show the features
                            self.chosenPatchNumber=1
                            self.horizontalSliderPatch.setMinimum(1)
                            self.horizontalSliderPatch.setMaximum(self.totalPatches)
                            self.horizontalSliderPatch.setValue(self.chosenPatchNumber)

                            self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                            self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                            self.overlay.show()

                            from loadf2 import loadImage_features_plot
                            self.wyPlot.setDisabled(True)
                            self.newf = loadImage_features_plot(self.matplotlibwidget_static,self.chosenPatchNumber)
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

                    elif self.modelDimension =='3D':
                        a=self.act[self.chosenLayerName]
                        if self.act[self.chosenLayerName].ndim == 5:
                            self.activations = self.act[self.chosenLayerName]
                            self.totalPatches = self.activations.shape[0]
                            self.totalPatchesSlices=self.activations.shape[1]

                            self.matplotlibwidget_static.mpl.getLayersFeatures_3D(self.activations, self.totalPatches,self.totalPatchesSlices)

                            self.chosenPatchNumber=1
                            self.chosenPatchSliceNumber=1
                            self.horizontalSliderPatch.setMinimum(1)
                            self.horizontalSliderPatch.setMaximum(self.totalPatches)
                            self.horizontalSliderPatch.setValue(self.chosenPatchNumber)
                            self.horizontalSliderSlice.setMinimum(1)
                            self.horizontalSliderSlice.setMaximum(self.totalPatchesSlices)
                            self.horizontalSliderSlice.setValue(self.chosenPatchSliceNumber)

                            self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                            self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                            self.overlay.show()

                            from loadf2 import loadImage_features_plot_3D
                            self.wyPlot.setDisabled(True)
                            self.newf = loadImage_features_plot_3D(self.matplotlibwidget_static, self.chosenPatchNumber,self.chosenPatchSliceNumber)
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

                else:
                    self.showChooseLayerDialog()

            else:
                self.showChooseButtonDialog()

        else:
            self.showChooseFileDialog()

    @pyqtSlot()
    def on_wySubsetSelection_clicked(self):
        # Show the Subset Selection
        if len(self.openfile_name) != 0:
            # show the weights
            # self.scrollArea.hide()
            self.W_F ='s'
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
            if self.twoInput==False:
                self.matplotlibwidget_static.mpl.getSubsetSelections(self.subset_selection, self.totalSS)

                self.createSubset(self.modelInput,self.subset_selection)
                self.matplotlibwidget_static.mpl.getSSResult(self.ssResult)

                self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                self.overlay.show()

                from loadf2 import loadImage_subset_selection_plot
                self.wyPlot.setDisabled(True)
                self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
                self.newf.trigger.connect(self.loadEnd2)
                self.newf.start()

                # self.matplotlibwidget_static.mpl.subset_selection_plot(self.chosenSSNumber)
            elif self.twoInput:
                if self.radioButton_3.isChecked(): # the 1st input
                    self.matplotlibwidget_static.mpl.getSubsetSelections(self.subset_selection, self.totalSS)
                    self.createSubset(self.modelInput,self.subset_selection)
                    self.matplotlibwidget_static.mpl.getSSResult(self.ssResult)

                    self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                    self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                    self.overlay.show()

                    from loadf2 import loadImage_subset_selection_plot
                    self.wyPlot.setDisabled(True)
                    self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
                    self.newf.trigger.connect(self.loadEnd2)
                    self.newf.start()

                elif self.radioButton_4.isChecked(): # the 2nd input
                    self.matplotlibwidget_static.mpl.getSubsetSelections(self.subset_selection_2, self.totalSS)
                    self.createSubset(self.modelInput2,self.subset_selection_2)
                    self.matplotlibwidget_static.mpl.getSSResult(self.ssResult)

                    self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                    self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                    self.overlay.show()

                    from loadf2 import loadImage_subset_selection_plot
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

    def clickList_1(self, qModelIndex):
        self.chosenActivationName = self.qList[qModelIndex.row()]

    def showChooseFileDialog(self):
        reply = QMessageBox.information(self,
                                        "Warning",
                                        "Please select one H5 File at first",
                                        QMessageBox.Ok )

    def showChooseLayerDialog(self):
        reply = QMessageBox.information(self,
                                        "Warning",
                                        "Please select one Layer at first",
                                        QMessageBox.Ok)

    def showChooseButtonDialog(self):
        reply = QMessageBox.information(self,
                                        "Warning",
                                        "Please select to plot the weights or the features",
                                        QMessageBox.Ok)

    def showNoWeights(self):
        reply = QMessageBox.information(self,
                                        "Warning",
                                        "This layer does not have weighst,please select other layers",
                                        QMessageBox.Ok)

    def showWeightsDimensionError(self):
        reply = QMessageBox.information(self,
                                        "Warning",
                                        "The diemnsion of the weights should be 0 or 4",
                                        QMessageBox.Ok)

    def showWeightsDimensionError3D(self):
        reply = QMessageBox.information(self,
                                        "Warning",
                                        "The diemnsion of the weights should be 0 or 5",
                                        QMessageBox.Ok)

    def showNoFeatures(self):
        reply = QMessageBox.information(self,
                                        "Warning",
                                        "This layer does not have feature maps, please select other layers",
                                        QMessageBox.Ok)

    def loadEnd2(self):
        self.overlay.killTimer(self.overlay.timer)
        self.overlay.hide()
        self.wyPlot.setDisabled(False)

    def alphaShouldBeNumber(self):
        reply = QMessageBox.information(self,
                                        "Warning",
                                        "Alpha should be a number!!!",
                                        QMessageBox.Ok)

    def GammaShouldBeNumber(self):
        reply = QMessageBox.information(self,
                                        "Warning",
                                        "Gamma should be a number!!!",
                                        QMessageBox.Ok)

    def createSubset(self,modelInput,subset_selection):
        class_idx = 0
        reg_param = 1 / (2e-4)

        input = modelInput  # tensor
        cost = -K.sum(K.log(input[:, class_idx] + 1e-8))  # tensor
        gradient = K.gradients(cost, input)  # list

        sess = tf.InteractiveSession()
        calcCost = network_visualization.TensorFlowTheanoFunction([input], cost)
        calcGrad = network_visualization.TensorFlowTheanoFunction([input], gradient)

        step_size = float(self.inputalpha)
        reg_param = float(self.inputGamma)

        test = subset_selection
        data_c = test
        oss_v = network_visualization.SubsetSelection(calcGrad, calcCost, data_c, alpha=reg_param, gamma=step_size)
        result = oss_v.optimize(np.random.uniform(0, 1.0, size=data_c.shape))
        result = result * test
        result[result>0]=1
        self.ssResult=result

    def showChooseInput(self):
        reply = QMessageBox.information(self,
                                        "Warning",
                                        "Please select to plot the input 1 or 2",
                                        QMessageBox.Ok)

class MatplotlibWidget(QWidget):

    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.initUi()

    def initUi(self):
        self.layout = QVBoxLayout(self)
        self.mpl = MyMplCanvas(self, width=15, height=15)
        self.layout.addWidget(self.mpl)

############################################################ class of grids
class Viewgroup(QtWidgets.QGridLayout):
    in_link = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super(Viewgroup, self).__init__(parent)

        self.mode = 1
        self.viewnr1 = 1
        self.viewnr2 = 1
        self.spin = None

        self.anewcanvas = None
        self.islinked = False
        self.skiplink = False
        self.skipdis = True

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
        self.linkon.setSizePolicy(self.sizePolicy)
        self.linkon.setText("")
        self.linkon.setIcon(self.icon2)
        self.imageedit.setSizePolicy(self.sizePolicy)
        self.imageedit.setText("")
        self.imageedit.setIcon(icon3)
        self.imrotate.setSizePolicy(self.sizePolicy)
        self.imrotate.setText("")
        self.imrotate.setIcon(icon4)

        self.Viewpanel = Activeview()
        self.addWidget(self.Viewpanel, 2, 0, 1, 4)

        self.anewscene = Activescene()
        self.Viewpanel.setScene(self.anewscene)
        self.Viewpanel.zooming_data.connect(self.updateZoom)

        self.modechange.clicked.connect(self.switchMode)
        self.imrotate.clicked.connect(self.rotateView)
        self.pathbox.currentIndexChanged.connect(self.loadScene)
        self.refbox.currentIndexChanged.connect(self.loadScene)
        self.imageedit.clicked.connect(self.setGrey)
        self.linkon.clicked.connect(self.linkPanel)
        self.oldindex = 0

    def switchMode(self):
        if self.mode ==1:
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
        if self.mode == 1:
            if self.pathbox.currentIndex() != 0:
                if self.viewnr1 == 1:
                    self.viewnr1 = 2
                    param = {'image': list1[self.spin - 1], 'mode': 2, 'shape':shapelist[self.spin - 1]}
                    self.anewcanvas = Canvas(param)
                elif self.viewnr1 == 2:
                    self.viewnr1 = 3
                    param = {'image': list1[self.spin - 1], 'mode': 3, 'shape':shapelist[self.spin - 1]}
                    self.anewcanvas = Canvas(param)
                else:
                    self.viewnr1 = 1
                    param = {'image': list1[self.spin - 1], 'mode': 1, 'shape':shapelist[self.spin - 1]}
                    self.anewcanvas = Canvas(param)
                self.loadImage()
            else:
                pass
        else:
            if self.refbox.currentIndex()!=0:
                if cnrlist[self.spin - 1] == 11:
                    if self.viewnr2 == 1:
                        self.viewnr2 = 2
                        param2 = {'image': list1[correslist[self.spin - 1]], 'mode': 5, 'color': problist[self.spin - 1],
                                  'hatch': hatchlist[self.spin - 1], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                                  'shape': shapelist[correslist[self.spin - 1]]}
                        self.anewcanvas = Canvas(param2)
                    elif self.viewnr2 == 2:
                        self.viewnr2 = 3
                        param3 = {'image': list1[correslist[self.spin - 1]], 'mode': 6, 'color': problist[self.spin - 1],
                                  'hatch': hatchlist[self.spin - 1], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                                  'shape': shapelist[correslist[self.spin - 1]]}
                        self.anewcanvas = Canvas(param3)
                    else:
                        self.viewnr2 = 1
                        param1 = {'image': list1[correslist[self.spin - 1]], 'mode': 4, 'color': problist[self.spin - 1],
                                  'hatch': hatchlist[self.spin - 1], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                                  'shape': shapelist[correslist[self.spin - 1]]}
                        self.anewcanvas = Canvas(param1)

                elif cnrlist[self.spin - 1] == 2:
                    if self.viewnr2 == 1:
                        self.viewnr2 = 2
                        param2 = {'image': list1[correslist[self.spin - 1]], 'mode': 8, 'color': problist[self.spin - 1],
                                  'cmap': cmap1, 'trans': vtr1, 'shape': shapelist[correslist[self.spin - 1]]}
                        self.anewcanvas = Canvas(param2)
                    elif self.viewnr2 == 2:
                        self.viewnr2 = 3
                        param3 = {'image': list1[correslist[self.spin - 1]], 'mode': 9, 'color': problist[self.spin - 1],
                                  'cmap': cmap1, 'trans': vtr1, 'shape': shapelist[correslist[self.spin - 1]]}
                        self.anewcanvas = Canvas(param3)
                    else:
                        self.viewnr2 = 1
                        param1 = {'image': list1[correslist[self.spin - 1]], 'mode': 7, 'color': problist[self.spin - 1],
                                  'cmap': cmap1, 'trans': vtr1, 'shape': shapelist[correslist[self.spin - 1]]}
                        self.anewcanvas = Canvas(param1)

                self.loadImage()
            else:
                pass

    def linkPanel(self):
        if self.pathbox.currentIndex() == 0 and self.refbox.currentIndex() ==0: # trigger condition
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
        self.graylabel.setText('G %s' % (self.anewcanvas.graylist))
        self.zoomlabel.setText('Z 1.0')
        self.anewcanvas.update_data.connect(self.updateSlices)
        self.anewcanvas.gray_data.connect(self.updateGray)
        self.anewcanvas.new_page.connect(self.newSliceview)

        if self.islinked == True:
            self.Viewpanel.zoom_link.disconnect()
            self.Viewpanel.move_link.disconnect()
            self.skiplink = False
            self.skipdis = True
            self.in_link.emit()

    def addPathd(self, pathDicom):
        region = os.path.split(pathDicom)
        proband = os.path.split(os.path.split(region[0])[0])[1]
        region = region[1]
        # self.pathbox.addItem('Proband: %s' % (proband) + ' Image: %s' % (region))
        self.pathbox.addItem('[' + (proband) + '][' + (region)+ ']')

    def addPathre(self, pathColor):
        self.refbox.addItem(pathColor)

    def loadScene(self, i):
        self.spin = i # position in combobox
        self.viewnr1 = 1
        self.viewnr2 = 1
        if i != 0:
            if self.mode == 1:
                param = {'image':list1[i - 1], 'mode':1, 'shape':shapelist[i - 1]}
                self.anewcanvas = Canvas(param)
            else:
                if cnrlist[i - 1] == 11:
                    param1 = {'image': list1[correslist[i - 1]], 'mode': 4, 'color': problist[i - 1],
                              'hatch': hatchlist[i - 1], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                              'shape': shapelist[correslist[i - 1]]}
                    self.anewcanvas = Canvas(param1)
                elif cnrlist[i - 1] == 2:
                    param1 = {'image': list1[correslist[i - 1]], 'mode': 7 , 'color':problist[i - 1],
                               'cmap':cmap1, 'trans':vtr1, 'shape': shapelist[correslist[i - 1]]}
                    self.anewcanvas = Canvas(param1)

            self.loadImage()
        else:
            if self.oldindex != 0:
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
            else:
                pass

        self.oldindex = i

    def newSliceview(self):
        self.graylabel.setText('G %s' % (self.anewcanvas.graylist))

    def updateSlices(self, elist):
        self.slicelabel.setText('S %s' % (elist[0] + 1) + '/ %s' % (elist[1]))

    def updateZoom(self, data):
        self.zoomlabel.setText('Z %s' % (data))

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
            greylist=[]
            greylist.append(minv)
            greylist.append(maxv)
            self.anewcanvas.setGreyscale(greylist)

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

        self.imagelist = QtWidgets.QComboBox()
        self.imagelist.addItem('closed')
        self.reflist = QtWidgets.QComboBox()
        self.reflist.addItem('closed')
        self.reflist.setDisabled(True)
        self.bimre = QtWidgets.QPushButton()
        self.bimre.setText('')
        self.bimre.setSizePolicy(self.sizePolicy)
        self.bimre.setIcon(icon1)
        self.blinkon = QtWidgets.QPushButton()
        self.blinkon.setText('')
        self.blinkon.setSizePolicy(self.sizePolicy)
        self.blinkon.setIcon(self.icon2)
        self.gridLayout_1.addWidget(self.imagelist, 0, 0, 1, 1)
        self.gridLayout_1.addWidget(self.reflist, 0, 1, 1, 1)
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
        self.imagelist.currentIndexChanged.connect(self.loadScene)
        self.ed31.clicked.connect(self.setGrey1)
        self.ed32.clicked.connect(self.setGrey2)
        self.ed33.clicked.connect(self.setGrey3)
        self.reflist.currentIndexChanged.connect(self.loadScene)
        self.blinkon.clicked.connect(self.linkPanel)

    def switchMode(self):
        if self.vmode == 1:
            self.vmode = 2
            # self.im_re.setText('Result')
            self.reflist.setDisabled(False)
            self.imagelist.setDisabled(True)
            self.imagelist.setCurrentIndex(0)
            self.reflist.setCurrentIndex(0)
        else:
            self.vmode = 1
            # self.im_re.setText('Image')
            self.reflist.setDisabled(True)
            self.imagelist.setDisabled(False)
            self.reflist.setCurrentIndex(0)
            self.imagelist.setCurrentIndex(0)

    def addPathim(self, pathDicom):
        region = os.path.split(pathDicom)
        proband = os.path.split(os.path.split(region[0])[0])[1]
        region = region[1]
        self.imagelist.addItem('Proband: %s' % (proband) + '   Image: %s' % (region))

    def addPathre(self, pathColor):
        self.reflist.addItem(pathColor)

    def linkPanel(self):
        if self.imagelist.currentIndex() == 0 and self.reflist.currentIndex() ==0: # trigger condition
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

    def loadScene(self, i):
        if i != 0:
            if self.vmode == 1:
                param1 = {'image': list1[i - 1], 'mode': 1, 'shape': shapelist[i - 1]}
                param2 = {'image': list1[i - 1], 'mode': 2, 'shape': shapelist[i - 1]}
                param3 = {'image': list1[i - 1], 'mode': 3, 'shape': shapelist[i - 1]}
                self.newcanvas1 = Canvas(param1)
                self.newcanvas2 = Canvas(param2)
                self.newcanvas3 = Canvas(param3)
            else:
                if cnrlist[i - 1] == 11:
                    param1 = {'image': list1[correslist[i - 1]], 'mode': 4 , 'color':problist[i - 1],
                              'hatch':hatchlist[i - 1], 'cmap':cmap3, 'hmap':hmap2, 'trans':vtr3,
                              'shape': shapelist[correslist[i - 1]]}
                    self.newcanvas1 = Canvas(param1)
                    param2 = {'image': list1[correslist[i - 1]], 'mode': 5 , 'color':problist[i - 1],
                              'hatch':hatchlist[i - 1], 'cmap':cmap3, 'hmap':hmap2, 'trans':vtr3,
                              'shape': shapelist[correslist[i - 1]]}
                    self.newcanvas2 = Canvas(param2)
                    param3 = {'image': list1[correslist[i - 1]], 'mode': 6 , 'color':problist[i - 1],
                              'hatch':hatchlist[i - 1], 'cmap':cmap3, 'hmap':hmap2, 'trans':vtr3,
                              'shape': shapelist[correslist[i - 1]]}
                    self.newcanvas3 = Canvas(param3)
                elif cnrlist[i - 1] == 2:
                    param1 = {'image': list1[correslist[i - 1]], 'mode': 7 , 'color':problist[i - 1],
                               'cmap':cmap1, 'trans':vtr1, 'shape': shapelist[correslist[i - 1]]}
                    self.newcanvas1 = Canvas(param1)
                    param2 = {'image': list1[correslist[i - 1]], 'mode': 8 , 'color':problist[i - 1],
                             'cmap':cmap1, 'trans':vtr1, 'shape': shapelist[correslist[i - 1]]}
                    self.newcanvas2 = Canvas(param2)
                    param3 = {'image': list1[correslist[i - 1]], 'mode': 9 , 'color':problist[i - 1],
                               'cmap':cmap1, 'trans':vtr1, 'shape': shapelist[correslist[i - 1]]}
                    self.newcanvas3 = Canvas(param3)
                # elif:
            self.loadImage()
        else:
            if self.oldindex != 0:
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
            else:
                pass

        self.oldindex = i

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
        self.grt1.setText('G %s' % (self.newcanvas1.graylist))
        self.zot1.setText('Z 1.0')
        self.newcanvas1.update_data.connect(self.updateSlices1)
        self.newcanvas1.gray_data.connect(self.updateGray1)
        self.newcanvas1.new_page.connect(self.newSliceview1)
        self.slt2.setText('S %s' % (self.newcanvas2.ind + 1) + '/ %s' % (self.newcanvas2.slices))
        self.grt2.setText('G %s' % (self.newcanvas2.graylist))
        self.zot2.setText('Z 1.0')
        self.newcanvas2.update_data.connect(self.updateSlices2)
        self.newcanvas2.gray_data.connect(self.updateGray2)
        self.newcanvas2.new_page.connect(self.newSliceview2)
        self.slt3.setText('S %s' % (self.newcanvas3.ind + 1) + '/ %s' % (self.newcanvas3.slices))
        self.grt3.setText('G %s' % (self.newcanvas3.graylist))
        self.zot3.setText('Z 1.0')
        self.newcanvas3.update_data.connect(self.updateSlices3)
        self.newcanvas3.gray_data.connect(self.updateGray3)
        self.newcanvas3.new_page.connect(self.newSliceview3)
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

    def newSliceview1(self):
        self.grt1.setText('G %s' % (self.newcanvas1.graylist))

    def newSliceview2(self):
        self.grt2.setText('G %s' % (self.newcanvas2.graylist))

    def newSliceview3(self):
        self.grt3.setText('G %s' % (self.newcanvas3.graylist))

    def updateSlices1(self, elist):
        self.slt1.setText('S %s' % (elist[0] + 1) + '/ %s' % (elist[1]))

    def updateZoom1(self, data):
        self.zot1.setText('Z %s' % (data))

    def updateGray1(self, elist):
        self.grt1.setText('G %s' % (elist))

    def updateSlices2(self, elist):
        self.slt2.setText('S %s' % (elist[0] + 1) + '/ %s' % (elist[1]))

    def updateZoom2(self, data):
        self.zot2.setText('Z %s' % (data))

    def updateGray2(self, elist):
        self.grt2.setText('G %s' % (elist))

    def updateSlices3(self, elist):
        self.slt3.setText('S %s' % (elist[0] + 1) + '/ %s' % (elist[1]))

    def updateZoom3(self, data):
        self.zot3.setText('Z %s' % (data))

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
            greylist=[]
            greylist.append(minv)
            greylist.append(maxv)
            self.newcanvas1.setGreyscale(greylist)

    def setGrey2(self):
        maxv, minv, ok = grey_window.getData()
        if ok and self.newcanvas2:
            greylist=[]
            greylist.append(minv)
            greylist.append(maxv)
            self.newcanvas2.setGreyscale(greylist)

    def setGrey3(self):
        maxv, minv, ok = grey_window.getData()
        if ok and self.newcanvas3:
            greylist=[]
            greylist.append(minv)
            greylist.append(maxv)
            self.newcanvas3.setGreyscale(greylist)

###################
sys._excepthook = sys.excepthook
def my_exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)

sys.excepthook = my_exception_hook

# class ExceptionHandler(QtCore.QObject):
#     errorSignal = QtCore.pyqtSignal()
#
#     def __init__(self):
#         super(ExceptionHandler, self).__init__()
#
#     def handler(self, exctype, value, traceback):
#         self.errorSignal.emit()
#         sys._excepthook(exctype, value, traceback)
# exceptionHandler = ExceptionHandler()
# sys._excepthook = sys.excepthook
# sys.excepthook = exceptionHandler.handler
#
# def something():
#     QtWidgets.QMessageBox.information('Warning', 'File error!')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyApp()
    mainWindow.showMaximized()

    mainWindow.show()
    # exceptionHandler.errorSignal.connect(something)
    sys.exit(app.exec_())
