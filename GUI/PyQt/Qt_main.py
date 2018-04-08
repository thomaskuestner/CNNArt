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
from activeview import Activeview
from activescene import Activescene
from canvas import Canvas
from Unpatch_eleven import*
from Unpatch_eight import*
from Unpatch_two import*

from DatabaseInfo import*
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import json

from loadingIcon import Overlay
import scipy.io as sio
from Grey_window import*
import sys
import h5py
# sys.path.insert(0, 'C:/Users/hansw/Desktop/Ma_code/tabYannick')
# from tabYannick.dlart import*
from matplotlib.figure import Figure

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from skimage import measure

#pyrcc5 C:\Users\hansw\Desktop\Ma_code\PyQt\resrc.qrc -o C:\Users\hansw\Desktop\Ma_code\PyQt\resrc_rc.py

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    update_data = QtCore.pyqtSignal(list)
    gray_data = QtCore.pyqtSignal(list)
    new_page = QtCore.pyqtSignal()

    def __init__(self):
        super(MyApp, self).__init__()
        self.setupUi(self)

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.maingrids = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        # self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.vision = 2
        self.gridson = False
        self.gridsnr = 2
        self.voxel_ndarray = []
        self.i = -1
        self.linked = False

        global pathlist, list1, shapelist, pnamelist, empty1, cmap1, cmap3, hmap1, hmap2, vtr1,  \
            vtr3, problist, hatchlist, correslist, cnrlist
        pathlist = []
        list1 = []
        shapelist = []
        pnamelist = []
        empty1 = []
        # cmap1 = mpl.colors.ListedColormap(['blue', 'red'])
        # cmap3 = mpl.colors.ListedColormap(['blue', 'purple', 'cyan', 'yellow', 'green'])
        # hmap1 = [None, '//', '\\', 'XX']
        # hmap2 = [None, '//', '\\', 'XX']
        # vtr1 = 0.3
        # vtr3 = 0.3

        with open('colors0.json', 'r') as json_data:
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

        self.openfile.clicked.connect(self.loadMR)
        self.bdswitch.clicked.connect(self.switchview)
        self.bgrids.clicked.connect(self.setlayout)
        self.bpatch.clicked.connect(self.loadpatch)
        self.bsetcolor.clicked.connect(self.setColor)

        self.bselectoron.clicked.connect(self.selectormode)
        self.bchoosemark.clicked.connect(self.chooseMark)
        self.artifactbox.setDisabled(True)
        self.brectangle.setDisabled(True)
        self.bellipse.setDisabled(True)
        self.blasso.setDisabled(True)
        self.brectangle.toggled.connect(lambda:formsclick(1))
        self.bellipse.toggled.connect(lambda:formsclick(2))
        self.blasso.toggled.connect(lambda: formsclick(3))
        self.selectoron = False
        self.x_clicked = None
        self.y_clicked = None
        self.mouse_second_clicked = False

        self.newfig = plt.figure(50) # 3
        self.newfig.set_facecolor("black")
        self.newax = self.newfig.add_subplot(111)
        # self.newax = plt.gca()  # for cooperation with pltc
        self.newax.axis('off')
        self.pltc = None
        self.newcanvas = FigureCanvas(self.newfig)  # must be defined because of selector next

        self.actionOpen_file.triggered.connect(self.loadMR)
        self.actionSave.triggered.connect(self.saveCurrent)
        self.actionLoad.triggered.connect(self.loadOld)
        self.actionColor.triggered.connect(self.defaultColor)

        ####################################################################### second tab
        # # initialize DeepLearningArt Application
        # self.deepLearningArtApp = DeepLearningArtApp()
        # self.deepLearningArtApp.setGUIHandle(self)
        #
        # # initialize TreeView Database
        # self.manageTreeView()
        #
        # # intialize TreeView Datasets
        # self.manageTreeViewDatasets()
        #
        # # initiliaze patch output path
        # self.Label_OutputPathPatching.setText(self.deepLearningArtApp.getOutputPathForPatching())
        #
        # # initialize markings path
        # self.Label_MarkingsPath.setText(self.deepLearningArtApp.getMarkingsPath())
        #
        # # initialize learning output path
        # self.Label_LearningOutputPath.setText(self.deepLearningArtApp.getLearningOutputPath())
        #
        # # initialize patching mode
        # self.ComboBox_Patching.setCurrentIndex(1)
        #
        # # initialize store mode
        # self.ComboBox_StoreOptions.setCurrentIndex(0)
        #
        # # initialize splitting mode
        # self.ComboBox_splittingMode.setCurrentIndex(DeepLearningArtApp.SIMPLE_RANDOM_SAMPLE_SPLITTING)
        # self.Label_SplittingParams.setText("using Test/Train="
        #                                    + str(self.deepLearningArtApp.getTrainTestDatasetRatio())
        #                                    + " and Valid/Train=" + str(
        #     self.deepLearningArtApp.getTrainValidationRatio()))
        #
        # # initialize combox box for DNN selection
        # self.ComboBox_DNNs.addItem("Select Deep Neural Network Model...")
        # self.ComboBox_DNNs.addItems(DeepLearningArtApp.deepNeuralNetworks.keys())
        # self.ComboBox_DNNs.setCurrentIndex(1)
        # self.deepLearningArtApp.setNeuralNetworkModel(self.ComboBox_DNNs.currentText())
        #
        # # initialize check boxes for used classes
        # self.CheckBox_Artifacts.setChecked(self.deepLearningArtApp.getUsingArtifacts())
        # self.CheckBox_BodyRegion.setChecked(self.deepLearningArtApp.getUsingBodyRegions())
        # self.CheckBox_TWeighting.setChecked(self.deepLearningArtApp.getUsingTWeighting())
        #
        # # initilize training parameters
        # self.DoubleSpinBox_WeightDecay.setValue(self.deepLearningArtApp.getWeightDecay())
        # self.DoubleSpinBox_Momentum.setValue(self.deepLearningArtApp.getMomentum())
        # self.CheckBox_Nesterov.setChecked(self.deepLearningArtApp.getNesterovEnabled())
        # self.CheckBox_DataAugmentation.setChecked(self.deepLearningArtApp.getDataAugmentationEnabled())
        # self.CheckBox_DataAug_horizontalFlip.setChecked(self.deepLearningArtApp.getHorizontalFlip())
        # self.CheckBox_DataAug_verticalFlip.setChecked(self.deepLearningArtApp.getVerticalFlip())
        # self.CheckBox_DataAug_Rotation.setChecked(False if self.deepLearningArtApp.getRotation() == 0 else True)
        # self.CheckBox_DataAug_zcaWeighting.setChecked(self.deepLearningArtApp.getZCA_Whitening())
        # self.CheckBox_DataAug_HeightShift.setChecked(False if self.deepLearningArtApp.getHeightShift() == 0 else True)
        # self.CheckBox_DataAug_WidthShift.setChecked(False if self.deepLearningArtApp.getWidthShift() == 0 else True)
        # self.CheckBox_DataAug_Zoom.setChecked(False if self.deepLearningArtApp.getZoom() == 0 else True)
        # self.check_dataAugmentation_enabled()
        #
        # ################################################################################################################
        #
        # ################################################################################################################
        # # Signals and Slots
        # ################################################################################################################
        #
        # # select database button clicked
        # self.Button_DB.clicked.connect(self.button_DB_clicked)
        # # self.Button_DB.clicked.connect(self.button_DB_clicked)
        #
        # # output path button for patching clicked
        # self.Button_OutputPathPatching.clicked.connect(self.button_outputPatching_clicked)
        #
        # # TreeWidgets
        # self.TreeWidget_Patients.clicked.connect(self.getSelectedPatients)
        # self.TreeWidget_Datasets.clicked.connect(self.getSelectedDatasets)
        #
        # # Patching button
        # self.Button_Patching.clicked.connect(self.button_patching_clicked)
        #
        # # mask marking path button clicekd
        # self.Button_MarkingsPath.clicked.connect(self.button_markingsPath_clicked)
        #
        # # combo box splitting mode is changed
        # self.ComboBox_splittingMode.currentIndexChanged.connect(self.splittingMode_changed)
        #
        # # "use current data" button clicked
        # self.Button_useCurrentData.clicked.connect(self.button_useCurrentData_clicked)
        #
        # # select dataset is clicked
        # self.Button_selectDataset.clicked.connect(self.button_selectDataset_clicked)
        #
        # # learning output path button clicked
        # self.Button_LearningOutputPath.clicked.connect(self.button_learningOutputPath_clicked)
        #
        # # train button clicked
        # self.Button_train.clicked.connect(self.button_train_clicked)
        #
        # # combobox dnns
        # self.ComboBox_DNNs.currentIndexChanged.connect(self.selectedDNN_changed)
        #
        # # # show Dataset for ArtGAN Button
        # # self.Button_ShowDataset.clicked.connect(self.button_showDataset_clicked)
        #
        # # data augmentation enbaled changed
        # self.CheckBox_DataAugmentation.stateChanged.connect(self.check_dataAugmentation_enabled)
###########################################################################################################################
        # # third tab
        # self.matplotlibwidget_static.hide()
        # self.matplotlibwidget_static_2.hide()
        # self.matplotlibwidget_static_3.hide()
        #
        # #in the listView_2 the select name will save in chosenActivationName
        # self.chosenActivationName = []
        # # the slider's value is the chosen patch's number
        # self.chosenPatchNumber = 1
        # self.openfile_name=''
        #
        # self.model={}
        # self.qList=[]
        # self.totalPatches=0
        # self.activations = {}
        #
        # # from the .h5 file extract the name of each layer and the total number of patches
        # # self.wyh5.clicked.connect(self.openfile)
        # self.listView_2.clicked.connect(self.clickList_1)
        #
        # #self.horizontalSlider_3.valueChanged.connect(self.sliderValue)
        # self.horizontalSlider_3.sliderReleased.connect(self.sliderValue)
        # self.horizontalSlider_3.valueChanged.connect(self.lcdNumber_3.display)
###################################################################################################################

        def formsclick(n):
            if n==1:
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

        def lasso_onselect(verts):
            # print(verts)
            p = path.Path(verts)

            patch = None
            col_str = None
            if self.artifactbox.currentIndex() == 0:
                col_str = "31"
                patch = patches.PathPatch(p, fill=False, edgecolor='red', lw=2)
            elif self.artifactbox.currentIndex() == 1:
                col_str = "32"
                patch = patches.PathPatch(p, fill=False, edgecolor='green', lw=2)
            elif self.artifactbox.currentIndex() == 2:
                col_str = "33"
                patch = patches.PathPatch(p, fill=False, edgecolor='blue', lw=2)
            self.newax.add_patch(patch)
            sepkey = os.path.split(self.selectorPath)
            sepkey = sepkey[1]
            layer_name = sepkey # region

            with open(self.markfile) as json_data:  # 'r' read
                saveFile = json.load(json_data)
                p = np.ndarray.tolist(p.vertices)  #

                if layer_name in saveFile:
                    number_str = str(self.ind) + "_" + col_str + "_" + str(len(self.newax.patches) - 1)
                    saveFile[layer_name][number_str] = {'vertices': p, 'codes': None}
                else:
                    number_str = str(self.ind) + "_" + col_str + "_" + str(len(self.newax.patches) - 1)
                    saveFile[layer_name] = {number_str: {'vertices': p, 'codes': None}}

            # with open(self.markfile, 'w') as json_data:
            #     json.dump(saveFile, json_data)

            with open(self.markfile, 'w') as json_data:
                json_data.write(json.dumps(saveFile))

            self.newcanvas.draw()

        def ronselect(eclick, erelease):
            col_str = None
            rect = None
            ell = None
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            p = np.array(([x1, y1, x2, y2]))

            sepkey = os.path.split(self.selectorPath)
            sepkey = sepkey[1]
            layer_name = sepkey

            if toggle_selector.RS.active and not toggle_selector.ES.active:
                if self.artifactbox.currentIndex() == 0:
                    col_str = "11"
                    rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=False,
                                         edgecolor="red", lw=2)
                elif self.artifactbox.currentIndex() == 1:
                    col_str = "12"
                    rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=False,
                                         edgecolor="green", lw=2)
                elif self.artifactbox.currentIndex() == 2:
                    col_str = "13"
                    rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=False,
                                         edgecolor="blue", lw=2)
                self.newax.add_patch(rect)
            elif toggle_selector.ES.active and not toggle_selector.RS.active:
                if self.artifactbox.currentIndex() == 0:
                    col_str = "21"
                    ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                                  width=np.abs(x1 - x2), height=np.abs(y1 - y2), edgecolor="red", fc='None', lw=2)
                elif self.artifactbox.currentIndex() == 1:
                    col_str = "22"
                    ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                                  width=np.abs(x1 - x2), height=np.abs(y1 - y2), edgecolor="green", fc='None', lw=2)
                elif self.artifactbox.currentIndex() == 2:
                    col_str = "23"
                    ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                                  width=np.abs(x1 - x2), height=np.abs(y1 - y2), edgecolor="blue", fc='None', lw=2)
                self.newax.add_patch(ell)

            with open(self.markfile) as json_data:  # 'r' read
                saveFile = json.load(json_data)
                p = np.ndarray.tolist(p)   # format from datapre
                # print(p)
                if layer_name in saveFile:
                    number_str = str(self.ind) + "_" + col_str + "_" + str(len(self.newax.patches) - 1)
                    # saveFile[layer_name].update({number_str: p})
                    saveFile[layer_name][number_str] = {'points': p}
                else:
                    number_str = str(self.ind) + "_" + col_str + "_" + str(len(self.newax.patches) - 1)
                    saveFile[layer_name] = {number_str : {'points': p}}
            # with open(self.markfile, 'w') as json_data:
            #     json.dump(saveFile, json_data)
            with open(self.markfile, 'w') as json_data:
                json_data.write(json.dumps(saveFile))

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

        toggle_selector.RS = RectangleSelector(self.newax, ronselect, button=[1])  # drawtype='box', useblit=False, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True

        toggle_selector.ES = EllipseSelector(self.newax, ronselect, drawtype='line', button=[1], minspanx=5,
                                                  minspany=5,
                                                  spancoords='pixels',
                                                  interactive=True)  # drawtype='line', minspanx=5, minspany=5, spancoords='pixels', interactive=True

        toggle_selector.LS = LassoSelector(self.newax, lasso_onselect, button=[1])

        toggle_selector.ES.set_active(False)
        toggle_selector.RS.set_active(False)
        toggle_selector.LS.set_active(False)

        # connect('key_press_event', toggle_selector)

        self.newcanvas.mpl_connect('scroll_event', self.newonscroll)
        self.newcanvas.mpl_connect('button_press_event', self.mouse_clicked)
        self.newcanvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.newcanvas.mpl_connect('button_release_event', self.mouse_release)

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
            for i in reversed(range(self.maingrids.count())):
                self.maingrids.itemAt(i).clearWidgets()
                for j in reversed(range(self.maingrids.itemAt(i).secondline.count())):
                    self.maingrids.itemAt(i).secondline.itemAt(j).widget().setParent(None)
                self.maingrids.removeItem(self.maingrids.itemAt(i)) # invisible
        else:
            for i in reversed(range(self.maingrids.count())):
                self.maingrids.itemAt(i).clearWidgets()
                for j in reversed(range(self.maingrids.itemAt(i).gridLayout_1.count())):
                    self.maingrids.itemAt(i).gridLayout_1.itemAt(j).widget().setParent(None)
                for j in reversed(range(self.maingrids.itemAt(i).gridLayout_2.count())):
                    self.maingrids.itemAt(i).gridLayout_2.itemAt(j).widget().setParent(None)
                for j in reversed(range(self.maingrids.itemAt(i).gridLayout_3.count())):
                    self.maingrids.itemAt(i).gridLayout_3.itemAt(j).widget().setParent(None)
                for j in reversed(range(self.maingrids.itemAt(i).gridLayout_4.count())):
                    self.maingrids.itemAt(i).gridLayout_4.itemAt(j).widget().setParent(None)
                self.maingrids.removeItem(self.maingrids.itemAt(i))

    def saveCurrent(self):
        with open('lastWorkspace.json', 'r') as json_data:
            lastState = json.load(json_data)
            lastState['mode'][0] = self.gridsnr
            if self.gridsnr == 2:
                lastState['layout'][0] = self.layoutlines
                lastState['layout'][1] = self.layoutcolumns
            else:
                lastState['layout'][0] = self.layout3D

            lastState['listA'] = list1
            lastState['Probs'] = problist
            lastState['Hatches'] = hatchlist
            lastState['Pathes'] = pathlist
            lastState['NResults'] = pnamelist
            lastState['NrClass'] = cnrlist
            lastState['Corres'] = correslist

        with open('lastWorkspace.json', 'w') as json_data:
            json_data.write(json.dumps(lastState))

    def loadOld(self):
        self.clearall()
        global pathlist, list1, pnamelist, problist, hatchlist, correslist, cnrlist

        with open('lastWorkspace.json', 'r') as json_data:
            lastState = json.load(json_data)
            list1 = lastState['listA']
            problist = lastState['Probs']
            hatchlist = lastState['Hatches']
            pathlist = lastState['Pathes']
            pnamelist = lastState['NResults']
            cnrlist = lastState['NrClass']
            correslist = lastState['Corres']

            gridsnr = lastState['mode'][0]
            if gridsnr == 2:
                if self.vision == 3:
                    self.switchview()
                self.layoutlines = lastState['layout'][0]
                self.layoutcolumns = lastState['layout'][1]
                self.linebox.setCurrentIndex(self.layoutlines - 1)
                self.columnbox.setCurrentIndex(self.layoutcolumns - 1)
            else:
                if self.vision == 2:
                    self.switchview()
                self.layout3D = lastState['layout'][0]
                self.linebox.setCurrentIndex(self.layout3D - 1)

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

                    self.maingrids.addLayout(blocklayout, i, j)
            if pathlist:
                n = 0
                for i in range(self.layoutlines):
                    for j in range(self.layoutcolumns):
                        if n < len(pathlist):
                            self.maingrids.itemAtPosition(i, j).pathbox.setCurrentIndex(n+1)
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
                self.maingrids.addLayout(blockline, i, 0)
            if pathlist:
                n = 0
                for i in range(self.layout3D):
                    if n < len(pathlist):
                        self.maingrids.itemAtPosition(i, 0).imagelist.setCurrentIndex(n+1)
                        n+=1
                    else:
                        break

    def linkMode(self):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids.itemAtPosition(i, j).islinked and not self.maingrids.itemAtPosition(i, j).skiplink:
                        self.maingrids.itemAtPosition(i, j).Viewpanel.zoom_link.connect(self.zoomAll)
                        self.maingrids.itemAtPosition(i, j).Viewpanel.move_link.connect(self.moveAll)
                        self.maingrids.itemAtPosition(i, j).anewcanvas.grey_link.connect(self.greyAll)
                        self.maingrids.itemAtPosition(i, j).anewcanvas.slice_link.connect(self.sliceAll)
                        self.maingrids.itemAtPosition(i, j).skiplink = True # avoid multi link
                    elif not self.maingrids.itemAtPosition(i, j).islinked and not self.maingrids.itemAtPosition(i, j).skipdis:
                        self.maingrids.itemAtPosition(i, j).Viewpanel.zoom_link.disconnect()
                        self.maingrids.itemAtPosition(i, j).Viewpanel.move_link.disconnect()
                        self.maingrids.itemAtPosition(i, j).anewcanvas.grey_link.disconnect()
                        self.maingrids.itemAtPosition(i, j).anewcanvas.slice_link.disconnect()
                        self.maingrids.itemAtPosition(i, j).skipdis = True
        else:
            for i in range(self.layout3D):
                if self.maingrids.itemAtPosition(i, 0).islinked and not self.maingrids.itemAtPosition(i, 0).skiplink:
                    self.maingrids.itemAtPosition(i, 0).Viewpanel1.zoom_link.connect(self.zoomAll)
                    self.maingrids.itemAtPosition(i, 0).Viewpanel1.move_link.connect(self.moveAll)
                    self.maingrids.itemAtPosition(i, 0).newcanvas1.grey_link.connect(self.greyAll)
                    self.maingrids.itemAtPosition(i, 0).Viewpanel2.zoom_link.connect(self.zoomAll)
                    self.maingrids.itemAtPosition(i, 0).Viewpanel2.move_link.connect(self.moveAll)
                    self.maingrids.itemAtPosition(i, 0).newcanvas2.grey_link.connect(self.greyAll)
                    self.maingrids.itemAtPosition(i, 0).Viewpanel3.zoom_link.connect(self.zoomAll)
                    self.maingrids.itemAtPosition(i, 0).Viewpanel3.move_link.connect(self.moveAll)
                    self.maingrids.itemAtPosition(i, 0).newcanvas3.grey_link.connect(self.greyAll)
                elif not self.maingrids.itemAtPosition(i, 0).islinked and not self.maingrids.itemAtPosition(i, 0).skipdis:
                    self.maingrids.itemAtPosition(i, 0).Viewpanel1.zoom_link.disconnect()
                    self.maingrids.itemAtPosition(i, 0).Viewpanel1.move_link.disconnect()
                    self.maingrids.itemAtPosition(i, 0).newcanvas1.grey_link.disconnect()
                    self.maingrids.itemAtPosition(i, 0).Viewpanel2.zoom_link.disconnect()
                    self.maingrids.itemAtPosition(i, 0).Viewpanel2.move_link.disconnect()
                    self.maingrids.itemAtPosition(i, 0).newcanvas2.grey_link.disconnect()
                    self.maingrids.itemAtPosition(i, 0).Viewpanel3.zoom_link.disconnect()
                    self.maingrids.itemAtPosition(i, 0).Viewpanel3.move_link.disconnect()
                    self.maingrids.itemAtPosition(i, 0).newcanvas3.grey_link.disconnect()

    def zoomAll(self, factor):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids.itemAtPosition(i, j).islinked:
                        self.maingrids.itemAtPosition(i, j).Viewpanel.linkedZoom(factor)
        else:
            for i in range(self.layout3D):
                if self.maingrids.itemAtPosition(i, 0).islinked:
                    self.maingrids.itemAtPosition(i, 0).Viewpanel1.linkedZoom(factor)
                    self.maingrids.itemAtPosition(i, 0).Viewpanel2.linkedZoom(factor)
                    self.maingrids.itemAtPosition(i, 0).Viewpanel3.linkedZoom(factor)

    def moveAll(self, movelist):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids.itemAtPosition(i, j).islinked:
                        self.maingrids.itemAtPosition(i, j).Viewpanel.linkedMove(movelist)
        else:
            for i in range(self.layout3D):
                if self.maingrids.itemAtPosition(i, 0).islinked:
                    self.maingrids.itemAtPosition(i, 0).Viewpanel1.linkedMove(movelist)
                    self.maingrids.itemAtPosition(i, 0).Viewpanel2.linkedMove(movelist)
                    self.maingrids.itemAtPosition(i, 0).Viewpanel3.linkedMove(movelist)

    def greyAll(self, glist):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids.itemAtPosition(i, j).islinked:
                        self.maingrids.itemAtPosition(i, j).anewcanvas.linkedGrey(glist)
        else:
            for i in range(self.layout3D):
                if self.maingrids.itemAtPosition(i, 0).islinked:
                    self.maingrids.itemAtPosition(i, 0).newcanvas1.linkedGrey(glist)
                    self.maingrids.itemAtPosition(i, 0).newcanvas2.linkedGrey(glist)
                    self.maingrids.itemAtPosition(i, 0).newcanvas3.linkedGrey(glist)

    def sliceAll(self, data):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids.itemAtPosition(i, j).islinked:
                        self.maingrids.itemAtPosition(i, j).anewcanvas.linkedSlice(data)

    def loadMR(self):
        # if self.gridson == False:
        #     QtWidgets.QMessageBox.information(self, 'Warning', 'Grids needed!')
        # else:
        with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])

        self.PathDicom = QtWidgets.QFileDialog.getExistingDirectory(self, "open file", dbinfo.sPathIn)
        # self.PathDicom = QtWidgets.QFileDialog.getExistingDirectory(self, "open file", "C:/Users/hansw/Videos/artefacts")
        if self.PathDicom:
            self.i = self.i + 1
            self.openfile.setDisabled(True)
            self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
            self.overlay.setGeometry(QtCore.QRect(950, 400, 171, 141))
            self.overlay.show()
            from loadf import loadImage
            self.newMR = loadImage(self.PathDicom)
            self.newMR.trigger.connect(self.loadEnd)
            self.newMR.start()
        else:
            pass

    # def plot_3d(self, image, threshold=-300):
    #     # Position the scan upright,
    #     # so the head of the patient would be at the top facing the camera
    #     p = image.transpose(2, 1, 0)
    #     verts, faces, a, b = measure.marching_cubes(p, threshold)
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(111, projection='3d')
    #     # Fancy indexing: `verts[faces]` to generate a collection of triangles
    #     mesh = Poly3DCollection(verts[faces], alpha=0.1)
    #     face_color = [0.5, 0.5, 1]
    #     mesh.set_facecolor(face_color)
    #     ax.add_collection3d(mesh)
    #     ax.set_xlim(0, p.shape[0])
    #     ax.set_ylim(0, p.shape[1])
    #     ax.set_zlim(0, p.shape[2])
    #     plt.show()

    def loadEnd(self):
        self.overlay.killTimer(self.overlay.timer)
        self.overlay.hide()
        # self.openfile.setText("load image")
        self.openfile.setDisabled(False)

        # self.plot_3d(self.newMR.svoxel, 200)

        pathlist.append(self.PathDicom)
        list1.append(self.newMR.voxel_ndarray)
        shapelist.append(self.newMR.new_shape)
        if self.gridson == True:
            if self.gridsnr == 2:
                for i in range(self.layoutlines):
                    for j in range(self.layoutcolumns):
                        self.maingrids.itemAtPosition(i, j).addPathd(self.PathDicom)
                for i in range(self.layoutlines):
                    for j in range(self.layoutcolumns):
                        if self.maingrids.itemAtPosition(i, j).mode == 1 and \
                                self.maingrids.itemAtPosition(i, j).pathbox.currentIndex() == 0:
                            self.maingrids.itemAtPosition(i, j).pathbox.setCurrentIndex(len(pathlist))
                            break
                    else:
                        continue
                    break
            else:
                for i in range(self.layout3D):
                    self.maingrids.itemAtPosition(i, 0).addPathim(self.PathDicom)
                for i in range(self.layout3D):
                    if self.maingrids.itemAtPosition(i, 0).vmode == 1 and \
                            self.maingrids.itemAtPosition(i, 0).imagelist.currentIndex() == 0:
                        self.maingrids.itemAtPosition(i, 0).imagelist.setCurrentIndex(len(pathlist))
                        break
        else:
            pass

    def unpatching2(self, result, orig):
        PatchSize = np.array((40.0, 40.0))
        PatchOverlay = 0.5
        imglay = fUnpatch2D(result, PatchSize, PatchOverlay, orig.shape, 0) # 0 for reference
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


    def loadpatch(self):
        # resultfile = QtWidgets.QFileDialog.getOpenFileName(self, "choose the result file",
        #                                 "C:/Users/hansw/Desktop/Ma_code/PyQt","mat files(*.mat);;h5 files(*.h5)")[0]
        resultfile = QtWidgets.QFileDialog.getOpenFileName(self, 'choose the result file', '',
                'mat files(*.mat);;h5 files(*.h5)', None, QtWidgets.QFileDialog.DontUseNativeDialog)[0]

        if resultfile:
            with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
            dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])
            PathDicom = QtWidgets.QFileDialog.getExistingDirectory(self, "choose the corresponding image", dbinfo.sPathIn)
            if PathDicom not in pathlist:
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
                    cnum = np.array(conten['prob_test'])
                    if cnum.shape[1] == 2:
                        IType = self.unpatching2(conten['prob_test'])

                        problist.append(IType)
                        hatchlist.append(empty1)
                        cnrlist.append(2)

                    # elif cnum.shape[1] == 8: 

                nameofCfile = os.path.split(resultfile)[1]
                nameofCfile = nameofCfile + '   class:' + str(cnum.shape[1])
                pnamelist.append(nameofCfile)
                if self.vision == 3:
                    for i in range(self.layout3D):
                        self.maingrids.itemAtPosition(i, 0).addPathre(nameofCfile)
                else:
                    for i in range(self.layoutlines):
                        for j in range(self.layoutcolumns):
                            self.maingrids.itemAtPosition(i, j).addPathre(nameofCfile)

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

##################################################
    def selectormode(self):
        if self.selectoron == False:
            self.selectoron = True
            if self.gridson == True:
                self.clearall()
                self.gridson = False    ###

            self.graylabel = QtWidgets.QLabel()
            self.slicelabel = QtWidgets.QLabel()
            self.maingrids.addWidget(self.graylabel, 0, 0, 1, 1)
            self.maingrids.addWidget(self.slicelabel, 0, 1, 1, 1)
            self.maingrids.addWidget(self.newcanvas, 1, 0, 20, 2)

            self.artifactbox.setDisabled(False)
            self.brectangle.setDisabled(False)
            self.bellipse.setDisabled(False)
            self.blasso.setDisabled(False)
        else:
            self.selectoron = False
            self.artifactbox.setDisabled(True)
            self.brectangle.setDisabled(True)
            self.bellipse.setDisabled(True)
            self.blasso.setDisabled(True)
            # self.newax.clear()
            for i in reversed(range(self.maingrids.count())):
                self.maingrids.itemAt(i).widget().setParent(None)

    def chooseMark(self):
        with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])
        self.selectorPath = QtWidgets.QFileDialog.getExistingDirectory(self, "choose the image to view", dbinfo.sPathIn)
        # self.selectorPath = QtWidgets.QFileDialog.getExistingDirectory(self, "choose the image to view",
        #                                                 "C:/Users/hansw/Videos/artefacts/MRPhysics/newProtocol")
        if self.selectorPath:
            # self.markfile = QtWidgets.QFileDialog.getOpenFileName(self, "choose the marking file",
            #                                                       "C:/Users/hansw/Desktop/Ma_code/PyQt/Markings", "") [0]
            self.markfile = QtWidgets.QFileDialog.getOpenFileName(self, 'choose the marking file', '',
                            'json files(*.json)', None, QtWidgets.QFileDialog.DontUseNativeDialog)[0]
            if self.markfile:
                files = sorted([os.path.join(self.selectorPath, file) for file in os.listdir(self.selectorPath)],
                               key=os.path.getctime)
                datasets = [dicom.read_file(f) \
                            for f in files]
                try:
                    self.imageforselector, pixel_space = dicom_numpy.combine_slices(datasets)
                except dicom_numpy.DicomImportException:
                    raise
                with open(self.markfile, 'r') as json_data:  ##### first time
                    self.loadFile = json.load(json_data)
                # print(self.loadFile)
            else:
                pass
        else:
            pass

        self.ind = 0
        self.slices = self.imageforselector.shape[2]

        self.newax.clear()
        self.pltc = self.newax.imshow(np.swapaxes(self.imageforselector[:, :, self.ind], 0, 1), cmap='gray', vmin=0,
                               vmax=2094)
        v_min, v_max = self.pltc.get_clim()
        self.graylist = []
        self.graylist.append(v_min)
        self.graylist.append(v_max)

        self.emitlist = []
        self.emitlist.append(self.ind)
        self.emitlist.append(self.slices)

        number_Patch = 0
        cur_no = "0"
        sepkey = os.path.split(self.selectorPath)
        sepkey = sepkey[1]
        if sepkey in self.loadFile:
            layer = self.loadFile[sepkey]
            while (cur_no + "_11_" + str(number_Patch)) in layer or (
                    cur_no + "_12_" + str(number_Patch)) in layer or (
                    cur_no + "_13_" + str(number_Patch)) in layer or (
                    cur_no + "_21_" + str(number_Patch)) in layer or (
                    cur_no + "_22_" + str(number_Patch)) in layer or (
                    cur_no + "_23_" + str(number_Patch)) in layer or (
                    cur_no + "_31_" + str(number_Patch)) in layer or (
                    cur_no + "_32_" + str(number_Patch)) in layer or (
                    cur_no + "_33_" + str(number_Patch)) in layer:
                patch = None
                if cur_no + "_11_" + str(number_Patch) in layer:
                    p = layer[cur_no + "_11_" + str(number_Patch)]
                    p = np.asarray(p['points'])
                    patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]),
                                          np.abs(p[1] - p[3]), fill=False,
                                          edgecolor="red", lw=2)
                elif cur_no + "_12_" + str(number_Patch) in layer:
                    p = layer[cur_no + "_12_" + str(number_Patch)]
                    p = np.asarray(p['points'])
                    patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]),
                                          np.abs(p[1] - p[3]), fill=False,
                                          edgecolor="green", lw=2)
                elif cur_no + "_13_" + str(number_Patch) in layer:
                    p = layer[cur_no + "_13_" + str(number_Patch)]
                    p = np.asarray(p['points'])
                    patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]),
                                          np.abs(p[1] - p[3]), fill=False,
                                          edgecolor="blue", lw=2)
                elif cur_no + "_21_" + str(number_Patch) in layer:
                    p = layer[cur_no + "_21_" + str(number_Patch)]
                    p = np.asarray(p['points'])
                    patch = Ellipse(
                        xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                        width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="red", fc='None', lw=2)
                elif cur_no + "_22_" + str(number_Patch) in layer:
                    p = layer[cur_no + "_22_" + str(number_Patch)]
                    p = np.asarray(p['points'])
                    patch = Ellipse(
                        xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                        width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="green", fc='None', lw=2)
                elif cur_no + "_23_" + str(number_Patch) in layer:
                    p = layer[cur_no + "_23_" + str(number_Patch)]
                    p = np.asarray(p['points'])
                    patch = Ellipse(
                        xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                        width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="blue", fc='None', lw=2)
                elif cur_no + "_31_" + str(number_Patch) in layer:
                    p = layer[cur_no + "_31_" + str(number_Patch)]
                    p = path.Path(np.asarray(p['vertices']), p['codes'])
                    patch = patches.PathPatch(p, fill=False, edgecolor='red', lw=2)
                elif cur_no + "_32_" + str(number_Patch) in layer:
                    p = layer[cur_no + "_32_" + str(number_Patch)]
                    p = path.Path(np.asarray(p['vertices']), p['codes'])
                    patch = patches.PathPatch(p, fill=False, edgecolor='green', lw=2)
                elif cur_no + "_33" \
                              "_" + str(number_Patch) in layer:
                    p = layer[cur_no + "_33_" + str(number_Patch)]
                    p = path.Path(np.asarray(p['vertices']), p['codes'])
                    patch = patches.PathPatch(p, fill=False, edgecolor='blue', lw=2)
                self.newax.add_patch(patch)
                number_Patch += 1

        self.slicelabel.setText('Slice %s' % (self.ind + 1) + '/ %s' % (self.slices))
        self.graylabel.setText('Grayscale Range %s' % (self.graylist))
        self.update_data.connect(self.updateSlices)
        self.gray_data.connect(self.updateGray)
        self.new_page.connect(self.newSliceview)

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

    def newslicesview(self):
        # plt.cla() # 1
        self.newax.clear()
        self.pltc = self.newax.imshow(np.swapaxes(self.imageforselector[:, :, self.ind], 0, 1), cmap='gray', vmin=0, vmax=2094)

        with open(self.markfile) as json_data:
            loadFile2 = json.load(json_data)
            number_Patch = 0
            sepkey = os.path.split(self.selectorPath)
            sepkey = sepkey[1]
            if sepkey in loadFile2:
                layer = loadFile2[sepkey]
                cur_no = str(self.ind)

                while (cur_no + "_11_" + str(number_Patch)) in layer or (cur_no + "_12_" + str(number_Patch)) in layer or (
                        cur_no + "_13_" + str(number_Patch)) in layer or (cur_no + "_21_" + str(number_Patch)) in layer or (
                        cur_no + "_22_" + str(number_Patch)) in layer or (cur_no + "_23_" + str(number_Patch)) in layer or (
                        cur_no + "_31_" + str(number_Patch)) in layer or (cur_no + "_32_" + str(number_Patch)) in layer or (
                        cur_no + "_33_" + str(number_Patch)) in layer:
                    patch = None
                    if cur_no + "_11_" + str(number_Patch) in layer:
                        p = layer[cur_no + "_11_" + str(number_Patch)]
                        p = np.asarray(p['points'])
                        patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]),
                                              np.abs(p[1] - p[3]), fill=False,
                                              edgecolor="red", lw=2)
                    elif cur_no + "_12_" + str(number_Patch) in layer:
                        p = layer[cur_no + "_12_" + str(number_Patch)]
                        p = np.asarray(p['points'])
                        patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]),
                                              np.abs(p[1] - p[3]), fill=False,
                                              edgecolor="green", lw=2)
                    elif cur_no + "_13_" + str(number_Patch) in layer:
                        p = layer[cur_no + "_13_" + str(number_Patch)]
                        p = np.asarray(p['points'])
                        patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]),
                                              np.abs(p[1] - p[3]), fill=False,
                                              edgecolor="blue", lw=2)
                    elif cur_no + "_21_" + str(number_Patch) in layer:
                        p = layer[cur_no + "_21_" + str(number_Patch)]
                        p = np.asarray(p['points'])
                        patch = Ellipse(
                            xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                            width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="red", fc='None', lw=2)
                    elif cur_no + "_22_" + str(number_Patch) in layer:
                        p = layer[cur_no + "_22_" + str(number_Patch)]
                        p = np.asarray(p['points'])
                        patch = Ellipse(
                            xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                            width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="green", fc='None', lw=2)
                    elif cur_no + "_23_" + str(number_Patch) in layer:
                        p = layer[cur_no + "_23_" + str(number_Patch)]
                        p = np.asarray(p['points'])
                        patch = Ellipse(
                            xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                            width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="blue", fc='None', lw=2)
                    elif cur_no + "_31_" + str(number_Patch) in layer:
                        p = layer[cur_no + "_31_" + str(number_Patch)]
                        p = path.Path(np.asarray(p['vertices']), p['codes'])
                        patch = patches.PathPatch(p, fill=False, edgecolor='red', lw=2)
                    elif cur_no + "_32_" + str(number_Patch) in layer:
                        p = layer[cur_no + "_32_" + str(number_Patch)]
                        p = path.Path(np.asarray(p['vertices']), p['codes'])
                        patch = patches.PathPatch(p, fill=False, edgecolor='green', lw=2)
                    elif cur_no + "_33" \
                                  "_" + str(number_Patch) in layer:
                        p = layer[cur_no + "_33_" + str(number_Patch)]
                        p = path.Path(np.asarray(p['vertices']), p['codes'])
                        patch = patches.PathPatch(p, fill=False, edgecolor='blue', lw=2)
                    self.newax.add_patch(patch)
                    number_Patch += 1

        self.newcanvas.draw()  # not self.newcanvas.show()

        v_min, v_max = self.pltc.get_clim()
        self.graylist[0] = v_min
        self.graylist[1] = v_max
        self.new_page.emit()

    def mouse_clicked(self, event):
        if event.button == 2:
            self.x_clicked = event.x
            self.y_clicked = event.y
            self.mouse_second_clicked = True
        elif event.button == 3:
            with open(self.markfile) as json_data:
                deleteMark = json.load(json_data)
                sepkey = os.path.split(self.selectorPath)
                sepkey = sepkey[1]
                layer = deleteMark[sepkey]
                cur_no = str(self.ind)
                cur_pa = str(len(self.newax.patches) - 1)

                if cur_no + "_11_" + cur_pa in layer:
                    layer.pop(cur_no + "_11_" + cur_pa, None)
                elif cur_no + "_12_" + cur_pa in layer:
                    layer.pop(cur_no + "_12_" + cur_pa, None)
                elif cur_no + "_13_" + cur_pa in layer:
                    layer.pop(cur_no + "_13_" + cur_pa, None)
                elif cur_no + "_21_" + cur_pa in layer:
                    layer.pop(cur_no + "_21_" + cur_pa, None)
                elif cur_no + "_22_" + cur_pa in layer:
                    layer.pop(cur_no + "_22_" + cur_pa, None)
                elif cur_no + "_23_" + cur_pa in layer:
                    layer.pop(cur_no + "_23_" + cur_pa, None)
                elif cur_no + "_31_" + cur_pa in layer:
                    layer.pop(cur_no + "_31_" + cur_pa, None)
                elif cur_no + "_32_" + cur_pa in layer:
                    layer.pop(cur_no + "_32_" + cur_pa, None)
                elif cur_no + "_33_" + cur_pa in layer:
                    layer.pop(cur_no + "_33_" + cur_pa, None)
            # with open(self.markfile, 'w') as json_data:
            #     json.dump(deleteMark, json_data)
            with open(self.markfile, 'w') as json_data:
                json_data.write(json.dumps(deleteMark))

            self.newslicesview()

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
            self.pltc.set_clim(vmin=v_min, vmax=v_max)
            self.graylist[0] = v_min.round(2)
            self.graylist[1] = v_max.round(2)
            self.gray_data.emit(self.graylist)

            self.newcanvas.draw()

    def mouse_release(self, event):
        if event.button == 2:
            self.mouse_second_clicked = False

    def newSliceview(self):
        self.graylabel.setText('Grayscale Range %s' % (self.graylist))

    def updateSlices(self, elist):
        self.slicelabel.setText('Slice %s' % (elist[0] + 1) + '/ %s' % (elist[1]))

    def updateGray(self, elist):
        self.graylabel.setText('Grayscale Range %s' % (elist))

    def closeEvent(self, QCloseEvent):
        reply = QMessageBox.question(self, 'Warning', 'Are you sure to exit?', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()

############################################################################################  the second tab
    # def button_train_clicked(self):
    #     # set epochs
    #     self.deepLearningArtApp.setEpochs(self.SpinBox_Epochs.value())
    #
    #     # handle check states of check boxes for used classes
    #     self.deepLearningArtApp.setUsingArtifacts(self.CheckBox_Artifacts.isChecked())
    #     self.deepLearningArtApp.setUsingBodyRegions(self.CheckBox_BodyRegion.isChecked())
    #     self.deepLearningArtApp.setUsingTWeighting(self.CheckBox_TWeighting.isChecked())
    #
    #     # set learning rates and batch sizes
    #     try:
    #         batchSizes = np.fromstring(self.LineEdit_BatchSizes.text(), dtype=np.int, sep=',')
    #         self.deepLearningArtApp.setBatchSizes(batchSizes)
    #         learningRates = np.fromstring(self.LineEdit_LearningRates.text(), dtype=np.float32, sep=',')
    #         self.deepLearningArtApp.setLearningRates(learningRates)
    #     except:
    #         raise ValueError("Wrong input format of learning rates! Enter values seperated by ','. For example: 0.1,0.01,0.001")
    #
    #     # set optimizer
    #     selectedOptimizer = self.ComboBox_Optimizers.currentText()
    #     if selectedOptimizer == "SGD":
    #         self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.SGD_OPTIMIZER)
    #     elif selectedOptimizer == "RMSprop":
    #         self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.RMS_PROP_OPTIMIZER)
    #     elif selectedOptimizer == "Adagrad":
    #         self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.ADAGRAD_OPTIMIZER)
    #     elif selectedOptimizer == "Adadelta":
    #         self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.ADADELTA_OPTIMIZER)
    #     elif selectedOptimizer == "Adam":
    #         self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.ADAM_OPTIMIZER)
    #     else:
    #         raise ValueError("Unknown Optimizer!")
    #
    #     # set weigth decay
    #     self.deepLearningArtApp.setWeightDecay(float(self.DoubleSpinBox_WeightDecay.value()))
    #     # set momentum
    #     self.deepLearningArtApp.setMomentum(float(self.DoubleSpinBox_Momentum.value()))
    #     # set nesterov enabled
    #     if self.CheckBox_Nesterov.checkState() == Qt.Checked:
    #         self.deepLearningArtApp.setNesterovEnabled(True)
    #     else:
    #         self.deepLearningArtApp.setNesterovEnabled(False)
    #
    #     # handle data augmentation
    #     if self.CheckBox_DataAugmentation.checkState() == Qt.Checked:
    #         self.deepLearningArtApp.setDataAugmentationEnabled(True)
    #         # get all checked data augmentation options
    #         if self.CheckBox_DataAug_horizontalFlip.checkState() == Qt.Checked:
    #             self.deepLearningArtApp.setHorizontalFlip(True)
    #         else:
    #             self.deepLearningArtApp.setHorizontalFlip(False)
    #
    #         if self.CheckBox_DataAug_verticalFlip.checkState() == Qt.Checked:
    #             self.deepLearningArtApp.setVerticalFlip(True)
    #         else:
    #             self.deepLearningArtApp.setVerticalFlip(False)
    #
    #         if self.CheckBox_DataAug_Rotation.checkState() == Qt.Checked:
    #             self.deepLearningArtApp.setRotation(True)
    #         else:
    #             self.deepLearningArtApp.setRotation(False)
    #
    #         if self.CheckBox_DataAug_zcaWeighting.checkState() == Qt.Checked:
    #             self.deepLearningArtApp.setZCA_Whitening(True)
    #         else:
    #             self.deepLearningArtApp.setZCA_Whitening(False)
    #
    #         if self.CheckBox_DataAug_HeightShift.checkState() == Qt.Checked:
    #             self.deepLearningArtApp.setHeightShift(True)
    #         else:
    #             self.deepLearningArtApp.setHeightShift(False)
    #
    #         if self.CheckBox_DataAug_WidthShift.checkState() == Qt.Checked:
    #             self.deepLearningArtApp.setWidthShift(True)
    #         else:
    #             self.deepLearningArtApp.setWidthShift(False)
    #
    #         if self.CheckBox_DataAug_Zoom.checkState() == Qt.Checked:
    #             self.deepLearningArtApp.setZoom(True)
    #         else:
    #             self.deepLearningArtApp.setZoom(False)
    #
    #
    #         # contrast improvement (contrast stretching, adaptive equalization, histogram equalization)
    #         # it is not recommended to set more than one of them to true
    #         if self.RadioButton_DataAug_contrastStretching.isChecked():
    #             self.deepLearningArtApp.setContrastStretching(True)
    #         else:
    #             self.deepLearningArtApp.setContrastStretching(False)
    #
    #         if self.RadioButton_DataAug_histogramEq.isChecked():
    #             self.deepLearningArtApp.setHistogramEqualization(True)
    #         else:
    #             self.deepLearningArtApp.setHistogramEqualization(False)
    #
    #         if self.RadioButton_DataAug_adaptiveEq.isChecked():
    #             self.deepLearningArtApp.setAdaptiveEqualization(True)
    #         else:
    #             self.deepLearningArtApp.setAdaptiveEqualization(False)
    #     else:
    #         # disable data augmentation
    #         self.deepLearningArtApp.setDataAugmentationEnabled(False)
    #
    #
    #     # start training process
    #     self.deepLearningArtApp.performTraining()
    #
    #
    #
    # def button_markingsPath_clicked(self):
    #     dir = self.openFileNamesDialog(self.deepLearningArtApp.getMarkingsPath())
    #     self.Label_MarkingsPath.setText(dir)
    #     self.deepLearningArtApp.setMarkingsPath(dir)
    #
    #
    #
    # def button_patching_clicked(self):
    #     if self.deepLearningArtApp.getSplittingMode() == DeepLearningArtApp.NONE_SPLITTING:
    #         QMessageBox.about(self, "My message box", "Select Splitting Mode!")
    #         return 0
    #
    #     self.getSelectedDatasets()
    #     self.getSelectedPatients()
    #
    #     # get patching parameters
    #     self.deepLearningArtApp.setPatchSizeX(self.SpinBox_PatchX.value())
    #     self.deepLearningArtApp.setPatchSizeY(self.SpinBox_PatchY.value())
    #     self.deepLearningArtApp.setPatchSizeZ(self.SpinBox_PatchZ.value())
    #     self.deepLearningArtApp.setPatchOverlapp(self.SpinBox_PatchOverlapp.value())
    #
    #     # get labling parameters
    #     if self.RadioButton_MaskLabeling.isChecked():
    #         self.deepLearningArtApp.setLabelingMode(DeepLearningArtApp.MASK_LABELING)
    #     elif self.RadioButton_PatchLabeling.isChecked():
    #         self.deepLearningArtApp.setLabelingMode(DeepLearningArtApp.PATCH_LABELING)
    #
    #     # get patching parameters
    #     if self.ComboBox_Patching.currentIndex() == 1:
    #         # 2D patching selected
    #         self.deepLearningArtApp.setPatchingMode(DeepLearningArtApp.PATCHING_2D)
    #     elif self.ComboBox_Patching.currentIndex() == 2:
    #         # 3D patching selected
    #         self.deepLearningArtApp.setPatchingMode(DeepLearningArtApp.PATCHING_3D)
    #     else:
    #         self.ComboBox_Patching.setCurrentIndex(1)
    #         self.deepLearningArtApp.setPatchingMode(DeepLearningArtApp.PATCHING_2D)
    #
    #     #using segmentation mask
    #     self.deepLearningArtApp.setUsingSegmentationMasks(self.CheckBox_SegmentationMask.isChecked())
    #
    #     # handle store mode
    #     self.deepLearningArtApp.setStoreMode(self.ComboBox_StoreOptions.currentIndex())
    #
    #     print("Start Patching for ")
    #     print("the Patients:")
    #     for x in self.deepLearningArtApp.getSelectedPatients():
    #         print(x)
    #     print("and the Datasets:")
    #     for x in self.deepLearningArtApp.getSelectedDatasets():
    #         print(x)
    #     print("with the following Patch Parameters:")
    #     print("Patch Size X: " + str(self.deepLearningArtApp.getPatchSizeX()))
    #     print("Patch Size Y: " + str(self.deepLearningArtApp.getPatchSizeY()))
    #     print("Patch Overlapp: " + str(self.deepLearningArtApp.getPatchOverlapp()))
    #
    #     #generate dataset
    #     self.deepLearningArtApp.generateDataset()
    #
    #     #check if attributes in DeepLearningArtApp class contains dataset
    #     if self.deepLearningArtApp.datasetAvailable() == True:
    #         # if yes, make the use current data button available
    #         self.Button_useCurrentData.setEnabled(True)
    #
    #
    #
    # def button_outputPatching_clicked(self):
    #     dir = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
    #     self.Label_OutputPathPatching.setText(dir)
    #     self.deepLearningArtApp.setOutputPathForPatching(dir)
    #
    #
    #
    # def getSelectedPatients(self):
    #     selectedPatients = []
    #     for i in range(self.TreeWidget_Patients.topLevelItemCount()):
    #         if self.TreeWidget_Patients.topLevelItem(i).checkState(0) == Qt.Checked:
    #             selectedPatients.append(self.TreeWidget_Patients.topLevelItem(i).text(0))
    #
    #     self.deepLearningArtApp.setSelectedPatients(selectedPatients)
    #
    #
    #
    # def button_DB_clicked(self):
    #     dir = self.openFileNamesDialog(self.deepLearningArtApp.getPathToDatabase())
    #     self.deepLearningArtApp.setPathToDatabase(dir)
    #     self.manageTreeView()
    #
    #
    #
    # def openFileNamesDialog(self, dir=None):
    #     if dir==None:
    #         # dir = "D:" + os.sep + "med_data" + os.sep + "MRPhysics"
    #         dir = 'C:' + os.sep + 'Users' + os.sep + 'hansw' + os.sep + 'Videos' + os.sep + 'artefacts'\
    #               + os.sep + 'MRPhysics'  + os.sep + 'newProtocol'
    #     options = QFileDialog.Options()
    #     options |=QFileDialog.DontUseNativeDialog
    #
    #     ret = QFileDialog.getExistingDirectory(self, "Select Directory", dir)
    #     # path to database
    #     dir = str(ret)
    #     return dir
    #
    #
    #
    # def manageTreeView(self):
    #     # all patients in database
    #     if os.path.exists(self.deepLearningArtApp.getPathToDatabase()):
    #         subdirs = os.listdir(self.deepLearningArtApp.getPathToDatabase())
    #         self.TreeWidget_Patients.setHeaderLabel("Patients:")
    #
    #         for x in subdirs:
    #             item = QTreeWidgetItem()
    #             item.setText(0, str(x))
    #             item.setCheckState(0, Qt.Unchecked)
    #             self.TreeWidget_Patients.addTopLevelItem(item)
    #
    #         self.Label_DB.setText(self.deepLearningArtApp.getPathToDatabase())
    #
    #
    #
    # def manageTreeViewDatasets(self):
    #     print(os.path.dirname(self.deepLearningArtApp.getPathToDatabase()))
    #     # manage datasets
    #     self.TreeWidget_Datasets.setHeaderLabel("Datasets:")
    #     for ds in DeepLearningArtApp.datasets.keys():
    #         dataset = DeepLearningArtApp.datasets[ds].getPathdata()
    #         item = QTreeWidgetItem()
    #         item.setText(0, dataset)
    #         item.setCheckState(0, Qt.Unchecked)
    #         self.TreeWidget_Datasets.addTopLevelItem(item)
    #
    #
    #
    # def getSelectedDatasets(self):
    #     selectedDatasets = []
    #     for i in range(self.TreeWidget_Datasets.topLevelItemCount()):
    #         if self.TreeWidget_Datasets.topLevelItem(i).checkState(0) == Qt.Checked:
    #             selectedDatasets.append(self.TreeWidget_Datasets.topLevelItem(i).text(0))
    #
    #     self.deepLearningArtApp.setSelectedDatasets(selectedDatasets)
    #
    #
    #
    # def selectedDNN_changed(self):
    #     self.deepLearningArtApp.setNeuralNetworkModel(self.ComboBox_DNNs.currentText())
    #
    #
    #
    # def button_useCurrentData_clicked(self):
    #     if self.deepLearningArtApp.datasetAvailable() == True:
    #         self.Label_currentDataset.setText("Current Dataset is used...")
    #         self.GroupBox_TrainNN.setEnabled(True)
    #     else:
    #         self.Button_useCurrentData.setEnabled(False)
    #         self.Label_currentDataset.setText("No Dataset selected!")
    #         self.GroupBox_TrainNN.setEnabled(False)
    #
    #
    #
    # def button_selectDataset_clicked(self):
    #     pathToDataset = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
    #     retbool, datasetName = self.deepLearningArtApp.loadDataset(pathToDataset)
    #     if retbool == True:
    #         self.Label_currentDataset.setText(datasetName + " is used as dataset...")
    #     else:
    #         self.Label_currentDataset.setText("No Dataset selected!")
    #
    #     if self.deepLearningArtApp.datasetAvailable() == True:
    #         self.GroupBox_TrainNN.setEnabled(True)
    #     else:
    #         self.GroupBox_TrainNN.setEnabled(False)
    #
    #
    #
    # def button_learningOutputPath_clicked(self):
    #     path = self.openFileNamesDialog(self.deepLearningArtApp.getLearningOutputPath())
    #     self.deepLearningArtApp.setLearningOutputPath(path)
    #     self.Label_LearningOutputPath.setText(path)
    #
    #
    #
    # def updateProgressBarTraining(self, val):
    #     self.ProgressBar_training.setValue(val)
    #
    #
    #
    # def splittingMode_changed(self):
    #
    #     if self.ComboBox_splittingMode.currentIndex() == 0:
    #         self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
    #         self.Label_SplittingParams.setText("Select splitting mode!")
    #     elif self.ComboBox_splittingMode.currentIndex() == 1:
    #         # call input dialog for editting ratios
    #         testTrainingRatio, retBool = QInputDialog.getDouble(self, "Enter Test/Training Ratio:",
    #                                                          "Ratio Test/Training Set:", 0.2, 0, 1, decimals=2)
    #         if retBool == True:
    #             validationTrainingRatio, retBool = QInputDialog.getDouble(self, "Enter Validation/Training Ratio",
    #                                                                   "Ratio Validation/Training Set: ", 0.2, 0, 1, decimals=2)
    #             if retBool == True:
    #                 self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.SIMPLE_RANDOM_SAMPLE_SPLITTING)
    #                 self.deepLearningArtApp.setTrainTestDatasetRatio(testTrainingRatio)
    #                 self.deepLearningArtApp.setTrainValidationRatio(validationTrainingRatio)
    #                 txtStr = "using Test/Train=" + str(testTrainingRatio) + " and Valid/Train=" + str(validationTrainingRatio)
    #                 self.Label_SplittingParams.setText(txtStr)
    #             else:
    #                 self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
    #                 self.ComboBox_splittingMode.setCurrentIndex(0)
    #                 self.Label_SplittingParams.setText("Select Splitting Mode!")
    #         else:
    #             self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
    #             self.ComboBox_splittingMode.setCurrentIndex(0)
    #             self.Label_SplittingParams.setText("Select Splitting Mode!")
    #     elif self.ComboBox_splittingMode.currentIndex() == 2:
    #         # cross validation splitting
    #         testTrainingRatio, retBool = QInputDialog.getDouble(self, "Enter Test/Training Ratio:",
    #                                                          "Ratio Test/Training Set:", 0.2, 0, 1, decimals=2)
    #
    #         if retBool == True:
    #             numFolds, retBool = QInputDialog.getInt(self, "Enter Number of Folds for Cross Validation",
    #                                                 "Number of Folds: ", 15, 0, 100000)
    #             if retBool == True:
    #                 self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.CROSS_VALIDATION_SPLITTING)
    #                 self.deepLearningArtApp.setTrainTestDatasetRatio(testTrainingRatio)
    #                 self.deepLearningArtApp.setNumFolds(numFolds)
    #                 self.Label_SplittingParams.setText("Test/Train Ratio: " + str(testTrainingRatio) + \
    #                                                       ", and " + str(numFolds) + " Folds")
    #             else:
    #                 self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
    #                 self.ComboBox_splittingMode.setCurrentIndex(0)
    #                 self.Label_SplittingParams.setText("Select Splitting Mode!")
    #         else:
    #             self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
    #             self.ComboBox_splittingMode.setCurrentIndex(0)
    #             self.Label_SplittingParams.setText("Select Splitting Mode!")
    #
    #     elif self.ComboBox_splittingMode.currentIndex() == 3:
    #         self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.PATIENT_CROSS_VALIDATION_SPLITTING)
    #
    #
    #
    # def check_dataAugmentation_enabled(self):
    #     if self.CheckBox_DataAugmentation.checkState() == Qt.Checked:
    #         self.CheckBox_DataAug_horizontalFlip.setEnabled(True)
    #         self.CheckBox_DataAug_verticalFlip.setEnabled(True)
    #         self.CheckBox_DataAug_Rotation.setEnabled(True)
    #         self.CheckBox_DataAug_zcaWeighting.setEnabled(True)
    #         self.CheckBox_DataAug_HeightShift.setEnabled(True)
    #         self.CheckBox_DataAug_WidthShift.setEnabled(True)
    #         self.CheckBox_DataAug_Zoom.setEnabled(True)
    #         self.RadioButton_DataAug_contrastStretching.setEnabled(True)
    #         self.RadioButton_DataAug_histogramEq.setEnabled(True)
    #         self.RadioButton_DataAug_adaptiveEq.setEnabled(True)
    #     else:
    #         self.CheckBox_DataAug_horizontalFlip.setEnabled(False)
    #         self.CheckBox_DataAug_verticalFlip.setEnabled(False)
    #         self.CheckBox_DataAug_Rotation.setEnabled(False)
    #         self.CheckBox_DataAug_zcaWeighting.setEnabled(False)
    #         self.CheckBox_DataAug_HeightShift.setEnabled(False)
    #         self.CheckBox_DataAug_WidthShift.setEnabled(False)
    #         self.CheckBox_DataAug_Zoom.setEnabled(False)
    #
    #         self.RadioButton_DataAug_contrastStretching.setEnabled(False)
    #         self.RadioButton_DataAug_contrastStretching.setAutoExclusive(False)
    #         self.RadioButton_DataAug_contrastStretching.setChecked(False)
    #         self.RadioButton_DataAug_contrastStretching.setAutoExclusive(True)
    #
    #         self.RadioButton_DataAug_histogramEq.setEnabled(False)
    #         self.RadioButton_DataAug_histogramEq.setAutoExclusive(False)
    #         self.RadioButton_DataAug_histogramEq.setChecked(False)
    #         self.RadioButton_DataAug_histogramEq.setAutoExclusive(True)
    #
    #
    #         self.RadioButton_DataAug_adaptiveEq.setEnabled(False)
    #         self.RadioButton_DataAug_adaptiveEq.setAutoExclusive(False)
    #         self.RadioButton_DataAug_adaptiveEq.setChecked(False)
    #         self.RadioButton_DataAug_adaptiveEq.setAutoExclusive(True)
##################################################################################################################################
#
#     def sliderValue(self):
#         self.chosenPatchNumber=self.horizontalSlider_3.value()
#         self.matplotlibwidget_static_2.mpl.feature_plot(self.chosenActivationName, self.chosenPatchNumber,self.activations)
#
#     @pyqtSlot()
#     def on_wyh5_clicked(self):
#         self.openfile_name = QFileDialog.getOpenFileName(self, 'Choose the file', '.', 'H5 files(*.h5)')[0]
#         self.model = h5py.File(self.openfile_name, 'r')
#
#         self.qList, self.totalPatches = self.show_activation_names()
#         self.horizontalSlider_3.setMinimum(1)
#         self.horizontalSlider_3.setMaximum(self.totalPatches)
#         self.textEdit_2.setPlainText(self.openfile_name)
#
#     @pyqtSlot()
#     def on_wy2_clicked(self):
#         # Show the structure of the model and plot the weights
#         if len(self.openfile_name) != 0:
#             # show the weights
#             self.matplotlibwidget_static_2.hide()
#             self.matplotlibwidget_static_3.hide()
#             self.matplotlibwidget_static.show()
#             self.matplotlibwidget_static.mpl.weights_plot(self.model)
#         else:
#             self.showChooseFileDialog()
#
#     @pyqtSlot()
#     def on_wy3_clicked(self):
#         # Show the layers' names of the model
#         if len(self.openfile_name)!=0:
#             self.matplotlibwidget_static.hide()
#             self.matplotlibwidget_static_3.hide()
#             # show the activations' name in the List
#             slm = QStringListModel();
#             slm.setStringList(self.qList)
#             self.listView_2.setModel(slm)
#         else:
#             self.showChooseFileDialog()
#
#     @pyqtSlot()
#     def on_wy4_clicked(self):
#         # Show the feature maps of the model
#         if len(self.openfile_name) != 0 :
#             if len(self.chosenActivationName)!=0:
#                 # choose which layer's feature maps to be plotted
#                 self.matplotlibwidget_static.hide()
#                 self.matplotlibwidget_static_3.hide()
#                 self.matplotlibwidget_static_2.show()
#                 self.matplotlibwidget_static_2.mpl.feature_plot(self.chosenActivationName, self.chosenPatchNumber,self.activations)
#             else:
#                 self.showChooseLayerDialog()
#         else:
#             self.showChooseFileDialog()
#
#     @pyqtSlot()
#     def on_wy5_clicked(self):
#         # Show the Subset Selection
#         if len(self.openfile_name) != 0:
#             # show the weights
#             self.matplotlibwidget_static.hide()
#             self.matplotlibwidget_static_2.hide()
#             self.matplotlibwidget_static_3.show()
#             self.matplotlibwidget_static_3.mpl.subset_selection_plot(self.model)
#         else:
#             self.showChooseFileDialog()
#
#     def clickList_1(self, qModelIndex):
#         self.chosenActivationName = self.qList[qModelIndex.row()]
#
#     def show_activation_names(self):
#         qList = []
#         totalPatches = 0
#         activations = self.model['activations']
#
#         for i in activations:
#             qList.append(i)
#             layerPath = 'activations' + '/' + i
#             self.activations[i] = self.model[layerPath]
#             if totalPatches < len(self.activations[i]):
#                 totalPatches =len(self.activations[i])
#
#         return qList, totalPatches
#
#     def showChooseFileDialog(self):
#         reply = QMessageBox.information(self,
#                                         "Warning",
#                                         "Please select one H5 File at first",
#                                         QMessageBox.Ok )
#
#     def showChooseLayerDialog(self):
#         reply = QMessageBox.information(self,
#                                         "Warning",
#                                         "Please select one Layer at first",
#                                         QMessageBox.Ok)
#
# class MyMplCanvas(FigureCanvas):
#
#     def __init__(self, parent=None, width=15, height=15):
#
#         plt.rcParams['font.family'] = ['SimHei']
#         plt.rcParams['axes.unicode_minus'] = False
#
#         self.fig = plt.figure(figsize=(width, height))
#         #self.openfile_name=''
#         self.model = {}
#         self.layerWeights = {}  # {layer name: weights value}
#         self.edgesInLayerName = [] #(input layer name, output layer name)
#         self.allLayerNames = []
#         self.axesDict = {}
#         self.activations = {}
#
#         FigureCanvas.__init__(self, self.fig)
#         self.setParent(parent)
#
#         FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
#         FigureCanvas.updateGeometry(self)
#
#     def weights_plot(self,model):
#
#         self.model = model
#         self.getLayersWeights()
#         edgesInLayerName = self.model['edgesInLayerName']
#         layer_by_depth = self.model['layer_by_depth']
#         maxCol = self.model['maxCol'].value + 1
#         maxRow = self.model['maxRow'].value
#
#         # plot all the layers
#         bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
#
#         for i in layer_by_depth:
#             layerPath = 'layer_by_depth' + '/' + i  # the i'th layer of the model
#             for j in self.model[layerPath]:
#                 layerPath2 = layerPath + '/' + j  # the j'th layer in layer i
#                 for ind in self.model[layerPath2]:
#                     layerPath3 = layerPath2 + '/' + ind
#                     #aaa=self.model[layerPath3].value
#                     layerName =self.model[layerPath3].value
#                     #layerName = str(self.model[layerPath3].value)[2:-1]
#                     #layerName = aaa
#                     self.allLayerNames.append(layerName)
#
#                     subplotNumber = (maxRow - 1 - int(i)) * maxCol + int(j) + 1
#                     self.ax = self.fig.add_subplot(maxRow, maxCol, subplotNumber)
#                     self.ax.text(0.5, 0.5, layerName, ha="center", va="center",
#                             bbox=bbox_props)
#                     self.ax.name = layerName
#                     self.axesDict[self.ax.name] = self.ax
#                     self.ax.set_axis_off()
#
#         edges = []
#         bbox_args = dict(boxstyle="round", fc="0.8")
#         arrow_args = dict(arrowstyle="->")
#
#         for i in edgesInLayerName:
#             inputLayer = str(i[0])[2:-1]
#             inputLayer = inputLayer.split(':')[0]
#             outputLayer = str(i[1])[2:-1]
#             outputLayer = outputLayer.split(':')[0]
#             edges.append((inputLayer, outputLayer))
#
#             self.ax_input = self.axesDict[inputLayer]
#             self.ax_output = self.axesDict[outputLayer]
#             an_o = self.ax_output.annotate('', xy=(.5, 0.9), xycoords='data',
#                                       # xytext=(.5, 1), textcoords='axes fraction',
#                                       ha="center", va="top",
#                                       bbox=bbox_args,
#                                       )
#             an_i = self.ax_input.annotate('', xy=(.5, 0.4), xycoords=an_o,
#                                      xytext=(.5, 0.2), textcoords='axes fraction',
#                                      ha="center", va="top",
#                                      bbox=bbox_args,
#                                      arrowprops=arrow_args)
#
#         self.fig.tight_layout()
#         self.draw()
#         self.fig.canvas.mpl_connect('button_press_event', self.on_click_axes)
#
#     def feature_plot(self,feature_map,ind,activations):
#
#         ind = ind-1
#         self.activations=activations
#         if activations[feature_map].ndim == 4:
#             featMap=activations[feature_map][ind]
#
#             # Compute nrows and ncols for images
#             n_mosaic = len(featMap)
#             nrows = int(np.round(np.sqrt(n_mosaic)))
#             ncols = int(nrows)
#             if (nrows ** 2) < n_mosaic:
#                 ncols += 1
#
#             self.fig.clear()
#             # self.draw()
#             # self.show()
#             self.plot_feature_mosaic(featMap, nrows, ncols)
#             self.fig.suptitle("Feature Maps of Patch #{} in Layer '{}'".format(ind+1, feature_map))
#             self.draw()
#         else:
#             pass
#
#     def subset_selection_plot(self,model):
#         self.model=model
#         subset_selection=self.getSubsetSelections()
#         nimgs = len(subset_selection)
#         nrows = int(np.round(np.sqrt(nimgs)))
#         ncols = int(nrows)
#         if (nrows ** 2) < nimgs:
#             ncols += 1
#
#         self.fig=self.plot_subset_mosaic(subset_selection, nrows, ncols, self.fig)
#
#     def on_click_axes(self,event):
#         ax = event.inaxes
#
#         if ax is None:
#             return
#
#         if event.button is 1:
#             f = plt.figure()
#
#             w = self.layerWeights[ax.name]
#             if w.ndim == 4:
#                 w = np.transpose(w, (3, 2, 0, 1))
#                 mosaic_number = w.shape[0]
#                 nrows = int(np.round(np.sqrt(mosaic_number)))
#                 ncols = int(nrows)
#
#                 if nrows ** 2 < mosaic_number:
#                     ncols += 1
#
#                 f = self.plot_weight_mosaic(w[:mosaic_number, 0], nrows, ncols, f)
#                 plt.suptitle("Weights of Layer '{}'".format(ax.name))
#                 f.show()
#             else:
#                 pass
#         else:
#             # No need to re-draw the canvas if it's not a left or right click
#             return
#         event.canvas.draw()
#
#     def getLayersWeights(self):
#         weights = self.model['weights']
#         for i in weights:
#             p = 'weights' + '/' + i
#             self.layerWeights[i] = self.model[p]
#
#     def getLayersFeatures(self):
#         model = h5py.File(self.openfile_name, 'r')
#         layersName = []
#         layersFeatures = {}
#
#         for i in model['layers']:
#             layerIndex = 'layers' + '/' + i
#
#             for n in model[layerIndex]:
#                 layerName = layerIndex + '/' + n
#                 layersName.append(n)
#
#                 featurePath = layerName + '/' + 'activation'
#                 layersFeatures[n] = model[featurePath]
#         # model.close()
#         return layersName, layersFeatures
#
#     def getSubsetSelections(self):
#
#         subset_selection=self.model['subset_selection']
#         return subset_selection
#
#     def plot_weight_mosaic(self,im, nrows, ncols, fig,**kwargs):
#
#         # Set default matplotlib parameters
#         if not 'interpolation' in kwargs.keys():
#             kwargs['interpolation'] = "none"
#
#         if not 'cmap' in kwargs.keys():
#             kwargs['cmap'] = "gray"
#
#         nimgs = len(im)
#         imshape = im[0].shape
#
#         mosaic = np.zeros(imshape)
#
#         for i in range(nimgs):
#             row = int(np.floor(i / ncols))
#             col = i % ncols
#
#             ax = fig.add_subplot(nrows, ncols,i+1)
#             ax.set_xlim(0,imshape[0]-1)
#             ax.set_ylim(0,imshape[1]-1)
#
#             mosaic = im[i]
#
#             ax.imshow(mosaic, **kwargs)
#             ax.set_axis_off()
#
#         fig.canvas.mpl_connect('button_press_event', self.on_click)
#         return fig
#
#     def plot_feature_mosaic(self,im, nrows, ncols, **kwargs):
#
#         # Set default matplotlib parameters
#         if not 'interpolation' in kwargs.keys():
#             kwargs['interpolation'] = "none"
#
#         if not 'cmap' in kwargs.keys():
#             kwargs['cmap'] = "gray"
#
#         nimgs = len(im)
#         imshape = im[0].shape
#
#         mosaic = np.zeros(imshape)
#         #fig.clear()
#
#         for i in range(nimgs):
#             row = int(np.floor(i / ncols))
#             col = i % ncols
#
#             ax = self.fig.add_subplot(nrows, ncols,i+1)
#             ax.set_xlim(0,imshape[0]-1)
#             ax.set_ylim(0,imshape[1]-1)
#
#             mosaic = im[i]
#
#             ax.imshow(mosaic, **kwargs)
#             ax.set_axis_off()
#         self.draw()
#         self.fig.canvas.mpl_connect('button_press_event', self.on_click)
#
#     def plot_subset_mosaic(self,im,nrows, ncols, fig,**kwargs):
#         if not 'interpolation' in kwargs.keys():
#             kwargs['interpolation'] = "none"
#
#         if not 'cmap' in kwargs.keys():
#             kwargs['cmap'] = "gray"
#         im = np.squeeze(im, axis=1)
#         nimgs = len(im)
#         imshape = im[0].shape
#
#         mosaic = np.zeros(imshape)
#
#         for i in range(nimgs):
#             row = int(np.floor(i / ncols))
#             col = i % ncols
#
#             ax = fig.add_subplot(nrows, ncols, i + 1)
#             ax.set_xlim(0, imshape[0] - 1)
#             ax.set_ylim(0, imshape[1] - 1)
#
#             mosaic = im[i]
#
#             ax.imshow(mosaic, **kwargs)
#             ax.set_axis_off()
#         # fig.suptitle("Subset Selection of Patch #{} in Layer '{}'".format(ind, feature_map))
#         fig.canvas.mpl_connect('button_press_event', self.on_click)
#         return fig
#
#     def on_click(self,event):
#         """Enlarge or restore the selected axis."""
#         ax = event.inaxes
#         if ax is None:
#             # Occurs when a region not in an axis is clicked...
#             return
#         if event.button is 1:
#             # On left click, zoom the selected axes
#             ax._orig_position = ax.get_position()
#             ax.set_position([0.1, 0.1, 0.85, 0.85])
#             for axis in event.canvas.figure.axes:
#                 # Hide all the other axes...
#                 if axis is not ax:
#                     axis.set_visible(False)
#         elif event.button is 3:
#             # On right click, restore the axes
#             try:
#                 ax.set_position(ax._orig_position)
#                 for axis in event.canvas.figure.axes:
#                     axis.set_visible(True)
#             except AttributeError:
#                 # If we haven't zoomed, ignore...
#                 pass
#         else:
#             # No need to re-draw the canvas if it's not a left or right click
#             return
#         event.canvas.draw()


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
                self.skipdis = False # cant skip now
                self.in_link.emit()
            else:
                self.linkon.setIcon(self.icon2)
                self.islinked = False
                self.skiplink = False
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
        self.spin = i
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
            # elif
            self.loadImage()
        else:
            if self.oldindex != 0:
                self.anewscene.clear()
                self.slicelabel.setText('')
                self.graylabel.setText('')
                self.zoomlabel.setText('')
                if self.islinked == True:
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
                self.skipdis = False # cant skip now
                self.in_link.emit()
            else:
                self.blinkon.setIcon(self.icon2)
                self.islinked = False
                self.skiplink = False
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