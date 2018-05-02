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

import pickle
import numpy as np
import pandas as pd
import codecs

#pyrcc5 C:\Users\hansw\Desktop\Ma_code\PyQt_main\resrc.qrc -o C:\Users\hansw\Desktop\Ma_code\PyQt_main\resrc_rc.py

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

        self.newfig = plt.figure(50) # 3
        self.newfig.set_facecolor("black")
        self.newax = self.newfig.add_subplot(111)
        self.newax.axis('off')
        self.pltc = None
        self.newcanvas = FigureCanvas(self.newfig)  # must be defined because of selector next
        self.keylist = []  # locate the key in combobox
        self.mrinmain = None

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
                p = np.ndarray.tolist(p.vertices)  #
                # print(type(p))
                if self.ind < 9:
                    ind = '0' + str(self.ind+1) # local ind
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
        # useblit: canvas fast update

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
                # shapelist = pd.Series(shapelist).to_json(orient='values')  # to str?
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
            # print(gridsnr)
            # print(shapelist)
            # print(pathlist)
            # print(pnamelist)
            # print(cnrlist)
            # print(correslist)

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
        if type(self.mrinmain)== 'numpy.ndarray': #
            self.updateList()

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
            self.newMR.trigger.connect(self.loadEnd)
            self.newMR.start()
        else:
            pass

    def loadSelect(self): # from loadEnd
        self.mrinmain = self.newMR.voxel_ndarray
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

        self.orderROIs()

    def orderROIs(self):
        with open(self.markfile, 'r') as json_data:
            saveFile = json.load(json_data)
        sepkey = os.path.split(self.selectorPath)[1]
        self.newkeylist=[]
        if sepkey in saveFile['layer']:   #### void key?
            for key in saveFile['layer'][sepkey].keys():
                newkey = (key[0]+key[1], key[2], key[3], key[4]+key[5])
                self.newkeylist.append(newkey) # dont cover original json file

            first = sorted(self.newkeylist, key=lambda e: e.__getitem__(2))
            second = sorted(first, key=lambda e: (e.__getitem__(2), e.__getitem__(3)))
            self.third = sorted(second, key=lambda e: (e.__getitem__(2), e.__getitem__(3), e.__getitem__(0)))
            self.updateList()

    def updateList(self):
        self.keylist = []
        self.labelWidget.clear()
        for element in self.third:
            oldkey =element[0]+element[1]+element[2]+element[3]   # format?
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

        sepkey = os.path.split(self.selectorPath)  # ('C:/Users/hansw/Videos/artefacts/MRPhysics/newProtocol/01_ab/dicom_sorted', 't1_tse_tra_Kopf_Motion_0003')
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
        self.newcanvas.draw()  # not self.newcanvas.show()

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
            self.pltc.set_clim(vmin=v_min, vmax=v_max)
            self.graylist[0] = v_min.round(2)
            self.graylist[1] = v_max.round(2)
            self.gray_data.emit(self.graylist)

            self.newcanvas.draw_idle()

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
            # elif
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