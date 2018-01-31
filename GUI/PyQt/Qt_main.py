import matplotlib
matplotlib.use('Qt5Agg')

import os
import dicom
import dicom_numpy  # package name is dicom-numpy in 3
import numpy as np
import matplotlib.pyplot as plt

from tkinter import * # for raised sunken
from matplotlib.widgets import LassoSelector, RectangleSelector, EllipseSelector
from matplotlib import path
import shelve
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import scipy.ndimage
#from PyQt4 import QtCore, QtGui, uic
#qtCreatorFile = "framework1.ui" # my Qt Designer file
#Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

from PyQt5 import QtWidgets, QtGui, QtCore
from framework1 import Ui_MainWindow
from CNN_setting import*
from Data_Preprocessing import*
from Layout_Choosing import*
from activescene import Activescene
from canvas import Canvas

from DatabaseInfo import*
import yaml

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        #super().__init__(parent)
        self.setupUi(self)

        self.gridson = False
        self.voxel_ndarray = []
        self.vision = 2
        self.list1 = []
        self.list2 = []
        self.list3 = []
        self.scenelist1 = []
        self.scenelist2 = []
        self.scenelist3 = []
        self.i = -1

        self.openfile.clicked.connect(self.load_data)
        self.grids1.clicked.connect(self.setlayout1)   # old: triggered
        self.grids2.clicked.connect(self.setlayout2)
        self.resetdicom.clicked.connect(self.resetcanvas)
        self.clearimage.clicked.connect(self.clearall)
        self.exit.clicked.connect(self.close)

    def load_scan(self, path):
        if path:
            slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key=lambda x: int(x.InstanceNumber))
            try:
                slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
            except:
                slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

            for s in slices:
                s.SliceThickness = slice_thickness
            return slices

    def resample(self, image, scan, new_spacing=[1, 1, 1]):
        # Determine current pixel spacing
        spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))  # scan is list
        spacing = np.array(list(spacing))

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
        return image, new_spacing

    def load_data(self):
        if self.gridson == False:
            QtWidgets.QMessageBox.information(self, 'Warning', 'Grids needed!')
        else:
            with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
            dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])

            self.PathDicom = QtWidgets.QFileDialog.getExistingDirectory(self, "open file", dbinfo.sPathIn)
            # self.PathDicom = QtWidgets.QFileDialog.getExistingDirectory(self, "open file", "C:/Users/hansw/Videos/artefacts")

            if self.PathDicom:
                print(self.PathDicom)
                files = sorted([os.path.join(self.PathDicom, file) for file in os.listdir(self.PathDicom)], key=os.path.getctime)
                datasets = [dicom.read_file(f) \
                            for f in files]
                try:
                    self.voxel_ndarray, pixel_space = dicom_numpy.combine_slices(datasets)
                except dicom_numpy.DicomImportException:
                    raise

                self.sscan = self.load_scan(self.PathDicom)
                if self.sscan:
                    self.simage = np.stack([s.pixel_array for s in self.sscan])
                    self.svoxel, spacing = self.resample(self.simage, self.sscan, [1, 1, 1])
                    self.svoxel = np.swapaxes(self.svoxel, 0, 2)

                self.list1.append(self.voxel_ndarray)
                self.a = np.rot90(self.svoxel, axes=(2, 0))
                self.list2.append(np.swapaxes(self.a, 0, 1))
                self.list3.append(np.swapaxes(self.svoxel, 1, 2))
                self.scenelist1.append(Activescene())
                self.scenelist2.append(Activescene())
                self.scenelist3.append(Activescene())

                self.i = self.i + 1
                self.putcanvas()
            else:
                pass

    def clearall(self):
        if self.gridson == True:
            for i in reversed(range(self.maingrids.count())):  # delete old widgets
                self.maingrids.itemAt(i).widget().setParent(None)
        else:
            QtWidgets.QMessageBox.information(self, 'Warning', 'No image!')

    def setlayout1(self):
        self.gridson = True
        self.vision = 2
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.maingrids = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layoutlines = self.combo_layline.currentIndex() + 1
        self.layoutcolumns = self.combo_laycolumn.currentIndex() + 1
        self.clearall()
        for i in range(self.layoutlines):
            for j in range(self.layoutcolumns):
                self.maingrids.addWidget(Activeview(), i, j)
        if  self.voxel_ndarray == []:
            pass
        else:
            self.switchcanvas()

    def setlayout2(self):
        self.gridson = True
        self.vision = 3
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.maingrids = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layout3D = self.combo_3D.currentIndex() + 1
        self.clearall()
        for i in range(self.layout3D):
            for j in range(3):
                self.maingrids.addWidget(Activeview(), i, j)
        if  self.voxel_ndarray == []:
            pass
        else:
            self.switchcanvas()

    ''' menubar version, with Layout_chooseing.py
    def setlayout(self):
        self.layoutlines, self.layoutcolumns, ok = Layout_window.getData()
        if ok:
            for i in reversed(range(self.maingrids.count())): # delete old widgets
                self.maingrids.itemAt(i).widget().setParent(None)
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    self.maingrids.addWidget(Activeview(), i, j)
        else:       # cancel clicked
            pass
    '''
    def putcanvas(self):
        if self.voxel_ndarray != []:   ###########
            if self.vision == 2:
                self.maxiimg = self.layoutlines * self.layoutcolumns
                if self.i <=self.maxiimg-1:
                    self.canvas = Canvas(self.list1[self.i])
                    self.scenelist1[self.i].addWidget(self.canvas)
                    self.maingrids.itemAt(self.i).widget().setScene(self.scenelist1[self.i])
                else:
                    QtWidgets.QMessageBox.information(self, 'Warning', 'More grids needed!')
            else:
                self.maxiimg = self.layout3D
                if self.i <=self.maxiimg-1:
                    self.canvas1 = Canvas(self.list1[self.i])
                    self.canvas2 = Canvas(self.list2[self.i])
                    self.canvas3 = Canvas(self.list3[self.i])
                    self.scenelist1[self.i].addWidget(self.canvas1)     #additem
                    self.scenelist2[self.i].addWidget(self.canvas2)
                    self.scenelist3[self.i].addWidget(self.canvas3)
                    self.maingrids.itemAt((self.i)*3).widget().setScene(self.scenelist1[self.i])
                    self.maingrids.itemAt((self.i)*3+1).widget().setScene(self.scenelist2[self.i])
                    self.maingrids.itemAt((self.i)*3+2).widget().setScene(self.scenelist3[self.i])
                else:
                    QtWidgets.QMessageBox.information(self, 'Warning', 'More grids needed!')

    def switchcanvas(self):
        if self.vision == 2:
            self.maxiimg = self.layoutlines * self.layoutcolumns
            if self.i <=self.maxiimg-1:
                for j in range(0, self.i+1):
                    self.canvas = Canvas(self.list1[j])
                    self.scenelist1[j].addWidget(self.canvas)
                    self.maingrids.itemAt(j).widget().setScene(self.scenelist1[j])
            else:
                QtWidgets.QMessageBox.information(self, 'Warning', 'More grids needed!')
        else:
            self.maxiimg = self.layout3D
            if self.i <=self.maxiimg-1:
                for j in range(0, self.i+1):
                    self.canvas1 = Canvas(self.list1[j])
                    self.canvas2 = Canvas(self.list2[j])
                    self.canvas3 = Canvas(self.list3[j])
                    self.scenelist1[j].addWidget(self.canvas1)
                    self.scenelist2[j].addWidget(self.canvas2)
                    self.scenelist3[j].addWidget(self.canvas3)
                    self.maingrids.itemAt(j*3).widget().setScene(self.scenelist1[j])
                    self.maingrids.itemAt(j*3+1).widget().setScene(self.scenelist2[j])
                    self.maingrids.itemAt(j*3+2).widget().setScene(self.scenelist3[j])
            else:
                QtWidgets.QMessageBox.information(self, 'Warning', 'More grids needed!')

    def resetcanvas(self):
        if self.gridson == False:
            QtWidgets.QMessageBox.information(self, 'Warning', 'No image!')
        else:
            self.clearall()
            if self.vision == 2:
                for i in range(self.layoutlines):
                    for j in range(self.layoutcolumns):
                        self.maingrids.addWidget(Activeview(), i, j)
                for j in range(0, self.i + 1):
                    self.canvas = Canvas(self.list1[j])
                    self.scenelist1[j].addWidget(self.canvas)
                    self.maingrids.itemAt(j).widget().setScene(self.scenelist1[j])
            else:
                for i in range(self.layout3D):
                    for j in range(3):
                        self.maingrids.addWidget(Activeview(), i, j)
                for j in range(0, self.i + 1):
                    self.canvas1 = Canvas(self.list1[j])
                    self.canvas2 = Canvas(self.list2[j])
                    self.canvas3 = Canvas(self.list3[j])
                    self.scenelist1[j].addWidget(self.canvas1)
                    self.scenelist2[j].addWidget(self.canvas2)
                    self.scenelist3[j].addWidget(self.canvas3)
                    self.maingrids.itemAt(j * 3).widget().setScene(self.scenelist1[j])
                    self.maingrids.itemAt(j * 3 + 1).widget().setScene(self.scenelist2[j])
                    self.maingrids.itemAt(j * 3 + 2).widget().setScene(self.scenelist3[j])
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyApp()
    mainWindow.showMaximized()
    newwindow1 = CNN_window()
    newwindow2 = DataPre_window()
    mainWindow.setting_CNN.clicked.connect(newwindow1.show)
    mainWindow.Datapre.clicked.connect(newwindow2.show)

    mainWindow.show()
    sys.exit(app.exec_())