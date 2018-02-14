import matplotlib
matplotlib.use('Qt5Agg')

import os
import dicom
import dicom_numpy  # package name is dicom-numpy in 3
import numpy as np
from matplotlib import pyplot as plt

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
from CNN_window import*
from Data_Prewindow import*
from Layout_Choosing import*
from Patches_window import*

from activescene import Activescene
from canvas import Canvas
# from canvas2 import Canvas2
from Unpatch_eleven import*
from Unpatch_two import*

from DatabaseInfo import*
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import json



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
        self.thicklist = []
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

        self.resultpatch.clicked.connect(self.addcolor)
        self.bpatch.clicked.connect(self.loadpatch)
        self.list4 = []
        self.list5 = []
        self.list6 = []
        self.empty1 = []
        self.cmap = []

        self.selectorbox = QtWidgets.QButtonGroup(self)
        self.selectorbox.addButton(self.brectangle, 11)
        self.selectorbox.addButton(self.bellipse, 12)
        self.selectorbox.addButton(self.blasso, 13)
        self.bselectoron.clicked.connect(self.selectormode)

        self.selectoron = False
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.maingrids = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.newfig = plt.figure(dpi=50)
        # self.newax = self.newfig.add_subplot(111)
        self.newax = plt.gca()
        self.pltc = None
        self.x_clicked = None
        self.y_clicked = None
        self.mouse_second_clicked = False

        self.newcanvas = FigureCanvas(self.newfig)
        self.newcanvas.mpl_connect('scroll_event', self.newonscroll)


        def lasso_onselect(verts):
            print(verts)
            p = path.Path(verts)
            with open(self.markfile, 'r') as json_data:  # 'r'
                saveFile = json.load(json_data)
            patch = None
            col_str = None
            if self.artefactbox.currentIndex() == 0:
                col_str = "31"
                patch = patches.PathPatch(p, fill=False, edgecolor='red', lw=2)
            elif self.artefactbox.currentIndex() == 1:
                col_str = "32"
                patch = patches.PathPatch(p, fill=False, edgecolor='green', lw=2)
            elif self.artefactbox.currentIndex() == 2:
                col_str = "33"
                patch = patches.PathPatch(p, fill=False, edgecolor='blue', lw=2)
            self.newax.add_patch(patch)
            sepkey = os.path.split(self.selectorPath)
            sepkey = sepkey[1]
            layer_name = sepkey
            if layer_name in saveFile:
                number_str = str(self.ind) + "_" + col_str + "_" + str(len(self.newax.patches) - 1)
                saveFile[layer_name].update({number_str: p})
            else:
                number_str = str(self.ind) + "_" + col_str + "_" + str(len(self.newax.patches) - 1)
                saveFile[layer_name] = {number_str: p}

            saveFile.close()
            self.newcanvas.draw()

        def ronselect(eclick, erelease):
            col_str = None
            rect = None
            ell = None
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            with open(self.markfile, 'r') as json_data:  # 'r'
                saveFile = json.load(json_data)
            p = np.array(([x1, y1, x2, y2]))

            sepkey = os.path.split(self.selectorPath)
            sepkey = sepkey[1]
            layer_name = sepkey

            if self.toggle_selector.RS.active and not self.toggle_selector.ES.active:
                if self.artefactbox.currentIndex() == 0:
                    col_str = "11"
                    rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=False,
                                         edgecolor="red", lw=2)
                elif self.artefactbox.currentIndex() == 1:
                    col_str = "12"
                    rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=False,
                                         edgecolor="green", lw=2)
                elif self.artefactbox.currentIndex() == 2:
                    col_str = "13"
                    rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=False,
                                         edgecolor="blue", lw=2)
                self.newax.add_patch(rect)
            elif self.toggle_selector.ES.active and not self.toggle_selector.RS.active:
                if self.artefactbox.currentIndex() == 0:
                    col_str = "21"
                    ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                                  width=np.abs(x1 - x2), height=np.abs(y1 - y2), edgecolor="red", fc='None', lw=2)
                elif self.artefactbox.currentIndex() == 1:
                    col_str = "22"
                    ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                                  width=np.abs(x1 - x2), height=np.abs(y1 - y2), edgecolor="green", fc='None', lw=2)
                elif self.artefactbox.currentIndex() == 2:
                    col_str = "23"
                    ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                                  width=np.abs(x1 - x2), height=np.abs(y1 - y2), edgecolor="blue", fc='None', lw=2)
                self.newax.add_patch(ell)

            if layer_name in saveFile:
                number_str = str(self.ind) + "_" + col_str + "_" + str(len(self.newax.patches) - 1)
                saveFile[layer_name].update({number_str: p})
            else:
                number_str = str(self.ind) + "_" + col_str + "_" + str(len(self.newax.patches) - 1)
                saveFile[layer_name] = {number_str: p}
            saveFile.close()

        def toggle_selector(event):
            if self.selectorbox.checkedId() == 11:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(True)
                toggle_selector.LS.set_active(False)
            elif self.selectorbox.checkedId() == 12:
                toggle_selector.ES.set_active(True)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(False)
            else:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(True)

        toggle_selector.RS = RectangleSelector(self.newax, ronselect, button=[1])  # drawtype='box', useblit=False, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True

        toggle_selector.ES = EllipseSelector(self.newax, ronselect, drawtype='line', button=[1], minspanx=5,
                                                  minspany=5,
                                                  spancoords='pixels',
                                                  interactive=True)  # drawtype='line', minspanx=5, minspany=5, spancoords='pixels', interactive=True

        toggle_selector.LS = LassoSelector(self.newax, lasso_onselect, button=[1])
        # connect('key_press_event', toggle_selector)

        self.newfig.canvas.mpl_connect('button_press_event', self.mouse_clicked)
        self.newfig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.newfig.canvas.mpl_connect('button_release_event', self.mouse_release)




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

    def resample(self, image, scan, new_spacing = [1, 1, 1]):
        # Determine current pixel spacing
        spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))  # scan is list
        spacing = np.array(list(spacing))

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
        return image, new_spacing, scan[0].SliceThickness

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
                #print(self.PathDicom)
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
                    self.svoxel, spacing, self.thickness = self.resample(self.simage, self.sscan, [1, 1, 1])
                    self.svoxel = np.swapaxes(self.svoxel, 0, 2)

                self.list1.append(self.voxel_ndarray)
                self.a = np.rot90(self.svoxel, axes=(2, 0))
                self.list2.append(np.swapaxes(self.a, 0, 1))
                self.list3.append(np.swapaxes(self.svoxel, 1, 2))
                #self.thicklist.append(self.thickness)
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

    def putcanvas(self):
        if self.voxel_ndarray != []:
            if self.vision == 2:
                self.maxiimg = self.layoutlines * self.layoutcolumns
                if self.i <=self.maxiimg-1:
                    self.canvas = Canvas(self.list1[self.i], 0, self.empty1, self.empty1, self.cmap)
                    self.scenelist1[self.i].addWidget(self.canvas)
                    self.maingrids.itemAt(self.i).widget().setScene(self.scenelist1[self.i])
                else:
                    QtWidgets.QMessageBox.information(self, 'Warning', 'More grids needed!')
            else:
                self.maxiimg = self.layout3D
                if self.i <=self.maxiimg-1:
                    self.canvas1 = Canvas(self.list1[self.i], 0, self.empty1, self.empty1,self.cmap)
                    self.canvas2 = Canvas(self.list2[self.i], 0, self.empty1, self.empty1,self.cmap)
                    self.canvas3 = Canvas(self.list3[self.i], 0, self.empty1, self.empty1,self.cmap)
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
                    self.canvas = Canvas(self.list1[j], 0, self.empty1, self.empty1, self.cmap)
                    self.scenelist1[j].clear()
                    self.scenelist1[j].addWidget(self.canvas)
                    self.maingrids.itemAt(j).widget().setScene(self.scenelist1[j])
            else:
                QtWidgets.QMessageBox.information(self, 'Warning', 'More grids needed!')
        else:
            self.maxiimg = self.layout3D
            if self.i <=self.maxiimg-1:
                for j in range(0, self.i+1):
                    self.canvas1 = Canvas(self.list1[j], 0, self.empty1, self.empty1, self.cmap)
                    self.canvas2 = Canvas(self.list2[j], 0, self.empty1, self.empty1, self.cmap)
                    self.canvas3 = Canvas(self.list3[j], 0, self.empty1, self.empty1,self.cmap)
                    self.scenelist1[j].clear()
                    self.scenelist2[j].clear()
                    self.scenelist3[j].clear()
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
                    self.canvas = Canvas(self.list1[j], 0, self.empty1, self.empty1, self.cmap)
                    self.scenelist1[j].clear()
                    self.scenelist1[j].addWidget(self.canvas)
                    self.maingrids.itemAt(j).widget().setScene(self.scenelist1[j])
            else:
                for i in range(self.layout3D):
                    for j in range(3):
                        self.maingrids.addWidget(Activeview(), i, j)
                for j in range(0, self.i + 1):
                    self.canvas1 = Canvas(self.list1[j], 0, self.empty1, self.empty1, self.cmap)
                    self.canvas2 = Canvas(self.list2[j], 0, self.empty1, self.empty1, self.cmap)
                    self.canvas3 = Canvas(self.list3[j], 0, self.empty1, self.empty1, self.cmap)
                    self.scenelist1[j].clear()
                    self.scenelist2[j].clear()
                    self.scenelist3[j].clear()
                    self.scenelist1[j].addWidget(self.canvas1)
                    self.scenelist2[j].addWidget(self.canvas2)
                    self.scenelist3[j].addWidget(self.canvas3)
                    self.maingrids.itemAt(j * 3).widget().setScene(self.scenelist1[j])
                    self.maingrids.itemAt(j * 3 + 1).widget().setScene(self.scenelist2[j])
                    self.maingrids.itemAt(j * 3 + 2).widget().setScene(self.scenelist3[j])

    def unpatching2(self, result):
        PatchSize = np.array((40.0, 40.0))
        PatchOverlay = 0.5
        imglay = fUnpatch2D(result, PatchSize, PatchOverlay, self.voxel_ndarray.shape, 0) # 0 for reference
        return imglay

    def unpatching11(self, result):
        PatchSize = np.array((40.0, 40.0))
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

        Type = UnpatchType(IndexType, domain, PatchSize, PatchOverlay, self.voxel_ndarray.shape)
        Arte = UnpatchArte(IndexArte, PatchSize, PatchOverlay, self.voxel_ndarray.shape)
        return Type, Arte

    def loadpatch(self):
        resultfile = QtWidgets.QFileDialog.getOpenFileName(self, "choose the result file",
                                        "C:/Users/hansw/Desktop/Ma_code/PyQt","Mat files(*.mat)")[0]
                    # last directory, C:/Users/hansw/Desktop/Ma_code/PyQt   , None, QtWidgets.QFileDialog.DontUseNativeDialog
        self.conten = sio.loadmat(resultfile)
        return self.conten

    def addcolor(self):
        if self.vision == 2:
            QtWidgets.QMessageBox.information(self, 'Info', 'Please view the result in 3D grids')
        else:
            self.linepos, self.classnr, self.cmap, ok = Patches_window.getData()
            if ok and self.voxel_ndarray != []:
                if self.classnr == 0:
                    self.imglay = self.unpatching2(self.conten['prob_test'])

                elif self.classnr == 1:  # multi class needed
                    self.IType, self.IArte = self.unpatching11(self.conten['prob_pre'])

                    self.canvas4 = Canvas(self.list1[self.linepos], 1, self.IType, self.IArte, self.cmap)
                    self.list4.append(self.canvas4)
                    self.scenelist1[self.linepos].clear()
                    self.scenelist1[self.linepos].addWidget(self.canvas4)

                    reverse1 = np.rot90(np.swapaxes(self.list2[self.linepos], 1, 0), axes=(0, 2))
                    self.canvas5 = Canvas(reverse1, 2, self.IType, self.IArte, self.cmap)
                    self.list5.append(self.canvas5)
                    self.scenelist2[self.linepos].clear()
                    self.scenelist2[self.linepos].addWidget(self.canvas5)

                    reverse2 = np.swapaxes(self.list3[self.linepos], 2, 1)
                    self.canvas6 = Canvas(reverse2, 3, self.IType, self.IArte, self.cmap)
                    self.list6.append(self.canvas6)
                    self.scenelist3[self.linepos].clear()
                    self.scenelist3[self.linepos].addWidget(self.canvas6)
            else:  # cancel clicked
                pass

    def selectormode(self):
        # self.selectorbox = selectorbox
        if self.selectoron == False:
            self.selectoron = True
            self.maingrids.addWidget(self.newcanvas)

            self.selectorPath = QtWidgets.QFileDialog.getExistingDirectory(self, "choose the image to view",
                                                            "C:/Users/hansw/Videos/artefacts/MRPhysics/newProtocol")
            if self.selectorPath:
                files = sorted([os.path.join(self.selectorPath, file) for file in os.listdir(self.selectorPath)],
                               key=os.path.getctime)
                datasets = [dicom.read_file(f) \
                            for f in files]
                try:
                    self.imageforselector, pixel_space = dicom_numpy.combine_slices(datasets)
                except dicom_numpy.DicomImportException:
                    raise
            else:
                self.selectoron = False
                for i in reversed(range(self.maingrids.count())):
                    self.maingrids.itemAt(i).widget().setParent(None)
                return

            self.markfile = QtWidgets.QFileDialog.getOpenFileName(self, "choose the marking file",
                                                               "C:/Users/hansw/Desktop/Ma_code/PyQt/Markings",
                                                               "")[0]  #slv files(*.slv)
            if self.markfile:
                with open(self.markfile, 'r') as json_data:   # 'r'
                    self.loadFile = json.load(json_data)
            else:
                self.selectoron = False
                for i in reversed(range(self.maingrids.count())):
                    self.maingrids.itemAt(i).widget().setParent(None)
                return

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

            self.ind = 0
            self.slices = self.imageforselector.shape[2]
            self.newslicesview()
        else:
            self.selectoron = False
            for i in reversed(range(self.maingrids.count())):  # delete old widgets
                self.maingrids.itemAt(i).widget().setParent(None)

    def newonscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1)  % self.slices
        else:
            self.ind = (self.ind - 1)  % self.slices
        if self.ind >= self.slices:
            self.ind = 0
        if self.ind <= -1:
            self.ind = self.slices - 1
        self.newslicesview()

    def newslicesview(self):
        plt.cla()
        self.pltc = plt.imshow(np.swapaxes(self.imageforselector[:, :, self.ind], 0, 1), cmap='gray', vmin=0, vmax=2094)
        self.newax.set_ylabel('slice %s' % (self.ind + 1))

        number_Patch = 0
        sepkey = os.path.split(self.selectorPath)
        sepkey = sepkey[1]
        if sepkey in self.loadFile:
            layer = self.loadFile[sepkey]
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


    def mouse_clicked(self, event):
        if event.button == 2:
            self.x_clicked = event.xdata
            self.y_clicked = event.ydata
            self.mouse_second_clicked = True

        elif event.button == 3:
            with open(self.markfile, 'r') as json_data:  # 'r'
                deleteMark = json.load(json_data)
            sepkey = os.path.split(self.selectorPath)
            sepkey = sepkey[1]
            layer = deleteMark[sepkey]
            cur_no = str(self.ind)
            cur_pa = str(len(self.newax.patches) - 1)
            if cur_no + "_11_" + cur_pa in layer:
                del deleteMark[sepkey][cur_no + "_11_" + cur_pa]
            elif cur_no + "_12_" + cur_pa in layer:
                del deleteMark[sepkey][cur_no + "_12_" + cur_pa]
            elif cur_no + "_13_" + cur_pa in layer:
                del deleteMark[sepkey][cur_no + "_13_" + cur_pa]
            elif cur_no + "_21_" + cur_pa in layer:
                del deleteMark[sepkey][cur_no + "_21_" + cur_pa]
            elif cur_no + "_22_" + cur_pa in layer:
                del deleteMark[sepkey][cur_no + "_22_" + cur_pa]
            elif cur_no + "_23_" + cur_pa in layer:
                del deleteMark[sepkey][cur_no + "_23_" + cur_pa]
            elif cur_no + "_31_" + cur_pa in layer:
                del deleteMark[sepkey][cur_no + "_31_" + cur_pa]
            elif cur_no + "_32_" + cur_pa in layer:
                del deleteMark[sepkey][cur_no + "_32_" + cur_pa]
            elif cur_no + "_33_" + cur_pa in layer:
                del deleteMark[sepkey][cur_no + "_33_" + cur_pa]

            deleteMark.close()
            self.newslicesview()

    def mouse_move(self, event):
        if self.mouse_second_clicked:
            factor = 10
            __x = event.xdata - self.x_clicked
            __y = event.ydata - self.y_clicked
            print(__x)
            print(__y)
            v_min, v_max = self.pltc.get_clim()
            print(v_min, v_max)
            if __x >= 0 and __y >= 0:
                print("h")
                __vmin = np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = np.abs(__x) * factor - np.abs(__y) * factor
            elif __x < 0 and __y >= 0:
                print("h")
                __vmin = -np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor - np.abs(__y) * factor

            elif __x < 0 and __y < 0:
                print("h")
                __vmin = -np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor + np.abs(__y) * factor

            else:
                print("h")
                __vmin = np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = np.abs(__x) * factor + np.abs(__y) * factor

            v_min += __vmin
            v_max += __vmax
            self.pltc.set_clim(vmin=v_min, vmax=v_max)
            self.newfig.canvas.draw()

    def mouse_release(self, event):
        # current_mrt_layer = self.get_currentLayerNumber()
        # self.mrt_layer_set[current_mrt_layer].set_v_min(int(self.v_min_label['text']))
        # self.mrt_layer_set[current_mrt_layer].set_v_max(int(self.v_max_label['text']))
        if event.button == 2:
            self.mouse_second_clicked = False

    # #menubar version, with Layout_chooseing.py
    # def setlayout(self):
    #     self.layoutlines, self.layoutcolumns, ok = Layout_window.getData()
    #     if ok:
    #         for i in reversed(range(self.maingrids.count())): # delete old widgets
    #             self.maingrids.itemAt(i).widget().setParent(None)
    #         for i in range(self.layoutlines):
    #             for j in range(self.layoutcolumns):
    #                 self.maingrids.addWidget(Activeview(), i, j)
    #     else:       # cancel clicked
    #         pass

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyApp()
    mainWindow.showMaximized()
    newwindow1 = CNN_window()
    newwindow2 = DataPre_window()
    #newwindow3 = Patches_window()
    mainWindow.setting_CNN.clicked.connect(newwindow1.show)
    mainWindow.Datapre.clicked.connect(newwindow2.show)
    #mainWindow.resultpatch.clicked.connect(newwindow3.show)

    mainWindow.show()
    sys.exit(app.exec_())