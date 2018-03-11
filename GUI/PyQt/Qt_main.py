import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import dicom
import dicom_numpy  # package name is dicom-numpy in 3
from matplotlib import pyplot as plt
from matplotlib.widgets import LassoSelector, RectangleSelector, EllipseSelector
from matplotlib import path
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

from framework1 import Ui_MainWindow
from Patches_window import*
from activeview import Activeview
from activescene import Activescene
from canvas import Canvas
from Unpatch_eleven import*
from Unpatch_two import*

from DatabaseInfo import*
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import json

from loadingIcon import Overlay
import scipy.io as sio
from Grey_window import*

# import h5py
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

        self.gridson = False
        self.voxel_ndarray = []
        self.vision = 2
        self.i = -1

        global list1, pathlist, list2, list3, ccnlist, empty1, cmap, cmap1, cmap2, cmap3, hmap1, hmap2, vtr1, vtr2, \
            vtr3, problist, hatchlist, correslist, cmaplist, hmaplist, vtrlist, cnrlist
        pathlist = []
        list1 = []
        list2 = []
        list3 = []
        ccnlist = []
        empty1 = []
        cmap1 = mpl.colors.ListedColormap(['blue', 'red'])
        cmap2 = mpl.colors.ListedColormap(['blue', 'red', 'green'])
        cmap3 = mpl.colors.ListedColormap(['blue', 'purple', 'cyan', 'yellow', 'green'])
        hmap1 = [None, '//', '\\', 'XX']
        hmap2 = [None, '//', '\\', 'XX']
        vtr1 = 0.3
        vtr2 = 0.3
        vtr3 = 0.3

        problist = []
        hatchlist = []
        correslist = []
        cmaplist = []
        hmaplist = []
        vtrlist = []
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

        # self.newfig.canvas.mpl_connect('scroll_event', self.newonscroll)
        # self.newfig.canvas.mpl_connect('button_press_event', self.mouse_clicked)
        # self.newfig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        # self.newfig.canvas.mpl_connect('button_release_event', self.mouse_release)
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
            self.linebox.addItem("5")
            self.linebox.addItem("6")
            self.linebox.addItem("7")
            self.linebox.addItem("8")
            self.linebox.addItem("9")
        else:
            self.vision = 2
            self.visionlabel.setText('2D')
            self.linebox.removeItem(8)
            self.linebox.removeItem(7)
            self.linebox.removeItem(6)
            self.linebox.removeItem(5)
            self.linebox.removeItem(4) # reversed sequence
            self.columnlabel.setDisabled(False)
            self.columnbox.setDisabled(False)

    def clearall(self):
        for i in reversed(range(self.maingrids.count())):
            self.maingrids.itemAt(i).clearWidgets()
            self.maingrids.removeItem(self.maingrids.itemAt(i))

    def setlayout(self):
        self.gridson = True
        if self.vision == 2:
            self.layoutlines = self.linebox.currentIndex() + 1
            self.layoutcolumns = self.columnbox.currentIndex() + 1
            self.clearall()
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    blocklayout = Viewgroup()
                    for dpath in pathlist:
                        blocklayout.addPathd(dpath)
                    self.maingrids.addLayout(blocklayout, i, j)
        else:
            self.layout3D = self.linebox.currentIndex() + 1
            self.clearall()
            for i in range(self.layout3D):
                blockline = Viewline()
                for dpath in pathlist:
                    blockline.addPathim(dpath)
                for cpath in ccnlist:
                    blockline.addPathre(cpath)
                self.maingrids.addLayout(blockline, i, 0)

    def loadMR(self):
        if self.gridson == False:
            QtWidgets.QMessageBox.information(self, 'Warning', 'Grids needed!')
        else:
            with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
            dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])

            self.PathDicom = QtWidgets.QFileDialog.getExistingDirectory(self, "open file", dbinfo.sPathIn)
            # self.PathDicom = QtWidgets.QFileDialog.getExistingDirectory(self, "open file", "C:/Users/hansw/Videos/artefacts")
            if self.PathDicom:
                self.i = self.i + 1
                self.openfile.setDisabled(True)
                self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                self.overlay.setGeometry(QtCore.QRect(830, 390, 171, 141))
                self.overlay.show()
                from loadf import loadImage
                self.newMR = loadImage(self.PathDicom)
                self.newMR.trigger.connect(self.loadEnd)
                self.newMR.start()
            else:
                pass

    def loadEnd(self):
        self.overlay.killTimer(self.overlay.timer)
        self.overlay.hide()
        # self.openfile.setText("load image")
        self.openfile.setDisabled(False)

        pathlist.append(self.PathDicom)
        list1.append(self.newMR.voxel_ndarray)

        bufferi = np.rot90(self.newMR.svoxel, axes=(2, 0))
        list2.append(np.swapaxes(bufferi, 0, 1))
        list3.append(np.swapaxes(self.newMR.svoxel, 1, 2))

        if self.vision ==2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    self.maingrids.itemAtPosition(i, j).addPathd(self.PathDicom)
        else:
            for i in range(self.layout3D):
                self.maingrids.itemAtPosition(i, 0).addPathim(self.PathDicom)

    def unpatching2(self, result, orig):
        PatchSize = np.array((40.0, 40.0))
        PatchOverlay = 0.5
        imglay = fUnpatch2D(result, PatchSize, PatchOverlay, orig.shape, 0) # 0 for reference
        return imglay

    # def unpatching8(self, result, orig):

    def unpatching11(self, result, orig):
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

        Type = UnpatchType(IndexType, domain, PatchSize, PatchOverlay, orig.shape)
        Arte = UnpatchArte(IndexArte, PatchSize, PatchOverlay, orig.shape)
        return Type, Arte

    def loadpatch(self):
        resultfile = QtWidgets.QFileDialog.getOpenFileName(self, "choose the result file",
                                        "C:/Users/hansw/Desktop/Ma_code/PyQt","mat files(*.mat);;h5 files(*.h5) ")[0]
                    # last directory, C:/Users/hansw/Desktop/Ma_code/PyQt   , None, QtWidgets.QFileDialog.DontUseNativeDialog
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
                    problist.append(IType)
                    hatchlist.append(IArte)
                    cmaplist.append(cmap3)
                    hmaplist.append(hmap2)
                    vtrlist.append(vtr3)
                    cnrlist.append(11)
                else:
                    cnum = np.array(conten['prob_test'])
                    if cnum.shape[1] == 2:
                        IType = self.unpatching2(conten['prob_test'])
                        problist.append(IType)
                        hatchlist.append(empty1)
                        cmaplist.append(cmap1)
                        hmaplist.append(empty1)
                        vtrlist.append(vtr1)
                        cnrlist.append(2)

                    elif cnum.shape[1] == 8: #########
                        cmaplist.append(cmap2)
                        hmaplist.append(hmap1)
                        vtrlist.append(vtr2)

                nameofCfile = os.path.split(resultfile)[1]
                nameofCfile = nameofCfile + '   class:' + str(cnum.shape[1])
                ccnlist.append(nameofCfile)
                if self.vision == 3:
                    for i in range(self.layout3D):
                        self.maingrids.itemAtPosition(i, 0).addPathre(nameofCfile)
            else:
                QtWidgets.QMessageBox.information(self, 'Warning', 'Please choose the right file!')
        else:
            pass

    def setColor(self):
        c1, c2, c3, h1, h2, v1, v2, v3, ok = Patches_window.getData()
        if ok:
            global cmap1, cmap2, cmap3, hmap1, hmap2, vtr1, vtr2, vtr3
            cmap1 = c1
            cmap2 = c2
            cmap3 = c3
            hmap1 = h1
            hmap2 = h2
            vtr1 = v1
            vtr2 = v2
            vtr3 = v3

##################################################
    def selectormode(self):
        if self.selectoron == False:
            self.selectoron = True
            if self.gridson == True:
                self.clearall()

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
        self.selectorPath = QtWidgets.QFileDialog.getExistingDirectory(self, "choose the image to view",
                                                        "C:/Users/hansw/Videos/artefacts/MRPhysics/newProtocol")
        if self.selectorPath:
            self.markfile = QtWidgets.QFileDialog.getOpenFileName(self, "choose the marking file",
                                                                  "C:/Users/hansw/Desktop/Ma_code/PyQt/Markings",
                                                                  "")[0]  # slv files(*.slv)
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
        # print("%s %s" % (event.button, event.step))
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

            #self.newfig.canvas.draw() #2
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

class Viewgroup(QtWidgets.QGridLayout):
    def __init__(self, parent=None):
        super(Viewgroup, self).__init__(parent)

        self.pathbox = QtWidgets.QComboBox()
        self.pathbox.addItem('closed')
        self.addWidget(self.pathbox, 0, 0, 1, 2)

        self.graylabel = QtWidgets.QLabel()
        self.zoomlabel = QtWidgets.QLabel()
        self.addWidget(self.graylabel, 1, 0, 1, 1)
        self.addWidget(self.zoomlabel, 1, 1, 1, 1)

        self.slicelabel = QtWidgets.QLabel()
        self.imageedit = QtWidgets.QPushButton()
        self.imageedit.setText('edit')
        self.addWidget(self.slicelabel, 2, 1, 1, 1)
        self.addWidget(self.imageedit, 2, 0, 1, 1)

        self.addWidget(Activeview(), 3, 0, 1, 2)
        self.itemAtPosition(3, 0).widget().setScene(Activescene())

        self.pathbox.currentIndexChanged.connect(self.loadScene)
        self.imageedit.clicked.connect(self.setGrey)
        self.oldindex = 0

    def addPathd(self, pathDicom):
        region = os.path.split(pathDicom)
        proband = os.path.split(os.path.split(region[0])[0])[1]
        region = region[1]
        self.pathbox.addItem('Proband: %s' % (proband) + '   Image: %s' % (region))

    def loadScene(self, i):
        if self.oldindex != 0:
            self.anewcanvas.update_data.disconnect()
            self.anewcanvas.gray_data.disconnect()
            self.itemAtPosition(3, 0).widget().disconnect()

        if i != 0:
            self.itemAtPosition(3, 0).widget().zoomback()
            param = {'image':list1[i-1], 'mode':0}
            self.anewcanvas = Canvas(param)
            anewscene = Activescene()
            anewscene.addWidget(self.anewcanvas)
            self.itemAtPosition(3, 0).widget().setScene(anewscene)
            self.slicelabel.setText('Slice %s' % (self.anewcanvas.ind + 1) + '/ %s' % (self.anewcanvas.slices))
            self.graylabel.setText('Grayscale Range %s' % (self.anewcanvas.graylist))
            self.zoomlabel.setText('Current Zooming  1.0')
            self.anewcanvas.update_data.connect(self.updateSlices)
            self.anewcanvas.gray_data.connect(self.updateGray)
            self.anewcanvas.new_page.connect(self.newSliceview)
            self.itemAtPosition(3, 0).widget().zooming_data.connect(self.updateZoom)
        else:
            self.itemAtPosition(3, 0).widget().setScene(Activescene())
            self.slicelabel.setText('')
            self.graylabel.setText('')
            self.zoomlabel.setText('')

        self.oldindex = i

    def newSliceview(self):
        self.graylabel.setText('Grayscale Range %s' % (self.anewcanvas.graylist))

    def updateSlices(self, elist):
        self.slicelabel.setText('Slice %s' % (elist[0] + 1) + '/ %s' % (elist[1]))

    def updateZoom(self, data):
        self.zoomlabel.setText('Current Zooming %s' % (data))

    def updateGray(self, elist):
        self.graylabel.setText('Grayscale Range %s' % (elist))

    def clearWidgets(self):
        for i in reversed(range(self.count())):
            widget = self.takeAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def setGrey(self):
        maxv, minv, ok = grey_window.getData()
        if ok:
            greylist=[]
            greylist.append(minv)
            greylist.append(maxv)
            self.anewcanvas.setGreyscale(greylist)

class Viewline(QtWidgets.QGridLayout):
    def __init__(self, parent=None):
        super(Viewline, self).__init__(parent)

        self.vmode = 1
        self.oldindex = 0
        self.im_re = QtWidgets.QLabel()
        self.im_re.setText('Image')
        self.addWidget(self.im_re, 0, 0, 1, 1)
        self.bimre = QtWidgets.QPushButton()
        self.bimre.setText('switch')
        self.addWidget(self.bimre, 0, 1, 1, 1)
        self.imagelist = QtWidgets.QComboBox()
        self.imagelist.addItem('closed')
        self.addWidget(self.imagelist, 0, 2, 1, 2)
        self.reflist = QtWidgets.QComboBox()
        self.reflist.addItem('closed')
        self.addWidget(self.reflist, 0, 4, 1, 2)
        self.reflist.setDisabled(True)

        self.grt1 = QtWidgets.QLabel()
        self.zot1 = QtWidgets.QLabel()
        self.grt2 = QtWidgets.QLabel()
        self.zot2 = QtWidgets.QLabel()
        self.grt3 = QtWidgets.QLabel()
        self.zot3 = QtWidgets.QLabel()
        self.addWidget(self.grt1, 1, 0, 1, 1)
        self.addWidget(self.zot1, 1, 1, 1, 1)
        self.addWidget(self.grt2, 1, 2, 1, 1)
        self.addWidget(self.zot2, 1, 3, 1, 1)
        self.addWidget(self.grt3, 1, 4, 1, 1)
        self.addWidget(self.zot3, 1, 5, 1, 1)

        self.ed31 = QtWidgets.QPushButton()
        self.ed31.setText('edit')
        self.ed32 = QtWidgets.QPushButton()
        self.ed32.setText('edit')
        self.ed33 = QtWidgets.QPushButton()
        self.ed33.setText('edit')
        self.slt1 = QtWidgets.QLabel()
        self.slt2 = QtWidgets.QLabel()
        self.slt3 = QtWidgets.QLabel()
        self.addWidget(self.ed31, 2, 0, 1, 1)
        self.addWidget(self.slt1, 2, 1, 1, 1)
        self.addWidget(self.ed32, 2, 2, 1, 1)
        self.addWidget(self.slt2, 2, 3, 1, 1)
        self.addWidget(self.ed33, 2, 4, 1, 1)
        self.addWidget(self.slt3, 2, 5, 1, 1)

        self.addWidget(Activeview(), 3, 0, 1, 2)
        self.addWidget(Activeview(), 3, 2, 1, 2)
        self.addWidget(Activeview(), 3, 4, 1, 2)
        self.itemAtPosition(3, 0).widget().setScene(Activescene())
        self.itemAtPosition(3, 2).widget().setScene(Activescene())
        self.itemAtPosition(3, 4).widget().setScene(Activescene())

        self.bimre.clicked.connect(self.switchMode)
        self.imagelist.currentIndexChanged.connect(self.loadScene)
        self.ed31.clicked.connect(self.setGrey1)
        self.ed32.clicked.connect(self.setGrey2)
        self.ed33.clicked.connect(self.setGrey3)
        self.reflist.currentIndexChanged.connect(self.loadScene)

    def switchMode(self):
        if self.vmode == 1:
            self.vmode = 2
            self.im_re.setText('Result')
            self.reflist.setDisabled(False)
            self.imagelist.setDisabled(True)
            self.imagelist.setCurrentIndex(0)
            self.reflist.setCurrentIndex(0)
        else:
            self.vmode = 1
            self.im_re.setText('Image')
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

    def loadScene(self, i):
        if self.oldindex != 0:
            self.newcanvas1.update_data.disconnect()
            self.newcanvas1.gray_data.disconnect()
            self.itemAtPosition(3, 0).widget().disconnect()
            self.newcanvas2.update_data.disconnect()
            self.newcanvas2.gray_data.disconnect()
            self.itemAtPosition(3, 2).widget().disconnect()
            self.newcanvas3.update_data.disconnect()
            self.newcanvas3.gray_data.disconnect()
            self.itemAtPosition(3, 4).widget().disconnect()
        if i != 0:
            if self.vmode == 1:
                param1 = {'image': list1[i - 1], 'mode': 0}
                param2 = {'image': list2[i - 1], 'mode': 0}
                param3 = {'image': list3[i - 1], 'mode': 0}
                self.newcanvas1 = Canvas(param1)
                self.newcanvas2 = Canvas(param2)
                self.newcanvas3 = Canvas(param3)
            else:
                if cnrlist[i - 1] == 11:
                    param1 = {'image': list1[correslist[i - 1]], 'mode': 1 , 'color':problist[i - 1],
                              'hatch':hatchlist[i - 1], 'cmap':cmap3, 'hmap':hmap2, 'trans':vtr3}
                    self.newcanvas1 = Canvas(param1)
                    reverse1 = np.rot90(np.swapaxes(list2[correslist[i - 1]], 1, 0), axes=(0, 2))
                    param2 = {'image': reverse1, 'mode': 2 , 'color':problist[i - 1],
                              'hatch':hatchlist[i - 1], 'cmap':cmap3, 'hmap':hmap2, 'trans':vtr3}
                    self.newcanvas2 = Canvas(param2)
                    reverse2 = np.swapaxes(list3[correslist[i - 1]], 2, 1)
                    param3 = {'image': reverse2, 'mode': 3 , 'color':problist[i - 1],
                              'hatch':hatchlist[i - 1], 'cmap':cmap3, 'hmap':hmap2, 'trans':vtr3}
                    self.newcanvas3 = Canvas(param3)
                elif cnrlist[i - 1] == 2:
                    param1 = {'image': list1[correslist[i - 1]], 'mode': 4 , 'color':problist[i - 1],
                               'cmap':cmap1, 'trans':vtr1}
                    self.newcanvas1 = Canvas(param1)
                    reverse1 = np.rot90(np.swapaxes(list2[correslist[i - 1]], 1, 0), axes=(0, 2))
                    param2 = {'image': reverse1, 'mode': 5 , 'color':problist[i - 1],
                             'cmap':cmap1, 'trans':vtr1}
                    self.newcanvas2 = Canvas(param2)
                    reverse2 = np.swapaxes(list3[correslist[i - 1]], 2, 1)
                    param3 = {'image': reverse2, 'mode': 6 , 'color':problist[i - 1],
                               'cmap':cmap1, 'trans':vtr1}
                    self.newcanvas3 = Canvas(param3)

                # elif:

            self.itemAtPosition(3, 0).widget().zoomback()
            anewscene = Activescene()
            anewscene.addWidget(self.newcanvas1)
            self.itemAtPosition(3, 0).widget().setScene(anewscene)
            self.itemAtPosition(3, 2).widget().zoomback()
            anewscene = Activescene()  # already set
            anewscene.addWidget(self.newcanvas2)
            self.itemAtPosition(3, 2).widget().setScene(anewscene)
            self.itemAtPosition(3, 4).widget().zoomback()
            anewscene = Activescene()
            anewscene.addWidget(self.newcanvas3)
            self.itemAtPosition(3, 4).widget().setScene(anewscene)

            self.slt1.setText('Slice %s' % (self.newcanvas1.ind + 1) + '/ %s' % (self.newcanvas1.slices))
            self.grt1.setText('Grayscale Range %s' % (self.newcanvas1.graylist))
            self.zot1.setText('Current Zooming  1.0')
            self.newcanvas1.update_data.connect(self.updateSlices1)
            self.newcanvas1.gray_data.connect(self.updateGray1)
            self.newcanvas1.new_page.connect(self.newSliceview1)
            self.itemAtPosition(3, 0).widget().zooming_data.connect(self.updateZoom1)
            self.slt2.setText('Slice %s' % (self.newcanvas2.ind + 1) + '/ %s' % (self.newcanvas2.slices))
            self.grt2.setText('Grayscale Range %s' % (self.newcanvas2.graylist))
            self.zot2.setText('Current Zooming  1.0')
            self.newcanvas2.update_data.connect(self.updateSlices2)
            self.newcanvas2.gray_data.connect(self.updateGray2)
            self.newcanvas2.new_page.connect(self.newSliceview2)
            self.itemAtPosition(3, 2).widget().zooming_data.connect(self.updateZoom2)
            self.slt3.setText('Slice %s' % (self.newcanvas3.ind + 1) + '/ %s' % (self.newcanvas3.slices))
            self.grt3.setText('Grayscale Range %s' % (self.newcanvas3.graylist))
            self.zot3.setText('Current Zooming  1.0')
            self.newcanvas3.update_data.connect(self.updateSlices3)
            self.newcanvas3.gray_data.connect(self.updateGray3)
            self.newcanvas3.new_page.connect(self.newSliceview3)
            self.itemAtPosition(3, 4).widget().zooming_data.connect(self.updateZoom3)
        else:
            self.itemAtPosition(3, 0).widget().setScene(Activescene())
            self.slt1.setText('')
            self.grt1.setText('')
            self.zot1.setText('')
            self.itemAtPosition(3, 2).widget().setScene(Activescene())
            self.slt2.setText('')
            self.grt2.setText('')
            self.zot2.setText('')
            self.itemAtPosition(3, 4).widget().setScene(Activescene())
            self.slt3.setText('')
            self.grt3.setText('')
            self.zot3.setText('')

        self.oldindex = i

    def newSliceview1(self):
        self.grt1.setText('Grayscale Range %s' % (self.newcanvas1.graylist))

    def newSliceview2(self):
        self.grt2.setText('Grayscale Range %s' % (self.newcanvas2.graylist))

    def newSliceview3(self):
        self.grt3.setText('Grayscale Range %s' % (self.newcanvas3.graylist))

    def updateSlices1(self, elist):
        self.slt1.setText('Slice %s' % (elist[0] + 1) + '/ %s' % (elist[1]))

    def updateZoom1(self, data):
        self.zot1.setText('Current Zooming %s' % (data))

    def updateGray1(self, elist):
        self.grt1.setText('Grayscale Range %s' % (elist))

    def updateSlices2(self, elist):
        self.slt2.setText('Slice %s' % (elist[0] + 1) + '/ %s' % (elist[1]))

    def updateZoom2(self, data):
        self.zot2.setText('Current Zooming %s' % (data))

    def updateGray2(self, elist):
        self.grt2.setText('Grayscale Range %s' % (elist))

    def updateSlices3(self, elist):
        self.slt3.setText('Slice %s' % (elist[0] + 1) + '/ %s' % (elist[1]))

    def updateZoom3(self, data):
        self.zot3.setText('Current Zooming %s' % (data))

    def updateGray3(self, elist):
        self.grt3.setText('Grayscale Range %s' % (elist))

    def clearWidgets(self):
        for i in reversed(range(self.count())):
            widget = self.takeAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def setGrey1(self):
        maxv, minv, ok = grey_window.getData()
        if ok:
            greylist=[]
            greylist.append(minv)
            greylist.append(maxv)
            self.newcanvas1.setGreyscale(greylist)

    def setGrey2(self):
        maxv, minv, ok = grey_window.getData()
        if ok:
            greylist=[]
            greylist.append(minv)
            greylist.append(maxv)
            self.newcanvas2.setGreyscale(greylist)

    def setGrey3(self):
        maxv, minv, ok = grey_window.getData()
        if ok:
            greylist=[]
            greylist.append(minv)
            greylist.append(maxv)
            self.newcanvas3.setGreyscale(greylist)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyApp()
    mainWindow.showMaximized()

    mainWindow.show()
    sys.exit(app.exec_())