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

from PyQt5 import QtWidgets
from framework1 import Ui_MainWindow  ##

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        #super(MyApp, self).__init__()
        super().__init__(parent)
        self.setupUi(self)

        self.tool_buttons = []
        self.activated_button = 0
        self.button1_activated = True
        self.x_clicked = None
        self.y_clicked = None
        self.mouse_second_clicked = False
        self.fig = self.canvas.figure  #
        self.ax = plt.gca()      # for seb
        self.pltc = None

        self.openfile = QtWidgets.QAction('Open', self)
        self.openfile.setShortcut('Ctrl+O')
        self.openfile.setStatusTip('Open new file')
        self.file = self.menubar.addMenu('&File')
        self.file.addAction(self.openfile)
        self.openfile.triggered.connect(self.load_data)

        self.slider.valueChanged.connect(self.null_slot)
        self.voxel_ndarray = []


        def onClick(i):
            old_button = self.activated_button
            self.activated_button = i
            if self.activated_button == 0:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(False)
                self.button1_activated = True
            if self.activated_button == 2:
                toggle_selector.ES.set_active(True)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(False)
                self.button1_activated = False
            if self.activated_button == 1:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(True)
                toggle_selector.LS.set_active(False)
                self.button1_activated = False
            if self.activated_button == 3:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(True)
                self.button1_activated = False
            if old_button == i:
                pass
            else:
                self.tool_buttons[old_button].configure(relief=RAISED)
                self.tool_buttons[self.activated_button].configure(relief=SUNKEN)
            return
        def lasso_onselect(verts):
            print (verts)
            p = path.Path(verts)
            current_mrt_layer = self.get_currentLayerNumber()
            proband = self.art_mod_label['text'][10:self.art_mod_label['text'].find('\n')]
            model = self.art_mod_label['text'][
                    self.art_mod_label['text'].find('\n') + 9:len(self.art_mod_label['text'])]
            print(proband)
            print(model)
            saveFile = shelve.open(self.Path_marking + proband + ".slv", writeback=True)
            print(p)
            patch = None
            col_str = None
            if self.chooseArtefact.get() == self.artefact_list[0]:
                col_str = "31"
                patch = patches.PathPatch(p, fill=False, edgecolor='red', lw=2)
            elif self.chooseArtefact.get() == self.artefact_list[1]:
                col_str = "32"
                patch = patches.PathPatch(p, fill=False, edgecolor='green', lw=2)
            elif self.chooseArtefact.get() == self.artefact_list[2]:
                col_str = "33"
                patch = patches.PathPatch(p, fill=False, edgecolor='blue', lw=2)
            self.ax.add_patch(patch)
            layer_name = model
            if saveFile.has_key(layer_name):
                number_str = str(self.mrt_layer_set[current_mrt_layer].get_current_Number()) + "_" + col_str + "_" + str(len(self.ax.patches) - 1)
                saveFile[layer_name].update({number_str: p})
            else:
                number_str = str(
                    self.mrt_layer_set[current_mrt_layer].get_current_Number()) + "_" + col_str + "_" + str(
                    len(self.ax.patches) - 1)
                saveFile[layer_name] = {number_str: p}

            saveFile.close()
            self.fig.canvas.draw_idle()

        def ronselect(eclick, erelease):
            'eclick and erelease are matplotlib events at press and release'
            col_str = None
            rect = None
            ell = None
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            current_mrt_layer = self.get_currentLayerNumber()
            proband = self.art_mod_label['text'][10:self.art_mod_label['text'].find('\n')]
            model = self.art_mod_label['text'][
                    self.art_mod_label['text'].find('\n') + 9:len(self.art_mod_label['text'])]
            print(proband)
            print(model)
            saveFile = shelve.open(self.Path_marking + proband +".slv", writeback=True)
            p = np.array(([x1,y1,x2,y2]))

            layer_name =  model

            if toggle_selector.RS.active and not toggle_selector.ES.active:
                if self.chooseArtefact.get() == self.artefact_list[0]:
                    col_str = "11"
                    rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=False,
                                         edgecolor="red", lw=2)
                elif self.chooseArtefact.get() == self.artefact_list[1]:
                    col_str = "12"
                    rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=False,
                                         edgecolor="green", lw=2)
                elif self.chooseArtefact.get() == self.artefact_list[2]:
                    col_str = "13"
                    rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=False,
                                         edgecolor="blue", lw=2)
                self.ax.add_patch(rect)
            elif toggle_selector.ES.active and not toggle_selector.RS.active:
                if self.chooseArtefact.get() == self.artefact_list[0]:
                    col_str = "21"
                    ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                                  width=np.abs(x1 - x2), height=np.abs(y1 - y2), edgecolor="red", fc='None', lw=2)
                elif self.chooseArtefact.get() == self.artefact_list[1]:
                    col_str = "22"
                    ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                                  width=np.abs(x1 - x2), height=np.abs(y1 - y2), edgecolor="green", fc='None', lw=2)
                elif self.chooseArtefact.get() == self.artefact_list[2]:
                    col_str = "23"
                    ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                                  width=np.abs(x1 - x2), height=np.abs(y1 - y2), edgecolor="blue", fc='None', lw=2)
                self.ax.add_patch(ell)

            if saveFile.has_key(layer_name):
                number_str = str(self.mrt_layer_set[current_mrt_layer].get_current_Number()) + "_" + col_str + "_" + str(len(self.ax.patches) - 1)
                saveFile[layer_name].update({number_str: p})
            else:
                number_str = str(
                    self.mrt_layer_set[current_mrt_layer].get_current_Number()) + "_" + col_str + "_" + str(
                    len(self.ax.patches) - 1)
                saveFile[layer_name] = {number_str: p}
            saveFile.close()
            print(' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata))
            print(' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata))
            print(' used button   : ', eclick.button)


        def toggle_selector(event):
            print(' Key pressed.')
            if self.activated_button == 2 and not toggle_selector.ES.active and (
                        toggle_selector.LS.active or toggle_selector.RS.active):
                toggle_selector.ES.set_active(True)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(False)
            if self.activated_button == 1 and not toggle_selector.RS.active and (
                        toggle_selector.LS.active or toggle_selector.ES.active):
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(True)
                toggle_selector.LS.set_active(False)
            if self.activated_button == 3 and (
                        toggle_selector.ES.active or toggle_selector.RS.active) and not toggle_selector.LS.active:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(True)

        toggle_selector.RS = RectangleSelector(self.ax, ronselect, button=[1]) #drawtype='box', useblit=False, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True

        toggle_selector.ES = EllipseSelector(self.ax, ronselect, drawtype='line', button=[1],  minspanx=5, minspany=5, spancoords='pixels',interactive=True) #drawtype='line', minspanx=5, minspany=5, spancoords='pixels', interactive=True

        toggle_selector.LS = LassoSelector(self.ax, lasso_onselect, button=[1])

        toggle_selector.ES.set_active(False)
        toggle_selector.RS.set_active(False)
        toggle_selector.LS.set_active(False)

        self.fig.canvas.mpl_connect('button_press_event', self.mouse_clicked)
        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self.mouse_release)

        self.axial.clicked.connect(self.updatea)
        self.axial.clicked.connect(self.View_A)
        self.sagittal.clicked.connect(self.updates)
        self.sagittal.clicked.connect(self.View_S)
        self.coronal.clicked.connect(self.updatec)
        self.coronal.clicked.connect(self.View_C)

    def null_slot(self):
        pass

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
        self.slider.setValue(1)
        self.PathDicom = QtWidgets.QFileDialog.getExistingDirectory(self, "open file", "C:/Users/hansw/Videos/artefacts")  # root directory
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
            self.ax1 = self.canvas.figure.add_subplot(111)

    def updatea(self):
        if self.voxel_ndarray == []:
            pass
        else:
            self.slider.valueChanged.disconnect()
            self.slices = self.voxel_ndarray.shape[2]
            self.slider.setRange(1, self.slices)
            self.slider.setValue(1)
            self.slider.valueChanged.connect(self.View_A)
            # spinbox
            #self.valueset.setRange(1, self.slices)
            #self.valueset.valueChanged.connect(self.slice_update)
            #self.valueset.setValue(1)


    def View_A(self):
        if self.voxel_ndarray == []:
            pass
        else:
            self.ax1.clear()
            self.ind = self.slider.value()
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel_ndarray[:, :, self.ind-1], 0, 1), cmap='gray', vmin=0, vmax=2094)
            self.ax1.set_ylabel('slice %s' % self.ind)

        #self.ax2 = self.canvas.figure.add_subplot(122)
        #self.ax2.clear()
        #self.im2 = self.ax2.imshow(np.swapaxes(self.voxel_ndarray[:, :, self.ind-1], 0, 1), cmap='gray', vmin=0, vmax=2094)

        self.canvas.draw()

    def updates(self):
        if self.voxel_ndarray == []:
            pass
        else:
            self.slider.valueChanged.disconnect()
            self.slices = self.svoxel.shape[0]
            self.slider.setRange(1, self.slices)
            self.slider.setValue(1)
            self.slider.valueChanged.connect(self.View_S)

    def View_S(self):
        if self.voxel_ndarray == []:
            pass
        else:
            self.ax1.clear()
            self.ind = self.slider.value()
            self.pltc = self.ax1.imshow(np.swapaxes(self.svoxel[self.ind-1, :, :], 0, 1), cmap='gray', vmin=0, vmax=2094)
            self.ax1.set_ylabel('slice %s' % self.ind)  # easy way
            self.canvas.draw()

    def updatec(self):
        if self.voxel_ndarray == []:
            pass
        else:
            self.slider.valueChanged.disconnect()
            self.slices = self.svoxel.shape[1]
            self.slider.setRange(1, self.slices)
            self.slider.setValue(1)
            self.slider.valueChanged.connect(self.View_C)

    def View_C(self):
        if self.voxel_ndarray == []:
            pass
        else:
            self.ax1.clear()
            self.ind = self.slider.value()
            self.pltc = self.ax1.imshow(np.swapaxes(self.svoxel[:, self.ind-1, :], 0, 1), cmap='gray', vmin=0, vmax=2094)
            self.ax1.set_ylabel('slice %s' % self.ind)  # easy way
            self.canvas.draw()
        
    def mouse_clicked(self, event):
        if event.button == 2 and self.button1_activated:
            self.x_clicked = event.xdata
            self.y_clicked = event.ydata
            self.mouse_second_clicked = True

    def mouse_move(self, event):
        if self.button1_activated and self.mouse_second_clicked:
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
            print(v_min, v_max)
            self.pltc.set_clim(vmin=v_min, vmax=v_max)
            self.fig.canvas.draw()  #

    def mouse_release(self, event):
        if event.button == 2:
            self.mouse_second_clicked = False

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyApp()
    mainWindow.show()
    sys.exit(app.exec_())