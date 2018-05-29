# -*- coding: utf-8 -*-
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import math
from PyQt5.QtCore import Qt, QTimer, QBasicTimer
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
import h5py
import network_visualization
from network_visualization import *


class loadImage_weights_plot_2D(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def __init__(self, matplotlibwidget_static,chosenLayerName):
        super(loadImage_weights_plot_2D, self).__init__()
        self.matplotlibwidget_static=matplotlibwidget_static
        self.chosenLayerName = chosenLayerName

    def run(self):

        self.matplotlibwidget_static.mpl.weights_plot_2D(self.chosenLayerName)
        self.trigger.emit()


class loadImage_weights_plot_3D(QtCore.QThread):
    trigger = QtCore.pyqtSignal()

    def __init__(self, matplotlibwidget_static, w,chosenWeightNumber,totalWeights,totalWeightsSlices):
        super(loadImage_weights_plot_3D, self).__init__()
        self.matplotlibwidget_static=matplotlibwidget_static
        self.w=w
        self.chosenWeightNumber=chosenWeightNumber
        self.totalWeights=totalWeights
        self.totalWeightsSlices=totalWeightsSlices

    def run(self):
        self.matplotlibwidget_static.mpl.weights_plot_3D(self.w, self.chosenWeightNumber, self.totalWeights,
                                                         self.totalWeightsSlices)
        self.trigger.emit()

class loadImage_features_plot(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def __init__(self, matplotlibwidget_static,chosenPatchNumber):
        super(loadImage_features_plot, self).__init__()
        self.matplotlibwidget_static = matplotlibwidget_static
        self.chosenPatchNumber =chosenPatchNumber

    def run(self):
        self.matplotlibwidget_static.mpl.features_plot(self.chosenPatchNumber)
        self.trigger.emit()

class loadImage_features_plot_3D(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def __init__(self, matplotlibwidget_static, chosenPatchNumber,chosenPatchSliceNumber):
        super(loadImage_features_plot_3D, self).__init__()
        self.matplotlibwidget_static = matplotlibwidget_static
        self.chosenPatchNumber =chosenPatchNumber
        self.chosenPatchSliceNumber=chosenPatchSliceNumber

    def run(self):
        self.matplotlibwidget_static.mpl.features_plot_3D(self.chosenPatchNumber, self.chosenPatchSliceNumber)
        self.trigger.emit()

class loadImage_subset_selection_plot(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def __init__(self, matplotlibwidget_static, chosenSSNumber):
        super(loadImage_subset_selection_plot, self).__init__()
        self.matplotlibwidget_static = matplotlibwidget_static
        self.chosenSSNumber=chosenSSNumber


    def run(self):
        # self.create_subset()
        self.matplotlibwidget_static.mpl.subset_selection_plot(self.chosenSSNumber)
        self.trigger.emit()



class Overlay(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        palette = QPalette(self.palette())
        palette.setColor(palette.Background, Qt.transparent)
        self.setPalette(palette)

        # self.timer = QBasicTimer()
        self.timer = QTimer()
        # self.timer.setInterval(1000)
        # self.timer.start()
        #
        # self.counter = 0

    def paintEvent(self, event):

        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), QBrush(QColor(255,255, 255, 127)))
        painter.setPen(QPen(Qt.NoPen))

        for i in range(6):
            if (self.counter / 5) % 6 == i:
                painter.setBrush(QBrush(QColor(127 + (self.counter % 5) * 32, 0, 0)))
            else:
                painter.setBrush(QBrush(QColor(0, 0, 0)))
            painter.drawEllipse(
                self.width() / 2 + 30 * math.cos(2 * math.pi * i / 6.0) - 10,
                self.height() / 2 + 30 * math.sin(2 * math.pi * i / 6.0) - 10,
                20, 20)

        painter.end()

    def showEvent(self, event):
        self.timer = self.startTimer(50)
        self.counter = 0

    def timerEvent(self, event):
        self.counter += 1
        self.update()

class MyMplCanvas(FigureCanvas):
    # wheel_scroll_W_signal = pyqtSignal(int)
    wheel_scroll_signal = pyqtSignal(int,str)
    # wheel_scroll_3D_signal = pyqtSignal(int)
    # wheel_scroll_SS_signal = pyqtSignal(int)

    def __init__(self, parent=None, width=15, height=15):

        plt.rcParams['font.family'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        self.fig = plt.figure(figsize=(width, height),dpi=500)
        #self.openfile_name=''
        self.model = {}

        self.w_count=0
        self.f_count=0
        self.s_count=0
        self.layerWeights = {}  # {layer name: weights value}
        self.edgesInLayerName = [] #(input layer name, output layer name)
        self.allLayerNames = []
        self.axesDict = {}

        self.activations = {}
        self.weights ={}
        self.totalWeights=0
        self.totalWeightsSlices =0
        self.chosenWeightNumber =0
        self.chosenWeightSliceNumber =0
        self.indW =0

        self.subset_selection = {}
        self.subset_selection2 = {}

        self.chosenLayerName=[]

        self.ind =0
        self.indFS =0
        self.nrows = 0
        self.ncols = 0
        self.totalPatches = 0
        self.totalPatchesSlices = 0

        # subset selection parameters
        self.totalSS =0
        self.ssResult={}
        self.chosenSSNumber =0
        # self.alpha=0.19
        # self.Gamma=0.0000001

        self.oncrollStatus=''


        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def loadImage(self):

        strImg = mpimg.imread('model.png')
        ax=self.fig.add_subplot(111)
        ax.imshow(strImg)
        ax.set_axis_off()


    def weights_plot_2D(self,chosenLayerName):
        self.fig.clf()
        self.chosenLayerName=chosenLayerName
        self.plot_weight_mosaic()

    def weights_plot_3D(self,w,chosenWeightNumber,totalWeights,totalWeightsSlices):
        self.weights=w
        self.chosenWeightNumber=chosenWeightNumber
        self.indW=self.chosenWeightNumber-1
        self.totalWeights=totalWeights
        self.totalWeightsSlices=totalWeightsSlices
        self.fig.clf()

        self.plot_weight_mosaic_3D(w)

    def features_plot(self,chosenPatchNumber):


        self.ind = chosenPatchNumber-1

        if self.activations.ndim == 4:
            featMap=self.activations[self.ind]

            # Compute nrows and ncols for images
            n_mosaic = len(featMap)
            self.nrows = int(np.round(np.sqrt(n_mosaic)))
            self.ncols = int(self.nrows)
            if (self.nrows ** 2) < n_mosaic:
                self.ncols += 1

            self.fig.clear()
            self.plot_feature_mosaic(featMap, self.nrows, self.ncols)
            self.fig.suptitle("Feature Maps of Patch #{} ".format(self.ind+1))
            self.draw()
        else:
            pass

    def features_plot_3D(self,chosenPatchNumber,chosenPatchSliceNumber):
        self.ind = chosenPatchNumber - 1
        self.indFS =chosenPatchSliceNumber -1

        if self.activations.ndim == 5:
            featMap = self.activations[self.ind][self.indFS]

            # Compute nrows and ncols for images
            n_mosaic = len(featMap)
            self.nrows = int(np.round(np.sqrt(n_mosaic)))
            self.ncols = int(self.nrows)
            if (self.nrows ** 2) < n_mosaic:
                self.ncols += 1

            self.fig.clear()
            self.plot_feature_mosaic_3D(featMap, self.nrows, self.ncols)
            self.fig.suptitle("#{} Feature Maps of Patch #{} ".format(self.indFS+1,self.ind + 1))
            self.draw()
        else:
            pass

    def subset_selection_plot(self, chosenSSNumber):

        self.chosenSSNumber =chosenSSNumber
        self.indSS = self.chosenSSNumber - 1
        # ss = self.subset_selection[self.indSS]
        ss = self.ssResult[self.indSS]
        ss=np.squeeze(ss,axis=0)

        self.fig.clear()
        self.plot_subset_mosaic(ss)
        self.draw()

    def plot_weight_mosaic(self,**kwargs):

        # Set default matplotlib parameters
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        #self.fig.suptitle("Weights of Layer '{}'".format(self.chosenLayerName))
        w = self.layerWeights[self.chosenLayerName]

        mosaic_number = w.shape[0]
        w = w[:mosaic_number, 0]
        nrows = int(np.round(np.sqrt(mosaic_number)))
        ncols = int(nrows)

        if nrows ** 2 < mosaic_number:
            ncols += 1

        imshape = w[0].shape

        for i in range(mosaic_number):

            ax = self.fig.add_subplot(nrows, ncols, i + 1)
            ax.set_xlim(0, imshape[0] - 1)
            ax.set_ylim(0, imshape[1] - 1)
            mosaic = w[i]
            ax.imshow(mosaic, **kwargs)
            ax.set_axis_off()

            self.fig.suptitle("Weights of Layer '{}'".format(self.chosenLayerName))
            self.draw()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def plot_weight_mosaic_3D(self,w,**kwargs):

        # Set default matplotlib parameters
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        mosaic_number = w.shape[0]
        w = w[:mosaic_number, 0] #(32,3,3,3)
        w=w[self.indW] #(3,3,3)

        nimgs = w.shape[0]
        nrows = int(np.round(np.sqrt(nimgs)))
        ncols = int(nrows)
        if (nrows ** 2) < nimgs:
            ncols += 1

        imshape = w[0].shape

        for i in range(nimgs):
            ax = self.fig.add_subplot(nrows, ncols, i + 1)
            ax.set_xlim(0, imshape[0] - 1)
            ax.set_ylim(0, imshape[1] - 1)

            mosaic = w[i]

            ax.imshow(mosaic, **kwargs)
            ax.set_axis_off()

        self.fig.suptitle("#{} Weights of the Layer".format(self.indW+1))
        self.draw()
        self.oncrollStatus ='onscrollW'
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        # self.fig.canvas.mpl_connect('scroll_event', self.onscrollW)

    def plot_feature_mosaic(self,im, nrows, ncols, **kwargs):

        # Set default matplotlib parameters
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        nimgs = len(im)
        imshape = im[0].shape

        for i in range(nimgs):

            ax = self.fig.add_subplot(nrows, ncols,i+1)
            ax.set_xlim(0,imshape[0]-1)
            ax.set_ylim(0,imshape[1]-1)

            mosaic = im[i]

            ax.imshow(mosaic, **kwargs)
            ax.set_axis_off()
        self.draw()
        self.oncrollStatus='onscroll'
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)

    def plot_feature_mosaic_3D(self,im, nrows, ncols, **kwargs):

        # Set default matplotlib parameters
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        nimgs = len(im)
        imshape = im[0].shape

        for i in range(nimgs):

            ax = self.fig.add_subplot(nrows, ncols,i+1)
            ax.set_xlim(0,imshape[0]-1)
            ax.set_ylim(0,imshape[1]-1)

            mosaic = im[i]

            ax.imshow(mosaic, **kwargs)
            ax.set_axis_off()
        self.draw()
        self.oncrollStatus = 'onscroll_3D'
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        # self.fig.canvas.mpl_connect('scroll_event', self.onscroll_3D)

    def plot_subset_mosaic(self,im,**kwargs):
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        # if len(im.shape) ==2:
        if im.ndim==2:
            imshape = im.shape

            ax = self.fig.add_subplot(111)
            ax.set_xlim(0, imshape[0] - 1)
            ax.set_ylim(0, imshape[1] - 1)
            ax.imshow(im, **kwargs)
            ax.set_axis_off()

        # elif len(im.shape) ==3:
        elif im.ndim == 3:
            im=np.transpose(im,(2,0,1))
            nimgs=im.shape[0]
            imshape = im[0].shape
            nrows = int(np.round(np.sqrt(nimgs)))
            ncols = int(nrows)
            if (nrows ** 2) < nimgs:
                ncols += 1

            for i in range(nimgs):

                ax = self.fig.add_subplot(nrows, ncols, i + 1)
                ax.set_xlim(0, imshape[0] - 1)
                ax.set_ylim(0, imshape[1] - 1)

                mosaic = im[i]

                ax.imshow(mosaic, **kwargs)
                ax.set_axis_off()
        else:
            print('the dimension of the subset selection is not right')

        self.oncrollStatus = 'onscrollSS'
        self.fig.suptitle("Subset Selection of Patch #{}".format(self.indSS+1))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        # self.fig.canvas.mpl_connect('scroll_event', self.onscrollSS)

    def on_click(self,event):
        """Enlarge or restore the selected axis."""
        ax = event.inaxes
        if ax is None:
            # Occurs when a region not in an axis is clicked...
            return
        if event.button is 1:
            # On left click, zoom the selected axes
            ax._orig_position = ax.get_position()
            ax.set_position([0.1, 0.1, 0.85, 0.85])
            for axis in event.canvas.figure.axes:
                # Hide all the other axes...
                if axis is not ax:
                    axis.set_visible(False)
        elif event.button is 3:
            # On right click, restore the axes
            try:
                ax.set_position(ax._orig_position)
                for axis in event.canvas.figure.axes:
                    axis.set_visible(True)
            except AttributeError:
                # If we haven't zoomed, ignore...
                pass
        else:
            # No need to re-draw the canvas if it's not a left or right click
            return
        event.canvas.draw()

    def onscrollW(self, event):

        if event.button == 'up':
            if self.indW == (self.totalWeights-1):
                pass
            else:
                self.indW+= 1
            # w = self.weights[self.indW]
            self.fig.clear()
            self.plot_weight_mosaic_3D(self.weights)
            self.draw()
            # self.wheel_scroll_W_signal.emit(self.indW+1)


        elif event.button == 'down':
            if self.indW -1<0:
                self.indW =0
            else:
                self.indW -= 1

            self.fig.clear()
            self.plot_weight_mosaic_3D(self.weights)
            self.draw()
            # self.wheel_scroll_W_signal.emit(self.indW+1)
        else:
            pass

    def onscroll(self, event):
        if self.oncrollStatus=='onscrollW':
            self.onscrollW(event)
            self.wheel_scroll_signal.emit(self.indW + 1,self.oncrollStatus)
        elif self.oncrollStatus=='onscroll':

            if event.button == 'up':
                if self.ind == (self.totalPatches - 1):
                    pass
                else:
                    self.ind += 1
                featMap = self.activations[self.ind]
                self.fig.clear()
                self.plot_feature_mosaic(featMap, self.nrows, self.ncols)
                self.fig.suptitle("Feature Maps of Patch #{} ".format(self.ind + 1))
                self.draw()
                self.wheel_scroll_signal.emit(self.ind + 1,self.oncrollStatus)


            elif event.button == 'down':
                if self.ind - 1 < 0:
                    self.ind = 0
                else:
                    self.ind -= 1
                featMap = self.activations[self.ind]
                self.fig.clear()
                self.plot_feature_mosaic(featMap, self.nrows, self.ncols)
                self.fig.suptitle("Feature Maps of Patch #{}".format(self.ind + 1))
                self.draw()
                self.wheel_scroll_signal.emit(self.ind + 1,self.oncrollStatus)
            else:
                pass

        elif self.oncrollStatus=='onscroll_3D':
            self.onscroll_3D(event)
            self.wheel_scroll_signal.emit(self.ind + 1,self.oncrollStatus)
        elif self.oncrollStatus=='onscrollSS':
            self.onscrollSS(event)
            self.wheel_scroll_signal.emit(self.indSS + 1,self.oncrollStatus)
        else:
            pass

    def onscroll_3D(self, event):

        if event.button == 'up':
            if self.ind == (self.totalPatches - 1):
                pass
            else:
                self.ind += 1
            featMap = self.activations[self.ind][self.indFS]
            self.fig.clear()
            self.plot_feature_mosaic_3D(featMap, self.nrows, self.ncols)
            self.fig.suptitle("#{} Feature Maps of Patch #{} ".format(self.indFS+1,self.ind + 1))
            self.draw()
            # self.wheel_scroll_3D_signal .emit(self.ind + 1)


        elif event.button == 'down':
            if self.ind - 1 < 0:
                self.ind = 0
            else:
                self.ind -= 1
            featMap = self.activations[self.ind][self.indFS]
            self.fig.clear()
            self.plot_feature_mosaic_3D(featMap, self.nrows, self.ncols)
            self.fig.suptitle("#{} Feature Maps of Patch #{} ".format(self.indFS + 1, self.ind + 1))
            self.draw()
            # self.wheel_scroll_3D_signal .emit(self.ind + 1)
        else:
            pass

    def onscrollSS(self, event):
        if event.button == 'up':
            if self.indSS == (self.totalSS-1):
                pass
            else:
                self.indSS+= 1
            # ss = self.subset_selection[self.indSS]
            ss = self.ssResult[self.indSS]
            ss = np.squeeze(ss, axis=0)
            self.fig.clear()
            self.plot_subset_mosaic(ss)
            self.draw()
            # self.wheel_scroll_signal.emit(self.ind+1)
            # self.wheel_scroll_SS_signal.emit(self.indSS+1)


        elif event.button == 'down':
            if self.indSS -1<0:
                self.indSS =0
            else:
                self.indSS -= 1

            ss = self.ssResult[self.indSS]
            ss = np.squeeze(ss, axis=0)
            # ss = self.subset_selection[self.indSS]
            self.fig.clear()
            self.plot_subset_mosaic(ss)
            # self.fig.suptitle("Feature Maps of Patch #{} in Layer '{}'".format(self.ind + 1, self.chosenLayerName))
            self.draw()
            # self.wheel_scroll_SS_signal.emit(self.indSS+1)
        else:
            pass

    def getLayersWeights(self,LayerWeights):
        self.layerWeights = LayerWeights

    def getLayersFeatures(self,activations,totalPatches):
        self.activations = activations
        self.totalPatches=totalPatches

    def getLayersFeatures_3D(self,activations, totalPatches,totalPatchesSlices):
        self.activations = activations
        self.totalPatches=totalPatches
        self.totalPatchesSlices=totalPatchesSlices

    def getSubsetSelections(self,subset_selection,totalSS):
        self.subset_selection = subset_selection
        self.totalSS=totalSS


    def getSSResult(self,ssResult):
        self.ssResult=ssResult

