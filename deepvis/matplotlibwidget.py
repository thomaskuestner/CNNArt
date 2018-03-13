# -*- coding: utf-8 -*-
import os
import numpy as np
import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from network_visualization import on_click,plot_mosaic,get_weights_mosaic,plot_weights,plot_all_weights,plot_feature_map,plot_all_feature_maps
import matplotlib.pyplot as plt

import sys
import random
import matplotlib

matplotlib.use("Qt5Agg")
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot,QStringListModel,pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSizePolicy, QWidget, QListView,QMessageBox
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbarfrom matplotlib.figure import Figure
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch
from Ui_matplotlib_pyqt_2List import Ui_MainWindow
import h5py
from keras.models import Sequential, load_model
from keras.utils.vis_utils import plot_model, model_to_dot
from network_visualization import plot_mosaic


def getLayersWeights():
    model = h5py.File('layer2ge.h5', 'r')
    layersName = []
    layersWeights = {}

    for i in model['layers']:
        layerIndex = 'layers' + '/' + i

        for n in model[layerIndex]:
            layerName = layerIndex + '/' + n
            layersName.append(n)

            weightsPath = layerName + '/' + 'weights'
            layersWeights[n] = model[weightsPath]
    #model.close()
    return layersName,layersWeights

def on_click_axes(event):
    """Enlarge or restore the selected axis."""

    ax = event.inaxes
    layersName, layersWeights = getLayersWeights()
    if ax is None:
        # Occurs when a region not in an axis is clicked...
        return
    if event.button is 1:
        #event.canvas.matplotlibwidget_static_2.setVisible(True)
        f = plt.figure()
        if ax.name=='arrow':
            return

        w = layersWeights[ax.name].value
        if w.ndim == 4:
            w = np.transpose(w, (3, 2, 0, 1))
            mosaic_number = w.shape[0]
            nrows = int(np.round(np.sqrt(mosaic_number)))
            ncols = int(nrows)

            if nrows ** 2 < mosaic_number:
                ncols += 1

            f = plot_mosaic(w[:mosaic_number, 0], nrows, ncols, f)
            plt.suptitle("Weights of Layer '{}'".format(ax.name))
            f.show()
        else:
            pass
    else:
        # No need to re-draw the canvas if it's not a left or right click
        return
    event.canvas.draw()

class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """


    def __init__(self, parent=None):
        """
        Constructor
        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.matplotlibwidget_static_2.hide()
        self.chosenActivationName=[]
        self.chosenPatchNumber=[]
        self.qList, self.allPatches = self.show_activation_names()
        self.listView.clicked.connect(self.clickList_1)
        self.listView_2.clicked.connect(self.clickList_2)

    @pyqtSlot()
    def on_pushButton_2_clicked(self):

        # show the activations' name in the List
        slm = QStringListModel();
        slm_2 = QStringListModel();
        slm.setStringList(self.qList)
        slm_2.setStringList(self.allPatches)
        self.listView.setModel(slm)
        self.listView_2.setModel(slm_2)




    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        self.matplotlibwidget_static_2.hide()
        self.matplotlibwidget_static_2.show()
        self.matplotlibwidget_static_2.mpl.feature_plot(self.chosenActivationName,int(self.chosenPatchNumber))



    def clickList_1(self, qModelIndex):
        self.chosenActivationName=self.qList[qModelIndex.row()]


    def clickList_2(self, qModelIndex):
        self.chosenPatchNumber =self.allPatches[qModelIndex.row()]



    def show_activation_names(self):
        qList=[]
        model = h5py.File('layer2ge.h5', 'r')
        layersName = []
        layersFeatures = {}
        totalPatches=0
        allPatches=[]

        for i in model['layers']:
            layerIndex = 'layers' + '/' + i

            for n in model[layerIndex]:
                qList.append(n)

                if int(i)==0:
                    layerName = layerIndex + '/' + n
                    layersName.append(n)

                    featurePath = layerName + '/' + 'activation'
                    layersFeatures[n] = model[featurePath]
                    totalPatches=layersFeatures[n].shape[0]

        for i in range(totalPatches):
            allPatches.append(str(i))
        return qList,allPatches

    






class MyMplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=15, height=15):
       
        plt.rcParams['font.family'] = ['SimHei']  
        plt.rcParams['axes.unicode_minus'] = False  

        self.fig = plt.figure(figsize=(width, height)) 

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

       
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

   



    def feature_plot(self,feature_map,ind):
        self.fig.clf()
        layersName, activations = self.getLayersFeatures()

        if activations[feature_map].ndim == 4:

            #for ind,featMap in enumerate(activations[feature_map]):

            featMap=activations[feature_map][ind]
            # Compute nrows and ncols for images
            n_mosaic = len(featMap)
            nrows = int(np.round(np.sqrt(n_mosaic)))
            ncols = int(nrows)
            if (nrows ** 2) < n_mosaic:
                ncols += 1

            plot_mosaic(featMap, nrows, ncols, self.fig)
            self.fig.suptitle("Feature Maps of Patch #{} in Layer '{}'".format(ind, feature_map))

        else:
            pass





    def getLayersFeatures(self):
        model = h5py.File('layer2ge.h5', 'r')
        layersName = []
        layersFeatures = {}

        for i in model['layers']:
            layerIndex = 'layers' + '/' + i

            for n in model[layerIndex]:
                layerName = layerIndex + '/' + n
                layersName.append(n)

                featurePath = layerName + '/' + 'activation'
                layersFeatures[n] = model[featurePath]
        # model.close()
        return layersName, layersFeatures


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.initUi()

    def initUi(self):
        self.layout = QVBoxLayout(self)
        self.mpl = MyMplCanvas(self, width=15, height=15)
        self.layout.addWidget(self.mpl)






if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
