# -*- coding: utf-8 -*-
import os
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use("Qt5Agg")
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import pyqtSlot,QStringListModel,pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSizePolicy, QWidget, QListView,QMessageBox,QFileDialog,QDialog,QPushButton
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from Ui_3 import Ui_MainWindow
import h5py
from activeview import *
from keras.utils.vis_utils import plot_model, model_to_dot
from keras.models import load_model
from loadf import *

import tensorflow as tf
import keras.backend as K
import network_visualization
from network_visualization import *

#from network_visualization import plot_mosaic,on_click


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
        self.inputalpha = '0.19'
        self.inputGamma = '0.0000001'

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

    def show_layer_names(self):
        qList = []
        activations = self.model['activations']

        for i in activations:
            qList.append(i)
            layerPath = 'activations' + '/' + i
            self.act[i] = self.model[layerPath]
            if self.act[i].ndim==5 and self.modelDimension=='3D':
                self.act[i]=np.transpose(self.act[i],(0,4,1,2,3))
        self.qList = qList

    def sliderValue(self):
        if self.W_F=='w':

            self.chosenWeightNumber=self.horizontalSliderPatch.value()
            self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
            self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
            self.overlay.show()

            from loadf import loadImage_weights_plot_3D
            self.wyPlot.setDisabled(True)
            self.newW3D = loadImage_weights_plot_3D(self.matplotlibwidget_static, self.w, self.chosenWeightNumber,
                                                    self.totalWeights, self.totalWeightsSlices)
            self.newW3D.trigger.connect(self.loadEnd)
            self.newW3D.start()

            # self.matplotlibwidget_static.mpl.weights_plot_3D(self.w, self.chosenWeightNumber, self.totalWeights,self.totalWeightsSlices)
        elif self.W_F=='f':

            if self.modelDimension=='2D':
                self.chosenPatchNumber=self.horizontalSliderPatch.value()
                self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
                self.overlay.show()

                from loadf import loadImage_features_plot
                self.wyPlot.setDisabled(True)
                self.newf = loadImage_features_plot(self.matplotlibwidget_static, self.chosenPatchNumber)
                self.newf.trigger.connect(self.loadEnd)
                self.newf.start()
                # self.matplotlibwidget_static.mpl.features_plot(self.chosenPatchNumber)
            elif self.modelDimension == '3D':

                self.chosenPatchNumber = self.horizontalSliderPatch.value()
                self.chosenPatchSliceNumber =self.horizontalSliderSlice.value()
                self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
                self.overlay.show()

                from loadf import loadImage_features_plot_3D
                self.wyPlot.setDisabled(True)
                self.newf = loadImage_features_plot_3D(self.matplotlibwidget_static, self.chosenPatchNumber,
                                                       self.chosenPatchSliceNumber)
                self.newf.trigger.connect(self.loadEnd)
                self.newf.start()
                # self.matplotlibwidget_static.mpl.features_plot_3D(self.chosenPatchNumber,self.chosenPatchSliceNumber)
        elif self.W_F=='s':

            self.chosenSSNumber = self.horizontalSliderPatch.value()
            self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
            self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
            self.overlay.show()

            from loadf import loadImage_subset_selection_plot
            self.wyPlot.setDisabled(True)
            self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
            self.newf.trigger.connect(self.loadEnd)
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


            self.model=h5py.File(self.openfile_name,'r')
            # self.model['modelDimension'] = str(self.model['modelDimension'].value)[3:5]
            a=self.model['modelDimension'].value
            self.modelDimension=a
            # self.modelDimension=str(a)[3:5]

            self.twoInput = self.model['twoInput'].value
            if self.twoInput:
                self.radioButton_3.show()
                self.radioButton_4.show()

            self.modelName=self.model['modelName'].value
            temp_model=load_model(self.modelName)
            plot_model(temp_model, 'model.png')
            if self.twoInput:
                self.modelInput=temp_model.input[0]
                self.modelInput2=temp_model.input[1]
            else:
                self.modelInput = temp_model.input

            self.weights =self.model['weights']

            if self.modelDimension =='3D':
                for i in self.weights:
                    self.LayerWeights[i] = self.weights[i].value
                    if self.LayerWeights[i].ndim == 5:
                        self.LayerWeights[i] = np.transpose(self.LayerWeights[i], (4,3,2,0,1))
            elif self.modelDimension =='2D':
                for i in self.weights:
                    # self.layerWeights[i] = self.weights[i].value
                    self.LayerWeights[i]=self.weights[i].value

                    if self.LayerWeights[i].ndim == 4:
                        self.LayerWeights[i] = np.transpose(self.LayerWeights[i], (3,2,0,1))
            else:
                print('the dimesnion of the weights should be 2D or 3D')


            self.show_layer_names()

            self.subset_selection =np.array(self.model['subset_selection'])
            if self.twoInput:
                self.subset_selection_2 = np.array(self.model['subset_selection_2'])

            # if self.modelDimension=='2D':
            #     self.subset_selection = np.transpose(np.array(self.subset_selection), (1, 0, 2, 3))
            # elif self.modelDimension=='3D':
            #     self.subset_selection = np.transpose(np.array(self.subset_selection), (1, 0, 2, 3, 4))
            # else:
            #     print('the subset selection data should be 2D or 3D')
            # self.subset_selection = np.squeeze(self.subset_selection, axis=1)
            self.totalSS = len(self.subset_selection)

            # show the activations' name in the List
            slm = QStringListModel();
            slm.setStringList(self.qList)
            self.listView.setModel(slm)

    @pyqtSlot()
    def on_wyShowArchitecture_clicked(self):
        # Show the structure of the model and plot the weights
        if len(self.openfile_name) != 0:
            # show the weights
            # self.matplotlibwidget_static_2.hide()
            # self.matplotlibwidget_static.hide()
            # self.matplotlibwidget_static_3.show()
            # self.scrollArea.show()
            # structure
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
                            from loadf import loadImage_weights_plot_2D
                            self.wyPlot.setDisabled(True)
                            self.newW2D = loadImage_weights_plot_2D(self.matplotlibwidget_static,self.chosenLayerName)
                            self.newW2D.trigger.connect(self.loadEnd)
                            self.newW2D.start()

                            # self.matplotlibwidget_static.mpl.weights_plot_2D(self.chosenLayerName)
                            self.matplotlibwidget_static.show()
                        elif self.LayerWeights[self.chosenLayerName].ndim==0:
                            self.showNoWeights()
                        else:
                            self.showWeightsDimensionError()

                    elif self.modelDimension == '3D':
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

                            from loadf import loadImage_weights_plot_3D
                            self.wyPlot.setDisabled(True)
                            self.newW3D = loadImage_weights_plot_3D(self.matplotlibwidget_static, self.w,self.chosenWeightNumber,self.totalWeights,self.totalWeightsSlices)
                            self.newW3D.trigger.connect(self.loadEnd)
                            self.newW3D.start()

                            # self.matplotlibwidget_static.mpl.weights_plot_3D(self.w,self.chosenWeightNumber,self.totalWeights,self.totalWeightsSlices)

                            self.matplotlibwidget_static.show()
                            self.horizontalSliderSlice.hide()
                            self.horizontalSliderPatch.show()
                            self.labelPatch.show()
                            self.labelSlice.hide()
                            self.lcdNumberSlice.hide()
                            self.lcdNumberPatch.show()
                        elif self.LayerWeights[self.chosenLayerName].ndim==0:
                            self.showNoWeights()
                        else:
                            self.showWeightsDimensionError3D()
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

                            from loadf import loadImage_features_plot
                            self.wyPlot.setDisabled(True)
                            self.newf = loadImage_features_plot(self.matplotlibwidget_static,self.chosenPatchNumber)
                            self.newf.trigger.connect(self.loadEnd)
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

                            from loadf import loadImage_features_plot_3D
                            self.wyPlot.setDisabled(True)
                            self.newf = loadImage_features_plot_3D(self.matplotlibwidget_static, self.chosenPatchNumber,self.chosenPatchSliceNumber)
                            self.newf.trigger.connect(self.loadEnd)
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

                from loadf import loadImage_subset_selection_plot
                self.wyPlot.setDisabled(True)
                self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
                self.newf.trigger.connect(self.loadEnd)
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

                    from loadf import loadImage_subset_selection_plot
                    self.wyPlot.setDisabled(True)
                    self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
                    self.newf.trigger.connect(self.loadEnd)
                    self.newf.start()

                elif self.radioButton_4.isChecked(): # the 2nd input
                    self.matplotlibwidget_static.mpl.getSubsetSelections(self.subset_selection_2, self.totalSS)
                    self.createSubset(self.modelInput2,self.subset_selection_2)
                    self.matplotlibwidget_static.mpl.getSSResult(self.ssResult)

                    self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                    self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                    self.overlay.show()

                    from loadf import loadImage_subset_selection_plot
                    self.wyPlot.setDisabled(True)
                    self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
                    self.newf.trigger.connect(self.loadEnd)
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

    def loadEnd(self):
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

sys._excepthook = sys.excepthook
def my_exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)

sys.excepthook = my_exception_hook

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
