'''
@author: Yannick Wilhelm
@email: yannick.wilhelm@gmx.de
@date: January 2018
'''

import sys
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import *
import pandas as pd
import seaborn as sn

from DLart.dlart import DeepLearningArtApp
from DLart.dlart_gui import Ui_DLArt_GUI
from utils.Label import *

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import copy

from matplotlib.figure import Figure
import scipy.io as sio
import dicom
import dicom_numpy as dicom_np
import random
from DLart.Constants_DLart import *
from config.PATH import *


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Set up the user interface from Designer
        #loadUi("dlart_gui.ui", self)
        self.ui = Ui_DLArt_GUI();
        self.ui.setupUi(self)

        # initialize DeepLearningArt Application
        self.deepLearningArtApp = DeepLearningArtApp()
        self.deepLearningArtApp.setGUIHandle(self)

        # initialize TreeView Database
        self.manageTreeView()

        # intialize TreeView Datasets
        self.manageTreeViewDatasets()

        # initiliaze patch output path
        self.ui.Label_OutputPathPatching.setText(self.deepLearningArtApp.getOutputPathForPatching())

        # initialize markings path
        self.ui.Label_MarkingsPath.setText(self.deepLearningArtApp.getMarkingsPath())

        # initialize learning output path
        self.ui.Label_LearningOutputPath.setText(self.deepLearningArtApp.getLearningOutputPath())

        #initialize patching mode
        self.ui.ComboBox_Patching.setCurrentIndex(1)

        #initialize store mode
        self.ui.ComboBox_StoreOptions.setCurrentIndex(0)

        #initialize splitting mode
        self.ui.ComboBox_splittingMode.setCurrentIndex(SIMPLE_RANDOM_SAMPLE_SPLITTING)
        self.ui.Label_SplittingParams.setText("using Test/Train="
                                              +str(self.deepLearningArtApp.getTrainTestDatasetRatio())
                                              +" and Valid/Train="+str(self.deepLearningArtApp.getTrainValidationRatio()))

        #initialize combox box for DNN selection
        self.ui.ComboBox_DNNs.addItem("Select Deep Neural Network Model...")
        self.ui.ComboBox_DNNs.addItems(DeepLearningArtApp.deepNeuralNetworks.keys())
        self.ui.ComboBox_DNNs.setCurrentIndex(1)
        self.deepLearningArtApp.setNeuralNetworkModel(self.ui.ComboBox_DNNs.currentText())

        #initialize check boxes for used classes
        self.ui.CheckBox_Artifacts.setChecked(self.deepLearningArtApp.getUsingArtifacts())
        self.ui.CheckBox_BodyRegion.setChecked(self.deepLearningArtApp.getUsingBodyRegions())
        self.ui.CheckBox_TWeighting.setChecked(self.deepLearningArtApp.getUsingTWeighting())

        #initilize training parameters
        self.ui.DoubleSpinBox_WeightDecay.setValue(self.deepLearningArtApp.getWeightDecay())
        self.ui.DoubleSpinBox_Momentum.setValue(self.deepLearningArtApp.getMomentum())
        self.ui.CheckBox_Nesterov.setChecked(self.deepLearningArtApp.getNesterovEnabled())
        self.ui.CheckBox_DataAugmentation.setChecked(self.deepLearningArtApp.getDataAugmentationEnabled())
        self.ui.CheckBox_DataAug_horizontalFlip.setChecked(self.deepLearningArtApp.getHorizontalFlip())
        self.ui.CheckBox_DataAug_verticalFlip.setChecked(self.deepLearningArtApp.getVerticalFlip())
        self.ui.CheckBox_DataAug_Rotation.setChecked(False if self.deepLearningArtApp.getRotation()==0 else True)
        self.ui.CheckBox_DataAug_zcaWeighting.setChecked(self.deepLearningArtApp.getZCA_Whitening())
        self.ui.CheckBox_DataAug_HeightShift.setChecked(False if self.deepLearningArtApp.getHeightShift()==0 else True)
        self.ui.CheckBox_DataAug_WidthShift.setChecked(False if self.deepLearningArtApp.getWidthShift()==0 else True)
        self.ui.CheckBox_DataAug_Zoom.setChecked(False if self.deepLearningArtApp.getZoom()==0 else True)
        self.check_dataAugmentation_enabled()

        self.multiclassCheckboxes = []


        ################################################################################################################
        #### Prediction Stuff
        ################################################################################################################

        # confusion matrix
        self.confusion_matrix_figure = Figure(figsize=(10, 10))
        self.canvas_confusion_matrix_figure = FigureCanvas(self.confusion_matrix_figure)
        self.toolbar_confusion_matrix_figure = NavigationToolbar(self.canvas_confusion_matrix_figure, self)
        self.ui.verticalLayout_conf_matrix.addWidget(self.canvas_confusion_matrix_figure)
        self.ui.verticalLayout_conf_matrix.addWidget(self.toolbar_confusion_matrix_figure)

        # accuracy plot
        self.accuracy_figure = Figure(figsize=(10,10))
        self.canvas_accuracy_figure = FigureCanvas(self.accuracy_figure)
        self.toolbar_accuracy_figure = NavigationToolbar(self.canvas_accuracy_figure, self)
        self.ui.verticalLayout_accuracy.addWidget(self.canvas_accuracy_figure)
        self.ui.verticalLayout_accuracy.addWidget(self.toolbar_accuracy_figure)

        # artifact map plot
        self.artifact_map_figure = Figure(figsize=(10, 10))
        self.canvas_artifact_map_figure = FigureCanvas(self.artifact_map_figure)
        self.toolbar_artifact_map_figure = NavigationToolbar(self.canvas_artifact_map_figure, self)
        self.ui.VerticalLayout_ArtifactMaps.addWidget(self.canvas_artifact_map_figure)
        self.ui.VerticalLayout_ArtifactMaps.addWidget(self.toolbar_artifact_map_figure)

        # segmentation masks plot
        self.segmentation_masks_figure = Figure(figsize=(10, 10))
        self.canvas_segmentation_masks_figure = FigureCanvas(self.segmentation_masks_figure)
        self.toolbar_segmentation_masks_figure = NavigationToolbar(self.canvas_segmentation_masks_figure, self)
        self.ui.verticalLayout_SegmentationMasks.addWidget(self.canvas_segmentation_masks_figure)
        self.ui.verticalLayout_SegmentationMasks.addWidget(self.toolbar_segmentation_masks_figure)

        # multiclass visualisation plot
        self.multiclassVisualisation_figure = Figure(figsize=(10,10))
        self.canvas_multiclassVisualisation_figure = FigureCanvas(self.multiclassVisualisation_figure)
        self.toolbar_multiclassVisualisation_figure = NavigationToolbar(self.canvas_multiclassVisualisation_figure, self)
        self.ui.verticalLayout_multiclassVisualisation.addWidget(self.canvas_multiclassVisualisation_figure)
        self.ui.verticalLayout_multiclassVisualisation.addWidget(self.toolbar_multiclassVisualisation_figure)


        # live training performance plot
        self.training_live_performance_figure = Figure(figsize=(10,10))
        self.canvas_training_live_performance_figure = FigureCanvas(self.training_live_performance_figure)
        self.ui.verticalLayout_training_performance.addWidget(self.canvas_training_live_performance_figure)
        ################################################################################################################

        ################################################################################################################
        # Signals and Slots
        ################################################################################################################

        #select database button clicked
        self.ui.Button_DB.clicked.connect(self.button_DB_clicked)
        # self.Button_DB.clicked.connect(self.button_DB_clicked)

        #output path button for patching clicked
        self.ui.Button_OutputPathPatching.clicked.connect(self.button_outputPatching_clicked)

        #TreeWidgets
        self.ui.TreeWidget_Patients.clicked.connect(self.getSelectedPatients)
        self.ui.TreeWidget_Datasets.clicked.connect(self.getSelectedDatasets)

        #Patching button
        self.ui.Button_Patching.clicked.connect(self.button_patching_clicked)

        #mask marking path button clicekd
        self.ui.Button_MarkingsPath.clicked.connect(self.button_markingsPath_clicked)

        # combo box splitting mode is changed
        self.ui.ComboBox_splittingMode.currentIndexChanged.connect(self.splittingMode_changed)

        # "use current data" button clicked
        self.ui.Button_useCurrentData.clicked.connect(self.button_useCurrentData_clicked)

        # select dataset is clicked
        self.ui.Button_selectDataset.clicked.connect(self.button_selectDataset_clicked)

        # learning output path button clicked
        self.ui.Button_LearningOutputPath.clicked.connect(self.button_learningOutputPath_clicked)

        # train button clicked
        self.ui.Button_train.clicked.connect(self.button_train_clicked)

        # combobox dnns
        self.ui.ComboBox_DNNs.currentIndexChanged.connect(self.selectedDNN_changed)

        # data augmentation enbaled changed
        self.ui.CheckBox_DataAugmentation.stateChanged.connect(self.check_dataAugmentation_enabled)

        # selection Button dataset prediction clicked
        self.ui.Button_selectDataset_prediction.clicked.connect(self.button_selectDataset_prediction_clicked)

        # select model prediction button clicked
        self.ui.Button_selectModel_prediction.clicked.connect(self.button_selectModel_prediction_clicked)

        # button predict clicked
        self.ui.Button_predict.clicked.connect(self.button_predict_clicked)

        # spinbox for selecting dicom slice index
        self.ui.SpinBox_SliceSelect.valueChanged.connect(self.spinBox_sliceSelect_changed)

        # multi-class visualisation button
        self.ui.Button_multiclassVisualisation.clicked.connect(self.button_multiclassVisualisation_clicked)

        # spinbox for selecting multi-class visualisation slice
        self.ui.spinBox_sliceSelection_multiclassVisualisation.valueChanged.connect(self.spinBox_sliceSelection_multiclassVisualisation_changed)

        # radio button multiclass visualisation
        self.ui.radioButton_probabilites.toggled.connect(self.button_multiclassVisualisation_clicked)
        self.ui.radioButton_classes.toggled.connect(self.button_multiclassVisualisation_clicked)

        ################################################################################################################


        ################################################################################################################
        ### colormaps for visualisation
        ################################################################################################################
        colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]  # green -> yellow -> red
        cmap_name = 'artifact_map_colors'
        self.artifact_colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

        # colormap for visualization of multi-class classification results
        one_colors = [(1, 0, 0), (1, 0, 0)]
        one_color_colormap = LinearSegmentedColormap.from_list('one_color', one_colors, N=100)
        one_color_colormap = one_color_colormap(np.arange(one_color_colormap.N))
        one_color_colormap[:, -1] = np.linspace(0, 0.4, 100)

        # red colormap
        red_colormap = ListedColormap(one_color_colormap)

        # blue colormap
        blue_colormap = copy.deepcopy(red_colormap)
        blue_colormap.colors[:, 0] = 0
        blue_colormap.colors[:, 2] = 1

        # green colormap
        green_colormap = copy.deepcopy(red_colormap)
        green_colormap.colors[:, 0] = 0
        green_colormap.colors[:, 1] = 1

        # yellow colormap
        yellow_colormap = copy.deepcopy(red_colormap)
        yellow_colormap.colors[:, 1] = 1

        # pink groundtruth colormap
        pinkcolors = [(1, 0, 1), (1, 0, 1)]
        pinkcolormap = LinearSegmentedColormap.from_list('ground_truth_colormap', pinkcolors, N=100)
        pinkcolormap = pinkcolormap(np.arange(pinkcolormap.N))
        pinkcolormap[:, -1] = np.linspace(0, 0.4, 100)
        self.ground_truth_colormap = ListedColormap(pinkcolormap)

        # cyan
        cyan_colormap = copy.deepcopy(red_colormap)
        cyan_colormap.colors[:,0] = 0
        cyan_colormap.colors[:,1] = 1
        cyan_colormap.colors[:,2] = 1

        # pink
        pink_colormap = copy.deepcopy(red_colormap)
        pink_colormap.colors[:, 2] = 1

        # orange
        orange_colormap = copy.deepcopy(red_colormap)
        orange_colormap.colors[:, 1] = 0.6

        # dark green
        dark_green_colormap = copy.deepcopy(red_colormap)
        dark_green_colormap.colors[:, 0] = 0
        dark_green_colormap.colors[:, 1] = 0.42

        # lila
        lila_colormap = copy.deepcopy(red_colormap)
        lila_colormap.colors[:, 0] = 0.4
        lila_colormap.colors[:, 2] = 0.4

        # strange blue
        strange_blue_colormap = copy.deepcopy(red_colormap)
        strange_blue_colormap.colors[:, 0] = 0.2
        strange_blue_colormap.colors[:, 1] = 0.4
        strange_blue_colormap.colors[:, 2] = 0.8

        # old green
        old_green_colormap = copy.deepcopy(red_colormap)
        old_green_colormap.colors[:, 0] = 0.4
        old_green_colormap.colors[:, 1] = 0.6

        #
        self.multiclass_colormaps = []
        self.multiclass_colormaps.append(green_colormap)
        self.multiclass_colormaps.append(red_colormap)
        self.multiclass_colormaps.append(blue_colormap)
        self.multiclass_colormaps.append(yellow_colormap)
        self.multiclass_colormaps.append(pink_colormap)
        self.multiclass_colormaps.append(cyan_colormap)
        self.multiclass_colormaps.append(orange_colormap)
        self.multiclass_colormaps.append(dark_green_colormap)
        self.multiclass_colormaps.append(lila_colormap)
        self.multiclass_colormaps.append(strange_blue_colormap)
        self.multiclass_colormaps.append(old_green_colormap)

        ################################################################################################################


    def button_multiclassVisualisation_clicked(self):
        unpatches_slices = self.deepLearningArtApp.getUnpatchedSlices()

        # plot artifact map for dataset class
        if unpatches_slices is not None:
            multiclass_probability_masks = unpatches_slices['multiclass_probability_masks']
            dicom_slices = unpatches_slices['dicom_slices']
            numClasses = multiclass_probability_masks.shape[2]

            index = int(self.ui.spinBox_sliceSelection_multiclassVisualisation.value())
            self.ui.spinBox_sliceSelection_multiclassVisualisation.setMaximum(int(dicom_slices.shape[-1]))

            if index >= 0 and index < dicom_slices.shape[-1]:
                if self.ui.radioButton_probabilites.isChecked():

                    classMappings = self.deepLearningArtApp.getClassMappingsForPrediction()

                    if len(self.multiclassCheckboxes) != 0:
                        for i in self.multiclassCheckboxes:
                            self.ui.verticalLayout_multiclassSelection.removeWidget(i)
                            i.deleteLater()
                            i = None

                    self.multiclassCheckboxes = []

                    if numClasses == 11:
                        for i in range(numClasses):
                            strName = ''
                            for labelKey in classMappings:
                                valPos = np.where(np.asarray(classMappings[labelKey], dtype=int)==1)
                                if valPos[0] == i:
                                    strName = str(Label.LABEL_STRINGS[labelKey])

                            cb = QCheckBox(strName)
                            c = self.multiclass_colormaps[i].colors[0, 0:3]
                            c = np.multiply(c, 255)
                            c = tuple(c)
                            cb.setStyleSheet("color: rgb" + str(c))
                            cb.stateChanged.connect(self.update_multiclass_visualisation)
                            self.ui.verticalLayout_multiclassSelection.addWidget(cb)
                            self.multiclassCheckboxes.append(cb)


                    if numClasses == 3:
                        for i in range(numClasses):
                            strName = Label.LABEL_STRINGS[i]

                            cb = QCheckBox(strName)
                            c = self.multiclass_colormaps[i].colors[0, 0:3]
                            c = np.multiply(c, 255)
                            c = tuple(c)
                            cb.setStyleSheet("color: rgb" + str(c))
                            cb.stateChanged.connect(self.update_multiclass_visualisation)
                            self.ui.verticalLayout_multiclassSelection.addWidget(cb)
                            self.multiclassCheckboxes.append(cb)


                    if numClasses == 8:
                        for i in range(numClasses):
                            strName = ''
                            for labelKey in classMappings:
                                valPos = np.where(np.asarray(classMappings[labelKey], dtype=int)==1)
                                if valPos[0] == i:
                                    strName = str(Label.LABEL_STRINGS[labelKey%100])

                            cb = QCheckBox(strName)
                            c = self.multiclass_colormaps[i].colors[0, 0:3]
                            c = np.multiply(c, 255)
                            c = tuple(c)
                            cb.setStyleSheet("color: rgb" + str(c))
                            cb.stateChanged.connect(self.update_multiclass_visualisation)
                            self.ui.verticalLayout_multiclassSelection.addWidget(cb)
                            self.multiclassCheckboxes.append(cb)

                    self.update_multiclass_visualisation()

                elif self.ui.radioButton_classes.isChecked():
                    if len(self.multiclassCheckboxes) != 0:
                        for i in self.multiclassCheckboxes:
                            self.ui.verticalLayout_multiclassSelection.removeWidget(i)
                            i.deleteLater()
                            i = None

                    self.multiclassCheckboxes = []

                    self.update_multiclass_visualisation()


    def update_multiclass_visualisation(self):
        unpatches_slices = self.deepLearningArtApp.getUnpatchedSlices()

        # plot artifact map for dataset class
        if unpatches_slices is not None:
            multiclass_probability_masks = unpatches_slices['multiclass_probability_masks']
            numClasses = multiclass_probability_masks.shape[2]
            dicom_slices = unpatches_slices['dicom_slices']
            dicom_masks = unpatches_slices['dicom_masks']

            index = int(self.ui.spinBox_sliceSelection_multiclassVisualisation.value())
            self.ui.spinBox_sliceSelection_multiclassVisualisation.setMaximum(int(dicom_slices.shape[-1]))

            if index >= 0 and index < dicom_slices.shape[-1]:
                dicom_slice = np.squeeze(dicom_slices[:, :, index])
                slice_prob_predictions = np.squeeze(multiclass_probability_masks[:, :, :, index])

                self.multiclassVisualisation_figure.clear()
                ax1 = self.multiclassVisualisation_figure.add_subplot(111)
                ax1.clear()
                ax1.imshow(dicom_slice, cmap='gray')

                if self.ui.radioButton_probabilites.isChecked():

                    for i in range(len(self.multiclassCheckboxes)):
                        if bool(self.multiclassCheckboxes[i].isChecked()):
                            prediction_slice = np.squeeze(multiclass_probability_masks[:, :, :, index])
                            prediction_slice = np.squeeze(prediction_slice[:, :, i])

                            map = ax1.imshow(prediction_slice, cmap=self.multiclass_colormaps[i], interpolation='nearest', vmin=0, vmax=1)
                            #cbaxes = self.multiclassVisualisation_figure.add_axes([0.8, 0.1, 0.1, 0.8])
                            self.multiclassVisualisation_figure.colorbar(mappable=map, ax=ax1)

                elif self.ui.radioButton_classes.isChecked():
                    IType = unpatches_slices['IType']
                    IArte = unpatches_slices['IArte']

                    if numClasses == 11:
                        cmap1 = ListedColormap(['blue', 'purple', 'cyan', 'yellow', 'green'])
                        im2 = ax1.imshow(np.squeeze(IType[:, :, index]), cmap=cmap1, alpha=.2, vmin=1, vmax=6)
                        self.multiclassVisualisation_figure.colorbar(mappable=im2, ax=ax1)

                    if numClasses == 8:
                        cmap1 = ListedColormap(['blue', 'yellow', 'green'])
                        im2 = ax1.imshow(np.squeeze(IType[:, :, index]), cmap=cmap1, alpha=.2, vmin=1, vmax=6)
                        self.multiclassVisualisation_figure.colorbar(mappable=im2, ax=ax1)

                    plt.rcParams['hatch.color'] = 'r'
                    im3 = ax1.contourf(np.squeeze(IArte[:, :, index]), hatches=[None, '//', '\\\\', 'XX'], colors='none', edges='r', levels=np.arange(5))

                    ax1.set_ylabel('slice %s' % index)


                self.multiclassVisualisation_figure.tight_layout()

                # save figures when clicking through the slices
                self.multiclassVisualisation_figure.savefig(DLART_OUT_PATH + str(index) + '.png')

                self.canvas_multiclassVisualisation_figure.draw()


    def button_predict_clicked(self):
        gpuId = self.ui.ComboBox_GPU_Prediction.currentIndex()
        self.deepLearningArtApp.setGPUPredictionId(gpuId)

        self.deepLearningArtApp.setDoUnpatching(bool(self.ui.CheckBox_Unpatching.isChecked()))

        bPredicted = self.deepLearningArtApp.performPrediction()

        if bPredicted:
            if not self.deepLearningArtApp.getUsingSegmentationMasksForPredictions():
                confusionMatrix = self.deepLearningArtApp.getConfusionMatrix()
                classificationReport = self.deepLearningArtApp.getClassificationReport()

                target_names = []

                classMappings = self.deepLearningArtApp.getClassMappingsForPrediction()

                # if len(classMappings[list(classMappings.keys())[0]]) == 3:
                #     for i in sorted(classMappings):
                #         i = i%100
                #         i = i%10
                #         if Label.LABEL_STRINGS[i] not in target_names:
                #             target_names.append(Label.LABEL_STRINGS[i])
                # elif len(classMappings[list(classMappings.keys())[0]]) == 8:
                #     for i in sorted(classMappings):
                #         i = i%100
                #         if Label.LABEL_STRINGS[i] not in target_names:
                #             target_names.append(Label.LABEL_STRINGS[i])
                # else:
                #     for i in sorted(self.deepLearningArtApp.getClassMappingsForPrediction()):
                #         target_names.append(Label.LABEL_STRINGS[i])

                acc_training = self.deepLearningArtApp.get_acc_training()
                acc_validation = self.deepLearningArtApp.get_acc_validation()
                acc_test = self.deepLearningArtApp.get_acc_test()

                self.plotTrainingResults(acc_training=acc_training, acc_validation=acc_validation)
                #self.plotConfusionMatrix(confusionMatrix, target_names)

                outputString = 'Test Accuracy: ' + str(acc_test) + '\n' + classificationReport
                self.ui.textEdit_summary.setText(outputString)

                # check for unpatching
                if self.deepLearningArtApp.getDoUnpatching():
                    self.plotArtifactMaps()

            else:

                acc_training = self.deepLearningArtApp.get_acc_training()
                acc_validation = self.deepLearningArtApp.get_acc_validation()
                acc_test = self.deepLearningArtApp.get_acc_test()

                self.plotTrainingResults(acc_training=acc_training, acc_validation=acc_validation)
                outputString = 'Test Accuracy: ' + str(acc_test)
                self.ui.textEdit_summary.setText(outputString)

                if self.deepLearningArtApp.getDoUnpatching():
                    self.plotSegmentationArtifactMaps()
                    self.plotSegmentationPredictions()



    def plotSegmentationPredictions(self):
        unpatched_slices = self.deepLearningArtApp.getUnpatchedSlices()

        if unpatched_slices is not None:
            predicted_segmentation_mask = unpatched_slices['predicted_segmentation_mask']
            dicom_slices = unpatched_slices['dicom_slices']
            dicom_masks = unpatched_slices['dicom_masks']

            index = int(self.ui.SpinBox_SliceSelect.value())
            self.ui.SpinBox_SliceSelect.setMaximum(int(dicom_slices.shape[-1]))

            if index >= 0 and index < dicom_slices.shape[-1]:
                pred_seg_mask_slice = np.squeeze(predicted_segmentation_mask[:, :, index])
                dicom_slice = np.squeeze(dicom_slices[:, :, index])
                dicom_mask_slice = np.squeeze(dicom_masks[:, :, index])

                self.segmentation_masks_figure.clear()
                ax1 = self.segmentation_masks_figure.add_subplot(121)
                ax1.clear()
                ax1.imshow(dicom_slice, cmap='gray')
                ax1.imshow(dicom_mask_slice, cmap=self.ground_truth_colormap, interpolation='nearest', alpha=1.)
                ax1.set_title('Ground Truth')

                ax2 = self.segmentation_masks_figure.add_subplot(122)
                ax2.clear()
                ax2.imshow(dicom_slice, cmap='gray')
                ax2.imshow(pred_seg_mask_slice, cmap=self.artifact_colormap, interpolation='nearest', alpha=.4)
                ax2.set_title('Predicted Segmentation Mask')

                self.segmentation_masks_figure.tight_layout()
                self.segmentation_masks_figure.savefig(DLART_OUT_PATH + str(index) + "_disc" + '.png')
                self.canvas_segmentation_masks_figure.draw()



    def plotSegmentationArtifactMaps(self):
        unpatched_slices = self.deepLearningArtApp.getUnpatchedSlices()

        if unpatched_slices is not None:
            probability_mask_background = unpatched_slices['probability_mask_background']
            probability_mask_foreground = unpatched_slices['probability_mask_foreground']
            dicom_slices = unpatched_slices['dicom_slices']
            dicom_masks = unpatched_slices['dicom_masks']

            index = int(self.ui.SpinBox_SliceSelect.value())
            self.ui.SpinBox_SliceSelect.setMaximum(int(dicom_slices.shape[-1]))

            if index >=0 and index < dicom_slices.shape[-1]:
                dicom_slice = np.squeeze(dicom_slices[:, :, index])
                prob_mask_back_slice = np.squeeze(probability_mask_background[:,:,index])
                prob_mask_fore_slice = np.squeeze(probability_mask_foreground[:,:,index])
                dicom_mask_slice = np.squeeze(dicom_masks[:,:,index])

                # plot artifact map
                self.artifact_map_figure.clear()
                ax1 = self.artifact_map_figure.add_subplot(131)
                ax1.clear()
                ax1.imshow(dicom_slice, cmap='gray')
                ax1.imshow(dicom_mask_slice, cmap = self.ground_truth_colormap, interpolation = 'nearest', alpha = 1.)
                ax1.set_title('Ground Truth')

                ax2 = self.artifact_map_figure.add_subplot(132)
                ax2.clear()
                ax2.imshow(dicom_slice, cmap='gray')
                ax2.imshow(prob_mask_fore_slice, cmap=self.artifact_colormap, interpolation='nearest', alpha=.4, vmin=0, vmax=1)
                ax2.set_title('Predicted Foreground')

                ax3 = self.artifact_map_figure.add_subplot(133)
                ax3.clear()
                ax3.imshow(dicom_slice, cmap='gray')
                map = ax3.imshow(prob_mask_back_slice, cmap=self.artifact_colormap, interpolation='nearest', alpha=.4, vmin=0, vmax=1)
                ax3.set_title('Predicted Background')

                self.artifact_map_figure.colorbar(mappable=map, ax=ax3)
                self.artifact_map_figure.tight_layout()

                self.artifact_map_figure.savefig(DLART_OUT_PATH + str(index) + '_cont' + '.png')

                self.canvas_artifact_map_figure.draw()


    def plotArtifactMaps(self):
        unpatches_slices = self.deepLearningArtApp.getUnpatchedSlices()

        # plot artifact map for dataset class
        if unpatches_slices is not None:
            multiclass_probability_masks = unpatches_slices['multiclass_probability_masks']
            dicom_slices = unpatches_slices['dicom_slices']
            dicom_masks = unpatches_slices['dicom_masks']

            index = int(self.ui.SpinBox_SliceSelect.value())
            self.ui.SpinBox_SliceSelect.setMaximum(int(dicom_slices.shape[-1]))

            if index >= 0 and index < dicom_slices.shape[-1]:

                slice = np.squeeze(dicom_slices[:, :, index])
                #probability_mask = multiclass_probability_masks[:, :, unpatches_slices['index_class'], index]
                probability_mask = multiclass_probability_masks[:, :, index]
                prob_mask = np.squeeze(probability_mask)
                label_mask = np.squeeze(dicom_masks[:, :, index])

                self.artifact_map_figure.clear()

                ax1 = self.artifact_map_figure.add_subplot(121)
                ax1.clear()
                ax1.imshow(slice, cmap='gray')
                ax1.imshow(label_mask, cmap = self.ground_truth_colormap, interpolation = 'nearest')
                ax1.set_title('Ground Truth')

                ax2 = self.artifact_map_figure.add_subplot(122)
                ax2.clear()
                ax2.imshow(slice, cmap='gray')
                map = ax2.imshow(prob_mask, cmap=self.artifact_colormap, interpolation='nearest', alpha=.4, vmin=0, vmax=1)
                ax2.set_title('Artifact Map')

                self.artifact_map_figure.colorbar(mappable=map, ax=ax2)
                self.artifact_map_figure.tight_layout()
                self.artifact_map_figure.savefig(DLART_OUT_PATH + str(index) + '_discrete' + '.png')
                self.canvas_artifact_map_figure.draw()



    def plotTrainingResults(self, acc_training, acc_validation):
        ax = self.accuracy_figure.add_subplot(111)
        ax.clear()
        acc_training = np.squeeze(acc_training)
        acc_validation = np.squeeze(acc_validation)
        x = np.arange(1, acc_training.shape[0]+1)
        ax.plot(x, np.squeeze(acc_training, 'b'))
        ax.plot(x, np.squeeze(acc_validation), 'r')
        ax.set_ylabel('Dice Coefficient')
        ax.set_xlabel('epochs')
        ax.grid(b=True, which='both')
        ax.legend(['Training Dice Coefficient', 'Validation Dice Coefficient'])
        self.canvas_accuracy_figure.draw()



    def plotConfusionMatrix(self, confusion_matrix, target_names):
        df_cm = pd.DataFrame(confusion_matrix,
                             index=[i for i in target_names],
                             columns=[i for i in target_names], )

        axes_confmat = self.confusion_matrix_figure.add_subplot(111)
        axes_confmat.clear()

        sn.heatmap(df_cm, annot=True, fmt='.3f', annot_kws={"size": 8} , ax=axes_confmat, linewidths=.2)
        axes_confmat.set_xlabel('Predicted Label')
        axes_confmat.set_ylabel('True Label')
        self.confusion_matrix_figure.tight_layout()

        self.canvas_confusion_matrix_figure.draw()



    def button_selectModel_prediction_clicked(self):
        pathToModel = self.openFileNamesDialog(self.deepLearningArtApp.getLearningOutputPath())
        self.deepLearningArtApp.setModelForPrediction(pathToModel)
        self.ui.Label_currentModel_prediction.setText(pathToModel)


    def button_selectDataset_prediction_clicked(self):
        pathToDataset = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
        self.deepLearningArtApp.setDatasetForPrediction(pathToDataset)
        self.ui.Label_currentDataset_prediction.setText(pathToDataset)


    def updateProgressBarTraining(self, val):
        if val >= 0 and val <= 100:
            self.ui.progressBar_Training.setValue(val)


    def plotTrainingLivePerformance(self, train_acc, val_acc, train_loss, val_loss):
        epochs = np.arange(1, len(train_acc)+1)

        self.training_live_performance_figure.clear()
        ax1 = self.training_live_performance_figure.add_subplot(211)
        ax1.clear()
        ax1.plot(epochs, train_acc, 'r')
        ax1.plot(epochs, val_acc, 'b')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Dice Coefficient')
        ax1.grid(b=True, which='both')
        ax1.legend(['Training Accuracy', 'Validation Accuracy'])
        ax1.set_xlim(1, self.deepLearningArtApp.getEpochs())

        ax2 = self.training_live_performance_figure.add_subplot(212)
        ax2.clear()
        ax2.plot(epochs, train_loss, 'r')
        ax2.plot(epochs, val_loss, 'b')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.grid(b=True, which='both')
        ax2.legend(['Training loss', 'Validation loss'])
        ax2.set_xlim(1, self.deepLearningArtApp.getEpochs())

        self.canvas_training_live_performance_figure.draw()

        QApplication.processEvents()
        #QTimer.singleShot(0, lambda: self.update)


    def button_train_clicked(self):
        # set gpu
        gpuId = self.ui.ComboBox_GPU.currentIndex()
        self.deepLearningArtApp.setGPUId(gpuId)

        # set epochs
        self.deepLearningArtApp.setEpochs(self.ui.SpinBox_Epochs.value())

        # handle check states of check boxes for used classes
        self.deepLearningArtApp.setUsingArtifacts(self.ui.CheckBox_Artifacts.isChecked())
        self.deepLearningArtApp.setUsingBodyRegions(self.ui.CheckBox_BodyRegion.isChecked())
        self.deepLearningArtApp.setUsingTWeighting(self.ui.CheckBox_TWeighting.isChecked())

        # set learning rates and batch sizes
        try:
            batchSizes = np.fromstring(self.ui.LineEdit_BatchSizes.text(), dtype=np.int, sep=',')
            self.deepLearningArtApp.setBatchSizes(batchSizes)
            learningRates = np.fromstring(self.ui.LineEdit_LearningRates.text(), dtype=np.float32, sep=',')
            self.deepLearningArtApp.setLearningRates(learningRates)
        except:
            raise ValueError("Wrong input format of learning rates! Enter values seperated by ','. For example: 0.1,0.01,0.001")

        # set optimizer
        selectedOptimizer = self.ui.ComboBox_Optimizers.currentText()
        if selectedOptimizer == "SGD":
            self.deepLearningArtApp.setOptimizer(SGD_OPTIMIZER)
        elif selectedOptimizer == "RMSprop":
            self.deepLearningArtApp.setOptimizer(RMS_PROP_OPTIMIZER)
        elif selectedOptimizer == "Adagrad":
            self.deepLearningArtApp.setOptimizer(ADAGRAD_OPTIMIZER)
        elif selectedOptimizer == "Adadelta":
            self.deepLearningArtApp.setOptimizer(ADADELTA_OPTIMIZER)
        elif selectedOptimizer == "Adam":
            self.deepLearningArtApp.setOptimizer(ADAM_OPTIMIZER)
        else:
            raise ValueError("Unknown Optimizer!")

        # set weigth decay
        self.deepLearningArtApp.setWeightDecay(float(self.ui.DoubleSpinBox_WeightDecay.value()))
        # set momentum
        self.deepLearningArtApp.setMomentum(float(self.ui.DoubleSpinBox_Momentum.value()))
        # set nesterov enabled
        if self.ui.CheckBox_Nesterov.checkState() == Qt.Checked:
            self.deepLearningArtApp.setNesterovEnabled(True)
        else:
            self.deepLearningArtApp.setNesterovEnabled(False)

        # handle data augmentation
        if self.ui.CheckBox_DataAugmentation.checkState() == Qt.Checked:
            self.deepLearningArtApp.setDataAugmentationEnabled(True)
            # get all checked data augmentation options
            if self.ui.CheckBox_DataAug_horizontalFlip.checkState() == Qt.Checked:
                self.deepLearningArtApp.setHorizontalFlip(True)
            else:
                self.deepLearningArtApp.setHorizontalFlip(False)

            if self.ui.CheckBox_DataAug_verticalFlip.checkState() == Qt.Checked:
                self.deepLearningArtApp.setVerticalFlip(True)
            else:
                self.deepLearningArtApp.setVerticalFlip(False)

            if self.ui.CheckBox_DataAug_Rotation.checkState() == Qt.Checked:
                self.deepLearningArtApp.setRotation(True)
            else:
                self.deepLearningArtApp.setRotation(False)

            if self.ui.CheckBox_DataAug_zcaWeighting.checkState() == Qt.Checked:
                self.deepLearningArtApp.setZCA_Whitening(True)
            else:
                self.deepLearningArtApp.setZCA_Whitening(False)

            if self.ui.CheckBox_DataAug_HeightShift.checkState() == Qt.Checked:
                self.deepLearningArtApp.setHeightShift(True)
            else:
                self.deepLearningArtApp.setHeightShift(False)

            if self.ui.CheckBox_DataAug_WidthShift.checkState() == Qt.Checked:
                self.deepLearningArtApp.setWidthShift(True)
            else:
                self.deepLearningArtApp.setWidthShift(False)

            if self.ui.CheckBox_DataAug_Zoom.checkState() == Qt.Checked:
                self.deepLearningArtApp.setZoom(True)
            else:
                self.deepLearningArtApp.setZoom(False)


            # contrast improvement (contrast stretching, adaptive equalization, histogram equalization)
            # it is not recommended to set more than one of them to true
            if self.ui.RadioButton_DataAug_contrastStretching.isChecked():
                self.deepLearningArtApp.setContrastStretching(True)
            else:
                self.deepLearningArtApp.setContrastStretching(False)

            if self.ui.RadioButton_DataAug_histogramEq.isChecked():
                self.deepLearningArtApp.setHistogramEqualization(True)
            else:
                self.deepLearningArtApp.setHistogramEqualization(False)

            if self.ui.RadioButton_DataAug_adaptiveEq.isChecked():
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
        self.ui.Label_MarkingsPath.setText(dir)
        self.deepLearningArtApp.setMarkingsPath(dir)



    def button_patching_clicked(self):
        if self.deepLearningArtApp.getSplittingMode() == NONE_SPLITTING:
            QMessageBox.about(self, "My message box", "Select Splitting Mode!")
            return 0

        self.getSelectedDatasets()
        self.getSelectedPatients()

        # random shuffle
        if self.ui.CheckBox_randomShuffle.isChecked():
            self.deepLearningArtApp.setIsRandomShuffle(True)
        else:
            self.deepLearningArtApp.setIsRandomShuffle(False)

        # get patching parameters
        self.deepLearningArtApp.setPatchSizeX(self.ui.SpinBox_PatchX.value())
        self.deepLearningArtApp.setPatchSizeY(self.ui.SpinBox_PatchY.value())
        self.deepLearningArtApp.setPatchSizeZ(self.ui.SpinBox_PatchZ.value())
        self.deepLearningArtApp.setPatchOverlapp(self.ui.SpinBox_PatchOverlapp.value())

        # get labling parameters
        if self.ui.RadioButton_MaskLabeling.isChecked():
            self.deepLearningArtApp.setLabelingMode(MASK_LABELING)
        elif self.ui.RadioButton_PatchLabeling.isChecked():
            self.deepLearningArtApp.setLabelingMode(PATCH_LABELING)

        # get patching parameters
        if self.ui.ComboBox_Patching.currentIndex() == 1:
            # 2D patching selected
            self.deepLearningArtApp.setPatchingMode(PATCHING_2D)
        elif self.ui.ComboBox_Patching.currentIndex() == 2:
            # 3D patching selected
            self.deepLearningArtApp.setPatchingMode(PATCHING_3D)
        else:
            self.ui.ComboBox_Patching.setCurrentIndex(1)
            self.deepLearningArtApp.setPatchingMode(PATCHING_2D)

        #using segmentation mask
        self.deepLearningArtApp.setUsingSegmentationMasks(self.ui.CheckBox_SegmentationMask.isChecked())

        # handle store mode
        self.deepLearningArtApp.setStoreMode(self.ui.ComboBox_StoreOptions.currentIndex())

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
            self.ui.Button_useCurrentData.setEnabled(True)



    def button_outputPatching_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
        self.ui.Label_OutputPathPatching.setText(dir)
        self.deepLearningArtApp.setOutputPathForPatching(dir)


    def spinBox_sliceSelect_changed(self):
        if self.deepLearningArtApp.usingSegmentationMasksForPrediction:
            self.plotSegmentationArtifactMaps()
            self.plotSegmentationPredictions()
        else:
            self.plotArtifactMaps()


    def spinBox_sliceSelection_multiclassVisualisation_changed(self):
        self.update_multiclass_visualisation()


    def getSelectedPatients(self):
        selectedPatients = []
        for i in range(self.ui.TreeWidget_Patients.topLevelItemCount()):
            if self.ui.TreeWidget_Patients.topLevelItem(i).checkState(0) == Qt.Checked:
                selectedPatients.append(self.ui.TreeWidget_Patients.topLevelItem(i).text(0))

        self.deepLearningArtApp.setSelectedPatients(selectedPatients)


    def button_DB_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getPathToDatabase())
        self.deepLearningArtApp.setPathToDatabase(dir)
        self.manageTreeView()


    def openFileNamesDialog(self, dir=None):
        if dir==None:
            dir = PATH_OUT + os.sep + "MRPhysics"

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
            self.ui.TreeWidget_Patients.setHeaderLabel("Patients:")

            for x in sorted(subdirs):
                item = QTreeWidgetItem()
                item.setText(0, str(x))
                item.setCheckState(0, Qt.Unchecked)
                self.ui.TreeWidget_Patients.addTopLevelItem(item)

            self.ui.Label_DB.setText(self.deepLearningArtApp.getPathToDatabase())


    def manageTreeViewDatasets(self):
        print(os.path.dirname(self.deepLearningArtApp.getPathToDatabase()))
        # manage datasets
        self.ui.TreeWidget_Datasets.setHeaderLabel("Datasets:")
        for ds in sorted(DeepLearningArtApp.datasets.keys()):
            dataset = DeepLearningArtApp.datasets[ds].getPathdata()
            item = QTreeWidgetItem()
            item.setText(0, dataset)
            item.setCheckState(0, Qt.Unchecked)
            self.ui.TreeWidget_Datasets.addTopLevelItem(item)


    def getSelectedDatasets(self):
        selectedDatasets = []
        for i in range(self.ui.TreeWidget_Datasets.topLevelItemCount()):
            if self.ui.TreeWidget_Datasets.topLevelItem(i).checkState(0) == Qt.Checked:
                selectedDatasets.append(self.ui.TreeWidget_Datasets.topLevelItem(i).text(0))

        self.deepLearningArtApp.setSelectedDatasets(selectedDatasets)


    def selectedDNN_changed(self):
        self.deepLearningArtApp.setNeuralNetworkModel(self.ui.ComboBox_DNNs.currentText())


    def button_useCurrentData_clicked(self):
        if self.deepLearningArtApp.datasetAvailable() == True:
            self.ui.Label_currentDataset.setText("Current Dataset is used...")
            self.ui.GroupBox_TrainNN.setEnabled(True)
        else:
            self.ui.Button_useCurrentData.setEnabled(False)
            self.ui.Label_currentDataset.setText("No Dataset selected!")
            self.ui.GroupBox_TrainNN.setEnabled(False)


    def button_selectDataset_clicked(self):
        pathToDataset = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
        retbool, datasetName = self.deepLearningArtApp.loadDataset(pathToDataset)
        if retbool == True:
            self.ui.Label_currentDataset.setText(datasetName + " is used as dataset...")
        else:
            self.ui.Label_currentDataset.setText("No Dataset selected!")

        if self.deepLearningArtApp.datasetAvailable() == True:
            self.ui.GroupBox_TrainNN.setEnabled(True)
        else:
            self.ui.GroupBox_TrainNN.setEnabled(False)


    def button_learningOutputPath_clicked(self):
        path = self.openFileNamesDialog(self.deepLearningArtApp.getLearningOutputPath())
        self.deepLearningArtApp.setLearningOutputPath(path)
        self.ui.Label_LearningOutputPath.setText(path)


    def splittingMode_changed(self):

        if self.ui.ComboBox_splittingMode.currentIndex() == 0:
            self.deepLearningArtApp.setSplittingMode(NONE_SPLITTING)
            self.ui.Label_SplittingParams.setText("Select splitting mode!")
        elif self.ui.ComboBox_splittingMode.currentIndex() == 1:
            # call input dialog for editting ratios
            testTrainingRatio, retBool = QInputDialog.getDouble(self, "Enter Test/Training Ratio:",
                                                             "Ratio Test/Training Set:", 0.2, 0, 1, decimals=2)
            if retBool == True:
                validationTrainingRatio, retBool = QInputDialog.getDouble(self, "Enter Validation/Training Ratio",
                                                                      "Ratio Validation/Training Set: ", 0.2, 0, 1, decimals=2)
                if retBool == True:
                    self.deepLearningArtApp.setSplittingMode(SIMPLE_RANDOM_SAMPLE_SPLITTING)
                    self.deepLearningArtApp.setTrainTestDatasetRatio(testTrainingRatio)
                    self.deepLearningArtApp.setTrainValidationRatio(validationTrainingRatio)
                    txtStr = "using Test/Train=" + str(testTrainingRatio) + " and Valid/Train=" + str(validationTrainingRatio)
                    self.ui.Label_SplittingParams.setText(txtStr)
                else:
                    self.deepLearningArtApp.setSplittingMode(NONE_SPLITTING)
                    self.ui.ComboBox_splittingMode.setCurrentIndex(0)
                    self.ui.Label_SplittingParams.setText("Select Splitting Mode!")
            else:
                self.deepLearningArtApp.setSplittingMode(NONE_SPLITTING)
                self.ui.ComboBox_splittingMode.setCurrentIndex(0)
                self.ui.Label_SplittingParams.setText("Select Splitting Mode!")
        elif self.ui.ComboBox_splittingMode.currentIndex() == 2:
            # cross validation splitting
            testTrainingRatio, retBool = QInputDialog.getDouble(self, "Enter Test/Training Ratio:",
                                                             "Ratio Test/Training Set:", 0.2, 0, 1, decimals=2)

            if retBool == True:
                numFolds, retBool = QInputDialog.getInt(self, "Enter Number of Folds for Cross Validation",
                                                    "Number of Folds: ", 15, 0, 100000)
                if retBool == True:
                    self.deepLearningArtApp.setSplittingMode(CROSS_VALIDATION_SPLITTING)
                    self.deepLearningArtApp.setTrainTestDatasetRatio(testTrainingRatio)
                    self.deepLearningArtApp.setNumFolds(numFolds)
                    self.ui.Label_SplittingParams.setText("Test/Train Ratio: " + str(testTrainingRatio) + \
                                                          ", and " + str(numFolds) + " Folds")
                else:
                    self.deepLearningArtApp.setSplittingMode(NONE_SPLITTING)
                    self.ui.ComboBox_splittingMode.setCurrentIndex(0)
                    self.ui.Label_SplittingParams.setText("Select Splitting Mode!")
            else:
                self.deepLearningArtApp.setSplittingMode(NONE_SPLITTING)
                self.ui.ComboBox_splittingMode.setCurrentIndex(0)
                self.ui.Label_SplittingParams.setText("Select Splitting Mode!")

        elif self.ui.ComboBox_splittingMode.currentIndex() == 3:
            self.deepLearningArtApp.setSplittingMode(PATIENT_CROSS_VALIDATION_SPLITTING)



    def check_dataAugmentation_enabled(self):
        if self.ui.CheckBox_DataAugmentation.checkState() == Qt.Checked:
            self.ui.CheckBox_DataAug_horizontalFlip.setEnabled(True)
            self.ui.CheckBox_DataAug_verticalFlip.setEnabled(True)
            self.ui.CheckBox_DataAug_Rotation.setEnabled(True)
            self.ui.CheckBox_DataAug_zcaWeighting.setEnabled(True)
            self.ui.CheckBox_DataAug_HeightShift.setEnabled(True)
            self.ui.CheckBox_DataAug_WidthShift.setEnabled(True)
            self.ui.CheckBox_DataAug_Zoom.setEnabled(True)
            self.ui.RadioButton_DataAug_contrastStretching.setEnabled(True)
            self.ui.RadioButton_DataAug_histogramEq.setEnabled(True)
            self.ui.RadioButton_DataAug_adaptiveEq.setEnabled(True)
        else:
            self.ui.CheckBox_DataAug_horizontalFlip.setEnabled(False)
            self.ui.CheckBox_DataAug_verticalFlip.setEnabled(False)
            self.ui.CheckBox_DataAug_Rotation.setEnabled(False)
            self.ui.CheckBox_DataAug_zcaWeighting.setEnabled(False)
            self.ui.CheckBox_DataAug_HeightShift.setEnabled(False)
            self.ui.CheckBox_DataAug_WidthShift.setEnabled(False)
            self.ui.CheckBox_DataAug_Zoom.setEnabled(False)

            self.ui.RadioButton_DataAug_contrastStretching.setEnabled(False)
            self.ui.RadioButton_DataAug_contrastStretching.setAutoExclusive(False)
            self.ui.RadioButton_DataAug_contrastStretching.setChecked(False)
            self.ui.RadioButton_DataAug_contrastStretching.setAutoExclusive(True)

            self.ui.RadioButton_DataAug_histogramEq.setEnabled(False)
            self.ui.RadioButton_DataAug_histogramEq.setAutoExclusive(False)
            self.ui.RadioButton_DataAug_histogramEq.setChecked(False)
            self.ui.RadioButton_DataAug_histogramEq.setAutoExclusive(True)


            self.ui.RadioButton_DataAug_adaptiveEq.setEnabled(False)
            self.ui.RadioButton_DataAug_adaptiveEq.setAutoExclusive(False)
            self.ui.RadioButton_DataAug_adaptiveEq.setChecked(False)
            self.ui.RadioButton_DataAug_adaptiveEq.setAutoExclusive(True)



    def plotImg(self, figure, canvas, index):
        art, ref = self.deepLearningArtApp.getArtRefPair(index)

        data = [random.random() for i in range(25)]

        axes_art = figure.add_subplot(121)
        axes_art.imshow(art, cmap='gray')
        axes_art.set_title('Artefact')

        axes_ref = figure.add_subplot(122)
        axes_ref.imshow(ref, cmap='gray')
        axes_ref.set_title('Reference')

        canvas.draw()




sys._excepthook = sys.excepthook

def my_exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)

sys.excepthook = my_exception_hook

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_())