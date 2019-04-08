# -*- coding: utf-8 -*-

import math

from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QPalette
from PyQt5.QtWidgets import QWidget


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
        palette.setColor(palette.Background, QtCore.Qt.transparent)
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
        painter.setPen(QPen(QtCore.Qt.NoPen))

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
