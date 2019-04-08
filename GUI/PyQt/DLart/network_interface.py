import os
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QTableWidget
from PyQt5.QtWidgets import (QTableWidgetItem)
from PyQt5.QtWidgets import QWidget

from config.PATH import SUBDIRS

subdir = SUBDIRS[1]


class DataSetsWindow(QWidget):
    def __init__(self, patients, sequences, parent=None):
        super(DataSetsWindow, self).__init__(parent)
        self.layout = QGridLayout()
        self.DataSetsHist = []
        for patient in patients:
            for sequence in sequences:
                self.DataSetsHist.append(patient + os.sep + subdir + os.sep + sequence)
        self.df = pd.read_csv('DLart/network_interface_datasets.csv')
        self.df = self.df.drop(self.df.index[:])
        self.createGUI()
        self.createRecord()
        self.currentrow = None
        self.setToolTip("Click the image to confirm selection")

    def createRecord(self):

        for i in range(len(self.DataSetsHist)):
            self.df.loc[i, 'image'] = self.DataSetsHist[i]
            self.df.loc[i, 'usingfor'] = self.table.cellWidget(i, 1).currentText()
        self.df.to_csv('DLart/network_interface_datasets.csv', index=False)

    def createGUI(self):

        length = len(self.DataSetsHist)
        self.table = QTableWidget(length, 2)
        self.table.setHorizontalHeaderLabels(["Images", "Using for"])
        self.table.verticalHeader().setVisible(True)
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)

        for i in range(length):
            imageItem = QTableWidgetItem(self.DataSetsHist[i])
            selectItem = QtWidgets.QComboBox()
            selectItem.addItems(['Training', 'Validation', 'Test'])
            self.table.setItem(i, 0, imageItem)
            self.table.setCellWidget(i, 1, selectItem)

        self.table.resizeColumnToContents(0)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.cellClicked.connect(self.updateRecord)

        self.layout.addWidget(self.table, 0, 0)
        self.setLayout(self.layout)
        self.setGeometry(800, 200, 550, 200)
        self.setWindowTitle("Select Datasets for Training & Test Network")
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

    def updateRecord(self):

        row = self.table.rowCount()
        self.currentrow = self.DataSetsHist[row-1]
        df = pd.read_csv('DLart/network_interface_datasets.csv')[:row]
        for r in range(row):
            df.loc[r, 'usingfor'] = self.table.cellWidget(r, 1).currentText()
        df.to_csv('DLart/network_interface_datasets.csv', index=False)
        return df

    def getCurrentRow(self):
        return self.currentrow


from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel

from pyqtgraph.parametertree import Parameter, ParameterTree


class NetworkInterface(QWidget):

    def __init__(self, params):
        super().__init__()

        self.param = Parameter.create(name='params', type='group', children=params)

        self.tree = ParameterTree()
        self.tree.setParameters(self.param, showTop=False)
        self.tree.setWindowTitle('Network Interface')

        self.window = QWidget()
        self.layout = QGridLayout()
        self.window.setLayout(self.layout)
        l = QLabel("Network Interface")
        l.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.layout.addWidget(l)
        self.layout.addWidget(self.tree)
        self.window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.window.setGeometry(500, 800, 800, 500)

    def updataParameter(self, params, opt):

        self.param = Parameter.create(name='params', type='group', children=params)
        self.tree.setParameters(self.param, showTop=False)
        self.layout.addWidget(self.tree)

        if opt == 1:
            print('Please wait for a moment, I am working in progress')
        else:
            print('interface updated')
        QtWidgets.QApplication.processEvents()

    def closeEvent(self, closeEvent):
        reply = QtWidgets.QMessageBox.question(self, 'Warning', 'Are you sure to exit?',
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:

            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()
