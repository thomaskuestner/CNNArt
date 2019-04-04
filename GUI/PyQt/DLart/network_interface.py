import os

import pandas as pd
import yaml
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QTableWidget
from PyQt5.QtWidgets import (QTableWidgetItem)
from PyQt5.QtWidgets import QWidget

with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    subdir = cfg['subdirs'][1]

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

        self.setWindowTitle("Select Data Sets for Training & Test Network")

    def updateRecord(self):

        row = self.table.rowCount()
        df = pd.read_csv('DLart/network_interface_datasets.csv')[:row]
        for r in range(row):
            df.loc[r, 'usingfor'] = self.table.cellWidget(r, 1).currentText()
        df.to_csv('DLart/network_interface_datasets.csv', index=False)
        return df

from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel

from pyqtgraph.parametertree import Parameter, ParameterTree


class NetworkInterface(QWidget):
    interfaceFinishUpdating = pyqtSignal(bool)

    def __init__(self, params):
        super().__init__()

        self.p = Parameter.create(name='params', type='group', children=params)
        self.p.sigTreeStateChanged.connect(self.change)

        for child in self.p.children():
            child.sigValueChanging.connect(self.valueChanging)
            for ch2 in child.children():
                ch2.sigValueChanging.connect(self.valueChanging)

        self.p.param('Save/Restore functionality', 'Save State').sigActivated.connect(self.save)
        self.p.param('Save/Restore functionality', 'Restore State').sigActivated.connect(self.restore)

        self.t = ParameterTree()
        self.t.setParameters(self.p, showTop=False)
        self.t.setWindowTitle('Network Interface')

        self.win = QWidget()
        self.layout = QGridLayout()
        self.win.setLayout(self.layout)
        l = QLabel("Network Interface")
        l.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.layout.addWidget(l)
        self.layout.addWidget(self.t)
        self.win.setGeometry(500, 800, 500, 800)

        ## test save/restore
        s = self.p.saveState()
        self.p.restoreState(s)

    def change(self, param, changes):
        print("tree changes:")
        for param, change, data in changes:
            path = self.p.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()

    def valueChanging(self, param, value):
        newparam = {}
        newparam[param.name()] = value

    def save(self):
        global state
        state = self.p.saveState()

    def restore(self):
        global state
        add = self.p['Save/Restore functionality', 'Restore State', 'Add missing items']
        rem = self.p['Save/Restore functionality', 'Restore State', 'Remove extra items']
        self.p.restoreState(state, addChildren=add, removeChildren=rem)

    def updataParameter(self, params):

        self.interfaceFinishUpdating.emit(False)

        self.p = Parameter.create(name='params', type='group', children=params)
        self.p.sigTreeStateChanged.connect(self.change)

        for child in self.p.children():
            child.sigValueChanging.connect(self.valueChanging)
            for ch2 in child.children():
                ch2.sigValueChanging.connect(self.valueChanging)

        self.p.param('Save/Restore functionality', 'Save State').sigActivated.connect(self.save)
        self.p.param('Save/Restore functionality', 'Restore State').sigActivated.connect(self.restore)

        self.t.setParameters(self.p, showTop=False)
        self.t.setWindowTitle('Network Interface')

        self.layout.addWidget(self.t)

        ## test save/restore
        s = self.p.saveState()
        self.p.restoreState(s)

        self.interfaceFinishUpdating.emit(True)


