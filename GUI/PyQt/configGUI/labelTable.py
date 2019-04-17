# create label list based on the markings.csv
import pandas
import numpy as np

from PyQt5.QtCore import QAbstractTableModel
import os

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox

import operator  # used for sorting

from PyQt5 import QtCore
from PyQt5.QtCore import QAbstractTableModel, Qt, pyqtSignal
from PyQt5.QtWidgets import *
from matplotlib.patches import PathPatch, Rectangle, Ellipse
from matplotlib.path import Path


class LabelTable(QTableView):

    statusChanged = pyqtSignal(bool)
    selectChanged = pyqtSignal(int)

    def __int__(self, parent=None):

# file is a .csv file containing the data extraction from marking_records.csv
# only the records with labelname can be saved
# file name can be self-defined by user, if not, the file willbe default marking_records.csv

        QTableView.__init__(self, parent)


    def set_table_model(self, labelfile, imagefile):
        self.labelfile = labelfile

        filename, file_extension = os.path.splitext(self.labelfile)
        if not file_extension == '.csv':
            QMessageBox.information(self, "Warning", "Please select one .csv File with label", QMessageBox.Ok)

        self.imagefile = imagefile

        self.df = pandas.read_csv(self.labelfile)

        header = ['On/Off', 'label name', 'slice', 'ID']
        self.dataListwithInfo = []
        dataList = []
        num = self.df[self.df['image'] == self.imagefile].index.values.astype(int)
        for i in range(0, len(num)):
            newItem = self.df.iloc[num[i]]
            self.dataListwithInfo.append(newItem)
            checkbox = QCheckBox("ON")
            if newItem['status'] == 0:
                checkbox.setChecked(True)
            else:
                checkbox.setChecked(False)
            dataList.append([checkbox, newItem['labelname'], newItem['slice'], str(num[i])])
        self.table_model = MyTableModel(self, dataList, header)
        self.table_model.viewChanged.connect(self.view_change_file)
        self.table_model.itemSelected.connect(self.selectRow)
        self.setModel(self.table_model)
        self.update()

    def update_model(self, labelfile):
        self.labelfile = labelfile

        filename, file_extension = os.path.splitext(self.labelfile)
        if not file_extension == '.csv':
            QMessageBox.information(self, "Warning", "Please select one .csv File with label", QMessageBox.Ok)

        self.df = pandas.read_csv(self.labelfile)

        header = ['On/Off', 'label name', 'slice', 'ID']
        self.dataListwithInfo = []
        dataList = []
        num = self.df[self.df['image'] == self.imagefile].index.values.astype(int)
        for i in range(0, len(num)):
            newItem = self.df.iloc[num[i]]
            self.dataListwithInfo.append(newItem)
            checkbox = QCheckBox("ON")
            if newItem['status'] == 0:
                checkbox.setChecked(True)
            else:
                checkbox.setChecked(False)
            dataList.append([checkbox, newItem['labelname'], newItem['slice'], str(num[i])])

        self.table_model = MyTableModel(self, dataList, header)
        self.table_model.viewChanged.connect(self.view_change_file)
        self.table_model.itemSelected.connect(self.selectRow)
        self.setModel(self.table_model)
        self.update()

    def view_all(self, value):

        self.df = pandas.read_csv(self.labelfile)

        header = ['On/Off', 'label name', 'slice', 'ID']
        self.dataListwithInfo = []
        dataList = []
        num = self.df[self.df['image'] == self.imagefile].index.values.astype(int)
        for i in range(0, len(num)):
            newItem = self.df.iloc[num[i]]
            self.dataListwithInfo.append(newItem)
            checkbox = QCheckBox("ON")
            if not value:
                self.df.loc[num[i],'status'] = 1
                checkbox.setChecked(True)
                checkbox.setText('ON')
                self.statusChanged.emit(True)
            else:
                self.df.loc[num[i], 'status'] = 0
                checkbox.setChecked(False)
                checkbox.setText('OFF')
                self.statusChanged.emit(False)
            dataList.append([checkbox, newItem['labelname'], newItem['slice'], str(num[i])])

        self.df.to_csv(self.labelfile, index=False)
        self.table_model = MyTableModel(self, dataList, header)
        self.table_model.viewChanged.connect(self.view_change_file)
        self.table_model.itemSelected.connect(self.selectRow)
        self.setModel(self.table_model)
        self.update()

    def get_table_model(self):
        return self.table_model

    def get_list(self):
        return self.dataListwithInfo

    def get_selectRow(self, clickedIndex):
        row = clickedIndex.row()
        self.selectChanged.emit(row)

    def view_change_file(self, status):

        ind = self.get_table_model().get_selectind()
        if self.df['labelshape'][ind] == 'lasso':
            path = Path(np.asarray(eval(self.df['path'][ind])))
            self.selectedItem = PathPatch(path, fill=True, alpha=.2, edgecolor=None)
        else:
            self.df.select_dtypes(include='object')
            self.selectedItem = eval(self.df['artist'][ind])
        if status:
            self.df.loc[ind, 'status'] = 0
            self.df.to_csv(self.labelfile, index=False)
            self.statusChanged.emit(True)
        else:
            self.df.loc[ind, 'status'] = 1
            self.df.to_csv(self.labelfile, index=False)
            self.statusChanged.emit(False)

class MyTableModel(QAbstractTableModel):
    """
    keep the method names
    they are an integral part of the model
    """
    layoutChanged = pyqtSignal()
    layoutAboutToBeChanged = pyqtSignal()
    viewChanged = pyqtSignal(bool)
    itemSelected = pyqtSignal()

    def __init__(self, parent, list, header, *args):
        QAbstractTableModel.__init__(self, parent, *args)

        self.mylist = list
        self.header = header
        self.selectind = 0

        # self.rowCheckStateMap = {}

    def setDataList(self, list):
        self.mylist = list
        self.layoutAboutToBeChanged.emit()
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(0), self.columnCount(0)))
        self.layoutChanged.emit()

    def rowCount(self, parent):
        return len(self.mylist)

    def columnCount(self, parent):
        return len(self.header)

    def data(self, index, role):
        if not index.isValid():
            return None
        if (index.column() == 0):
            value = self.mylist[index.row()][index.column()].text()
        else:
            value = self.mylist[index.row()][index.column()]
        if role == QtCore.Qt.EditRole:
            return value
            itemSelected.emit()
        elif role == QtCore.Qt.DisplayRole:
            return value
            itemSelected.emit()
        elif role == QtCore.Qt.CheckStateRole:
            if index.column() == 0:
                # print(">>> data() row,col = %d, %d" % (index.row(), index.column()))
                if self.mylist[index.row()][index.column()].isChecked():
                    return QtCore.Qt.Checked
                else:
                    return QtCore.Qt.Unchecked

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.header[col]
        return None

    def sort(self, col, order):
        self.mylist = self.get_mylist()

        """sort table by given column number col"""
        # print(">>> sort() col = ", col)
        if col != 0:
            self.layoutAboutToBeChanged.emit()
            self.mylist = sorted(self.mylist, key=lambda x: x[col])
            if order == Qt.DescendingOrder:
                self.mylist = sorted(self.mylist, reverse=True, key=lambda x: x[col])
            self.layoutChanged.emit()


    def flags(self, index):
        if not index.isValid():
            return None
        # print(">>> flags() index.column() = ", index.column())
        if index.column() == 0:
            # return Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable
        else:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        # print(">>> setData() role = ", role)
        # print(">>> setData() index.column() = ", index.column())
        # print(">>> setData() value = ", value)
        if role == QtCore.Qt.CheckStateRole and index.column() == 0:
            if value == QtCore.Qt.Checked:
                self.mylist[index.row()][index.column()].setChecked(True)
                self.mylist[index.row()][index.column()].setText("ON")
                self.viewChanged.emit(True)
                # if studentInfos.size() > index.row():
                #     emit StudentInfoIsChecked(studentInfos[index.row()])
            else:
                self.mylist[index.row()][index.column()].setChecked(False)
                self.mylist[index.row()][index.column()].setText("OFF")
                self.viewChanged.emit(False)
        else:
            pass
        row = index.row()
        self.selectind = int(self.mylist[row][-1])
        self.dataChanged.emit(index, index)
        return True

    def get_mylist(self):
        return self.mylist

    def get_selectind(self):
        return self.selectind