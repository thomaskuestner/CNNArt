from PyQt5 import QtWidgets
from cPatches import Ui_Patches

import numpy as np
import matplotlib.pyplot as plt
import os
from Unpatch_eleven import*
import scipy.io as sio
import matplotlib as mpl
from collections import Counter

class Patches_window(QtWidgets.QDialog,Ui_Patches):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # self.loadresult.clicked.connect(self.loadoverlay)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.data = {'ok': 1}

        self.listWidget.insertItem(0, '11 classes')
        self.listWidget.insertItem(0, '2 classes')
        self.listWidget.currentRowChanged.connect(self.stackedWidget.setCurrentIndex)

        self.twobox = QtWidgets.QButtonGroup(self)
        self.twobox.addButton(self.t1, 11)
        self.twobox.addButton(self.t2, 12)
        self.twobox.buttonClicked.connect(self.ccmode_two)

        self.elebox = QtWidgets.QButtonGroup(self)  
        self.elebox.addButton(self.e1, 11)
        self.elebox.addButton(self.e2, 12)
        self.elebox.buttonClicked.connect(self.ccmode_eleven)

        self.b_111.clicked.connect(self.color111)
        self.b_112.clicked.connect(self.color112)
        self.b_113.clicked.connect(self.color113)
        self.b_114.clicked.connect(self.color114)
        self.b_115.clicked.connect(self.color115)

        self.diylist = []
        self.filelist = []

    def ccmode_two(self):
        if self.twobox.checkedId() == 11:
            self.colormode = 1
        else:
            self.colormode = 2

    def ccmode_eleven(self):
        if self.elebox.checkedId() == 11:
            self.colormode = 1
        else:
            self.colormode = 2
    # def ccmode_eight

    def color111(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist.append(col.name())

    def color112(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist.append(col.name())

    def color113(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist.append(col.name())

    def color114(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist.append(col.name())

    def color115(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist.append(col.name())

    def ccsetting(self):
        self.linepos = self.colorlinepos.currentIndex()
        self.classnr = self.listWidget.currentRow()
        print(self.classnr)
        if self.classnr == 1: ####
            if self.colormode == 1:
                self.cmap = mpl.colors.ListedColormap(['blue', 'purple', 'cyan', 'yellow', 'green'])
            else:
                self.cmap = mpl.colors.ListedColormap(self.diylist)
        # elif:
        return self.linepos, self.classnr, self.cmap

    def getData(parent=None):
        dialog = Patches_window(parent)
        ok = dialog.exec_()
        linepos, classnr, cmap = dialog.ccsetting()
        return linepos, classnr, cmap, ok











