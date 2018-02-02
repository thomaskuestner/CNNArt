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

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.data = {'ok': 1}

        self.listWidget.insertItem(0, '11 classes')
        self.listWidget.insertItem(0, '2 classes')
        self.listWidget.currentRowChanged.connect(self.stackedWidget.setCurrentIndex)

        self.twobox = QtWidgets.QButtonGroup(self)
        self.twobox.addButton(self.t1, 11)
        self.twobox.addButton(self.t2, 12)
        self.twobox.buttonClicked.connect(self.ccmode_t)

        self.elebox = QtWidgets.QButtonGroup(self)  
        self.elebox.addButton(self.e1, 11)
        self.elebox.addButton(self.e2, 12)
        self.elebox.buttonClicked.connect(self.ccmode_e)

    def ccmode_t(self):
        if self.twobox.checkedId() == 11:
            self.colormode = 1
        else:
            self.colormode = 2

    def ccmode_e(self):
        if self.elebox.checkedId() == 11:
            self.colormode = 1
        else:
            self.colormode = 2

    def ccsetting(self):
        self.linepos = self.colorlinepos.currentIndex()
        self.classnr = self.listWidget.currentRow()
        print(self.classnr)
        return self.linepos, self.classnr

    def getData(parent=None):
        dialog = Patches_window(parent)
        ok = dialog.exec_()
        linepos, classnr = dialog.ccsetting()
        return linepos, classnr, ok











