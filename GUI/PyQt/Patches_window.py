from PyQt5 import QtWidgets
from cPatches import Ui_Patches

import matplotlib as mpl
from collections import Counter

class Patches_window(QtWidgets.QDialog,Ui_Patches):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.colormode2 = 1
        self.colormode8 = 1
        self.colormode11 = 1

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.data = {'ok': 1}

        self.listWidget.insertItem(0, '11 classes')
        self.listWidget.insertItem(0, '8 classes')
        self.listWidget.insertItem(0, '2 classes')
        self.listWidget.currentRowChanged.connect(self.stackedWidget.setCurrentIndex)
        self.listWidget.setCurrentRow(0)

        self.twobox = QtWidgets.QButtonGroup(self)
        self.twobox.addButton(self.t1, 11)
        self.twobox.addButton(self.t2, 12)
        self.twobox.buttonClicked.connect(self.ccmode_two)
        self.t1.setChecked(True)

        self.eigbox = QtWidgets.QButtonGroup(self)
        self.eigbox.addButton(self.i1, 11)
        self.eigbox.addButton(self.i2, 12)
        self.eigbox.buttonClicked.connect(self.ccmode_eight)
        self.i1.setChecked(True)

        self.elebox = QtWidgets.QButtonGroup(self)  
        self.elebox.addButton(self.e1, 11)
        self.elebox.addButton(self.e2, 12)
        self.elebox.buttonClicked.connect(self.ccmode_eleven)
        self.e1.setChecked(True)

        self.b_21.clicked.connect(self.color21)
        self.b_22.clicked.connect(self.color22)

        self.b_81.clicked.connect(self.color81)
        self.b_82.clicked.connect(self.color82)
        self.b_83.clicked.connect(self.color83)
        
        self.b_111.clicked.connect(self.color111)
        self.b_112.clicked.connect(self.color112)
        self.b_113.clicked.connect(self.color113)
        self.b_114.clicked.connect(self.color114)
        self.b_115.clicked.connect(self.color115)
        
        self.b_21.setStyleSheet("background-color: blue")
        self.b_22.setStyleSheet("background-color: red")
        
        self.b_81.setStyleSheet("background-color: blue")
        self.b_82.setStyleSheet("background-color: red")
        self.b_83.setStyleSheet("background-color: green")

        self.b_111.setStyleSheet("background-color: blue")
        self.b_112.setStyleSheet("background-color: purple")
        self.b_113.setStyleSheet("background-color: yellow")
        self.b_114.setStyleSheet("background-color: cyan")
        self.b_115.setStyleSheet("background-color: green")

        self.diylist1 = ['blue', 'red']
        self.diylist2 = ['blue', 'red', 'green']
        self.diylist3 = ['blue', 'purple', 'cyan', 'yellow', 'green']
        self.hmap1 = [None, '//', '\\', 'XX']
        self.hmap2 = [None, '//', '\\', 'XX']

        self.cb_81.currentIndexChanged.connect(self.hatch81)
        self.cb_82.currentIndexChanged.connect(self.hatch82)
        self.cb_83.currentIndexChanged.connect(self.hatch83)
        self.cb_111.currentIndexChanged.connect(self.hatch111)
        self.cb_112.currentIndexChanged.connect(self.hatch112)
        self.cb_113.currentIndexChanged.connect(self.hatch113)

        self.a = 0.3
        self.b = 0.3
        self.c = 0.3

    def ccmode_two(self):
        if self.twobox.checkedId() == 11:
            self.colormode2 = 1
        else:
            self.colormode2 = 2

    def ccmode_eight(self):
        if self.eigbox.checkedId() == 11:
            self.colormode8 = 1
        else:
            self.colormode8 = 2

    def ccmode_eleven(self):
        if self.elebox.checkedId() == 11:
            self.colormode11 = 1
        else:
            self.colormode11 = 2

    def color21(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist1[0] = col.name()
            self.b_21.setStyleSheet("background-color:" + col.name())

    def color22(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist1[1] = col.name()
            self.b_22.setStyleSheet("background-color:" + col.name())

    def color81(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist2[0] = col.name()
            self.b_81.setStyleSheet("background-color:" + col.name())

    def color82(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist2[1] = col.name()
            self.b_82.setStyleSheet("background-color:" + col.name())

    def color83(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist2[2] = col.name()
            self.b_83.setStyleSheet("background-color:" + col.name())

    def color111(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist3[0] = col.name()
            self.b_111.setStyleSheet("background-color:" + col.name())

    def color112(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist3[1] = col.name()
            self.b_112.setStyleSheet("background-color:" + col.name())

    def color113(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist3[2] = col.name()
            self.b_113.setStyleSheet("background-color:" + col.name())

    def color114(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist3[3] = col.name()
            self.b_114.setStyleSheet("background-color:" + col.name())

    def color115(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist3[4] = col.name()
            self.b_115.setStyleSheet("background-color:" + col.name())

    def hatch81(self):
        self.hmap1[0] = self.cb_81.currentText()

    def hatch82(self):
        self.hmap1[1] = self.cb_82.currentText()

    def hatch83(self):
        self.hmap1[2] = self.cb_83.currentText()

    def hatch111(self):
        self.hmap2[0] = self.cb_111.currentText()

    def hatch112(self):
        self.hmap2[1] = self.cb_112.currentText()

    def hatch113(self):
        self.hmap2[2] = self.cb_113.currentText()

    def ccsetting(self):
        if self.colormode2 == 1:
            self.cmap1 = mpl.colors.ListedColormap(['blue', 'red'])
        else:
            self.cmap1 = mpl.colors.ListedColormap(self.diylist1)

        if self.colormode8 == 1:
            self.cmap2 = mpl.colors.ListedColormap(['blue', 'red', 'green'])
        else:
            self.cmap2 = mpl.colors.ListedColormap(self.diylist2)

        if self.colormode11 == 1:
            self.cmap3 = mpl.colors.ListedColormap(['blue', 'purple', 'cyan', 'yellow', 'green'])
        else:
            self.cmap3 = mpl.colors.ListedColormap(self.diylist3)

        if self.trans1.text():
            self.vtr1 = self.trans1.text()
        else:
            self.vtr1 = self.a
        if self.trans2.text():
            self.vtr2 = self.trans2.text()
        else:
            self.vtr2 = self.b
        if self.trans3.text():
            self.vtr3 = self.trans3.text()
        else:
            self.vtr3 = self.c

        if not (0<=self.vtr1<=1 and 0<=self.vtr1<=1 and 0<=self.vtr1<=1):
            QtWidgets.QMessageBox.information(self, 'Error', 'Out of range!')
            self.vtr1 = self.a
            self.vtr2 = self.b
            self.vtr3 = self.c

        try:
            self.vtr1 = float(self.vtr1)
            self.vtr2 = float(self.vtr2)
            self.vtr3 = float(self.vtr3)
        except Exception:
            QtWidgets.QMessageBox.information(self, 'Error', 'Input can only be a number!')
            self.vtr1 = self.a
            self.vtr2 = self.b
            self.vtr3 = self.c

        # self.classnr = self.cnrbox.currentIndex
        return self.cmap1, self.cmap2, self.cmap3, self.hmap1, self.hmap2, self.vtr1, self.vtr2, self.vtr3

    def getData(parent=None):
        dialog = Patches_window(parent)
        ok = dialog.exec_()
        cmap1, cmap2, cmap3, hmap1, hmap2, vtr1, vtr2, vtr3 = dialog.ccsetting()
        return cmap1, cmap2, cmap3, hmap1, hmap2, vtr1, vtr2, vtr3, ok