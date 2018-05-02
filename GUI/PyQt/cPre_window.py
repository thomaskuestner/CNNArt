from PyQt5 import QtWidgets
from PreColor import Ui_PreColor
import json
import matplotlib as mpl

class cPre_window(QtWidgets.QDialog, Ui_PreColor):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        with open('colors0.json', 'r') as json_data:
            self.dcolors = json.load(json_data)

            self.diylist1 = self.dcolors['class2']['colors']
            self.diylist3 = self.dcolors['class11']['colors']
            self.hmap1 = self.dcolors['class8']['hatches']
            self.hmap2 = self.dcolors['class11']['hatches']
            self.dev1 = self.dcolors['class2']['trans'][0]
            self.dev2 = self.dcolors['class11']['trans'][0]

        for n, i in enumerate(self.hmap1):
            if i == '\\\\':
                self.hmap1[n] = '\\'

        for n, i in enumerate(self.hmap2):
            if i == '\\\\':
                self.hmap2[n] = '\\'

        self.hlist = ['//', '\\', 'XX', '--', '**']

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.data = {'ok': 1}

        self.listWidget.insertItem(0, '11 classes')
        self.listWidget.insertItem(0, '8 classes')
        self.listWidget.insertItem(0, '2 classes')
        self.listWidget.currentRowChanged.connect(self.stackedWidget.setCurrentIndex)
        self.listWidget.setCurrentRow(0)

        self.b_21.clicked.connect(self.color21)
        self.b_22.clicked.connect(self.color22)

        self.b_111.clicked.connect(self.color111)
        self.b_112.clicked.connect(self.color112)
        self.b_113.clicked.connect(self.color113)
        self.b_114.clicked.connect(self.color114)
        self.b_115.clicked.connect(self.color115)

        self.b_21.setStyleSheet('background-color:' + self.diylist1[0])
        self.b_22.setStyleSheet('background-color:' + self.diylist1[1])

        self.b_111.setStyleSheet('background-color:' + self.diylist3[0])
        self.b_112.setStyleSheet('background-color:' + self.diylist3[1])
        self.b_113.setStyleSheet('background-color:' + self.diylist3[2])
        self.b_114.setStyleSheet('background-color:' + self.diylist3[3])
        self.b_115.setStyleSheet('background-color:' + self.diylist3[4])

        self.cb_81.currentIndexChanged.connect(self.hatch81)
        self.cb_82.currentIndexChanged.connect(self.hatch82)
        self.cb_83.currentIndexChanged.connect(self.hatch83)
        self.cb_111.currentIndexChanged.connect(self.hatch111)
        self.cb_112.currentIndexChanged.connect(self.hatch112)
        self.cb_113.currentIndexChanged.connect(self.hatch113)

        self.cb_81.setCurrentIndex(self.hlist.index(self.hmap1[1]))
        self.cb_82.setCurrentIndex(self.hlist.index(self.hmap1[2]))
        self.cb_83.setCurrentIndex(self.hlist.index(self.hmap1[3]))
        self.cb_111.setCurrentIndex(self.hlist.index(self.hmap2[1]))
        self.cb_112.setCurrentIndex(self.hlist.index(self.hmap2[2]))
        self.cb_113.setCurrentIndex(self.hlist.index(self.hmap2[3]))

        self.trans1.setText(str(self.dev1))
        self.trans3.setText(str(self.dev2))

    def color21(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist1[0] = col.name()
            self.b_21.setStyleSheet('background-color:' + col.name())

    def color22(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist1[1] = col.name()
            self.b_22.setStyleSheet('background-color:' + col.name())

    def color111(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist3[0] = col.name()

            self.b_111.setStyleSheet('background-color:' + col.name())

    def color112(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist3[1] = col.name()
            self.b_112.setStyleSheet('background-color:' + col.name())

    def color113(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist3[2] = col.name()
            self.b_113.setStyleSheet('background-color:' + col.name())

    def color114(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist3[3] = col.name()
            self.b_114.setStyleSheet('background-color:' + col.name())

    def color115(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.diylist3[4] = col.name()
            self.b_115.setStyleSheet('background-color:' + col.name())

    def hatch81(self):
        self.hmap1[1] = self.cb_81.currentText()

    def hatch82(self):
        self.hmap1[2] = self.cb_82.currentText()

    def hatch83(self):
        self.hmap1[3] = self.cb_83.currentText()

    def hatch111(self):
        self.hmap2[1] = self.cb_111.currentText()

    def hatch112(self):
        self.hmap2[2] = self.cb_112.currentText()

    def hatch113(self):
        self.hmap2[3] = self.cb_113.currentText()

    def ccsetting(self):

        if self.trans1.text().isdigit():
            self.vtr1 = self.trans1.text()
        else:  # self.vtr1 = float(self.trans1.text())
            try:
                self.vtr1 = float(self.trans1.text())
            except Exception:
                QtWidgets.QMessageBox.information(self, 'Error', 'Input can only be a number!')
                self.vtr1 = self.dev1

        self.cmap1 = mpl.colors.ListedColormap(self.diylist1)

        if self.trans3.text().isdigit():
            self.vtr3 = self.trans3.text()
        else:  # self.vtr3 = float(self.trans3.text())
            try:
                self.vtr3 = float(self.trans3.text())
            except Exception:
                QtWidgets.QMessageBox.information(self, 'Error', 'Input can only be a number!')
                self.vtr3 = self.dev2

        self.cmap3 = mpl.colors.ListedColormap(self.diylist3)

        if not (0 <= self.vtr1 <= 1 and 0 <= self.vtr3 <= 1):
            QtWidgets.QMessageBox.information(self, 'Error', 'Out of range!')
            self.vtr1 = self.dev1
            self.vtr3 = self.dev2

        with open('colors0.json', 'r') as json_data:
            self.colors = json.load(json_data)

            self.colors['class2']['colors'] = self.diylist1
            self.colors['class11']['colors'] = self.diylist3
            self.colors['class8']['hatches'] = self.hmap1
            self.colors['class11']['hatches'] = self.hmap2
            self.colors['class2']['trans'][0] = self.vtr1
            self.colors['class11']['trans'][0] = self.vtr3

        with open('colors0.json', 'w') as json_data:
            json_data.write(json.dumps(self.colors))

        return self.cmap1, self.cmap3, self.hmap1, self.hmap2, self.vtr1, self.vtr3

    def getData(parent=None):
        dialog = cPre_window(parent)
        ok = dialog.exec_()
        cmap1, cmap3, hmap1, hmap2, vtr1, vtr3 = dialog.ccsetting()
        return cmap1, cmap3, hmap1, hmap2, vtr1, vtr3, ok