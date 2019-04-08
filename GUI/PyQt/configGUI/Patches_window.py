import matplotlib
import pandas

from PyQt5 import QtWidgets

import json
from PyQt5.QtWidgets import QColorDialog

from configGUI.cPatches import Ui_Patches

class Patches_window(QtWidgets.QDialog,Ui_Patches):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        with open('configGUI/colors0.json', 'r') as json_data:
            self.dcolors = json.load(json_data)
            self.diylist1d = self.dcolors['class2']['colors']
            self.diylist3d = self.dcolors['class11']['colors']
            self.hmap1d = self.dcolors['class8']['hatches']
            self.hmap2d = self.dcolors['class11']['hatches']
            self.dev1d = self.dcolors['class2']['trans'][0]
            self.dev2d = self.dcolors['class11']['trans'][0]
            self.colormapsd = self.dcolors['classes']['colors']
            self.transd = self.dcolors['classes']['trans'][0]

        with open('configGUI/colors1.json', 'r') as json_data:
            self.colors = json.load(json_data)
            self.diylist1 = self.colors['class2']['colors']
            self.diylist3 = self.colors['class11']['colors']
            self.hmap1 = self.colors['class8']['hatches']
            self.hmap2 = self.colors['class11']['hatches']
            self.dev1 = self.colors['class2']['trans'][0]
            self.dev2 = self.colors['class11']['trans'][0]
            self.trans = self.colors['classes']['trans'][0]
            self.colormaps = []
            self.classes = []
        self.patch_color_df = pandas.read_csv('configGUI/patch_color.csv')
        count = self.patch_color_df['class'].count()
        for i in range(count):
            self.colormaps.append(self.patch_color_df.iloc[i]['color'])
            self.classes.append(self.patch_color_df.iloc[i]['class'])

        for n, i in enumerate(self.hmap1):
            if i == '\\\\':
                self.hmap1[n] = '\\'

        for n, i in enumerate(self.hmap2):
            if i == '\\\\':
                self.hmap2[n] = '\\'

        self.hlist = ['//', '\\', 'XX', '--', '**']

        self.colormode2 = 1
        self.colormode8 = 1
        self.colormode11 = 1
        self.colormodem = 1

        self.color_dialog = QColorDialog()

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.data = {'ok': 1}
        self.labellist = []
        self.buttonlist = []
        count = len(self.colormaps)
        if count>0:
            self.listWidget.insertItem(0, '%d classes' % count)
            for i in range(count):
                label = QtWidgets.QRadioButton('class: %s' % self.classes[i])
                self.labellist.append(label)
                button = QtWidgets.QPushButton()
                button.setText("")
                row = i + 3
                self.gridLayout_4.addWidget(label, row, 0, 1, 1)
                self.gridLayout_4.addWidget(button, row, 1, 1, 1)
                button.setStyleSheet('background-color:' + self.colormaps[i])
                button.clicked.connect(self.colorm)
                self.buttonlist.append(button)
            self.labellist[0].setChecked(True)
        self.label_trans = QtWidgets.QLabel('transparency 0-1')
        self.text_trans =  QtWidgets.QLineEdit('0.3')
        row = row + 1
        self.gridLayout_4.addWidget(self.label_trans, row, 0, 1, 1)
        self.gridLayout_4.addWidget(self.text_trans, row, 1, 1, 1)

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

        self.multibox = QtWidgets.QButtonGroup(self)
        self.multibox.addButton(self.radioButton, 11)
        self.multibox.addButton(self.radioButton_2, 12)
        self.multibox.buttonClicked.connect(self.ccmode_multi)
        self.radioButton.setChecked(True)

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

        self.vtr1 = self.dev1d
        self.vtr3 = self.dev2d
        self.trans1.setText(str(self.dev1d))
        self.trans3.setText(str(self.dev2d))

        self.label_classes.setText('%d classes' % count)

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

    def ccmode_multi(self):
        if self.multibox.checkedId() == 11:
            self.colormodem = 1
        else:
            self.colormodem = 2

    def colorm(self):
        for i in range(len(self.labellist)):
            if self.labellist[i].isChecked():
                col = self.color_dialog.getColor()
                self.colormaps[i] = col.name()
                self.buttonlist[i].setStyleSheet('background-color:' + col.name())

    def color21(self):
        col = self.color_dialog.getColor()
        self.diylist1[0] = col.name()
        self.b_21.setStyleSheet('background-color:' + col.name())

    def color22(self):
        col = self.color_dialog.getColor()
        self.diylist1[1] = col.name()
        self.b_22.setStyleSheet('background-color:' + col.name())

    def color111(self):
        col = self.color_dialog.getColor()
        self.diylist3[0] = col.name()
        self.b_111.setStyleSheet('background-color:' + col.name())

    def color112(self):
        col = self.color_dialog.getColor()
        self.diylist3[1] = col.name()
        self.b_112.setStyleSheet('background-color:' + col.name())

    def color113(self):
        col = self.color_dialog.getColor()
        self.diylist3[2] = col.name()
        self.b_113.setStyleSheet('background-color:' + col.name())

    def color114(self):
        col = self.color_dialog.getColor()
        if col.isValid():
            self.diylist3[3] = col.name()
            self.b_114.setStyleSheet('background-color:' + col.name())

    def color115(self):
        col = self.color_dialog.getColor()
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
        if self.colormode2 == 2:
            if self.trans1.text().isdigit():
                self.vtr1 = self.trans1.text()
            else:#self.vtr1 = float(self.trans1.text())
                try:
                    self.vtr1 = float(self.trans1.text())
                except Exception:
                    QtWidgets.QMessageBox.information(self, 'Error', 'Input can only be a number!')
                    self.vtr1 = self.dev1d

            self.cmap1 = self.diylist1
        else:
            self.vtr1 =self.dev1d
            self.cmap1 = self.diylist1d

        if self.colormode11 == 2:
            if self.trans3.text().isdigit():
                self.vtr3 = self.trans3.text()
            else:#self.vtr3 = float(self.trans3.text())
                try:
                    self.vtr3 = float(self.trans3.text())
                except Exception:
                    QtWidgets.QMessageBox.information(self, 'Error', 'Input can only be a number!')
                    self.vtr3 = self.dev2d

            self.cmap3 = self.diylist3
        else:
            self.vtr1 =self.dev2d
            self.cmap3 = self.diylist3d
            self.hmap2 = self.hmap2d

        if self.colormode8 == 1:
            self.hmap1 = self.hmap1d

        if self.colormodem == 1:
            self.cmaps = self.colormapsd
            for i in range(len(self.labellist)):
                self.patch_color_df.loc[i, 'color'] = matplotlib.colors.to_hex(self.colormapsd[i])
            self.vtrs = self.transd
        else:
            self.cmaps = self.colormaps
            self.patch_color_df = pandas.read_csv('configGUI/patch_color.csv')
            for i in range(len(self.labellist)):
                self.patch_color_df.loc[i, 'color'] = matplotlib.colors.to_hex(self.colormaps[i])
            self.patch_color_df.to_csv('configGUI/patch_color.csv', index=False)
            if self.text_trans.text().isdigit():
                self.vtrs = self.text_trans.text()
            else:
                try:
                    self.vtrs = float(self.text_trans.text())
                except Exception:
                    QtWidgets.QMessageBox.information(self, 'Error', 'Input can only be a number!')
                    self.vtrs = self.transd

        if not (0<=self.vtr1<=1 and 0<=self.vtr3<=1):
            QtWidgets.QMessageBox.information(self, 'Error', 'Out of range!')
            self.vtr1 = self.dev1d
            self.vtr3 = self.dev2d

        with open('configGUI/colors1.json', 'r') as json_data:
            self.colors = json.load(json_data)

            self.colors['class2']['colors'] = self.diylist1
            self.colors['class11']['colors'] = self.diylist3
            self.colors['class8']['hatches'] = self.hmap1
            self.colors['class11']['hatches'] = self.hmap2
            self.colors['class2']['trans'][0] = self.vtr1
            self.colors['class11']['trans'][0] = self.vtr3
            self.colors['classes']['colors'] = self.colormaps
            self.colors['classes']['trans'][0] = self.vtrs

        with open('configGUI/colors1.json', 'w') as json_data:
            json_data.write(json.dumps(self.colors))
        return self.cmap1, self.cmap3, self.hmap1, self.hmap2, self.vtr1, self.vtr3, self.cmaps, self.vtrs

    def getData(parent=None):
        dialog = Patches_window(parent)
        ok = dialog.exec_()
        cmap1, cmap3, hmap1, hmap2, vtr1, vtr3, cmaps, vtrs = dialog.ccsetting()
        return cmap1, cmap3, hmap1, hmap2, vtr1, vtr3, cmaps, vtrs, ok