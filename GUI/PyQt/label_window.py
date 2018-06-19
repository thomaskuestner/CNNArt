from PyQt5 import QtWidgets
from editlabel import Ui_edit_label
import json
import matplotlib as mpl

class Label_window(QtWidgets.QDialog,Ui_edit_label):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.data = {'ok': 1}

        self.breset.clicked.connect(self.resetLabel)
        self.b1.clicked.connect(self.color1)
        self.b2.clicked.connect(self.color2)
        self.b3.clicked.connect(self.color3)
        self.b4.clicked.connect(self.color4)
        self.b5.clicked.connect(self.color5)
        self.b6.clicked.connect(self.color6)
        self.b7.clicked.connect(self.color7)
        self.b8.clicked.connect(self.color8)

        self.bsetpath.clicked.connect(self.setPath)

        self.setInit()

    def setInit(self):
        with open('editlabel.json', 'r') as json_data:
            self.infos = json.load(json_data)
            self.namelist = self.infos['names']
            self.colorlist = self.infos['colors']
            self.pathROI = self.infos['path'][0]
        self.b1.setStyleSheet('background-color:' + self.colorlist[0])
        self.b2.setStyleSheet('background-color:' + self.colorlist[1])
        self.b3.setStyleSheet('background-color:' + self.colorlist[2])
        self.b4.setStyleSheet('background-color:' + self.colorlist[3])
        self.b5.setStyleSheet('background-color:' + self.colorlist[4])
        self.b6.setStyleSheet('background-color:' + self.colorlist[5])
        self.b7.setStyleSheet('background-color:' + self.colorlist[6])
        self.b8.setStyleSheet('background-color:' + self.colorlist[7])

        self.lineEdit.setText(self.namelist[0])
        self.lineEdit_2.setText(self.namelist[1])
        self.lineEdit_3.setText(self.namelist[2])
        self.lineEdit_4.setText(self.namelist[3])
        self.lineEdit_5.setText(self.namelist[4])
        self.lineEdit_6.setText(self.namelist[5])
        self.lineEdit_7.setText(self.namelist[6])
        self.lineEdit_8.setText(self.namelist[7])

        self.pathline.setText(self.pathROI)

    def resetLabel(self):
        labels = {'names': ['label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8'],
                  'colors': ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink'],
                  'path':[self.pathROI]}
        wFile = json.dumps(labels)
        with open('editlabel.json', 'w') as json_data:
            json_data.write(wFile)
        self.setInit()

    def setPath(self):
        self.pathROI = QtWidgets.QFileDialog.getExistingDirectory(self, "choose the image to view", "")
        self.pathline.setText(self.pathROI)

    def color1(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.colorlist[0] = col.name()
            self.b1.setStyleSheet('background-color:' + col.name())

    def color2(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.colorlist[1] = col.name()
            self.b2.setStyleSheet('background-color:' + col.name())

    def color3(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.colorlist[2] = col.name()
            self.b3.setStyleSheet('background-color:' + col.name())

    def color4(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.colorlist[3] = col.name()
            self.b4.setStyleSheet('background-color:' + col.name())

    def color5(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.colorlist[4] = col.name()
            self.b5.setStyleSheet('background-color:' + col.name())

    def color6(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.colorlist[5] = col.name()
            self.b6.setStyleSheet('background-color:' + col.name())

    def color7(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.colorlist[6] = col.name()
            self.b7.setStyleSheet('background-color:' + col.name())

    def color8(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.colorlist[7] = col.name()
            self.b8.setStyleSheet('background-color:' + col.name())

    def returnInfo(self):
        with open('editlabel.json', 'r') as json_data:
            self.infos = json.load(json_data)

            self.namelist[0] = self.lineEdit.text()
            self.namelist[1] = self.lineEdit_2.text()
            self.namelist[2] = self.lineEdit_3.text()
            self.namelist[3] = self.lineEdit_4.text()
            self.namelist[4] = self.lineEdit_5.text()
            self.namelist[5] = self.lineEdit_6.text()
            self.namelist[6] = self.lineEdit_7.text()
            self.namelist[7] = self.lineEdit_8.text()


            self.infos['names'] = self.namelist
            self.infos['colors'] = self.colorlist
            self.infos['path'][0] = self.pathROI
        with open('editlabel.json', 'w') as json_data:
            json_data.write(json.dumps(self.infos))
        return self.namelist, self.colorlist, self.pathROI

    def getData(parent=None):
        dialog = Label_window(parent)
        ok = dialog.exec_()
        namelist, colorlist, pathROI = dialog.returnInfo()
        return namelist, colorlist, pathROI, ok