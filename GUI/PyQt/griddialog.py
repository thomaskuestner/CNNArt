# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'griddialog.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.formLayoutWidget = QtWidgets.QWidget(Dialog)
        self.formLayoutWidget.setGeometry(QtCore.QRect(50, 20, 321, 171))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.formLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.combo_layline = QtWidgets.QComboBox(self.formLayoutWidget)
        self.combo_layline.setObjectName("combo_layline")
        self.combo_layline.addItem("")
        self.combo_layline.addItem("")
        self.combo_layline.addItem("")
        self.combo_layline.addItem("")
        self.gridLayout.addWidget(self.combo_layline, 1, 1, 1, 1)
        self.combo_laycolumn = QtWidgets.QComboBox(self.formLayoutWidget)
        self.combo_laycolumn.setObjectName("combo_laycolumn")
        self.combo_laycolumn.addItem("")
        self.combo_laycolumn.addItem("")
        self.combo_laycolumn.addItem("")
        self.combo_laycolumn.addItem("")
        self.gridLayout.addWidget(self.combo_laycolumn, 2, 1, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Choose the layout:"))
        self.label_3.setText(_translate("Dialog", "Colomns"))
        self.label_2.setText(_translate("Dialog", "Lines"))
        self.combo_layline.setItemText(0, _translate("Dialog", "1"))
        self.combo_layline.setItemText(1, _translate("Dialog", "2"))
        self.combo_layline.setItemText(2, _translate("Dialog", "3"))
        self.combo_layline.setItemText(3, _translate("Dialog", "4"))
        self.combo_laycolumn.setItemText(0, _translate("Dialog", "1"))
        self.combo_laycolumn.setItemText(1, _translate("Dialog", "2"))
        self.combo_laycolumn.setItemText(2, _translate("Dialog", "3"))
        self.combo_laycolumn.setItemText(3, _translate("Dialog", "4"))

