# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'editgray.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_GDialog(object):
    def setupUi(self, GDialog):
        GDialog.setObjectName("GDialog")
        GDialog.resize(230, 117)
        self.verticalLayout = QtWidgets.QVBoxLayout(GDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.minvv = QtWidgets.QLabel(GDialog)
        self.minvv.setObjectName("minvv")
        self.gridLayout.addWidget(self.minvv, 0, 0, 1, 1)
        self.maxvv = QtWidgets.QLabel(GDialog)
        self.maxvv.setObjectName("maxvv")
        self.gridLayout.addWidget(self.maxvv, 1, 0, 1, 1)
        self.minv = QtWidgets.QLineEdit(GDialog)
        self.minv.setObjectName("minv")
        self.gridLayout.addWidget(self.minv, 0, 1, 1, 1)
        self.maxv = QtWidgets.QLineEdit(GDialog)
        self.maxv.setObjectName("maxv")
        self.gridLayout.addWidget(self.maxv, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(GDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(GDialog)
        self.buttonBox.accepted.connect(GDialog.accept)
        self.buttonBox.rejected.connect(GDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(GDialog)

    def retranslateUi(self, GDialog):
        _translate = QtCore.QCoreApplication.translate
        GDialog.setWindowTitle(_translate("GDialog", "greyscale"))
        self.minvv.setText(_translate("GDialog", "min"))
        self.maxvv.setText(_translate("GDialog", "max"))

