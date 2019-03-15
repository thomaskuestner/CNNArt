from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib import colors

from configGUI.LabelChoose import Ui_Dialog
from configGUI.colorDialog import ColorDialog
from configGUI.lib import labelValidator


class LabelDialog(Ui_Dialog):

    def __init__(self, parent=None, listItem=None):

        super(LabelDialog, self).__init__(parent)
        self.setupUi(self)
        self.lineEdit.setValidator(labelValidator())
        self.lineEdit.editingFinished.connect(self.postProcess)
        self.colorDialog = ColorDialog(parent=self)
        self.pushButton.clicked.connect(self.chooseColor)
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Cancel).setText('Close')
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setText('Apply')
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).clicked.connect(self.validate)
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Cancel).clicked.connect(self.invalidate)

        if listItem is not None and len(listItem) > 0:
            for item in listItem:
                self.listWidget.addItem(item)
            self.listWidget.itemClicked.connect(self.listItemClick)
            self.listWidget.itemDoubleClicked.connect(self.listItemDoubleClick)

        self.labelName = ''

    def validate(self):
        if self.lineEdit.text().strip():
            # self.text
            self.accept()

    def invalidate(self):
        self.reject()

    def postProcess(self):
        self.lineEdit.setText(self.lineEdit.text())

    def popUp(self, text='', move=True):
        if text is not None:
            self.lineEdit.setText(text)
            self.lineEdit.setSelection(0, len(text))
            self.labelName = text
        self.lineEdit.setFocus(Qt.PopupFocusReason)
        if move:
            self.move(QCursor.pos())
        return self.lineEdit.text() if self.exec_() else None

    def listItemClick(self, tQListWidgetItem):
        try:
            text = tQListWidgetItem.text().trimmed()
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            text = tQListWidgetItem.text().strip()
        self.lineEdit.setText(text)
        
    def listItemDoubleClick(self, tQListWidgetItem):
        self.listItemClick(tQListWidgetItem)
        self.validate()

    def chooseColor(self):
        color = self.colorDialog.getColor(u'Choose label color', default='b')
        return color

    def getColor(self):
        c = self.colorDialog.currentColor()
        color = colors.to_hex(c.name(),keep_alpha=True)
        return color

    def getName(self):
        return self.labelName

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "Choose label"))

# import configGUI.resrc_rc
#
# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     labelDialog = QtWidgets.QDialog()
#     labelDialog.show()
#     sys.exit(app.exec_())