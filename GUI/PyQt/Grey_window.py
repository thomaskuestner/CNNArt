from PyQt5 import QtWidgets
from editgray import*

class grey_window(QtWidgets.QDialog, Ui_GDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.data = {'ok': 1}

        self.a = 2094
        self.b = 0

    def setValue(self):
        if self.maxv.text() and self.minv.text():
            self.maxvalue = self.maxv.text()
            self.minvalue = self.minv.text()
            try:
                self.maxvalue = int(self.maxvalue)
                self.minvalue = int(self.minvalue)
                return self.maxvalue, self.minvalue
            except Exception:
                QtWidgets.QMessageBox.information(self, 'Error', 'Input can only be a number')
                return self.a, self.b
        else:
            return self.a, self.b

    def getData(parent = None): # static function for communication
        dialog = grey_window(parent)
        ok = dialog.exec_()
        maxvalue, minvalue = dialog.setValue()
        return maxvalue, minvalue, ok
