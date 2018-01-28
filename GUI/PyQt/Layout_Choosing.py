from PyQt5 import QtWidgets
from griddialog import*
from activeview import*

class Layout_window(QtWidgets.QDialog,Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        #self.Finish.clicked.connect(self.setlayout)
        #self.Finish.clicked.connect(self.close)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.data = {'ok': 1}
    def setlayout(self):
        self.layoutlines = self.combo_layline.currentIndex() + 1
        self.layoutcolumns = self.combo_laycolumn.currentIndex() + 1
        return self.layoutlines, self.layoutcolumns

    def getData(parent = None): # static function for communication
        dialog = Layout_window(parent)
        ok = dialog.exec_()
        layoutlines, layoutcolumns = dialog.setlayout()
        return layoutlines, layoutcolumns, ok
