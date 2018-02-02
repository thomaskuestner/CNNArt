from PyQt5 import QtWidgets
from DataPre import Ui_DataPre

class DataPre_window(QtWidgets.QMainWindow,Ui_DataPre):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.finish.clicked.connect(self.close)
''''
    def __init__(self):
        super(Ui_DataPre, self).__init__()
        self.setupUi(self)
        self.finish.clicked.connect(self.close)     # in this format, the class name must be set as Ui_DataPre/DataPre
        '''