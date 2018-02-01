from PyQt5 import QtWidgets
from setCNN import Ui_setCNN

class CNN_window(QtWidgets.QMainWindow,Ui_setCNN):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.start.clicked.connect(self.close)
'''
    def __init__(self):
        super(Ui_setCNN, self).__init__()
        self.setupUi(self)
        self.start.clicked.connect(self.close)                  # in this format, the class name must be set as Ui_setCNN/setCNN
'''