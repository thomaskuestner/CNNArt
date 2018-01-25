from PyQt5 import QtCore, QtGui, QtWidgets

class Activescene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        super(Activescene, self).__init__(parent)