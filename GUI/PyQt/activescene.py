from PyQt5 import QtCore, QtGui, QtWidgets

class Activescene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        super(Activescene, self).__init__(parent)

        #self.setBackgroundBrush(QtCore.Qt.black)
        #QtWidgets.QGraphicsScene.setSceneRect(self, -5000, -5000, 10000, 10000)



    # def mousePressEvent(self, event):
    #     if event.button() == QtCore.Qt.LeftButton:
    #         self.__prevMousePos = event.pos()
    #     else:
    #         super(Activescene, self).mousePressEvent(event)
    #
    # def mouseMoveEvent(self, event):
    #     if event.buttons() == QtCore.Qt.LeftButton:
    #         offset = self.__prevMousePos - event.pos()
    #         self.__prevMousePos = event.pos()
    #         self.verticalScrollBar().setValue(self.verticalScrollBar().value() + offset.y())
    #         self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + offset.x())
    #
    #     else:
    #         super(Activescene, self).mouseMoveEvent(event)