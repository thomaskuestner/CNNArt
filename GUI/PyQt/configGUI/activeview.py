from PyQt5 import QtCore, QtGui, QtWidgets

class Activeview(QtWidgets.QGraphicsView):
    zooming_data = QtCore.pyqtSignal(float)
    zoom_link = QtCore.pyqtSignal(float)
    move_link = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super(Activeview, self).__init__(parent)
        # self.setStyleSheet("border: 0px")
        self.selfhandle = False
        self.left = 1
        # self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        # self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        # self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        # self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag) #
        self.setMinimumSize(500, 500)

        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        self.setBackgroundBrush(brush)
        #self.setAutoFillBackground(True)

        self.zoomdata = 1
        self.movelist = []
        self.movelist.append(0)
        self.movelist.append(0)

    # def updateCanvas(self, data):
    #     self.__dyCanvas = data
    #
    # def getCanvas(self):
    #     return self.__dyCanvas

    def stopMove(self, n):
        if n == 0:
            self.left = 1
        else:
            self.left = 0

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.left == 1:
                self.__prevMousePos = event.pos()
                self.selfhandle = True
            else:
                self.selfhandle = False
        if event.button() == QtCore.Qt.RightButton:
            self._dragPos = event.pos()
            self.selfhandle = True
        super(Activeview, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            if self.left == 1:
                offset = self.__prevMousePos - event.pos()
                self.__prevMousePos = event.pos()
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() + offset.y())
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + offset.x())
                self.movelist[0] = offset.y()
                self.movelist[1] = offset.x()
                self.move_link.emit(self.movelist)
        if event.buttons() == QtCore.Qt.RightButton:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            factor = 1.15
            if diff.y() > 0:  # x() is left/right
                factor = 1.0 / factor
            self.scale(factor, factor)
            self.zoomdata = self.zoomdata * factor
            self.zooming_data.emit(round(self.zoomdata, 2))
            self.zoom_link.emit(factor)
        super(Activeview, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.left == 1:
                self.selfhandle = False
        if event.button() == QtCore.Qt.RightButton:
            self.selfhandle = False
        super(Activeview, self).mouseReleaseEvent(event)

    def zoomback(self):
        factor = 1/self.zoomdata
        self.scale(factor, factor)
        self.zoomdata = 1

    def linkedZoom(self, factor):
        if self.selfhandle == False:
            self.scale(factor, factor)
            self.zoomdata = self.zoomdata * factor
            self.zooming_data.emit(round(self.zoomdata, 2))
        else:
            pass

    def linkedMove(self, movelist):
        if self.selfhandle == False:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + movelist[0])
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + movelist[1])
        else:
            pass