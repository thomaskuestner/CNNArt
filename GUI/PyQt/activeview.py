from PyQt5 import QtCore, QtGui, QtWidgets


class Activeview(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(Activeview, self).__init__(parent)

        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setMinimumSize(500, 500)

        #self.setAutoFillBackground(True) #

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.__prevMousePos = event.pos()

        elif event.button() == QtCore.Qt.RightButton:
            self._dragPos = event.pos()
        else:
            super(Activeview, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            offset = self.__prevMousePos - event.pos()
            self.__prevMousePos = event.pos()
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + offset.y())
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + offset.x())

        elif event.buttons() == QtCore.Qt.RightButton:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            factor = 1.2
            if diff.y() > 0:  # x() is left/right , still not for sure
                factor = 1.0 / factor
            self.scale(factor, factor)
        else:
            super(Activeview, self).mouseMoveEvent(event)

    ''' old test version
        self._isPanning = True
        self._mousePressed = False

    def mousePressEvent(self,  event):
        if event.button() == QtCore.Qt.RightButton:
            self._mousePressed = True
            if self._isPanning:
                #self.setCursor(QtCore.Qt.ClosedHandCursor)
                self._dragPos = event.pos()
                event.accept()
            else:
                super(Activeview, self).mousePressEvent(event)

        elif event.button() == QtCore.Qt.LeftButton:
            if self._isPanning:
                self.__prevMousePos = event.pos()
            else:
                super(Activeview, self).mousePressEvent(event)


    def mouseMoveEvent(self, event):
        if self._mousePressed & QtCore.Qt.RightButton:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            factor = 1.2
            if diff.x() > 0:
                factor = 1.0 / factor
            self.scale(factor, factor)
            event.accept()

        elif self._mousePressed & QtCore.Qt.LeftButton:
            if self._isPanning:
                offset = self.__prevMousePos - event.pos()
                self.__prevMousePos = event.pos()

                self.verticalScrollBar().setValue(self.verticalScrollBar().value() + offset.y())
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + offset.x())
            else:
                super(Activeview, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:

            if event.modifiers() & Qt.ControlModifier:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self._isPanning = False
                self.setCursor(Qt.ArrowCursor)
            self._mousePressed = False

        elif event.button() == QtCore.Qt.LeftButton:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                self.setCursor(QtCore.Qt.OpenHandCursor)
            else:
                self._isPanning = False
                self.setCursor(QtCore.Qt.ArrowCursor)
            self._mousePressed = False
            super(Activeview, self).mouseReleaseEvent(event)

    def wheelEvent(self, event):
        zoom_in = 1.15
        zoom_out = 1.0 / zoom_in
        if event.angleDelta().y() > 0:
            self.scale(zoom_in, zoom_in)
        else:
            self.scale(zoom_out, zoom_out)
    '''