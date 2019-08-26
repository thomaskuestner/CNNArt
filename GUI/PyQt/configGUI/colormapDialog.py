import logging
import sys

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, qVersion, QSize
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QWidget, QHBoxLayout, QComboBox, QPushButton, QCheckBox, QButtonGroup, \
    QLabel, QLineEdit, QApplication, QDialogButtonBox
from silx.gui.plot import PlotWidget
QTVERSION = qVersion()
_logger = logging.getLogger(__name__)
class MyQLineEdit(QLineEdit):
    def __init__(self,parent=None,name=""):
        QLineEdit.__init__(self,parent)

    def focusInEvent(self,event):
        self.setPaletteBackgroundColor(QColor('yellow'))

    def focusOutEvent(self,event):
        self.setPaletteBackgroundColor(QColor('white'))
        self.returnPressed[()].emit()

    def setPaletteBackgroundColor(self, color):
        _logger.debug("setPalettebackgroundColor not implemented yet")
        pass

_COLORMAP_NAMES = ('Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'YlGn')

_COLOR_NAMES = ('gray', 'purple', 'blue', 'green', 'orange', 'red',
                'gold', 'darkorange', 'darkmagenta', 'darkred', 'darkviolet',
                'cyan', 'darkslateblue', 'greenyellow', 'dodgerblu', 'chartreuse')

class ColormapDialog(QDialog):

    sigColormapChanged = pyqtSignal(object)
    def __init__(self, parent=None, name="Colormap Dialog"):
        QDialog.__init__(self, parent)
        self.setWindowTitle(name)
        self.title = name


        self.colormapList = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

        # histogramData is tupel(bins, counts)
        self.histogramData = None

        # default values
        self.dataMin   = -10
        self.dataMax   = 10
        self.minValue  = 0
        self.maxValue  = 1

        self.colormapIndex  = 2
        self.colormapType   = 0

        self.autoscale   = False
        self.autoscale90 = False
        # main layout
        vlayout = QVBoxLayout(self)
        vlayout.setContentsMargins(10, 10, 10, 10)
        vlayout.setSpacing(0)

        # layout 1 : -combo to choose colormap
        #            -autoscale button
        #            -autoscale 90% button
        hbox1    = QWidget(self)
        hlayout1 = QHBoxLayout(hbox1)
        vlayout.addWidget(hbox1)
        hlayout1.setContentsMargins(0, 0, 0, 0)
        hlayout1.setSpacing(10)

        # combo
        self.combo = QComboBox(hbox1)
        for colormap in self.colormapList:
            self.combo.addItem(colormap)
        self.combo.activated[int].connect(self.colormapChange)
        hlayout1.addWidget(self.combo)

        # autoscale
        self.autoScaleButton = QPushButton("Autoscale", hbox1)
        self.autoScaleButton.setCheckable(True)
        self.autoScaleButton.setAutoDefault(False)
        self.autoScaleButton.toggled[bool].connect(self.autoscaleChange)
        hlayout1.addWidget(self.autoScaleButton)

        # autoscale 90%
        self.autoScale90Button = QPushButton("Autoscale 90%", hbox1)
        self.autoScale90Button.setCheckable(True)
        self.autoScale90Button.setAutoDefault(False)

        self.autoScale90Button.toggled[bool].connect(self.autoscale90Change)
        hlayout1.addWidget(self.autoScale90Button)

        # hlayout
        hbox0    = QWidget(self)
        self.__hbox0 = hbox0
        hlayout0 = QHBoxLayout(hbox0)
        hlayout0.setContentsMargins(0, 0, 0, 0)
        hlayout0.setSpacing(0)
        vlayout.addWidget(hbox0)
        #hlayout0.addStretch(10)

        self.buttonGroup = QButtonGroup()
        g1 = QCheckBox(hbox0)
        g1.setText("Linear")
        g2 = QCheckBox(hbox0)
        g2.setText("Logarithmic")
        g3 = QCheckBox(hbox0)
        g3.setText("Gamma")
        self.buttonGroup.addButton(g1, 0)
        self.buttonGroup.addButton(g2, 1)
        self.buttonGroup.addButton(g3, 2)
        self.buttonGroup.setExclusive(True)
        if self.colormapType == 1:
            self.buttonGroup.button(1).setChecked(True)
        elif self.colormapType == 2:
            self.buttonGroup.button(2).setChecked(True)
        else:
            self.buttonGroup.button(0).setChecked(True)
        hlayout0.addWidget(g1)
        hlayout0.addWidget(g2)
        hlayout0.addWidget(g3)
        vlayout.addWidget(hbox0)
        self.buttonGroup.buttonClicked[int].connect(self.buttonGroupChange)
        vlayout.addSpacing(20)

        hboxlimits = QWidget(self)
        hboxlimitslayout = QHBoxLayout(hboxlimits)
        hboxlimitslayout.setContentsMargins(0, 0, 0, 0)
        hboxlimitslayout.setSpacing(0)

        self.slider = None

        vlayout.addWidget(hboxlimits)

        vboxlimits = QWidget(hboxlimits)
        vboxlimitslayout = QVBoxLayout(vboxlimits)
        vboxlimitslayout.setContentsMargins(0, 0, 0, 0)
        vboxlimitslayout.setSpacing(0)
        hboxlimitslayout.addWidget(vboxlimits)

        # hlayout 2 : - min label
        #             - min texte
        hbox2    = QWidget(vboxlimits)
        self.__hbox2 = hbox2
        hlayout2 = QHBoxLayout(hbox2)
        hlayout2.setContentsMargins(0, 0, 0, 0)
        hlayout2.setSpacing(0)
        #vlayout.addWidget(hbox2)
        vboxlimitslayout.addWidget(hbox2)
        hlayout2.addStretch(10)

        self.minLabel  = QLabel(hbox2)
        self.minLabel.setText("Minimum")
        hlayout2.addWidget(self.minLabel)

        hlayout2.addSpacing(5)
        hlayout2.addStretch(1)
        self.minText  = MyQLineEdit(hbox2)
        self.minText.setFixedWidth(150)
        self.minText.setAlignment(QtCore.Qt.AlignRight)
        self.minText.returnPressed[()].connect(self.minTextChanged)
        hlayout2.addWidget(self.minText)

        # hlayout 3 : - min label
        #             - min text
        hbox3    = QWidget(vboxlimits)
        self.__hbox3 = hbox3
        hlayout3 = QHBoxLayout(hbox3)
        hlayout3.setContentsMargins(0, 0, 0, 0)
        hlayout3.setSpacing(0)
        #vlayout.addWidget(hbox3)
        vboxlimitslayout.addWidget(hbox3)

        hlayout3.addStretch(10)
        self.maxLabel = QLabel(hbox3)
        self.maxLabel.setText("Maximum")
        hlayout3.addWidget(self.maxLabel)

        hlayout3.addSpacing(5)
        hlayout3.addStretch(1)

        self.maxText = MyQLineEdit(hbox3)
        self.maxText.setFixedWidth(150)
        self.maxText.setAlignment(QtCore.Qt.AlignRight)

        self.maxText.returnPressed[()].connect(self.maxTextChanged)
        hlayout3.addWidget(self.maxText)


        # Graph widget for color curve...
        self.c = PlotWidget(self, backend=None)
        self.c.setGraphXLabel("Data Values")
        self.c.setInteractiveMode('select')

        self.marge = (abs(self.dataMax) + abs(self.dataMin)) / 6.0
        self.minmd = self.dataMin - self.marge
        self.maxpd = self.dataMax + self.marge

        self.c.setGraphXLimits(self.minmd, self.maxpd)
        self.c.setGraphYLimits(-11.5, 11.5)

        x = [self.minmd, self.dataMin, self.dataMax, self.maxpd]
        y = [-10, -10, 10, 10 ]
        self.c.addCurve(x, y,
                        legend="ConstrainedCurve",
                        color='black',
                        symbol='o',
                        linestyle='-')
        self.markers = []
        self.__x = x
        self.__y = y
        labelList = ["","Min", "Max", ""]
        for i in range(4):
            if i in [1, 2]:
                draggable = True
                color = "blue"
            else:
                draggable = False
                color = "black"
            #TODO symbol
            legend = "%d" % i
            self.c.addXMarker(x[i],
                              legend=legend,
                              text=labelList[i],
                              draggable=draggable,
                              color=color)
            self.markers.append((legend, ""))

        self.c.setMinimumSize(QSize(250,200))
        vlayout.addWidget(self.c)

        self.c.sigPlotSignal.connect(self.chval)

        # colormap window can not be resized
        self.setFixedSize(vlayout.minimumSize())
        self.buttonBox = QDialogButtonBox()
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        vlayout.addWidget(self.buttonBox)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def plotHistogram(self, data=None):
        if data is not None:
            self.histogramData = data
        if self.histogramData is None:
            return False
        bins, counts = self.histogramData
        self.c.addCurve(bins, counts,
                        "Histogram",
                        color='darkYellow',
                        histogram='center',
                        yaxis='right',
                        fill=True)

    def _update(self):
        _logger.debug("colormap _update called")
        self.marge = (abs(self.dataMax) + abs(self.dataMin)) / 6.0
        self.minmd = self.dataMin - self.marge
        self.maxpd = self.dataMax + self.marge
        self.c.setGraphXLimits(self.minmd, self.maxpd)
        self.c.setGraphYLimits( -11.5, 11.5)

        self.__x = [self.minmd, self.dataMin, self.dataMax, self.maxpd]
        self.__y = [-10, -10, 10, 10]
        self.c.addCurve(self.__x, self.__y,
                        legend="ConstrainedCurve",
                        color='black',
                        symbol='o',
                        linestyle='-')
        self.c.clearMarkers()
        for i in range(4):
            if i in [1, 2]:
                draggable = True
                color = "blue"
            else:
                draggable = False
                color = "black"
            key = self.markers[i][0]
            label = self.markers[i][1]
            self.c.addXMarker(self.__x[i],
                              legend=key,
                              text=label,
                              draggable=draggable,
                              color=color)
        self.sendColormap()

    def buttonGroupChange(self, val):
        _logger.debug("buttonGroup asking to update colormap")
        self.setColormapType(val, update=True)
        self._update()

    def setColormapType(self, val, update=False):
        self.colormapType = val
        if self.colormapType == 1:
            self.buttonGroup.button(1).setChecked(True)
        elif self.colormapType == 2:
            self.buttonGroup.button(2).setChecked(True)
        else:
            self.colormapType = 0
            self.buttonGroup.button(0).setChecked(True)
        if update:
            self._update()

    def chval(self, ddict):
        _logger.debug("Received %s", ddict)
        if ddict['event'] == 'markerMoving':
            diam = int(ddict['label'])
            x = ddict['x']
            if diam == 1:
                self.setDisplayedMinValue(x)
            elif diam == 2:
                self.setDisplayedMaxValue(x)
        elif ddict['event'] == 'markerMoved':
            diam = int(ddict['label'])
            x = ddict['x']
            if diam == 1:
                self.setMinValue(x)
            if diam == 2:
                self.setMaxValue(x)

    """
    Colormap
    """
    def setColormap(self, colormap):
        self.colormapIndex = colormap
        if QTVERSION < '4.0.0':
            self.combo.setCurrentItem(colormap)
        else:
            self.combo.setCurrentIndex(colormap)

    def colormapChange(self, colormap):
        self.colormapIndex = colormap
        self.sendColormap()

    # AUTOSCALE
    """
    Autoscale
    """
    def autoscaleChange(self, val):
        self.autoscale = val
        self.setAutoscale(val)
        self.sendColormap()

    def setAutoscale(self, val):
        _logger.debug("setAutoscale called %s", val)
        if val:
            self.autoScaleButton.setChecked(True)
            self.autoScale90Button.setChecked(False)
            #self.autoScale90Button.setDown(False)
            self.setMinValue(self.dataMin)
            self.setMaxValue(self.dataMax)
            self.maxText.setEnabled(0)
            self.minText.setEnabled(0)
            self.c.setEnabled(False)
            #self.c.disablemarkermode()
        else:
            self.autoScaleButton.setChecked(False)
            self.autoScale90Button.setChecked(False)
            self.minText.setEnabled(1)
            self.maxText.setEnabled(1)
            self.c.setEnabled(True)
            #self.c.enablemarkermode()

    """
    set rangeValues to dataMin ; dataMax-10%
    """
    def autoscale90Change(self, val):
        self.autoscale90 = val
        self.setAutoscale90(val)
        self.sendColormap()

    def setAutoscale90(self, val):
        if val:
            self.autoScaleButton.setChecked(False)
            self.setMinValue(self.dataMin)
            self.setMaxValue(self.dataMax - abs(self.dataMax/10))
            self.minText.setEnabled(0)
            self.maxText.setEnabled(0)
            self.c.setEnabled(False)
        else:
            self.autoScale90Button.setChecked(False)
            self.minText.setEnabled(1)
            self.maxText.setEnabled(1)
            self.c.setEnabled(True)
            self.c.setFocus()



    # MINIMUM
    """
    change min value and update colormap
    """
    def setMinValue(self, val):
        v = float(str(val))
        self.minValue = v
        self.minText.setText("%g" % v)
        self.__x[1] = v
        key = self.markers[1][0]
        label = self.markers[1][1]
        self.c.addXMarker(v, legend=key, text=label, color="blue", draggable=True)
        self.c.addCurve(self.__x,
                        self.__y,
                        legend="ConstrainedCurve",
                        color='black',
                        symbol='o',
                        linestyle='-')
        self.sendColormap()

    """
    min value changed by text
    """
    def minTextChanged(self):
        text = str(self.minText.text())
        if not len(text):
            return
        val = float(text)
        self.setMinValue(val)
        if self.minText.hasFocus():
            self.c.setFocus()

    """
    change only the displayed min value
    """
    def setDisplayedMinValue(self, val):
        val = float(val)
        self.minValue = val
        self.minText.setText("%g"%val)
        self.__x[1] = val
        key = self.markers[1][0]
        label = self.markers[1][1]
        self.c.addXMarker(val, legend=key, text=label, color="blue", draggable=True)
        self.c.addCurve(self.__x, self.__y,
                        legend="ConstrainedCurve",
                        color='black',
                        symbol='o',
                        linestyle='-')
    # MAXIMUM
    """
    change max value and update colormap
    """
    def setMaxValue(self, val):
        v = float(str(val))
        self.maxValue = v
        self.maxText.setText("%g"%v)
        self.__x[2] = v
        key = self.markers[2][0]
        label = self.markers[2][1]
        self.c.addXMarker(v, legend=key, text=label, color="blue", draggable=True)
        self.c.addCurve(self.__x, self.__y,
                        legend="ConstrainedCurve",
                        color='black',
                        symbol='o',
                        linestyle='-')
        self.sendColormap()

    """
    max value changed by text
    """
    def maxTextChanged(self):
        text = str(self.maxText.text())
        if not len(text):return
        val = float(text)
        self.setMaxValue(val)
        if self.maxText.hasFocus():
            self.c.setFocus()

    """
    change only the displayed max value
    """
    def setDisplayedMaxValue(self, val):
        val = float(val)
        self.maxValue = val
        self.maxText.setText("%g"%val)
        self.__x[2] = val
        key = self.markers[2][0]
        label = self.markers[2][1]
        self.c.addXMarker(val, legend=key, text=label, color="blue", draggable=True)
        self.c.addCurve(self.__x, self.__y,
                        legend="ConstrainedCurve",
                        color='black',
                        symbol='o',
                        linestyle='-')


    # DATA values
    """
    set min/max value of data source
    """
    def setDataMinMax(self, minVal, maxVal, update=True):
        if minVal is not None:
            vmin = float(str(minVal))
            self.dataMin = vmin
        if maxVal is not None:
            vmax = float(str(maxVal))
            self.dataMax = vmax

        if update:
            # are current values in the good range ?
            self._update()

    def getColormap(self):

        if self.minValue > self.maxValue:
            vmax = self.minValue
            vmin = self.maxValue
        else:
            vmax = self.maxValue
            vmin = self.minValue
        cmap = [self.colormapIndex, self.autoscale,
                vmin, vmax,
                self.dataMin, self.dataMax,
                self.colormapType]
        return cmap if self.exec_() else None

    """
    send 'ColormapChanged' signal
    """
    def sendColormap(self):
        _logger.debug("sending colormap")
        try:
            cmap = self.getColormap()
            self.sigColormapChanged.emit(cmap)
        except:
            sys.excepthook(sys.exc_info()[0],
                           sys.exc_info()[1],
                           sys.exc_info()[2])

    def colormapListToDict(colormapList):
        """Convert colormap from this dialog to :class:`PlotBackend`.
        :param colormapList: Colormap as returned by :meth:`getColormap`.
        :type colormapList: list or tuple
        :return: Colormap as used in :class:`PlotBackend`.
        :rtype: dict
        """
        index, autoscale, vMin, vMax, dataMin, dataMax, cmapType = colormapList
        # Warning, gamma cmapType is not supported in plot backend
        # Here it is silently replaced by linear as the default colormap
        return {
            'name': _COLORMAP_NAMES[index],
            'autoscale': autoscale,
            'vmin': vMin,
            'vmax': vMax,
            'normalization': 'log' if cmapType == 1 else 'linear',
            'colors': 256
        }

    def getColormap_name(self):
        self.getColormap()
        return _COLORMAP_NAMES[self.colormapIndex]

    def getColor(self):
        return _COLOR_NAMES[self.colormapIndex]

    def colormapDictToList(colormapDict):
        """Convert colormap from :class:`PlotBackend` to this dialog.
        :param dict colormapDict: Colormap as used in :class:`PlotBackend`.
        :return: Colormap as returned by :meth:`getColormap`.
        :rtype: list
        """
        cmapIndex = _COLORMAP_NAMES.index(colormapDict['name'])
        cmapType = 1 if colormapDict['normalization'].startswith('log') else 0
        return [cmapIndex,
                colormapDict['autoscale'],
                colormapDict['vmin'],
                colormapDict['vmax'],
                0,  # dataMin is not defined in PlotBackend colormap
                0,  # dataMax is not defined in PlotBackend colormap
                cmapType]
