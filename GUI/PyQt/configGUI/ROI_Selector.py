# for ROI selector
import json
import sys

from PyQt5 import QtGui, QtCore
import pydicom

from pydicom.data.data_manager import get_files
import pyqtgraph as pg


with open('configGUI/lastWorkspace.json', 'r') as json_data:
    lastState = json.load(json_data)
pathlist = lastState['Pathes']
ind = lastState['Index'][-1]

pg.setConfigOptions(imageAxisOrder='row-major')

approi = QtGui.QApplication([])
win = QtGui.QMainWindow()
win.resize(800,800)
imv = pg.ImageView()
win.setCentralWidget(imv)
win.show()

ind1 = ind+1
win.setWindowTitle('Select ROI in XY on slice ' + str(ind1))

dirname = get_files(pathlist[-1], '*')
dirname.sort()
filename = dirname[ind]

dataset = pydicom.dcmread(filename)
#ax = {'t': None, 'x': 1, 'y': 0, 'c': 2}

imv.setImage(dataset.pixel_array)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

