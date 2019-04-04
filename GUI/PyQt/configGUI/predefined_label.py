import codecs
import os

import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtProperty, Qt, QVariant, QSize
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QWidget, QTableWidget, QLineEdit, QLabel, QPushButton, QDialog
from PyQt5.QtWidgets import (QComboBox, QGridLayout,
                             QItemEditorCreatorBase, QItemEditorFactory, QTableWidgetItem)


class ColorListEditor(QComboBox):
    def __init__(self, widget=None):
        super(ColorListEditor, self).__init__(widget)

        self.populateList()

    def getColor(self):
        color = self.itemData(self.currentIndex(), Qt.DecorationRole)
        return color

    def setColor(self, color):
        self.setCurrentIndex(self.findData(color, Qt.DecorationRole))

    color = pyqtProperty(QColor, getColor, setColor, user=True)

    def populateList(self):
        for i, colorName in enumerate(QColor.colorNames()):
            color = QColor(colorName)
            self.insertItem(i, colorName)
            self.setItemData(i, color, Qt.DecorationRole)

class ColorListItemEditorCreator(QItemEditorCreatorBase):
    def createWidget(self, parent):
        return ColorListEditor(parent)

class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        factory = QItemEditorFactory()
        factory.registerEditor(QVariant.Color, ColorListItemEditorCreator())
        QItemEditorFactory.setDefaultFactory(factory)

        self.labelHist = []
        self.loadPredefinedClasses('configGUI/predefined_classes.txt')
        self.createGUI()

    def createGUI(self):
        length = len(self.labelHist)
        tableData = pd.read_csv('configGUI/predefined_label.csv')

        self.table = QTableWidget(length, 2)
        self.table.setHorizontalHeaderLabels(["Label Name", "Label Color"])
        self.table.verticalHeader().setVisible(True)
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)

        self.table.cellDoubleClicked.connect(self.editCell)
        self.table.cellChanged.connect(self.cellChanged)

        for i in range(length):
            name = tableData.at[i,'label name']
            nameItem = QTableWidgetItem(name)
            nameItem.setFlags(nameItem.flags() | Qt.ItemIsEditable)
            color = QColor(tableData.at[i,'label color'])
            colorItem = QTableWidgetItem()
            colorItem.setData(Qt.DisplayRole, color)
            colorItem.setFlags(colorItem.flags() | Qt.ItemIsEditable)
            self.table.setItem(i, 0, nameItem)
            self.table.setItem(i, 1, colorItem)

        self.table.resizeColumnToContents(0)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.layout = QGridLayout()
        self.layout.addWidget(self.table, 0, 0)
        self.setLayout(self.layout)

        self.setWindowTitle("Predefined Label Editor")

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)

    def editCell(self,r,c):
        if c==0:
            nameDialog = NameDialog(self.table.item(r,c))
            name = nameDialog.popUp()
            nameItem = QTableWidgetItem(name)
            self.table.setItem(r,0,nameItem)
            colorItem = QTableWidgetItem()
            colorItem.setData(Qt.DisplayRole, QColor('blue'))
            self.table.setItem(r,1,colorItem)
            if r < len(self.labelHist):
                self.labelHist[r]=name
            else:
                self.labelHist.append(name)
            with open('configGUI/predefined_classes.txt', 'w') as f:
                for item in self.labelHist:
                    f.write("%s\n" % item.text())

    def cellChanged(self,r,c):
        length = len(self.labelHist)
        if r >= length -1:
            rowPosition = self.table.rowCount()
            self.table.insertRow(rowPosition)

        self.table.update()
        tableData = self.read_table(self.table)

    def read_table(self, table):
        row = table.rowCount()
        tableData = pd.read_csv('configGUI/predefined_label.csv')
        for r in range(row):
            if table.item(r, 0) is not None:
                tableData.loc[r,'label name'] = table.item(r,0).text()
            else:
                pass
            if table.item(r, 1) is not None:
                tableData.loc[r,'label color'] = table.item(r,1).text()
            else:
                pass
            tableData.to_csv('configGUI/predefined_label.csv',index=False)
        return tableData

class NameDialog(QDialog):
    def __init__(self,name):
        QDialog.__init__(self)

        self.setMinimumSize(QSize(320, 140))
        self.setWindowTitle("Type in a Label Name")

        self.text = name

        self.nameLabel = QLabel(self)
        self.nameLabel.setText('Name:')
        self.line = QLineEdit(self)

        self.line.move(80, 20)
        self.line.resize(200, 32)
        self.nameLabel.move(20, 20)

        pybutton = QPushButton('OK', self)
        pybutton.clicked.connect(self.accept)
        pybutton.resize(200,32)
        pybutton.move(80, 60)

    def popUp(self):

        return self.line.text() if self.exec_() and self.line.text() else self.text

if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)

    window = Window()
    window.show()

sys.exit(app.exec_())