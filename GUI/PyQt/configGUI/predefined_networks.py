# for generating a network table for editing predefined networks name and path
import pandas as pd

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication, QWidget, QTableWidget, QLineEdit, QLabel, QPushButton, QDialog
from PyQt5.QtWidgets import (QGridLayout,
                             QTableWidgetItem)


class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.createGUI()

    def createGUI(self):

        tableData = pd.read_csv('DLart/networks.csv')
        self.networkHist = []
        for i in range(pd.DataFrame.count(tableData)['name']):
            self.networkHist.append(tableData.iloc[i]['name'])
        length = len(self.networkHist)
        self.table = QTableWidget(length, 2)
        self.table.setHorizontalHeaderLabels(["Network Name", "Network Path"])
        self.table.verticalHeader().setVisible(True)
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)

        self.table.cellDoubleClicked.connect(self.editCell)
        self.table.cellChanged.connect(self.cellChanged)

        for i in range(length):
            name = tableData.at[i,'name']
            nameItem = QTableWidgetItem(name)
            nameItem.setFlags(nameItem.flags() | Qt.ItemIsEditable)
            path = tableData.at[i,'path'].replace(".", "/") + '.py'
            pathItem = QTableWidgetItem(path)
            self.table.setItem(i, 0, nameItem)
            self.table.setItem(i, 1, pathItem)

        self.table.resizeColumnToContents(0)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.layout = QGridLayout()
        self.layout.addWidget(self.table, 0, 0)
        self.setLayout(self.layout)

        self.setWindowTitle("Predefined Neural Networks Editor")

    def editCell(self,r,c):
        if c==0:
            nameDialog = NameDialog(self.table.item(r,c))
            name = nameDialog.popUp()
            nameItem = QTableWidgetItem(name)
            self.table.setItem(r,0,nameItem)
            if r < len(self.networkHist):
                self.networkHist[r] = name
            else:
                self.networkHist.append(name)
        elif c==1:
            dialog = QtWidgets.QFileDialog()
            options = dialog.Options()
            options |= QtWidgets.QFileDialog.Directory
            options |= QtWidgets.QFileDialog.ExistingFile
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            dialog.setOptions(options)
            file_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose the file', '.', 'python files(*.py)')[0]
            list = file_path.split('/')
            ind = list.index('networks')
            model_path = ''
            for item in list[ind:-1]:
                model_path = model_path + item + '/'
            model_path += list[-1]
            pathItem = QTableWidgetItem(model_path)
            self.table.setItem(r,1,pathItem)

    def cellChanged(self,r,c):
        length = len(self.networkHist)
        if r >= length -1:
            rowPosition = self.table.rowCount()
            self.table.insertRow(rowPosition)

        self.table.update()
        tableData = self.read_table(self.table)

    def read_table(self, table):
        row = table.rowCount()
        tableData = pd.read_csv('DLart/networks.csv')
        for r in range(row):
            if table.item(r, 0) is not None:
                tableData.loc[r,'name'] = table.item(r,0).text()
            else:
                pass
            if table.item(r, 1) is not None:
                tableData.loc[r,'path'] = table.item(r,1).text().split('.')[0].replace("/", ".")
            else:
                pass
            tableData.to_csv('DLart/networks.csv',index=False)
        return tableData

class NameDialog(QDialog):
    def __init__(self,name):
        QDialog.__init__(self)

        self.setMinimumSize(QSize(320, 140))
        self.setWindowTitle("Type in a Network Name")

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