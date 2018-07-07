from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib as mpl
import numpy as np
from PyQt5 import QtCore

class Canvas(FigureCanvas):
    update_data = QtCore.pyqtSignal(list)
    gray_data = QtCore.pyqtSignal(list)
    new_page = QtCore.pyqtSignal()
    slice_link = QtCore.pyqtSignal(int)
    grey_link = QtCore.pyqtSignal(list)

    def __init__(self, param, parent=None):
        self.figure = plt.figure()
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)

        self.wheel_clicked = False # only for link functions
        # self.dragging_threshold = 5
        # self.__mousePressPos = False

        self.figure.set_facecolor("black") # white region outsides the (ax)dicom image

        self.voxel = param.get('image')
        self.shape = param.get('shape')
        self.Y = param.get('color')
        self.Z = param.get('hatch')
        self.mode = param.get('mode')
        self.cmap = param.get('cmap')
        self.hmap = param.get('hmap')
        self.trans = param.get('trans')

        if self.mode == 1 or self.mode ==4 or self.mode ==7:
            self.slices = self.voxel.shape[2]
            self.ind = self.slices // 2
        elif self.mode == 2 or self.mode ==5 or self.mode ==8:
            self.slices = self.voxel.shape[0]
            self.ind = self.slices // 2
        elif self.mode == 3 or self.mode ==6 or self.mode ==9:
            self.slices = self.voxel.shape[1]
            self.ind = self.slices//2

        self.x_clicked = None
        self.y_clicked = None
        self.wheel_clicked = False
        self.ax1 = self.figure.add_subplot(111)
        # self.ax1 = plt.gca()

        self.figure.canvas.mpl_connect('key_press_event', self.click)
        self.figure.canvas.mpl_connect('button_press_event', self.mouse_clicked)
        self.figure.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.figure.canvas.mpl_connect('button_release_event', self.mouse_release)
        self.figure.canvas.mpl_connect('scroll_event', self.onscroll)

        self.emitlist = []
        self.emitlist.append(self.ind)
        self.emitlist.append(self.slices)

        self.gchange = []
        self.gchange.append(0)
        self.gchange.append(0)

        self.View_A()

    def onscroll(self, event):
        self.wheel_roll = True
        if event.button == 'up':
            self.ind = (self.ind + 1)
            self.slice_link.emit(0)
        else:
            self.ind = (self.ind - 1)
            self.slice_link.emit(1)
        self.afterscroll()

    def afterscroll(self):
        if self.ind >= self.slices:
            self.ind = 0
        if self.ind <= -1:
            self.ind = self.slices - 1

        self.ax1.clear()
        self.emitlist[0] = self.ind
        self.update_data.emit(self.emitlist)
        self.View_A()

    def click(self, event):
        self.v_min, self.v_max = self.pltc.get_clim()
        if event.key == 'w':
            self.wheel_roll = True
            self.ind = (self.ind + 1)
            self.slice_link.emit(0)
            self.afterscroll()
        elif event.key == 'q':
            self.wheel_roll = True
            self.ind = (self.ind - 1)
            self.slice_link.emit(1)
            self.afterscroll()
        elif event.key == 'left':
            self.wheel_clicked = True
            self.factor1 = -20
            self.factor2 = -20
            self.afteradjust()
        elif event.key == 'right':
            self.wheel_clicked = True
            self.factor1 = 20
            self.factor2 = 20
            self.afteradjust()
        elif event.key == 'down':
            self.wheel_clicked = True
            self.factor1 = 20
            self.factor2 = -20
            self.afteradjust()
        elif event.key == 'up':
            self.wheel_clicked = True
            self.factor1 = -20
            self.factor2 = 20
            self.afteradjust()

    def afteradjust(self):
        if (float(self.factor1 - self.factor2)) / (self.v_max - self.factor1 + 0.001) > 1:
            nmb = (float(self.factor1 - self.factor2)) / (self.v_max - self.factor1 + 0.001) + 1
            self.factor1 = (float(self.factor1 - self.factor2)) / nmb * (self.factor1 / (self.factor1 - self.factor2))
            self.factor2 = (float(self.factor1 - self.factor2)) / nmb * (self.factor2 / (self.factor1 - self.factor2))

        self.v_min += self.factor1
        self.v_max += self.factor2

        self.gchange[0] = self.factor1
        self.gchange[1] = self.factor2
        self.grey_link.emit(self.gchange)  ###

        self.pltc.set_clim(vmin=self.v_min, vmax=self.v_max)
        self.graylist[0] = self.v_min
        self.graylist[1] = self.v_max
        self.gray_data.emit(self.graylist)
        self.figure.canvas.draw()
        self.wheel_clicked = False

    def View_A(self):
        if self.mode == 1:
            self.ax1.axis('off')
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel[:, :, self.ind], 0, 1), cmap='gray', vmin=0, vmax=2094)
            self.draw_idle()
        elif self.mode == 2:
            self.ax1.axis('off')
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel[self.ind, :, :], 0, 1), cmap='gray', vmin=0, vmax=2094,
                                        extent=[0, self.shape[1], self.shape[2], 0], interpolation='sinc')
            self.draw_idle()
        elif self.mode == 3:
            self.ax1.axis('off')
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel[:, self.ind, :], 0, 1), cmap='gray', vmin=0, vmax=2094,
                                        extent = [0, self.shape[0], self.shape[2], 0], interpolation = 'sinc')
            self.draw_idle()
        elif self.mode == 4:
            self.ax1.axis('off')
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel[:, :, self.ind], 0, 1), cmap='gray', vmin=0, vmax=2094)

            self.im2 = self.ax1.imshow(np.swapaxes(self.Y[:, :, self.ind], 0, 1), cmap=self.cmap, alpha=.3)
            plt.rcParams['hatch.color'] = 'r'
            self.im3 = self.ax1.contourf(np.transpose(self.Z[:, :, self.ind]), hatches=self.hmap,
                                        colors='none', levels=np.arange(5))
            self.draw_idle()
        elif self.mode == 5:
            self.ax1.axis('off')
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel[self.ind, :, :], 0, 1), cmap='gray', vmin=0, vmax=2094,
                                        extent=[0, self.shape[1], self.shape[2], 0], interpolation='sinc')
            self.im2 = self.ax1.imshow(np.swapaxes(self.Y[self.ind, :, :], 0, 1), cmap=self.cmap, alpha=.3,
                                       extent=[0, self.shape[1], self.shape[2], 0])
            plt.rcParams['hatch.color'] = 'r'
            self.im3 = self.ax1.contourf(np.transpose(self.Z[self.ind, :, :]), hatches=self.hmap,
                                     colors='none', levels=np.arange(5),
                                        extent=[0, self.shape[1], self.shape[2], 0])
            self.draw_idle()
        elif self.mode == 6:
            self.ax1.axis('off')
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel[:, self.ind, :], 0, 1), cmap='gray', vmin=0, vmax=2094,
                                        extent=[0, self.shape[0], self.shape[2], 0], interpolation='sinc')
            self.im2 = self.ax1.imshow(np.swapaxes(self.Y[:, self.ind, :], 0, 1), cmap=self.cmap, alpha=.3,
                                       extent=[0, self.shape[0], self.shape[2], 0])
            plt.rcParams['hatch.color'] = 'r'
            self.im3 = self.ax1.contourf(np.transpose(self.Z[:, self.ind, :]), hatches=self.hmap,
                                        colors='none', levels=np.arange(5),
                                        extent=[0, self.shape[0], self.shape[2], 0])
            self.draw_idle()

        elif self.mode == 7:
            self.ax1.axis('off')
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel[:, :, self.ind], 0, 1), cmap='gray', vmin=0, vmax=2094)

            self.im2 = self.ax1.imshow(np.swapaxes(self.Y[:, :, self.ind], 0, 1), cmap=self.cmap, alpha=.3)
            self.draw_idle()
        elif self.mode == 8:
            self.ax1.axis('off')
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel[self.ind, :, :], 0, 1), cmap='gray', vmin=0, vmax=2094,
                                        extent=[0, self.shape[1], self.shape[2], 0], interpolation='sinc')
            self.im2 = self.ax1.imshow(np.swapaxes(self.Y[self.ind, :, :], 0, 1), cmap=self.cmap, alpha=.3,
                                       extent=[0, self.shape[1], self.shape[2], 0])
            self.draw_idle()
        elif self.mode == 9:
            self.ax1.axis('off')
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel[:, self.ind, :], 0, 1), cmap='gray', vmin=0, vmax=2094,
                                        extent=[0, self.shape[0], self.shape[2], 0], interpolation='sinc')
            self.im2 = self.ax1.imshow(np.swapaxes(self.Y[:, self.ind, :], 0, 1), cmap=self.cmap, alpha=.3,
                                       extent=[0, self.shape[0], self.shape[2], 0])
            self.draw_idle()

        self.wheel_roll = False ###

        v_min, v_max = self.pltc.get_clim()
        self.graylist = []
        self.graylist.append(v_min)
        self.graylist.append(v_max)

        self.new_page.emit()

    def mouse_clicked(self, event):
        if event.button == 2:
            self.x_clicked = event.x
            self.y_clicked = event.y
            self.wheel_clicked = True

    def mouse_move(self, event):
        # if self.wheel_clicked:
        if event.button == 2:
            factor = 1
            __x = event.x - self.x_clicked
            __y = event.y - self.y_clicked
            v_min, v_max = self.pltc.get_clim()
            if __x >= 0 and __y >= 0:
                __vmin = np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y >= 0:
                __vmin = -np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y < 0:
                __vmin = -np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor - np.abs(__y) * factor
            else:
                __vmin = np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = np.abs(__x) * factor - np.abs(__y) * factor

            if (float(__vmin - __vmax)) / (v_max - v_min + 0.001) > 1:
                nmb = (float(__vmin - __vmax)) / (v_max - v_min + 0.001) + 1
                __vmin = (float(__vmin - __vmax)) / nmb * (__vmin / (__vmin - __vmax))
                __vmax = (float(__vmin - __vmax)) / nmb * (__vmax / (__vmin - __vmax))

            v_min += __vmin
            v_max += __vmax
            if v_min < v_max:
                self.gchange[0] = __vmin
                self.gchange[1] = __vmax
                self.grey_link.emit(self.gchange)   ###

                self.pltc.set_clim(vmin=v_min, vmax=v_max)
                self.graylist[0] = v_min.round(2)
                self.graylist[1] = v_max.round(2)
                self.gray_data.emit(self.graylist)
                self.figure.canvas.draw()
            else:
                v_min -= __vmin
                v_max -= __vmax
    def mouse_release(self, event):
        if event.button == 2:
            self.wheel_clicked = False

    # def mousePressEvent(self, event):
    #     if event.button() == QtCore.Qt.LeftButton:
    #         self.__mousePressPos = event.globalPos()                # global
    #         self.__mouseMovePos = event.globalPos() - self.pos()    # local
    #     super(Canvas, self).mousePressEvent(event)
    #
    # def mouseMoveEvent(self, event):
    #     if event.buttons() & QtCore.Qt.LeftButton:
    #         globalPos = event.globalPos()
    #         moved = globalPos - self.__mousePressPos
    #         if moved.manhattanLength() > self.dragging_threshold:
    #             # move when user drag window more than dragging_threshould
    #             diff = globalPos - self.__mouseMovePos
    #             self.move(diff)
    #             self.__mouseMovePos = globalPos - self.pos()
    #     super(Canvas, self).mouseMoveEvent(event)
    #
    # def mouseReleaseEvent(self, event):
    #     if event.button() == QtCore.Qt.LeftButton:
    #         moved = event.globalPos() - self.__mousePressPos
    #         if moved.manhattanLength() > self.dragging_threshold:
    #             # do not call click event or so on
    #             event.ignore()
    #     super(Canvas, self).mouseReleaseEvent(event)

    def setGreyscale(self, glist):
        v_min = glist[0]
        v_max = glist[1]
        self.pltc.set_clim(vmin=v_min, vmax=v_max)
        glist[0] = v_min
        glist[1] = v_max
        self.gray_data.emit(glist)
        self.figure.canvas.draw()

    def linkedSlice(self, data):
        if self.wheel_roll == False:
            if data == 0:
                self.ind = self.ind + 1
            else:
                self.ind = self.ind - 1
            if self.ind >= self.slices:
                self.ind = 0
            if self.ind <= -1:
                self.ind = self.slices - 1
            self.ax1.clear()
            self.emitlist[0] = self.ind
            self.update_data.emit(self.emitlist)
            self.View_A()
        else:
            pass

    def linkedGrey(self, glist):
        if self.wheel_clicked == False:
            v_min, v_max = self.pltc.get_clim()
            __vmin = glist[0]
            __vmax = glist[1]
            v_min += __vmin
            v_max += __vmax
            self.pltc.set_clim(vmin=v_min, vmax=v_max)
            # if type(v_min)=='int':
            try:
                # print(type(v_min))
                self.graylist[0] = v_min.round(2)
                self.graylist[1] = v_max.round(2)
            except:
                self.graylist[0] = v_min
                self.graylist[1] = v_max
            self.gray_data.emit(self.graylist)
            self.figure.canvas.draw()
        else:
            pass