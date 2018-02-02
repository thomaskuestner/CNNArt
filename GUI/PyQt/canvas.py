from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib as mpl
import numpy as np

class Canvas(FigureCanvas):
    def __init__(self, voxel, model, Y ,Z, parent=None):
        self.figure = plt.figure()
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)

        self.voxel = voxel
        self.Y = Y
        self.Z = Z
        self.model = model

        if self.model == 0 or self.model ==1:
            self.slices = self.voxel.shape[2]
            self.ind = self.slices // 2
        elif self.model == 2:
            self.slices = self.voxel.shape[0]
            self.ind = self.slices // 2
        elif self.model == 3:
            self.slices = self.voxel.shape[1]
            self.ind = self.slices//2

        self.x_clicked = None
        self.y_clicked = None
        self.mouse_second_clicked = False
        self.ax1 = self.figure.add_subplot(111)
        self.cmap1 = mpl.colors.ListedColormap(['blue', 'purple', 'cyan', 'yellow', 'green'])

        self.figure.canvas.mpl_connect('button_press_event', self.mouse_clicked)
        self.figure.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.figure.canvas.mpl_connect('button_release_event', self.mouse_release)
        self.figure.canvas.mpl_connect('scroll_event', self.onscroll)

        self.View_A()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) #% self.slices
        else:
            self.ind = (self.ind - 1) #% self.slices
        if self.ind >= self.slices:
            self.ind = 0
        if self.ind <= -1:
            self.ind = self.slices - 1
        # plt.cla() # not clf
        self.View_A()

    def View_A(self):
        if self.model == 0:
            self.ax1.axis('off')
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel[:, :, self.ind], 0, 1), cmap='gray', vmin=0, vmax=2094)
            self.ax1.set_ylabel('slice %s' % (self.ind + 1))
            self.draw()
        elif self.model == 1:
            self.ax1.axis('off')
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel[:, :, self.ind], 0, 1), cmap='gray', vmin=0, vmax=2094)

            self.im2 = self.ax1.imshow(np.swapaxes(self.Y[:, :, self.ind], 0, 1), cmap=self.cmap1, alpha=.3, vmin=1,
                                     vmax=6)
            plt.rcParams['hatch.color'] = 'r'
            self.im3 = self.ax1.contourf(np.transpose(self.Z[:, :, self.ind]), hatches=[None, '//', '\\', 'XX'],
                                        colors='none', edges='r', levels=np.arange(5))
            self.ax1.set_ylabel('slice %s' % (self.ind + 1))
            self.draw()
        elif self.model == 2:
            self.ax1.axis('off')
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel[self.ind, :, :], 0, 1), cmap='gray', vmin=0, vmax=2094)
            self.im2 = self.ax1.imshow(np.swapaxes(self.Y[self.ind, :, :], 0, 1), cmap=self.cmap1, alpha=.3, vmin=1,
                                       vmax=6, extent=[0, self.voxel.shape[1], self.voxel.shape[2], 0])
            plt.rcParams['hatch.color'] = 'r'
            self.im3 = self.ax1.contourf(np.transpose(self.Z[self.ind, :, :]), hatches=[None, '//', '\\', 'XX'],
                                     colors='none', edges='r', levels=np.arange(5),
                                        extent=[0, self.voxel.shape[1], self.voxel.shape[2], 0])
            self.ax1.set_ylabel('slice %s' % (self.ind + 1))
            self.draw()
        elif self.model == 3:
            self.ax1.axis('off')
            self.pltc = self.ax1.imshow(np.swapaxes(self.voxel[:, self.ind, :], 0, 1), cmap='gray', vmin=0, vmax=2094)

            self.im2 = self.ax1.imshow(np.swapaxes(self.Y[:, self.ind, :], 0, 1), cmap=self.cmap1, alpha=.3, vmin=1,
                                      vmax=6, extent=[0, self.voxel.shape[0], self.voxel.shape[2], 0])
            plt.rcParams['hatch.color'] = 'r'
            self.im3 = self.ax1.contourf(np.transpose(self.Z[:, self.ind, :]), hatches=[None, '//', '\\', 'XX'],
                                        colors='none', edges='r', levels=np.arange(5),
                                        extent=[0, self.voxel.shape[0], self.voxel.shape[2], 0])
            self.ax1.set_ylabel('slice %s' % (self.ind + 1))
            self.draw()

    def mouse_clicked(self, event):
        if event.button == 2:
            self.x_clicked = event.x
            self.y_clicked = event.y
            self.mouse_second_clicked = True

    def mouse_move(self, event):
        if self.mouse_second_clicked:
            factor = 10
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
            self.pltc.set_clim(vmin=v_min.round(2), vmax=v_max.round(2))
            self.figure.canvas.draw()

    def mouse_release(self, event):
        if event.button == 2:
            self.mouse_second_clicked = False

''' canvas drag version
class Canvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()  # Figure(figsize=(5, 5), dpi=100) if not draw in QT designer
        self.ax1 = self.fig.add_subplot(111)
        #self.ax1.plot([1, 2, 3], [1, 2, 3], linewidth=2, color="#c6463d", label="line1")

        self.ax1.axis('off')

        FigureCanvas.__init__(self, self.fig)

        self.draggable = True
        self.dragging_threshold = 5
        self.__mousePressPos = None
        self.__mouseMovePos = None

    def mousePressEvent(self, event):
        if self.draggable and event.button() == QtCore.Qt.LeftButton:
            self.__mousePressPos = event.globalPos()                # global
            self.__mouseMovePos = event.globalPos() - self.pos()    # local
        super(Canvas, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.draggable and event.buttons() & QtCore.Qt.LeftButton:
            globalPos = event.globalPos()
            moved = globalPos - self.__mousePressPos
            if moved.manhattanLength() > self.dragging_threshold:
                # move when user drag window more than dragging_threshould
                diff = globalPos - self.__mouseMovePos
                self.move(diff)
                self.__mouseMovePos = globalPos - self.pos()
        super(Canvas, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.__mousePressPos is not None:
            if event.button() == QtCore.Qt.LeftButton:
                moved = event.globalPos() - self.__mousePressPos
                if moved.manhattanLength() > self.dragging_threshold:
                    # do not call click event or so on
                    event.ignore()
                self.__mousePressPos = None
        super(Canvas, self).mouseReleaseEvent(event)
    '''