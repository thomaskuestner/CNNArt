from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
import numpy as np

class Canvas2(FigureCanvas):
    def __init__(self,X1, parent=None):
        self.figure2 = plt.figure()
        FigureCanvas.__init__(self, self.figure2)
        self.setParent(parent)

        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.Z = Z
        self.view = view
        self.thickness = thickness

        self.slices1 = self.X1.shape[2]
        self.ind1 = self.slices1//2
        self.slices2 = self.X1.shape[0]
        self.ind2 = self.slices2//2
        self.slices3 = self.X1.shape[1]
        self.ind3 = self.slices3//2

        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')

        self.figure2.canvas.mpl_connect('scroll_event', self.onscroll)

        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind1 = (self.ind1 + 1) #% self.slices
        else:
            self.ind1 = (self.ind1 - 1) #% self.slices
        if self.ind1 >= self.X1.shape[2]:
            self.ind1 = 0
        if self.ind1 <= -1:
            self.ind1 = self.X1.shape[2] - 1

        self.update()

    def update(self):
        self.cmap1 = mpl.colors.ListedColormap(['blue', 'purple','cyan', 'yellow', 'green'])

        if self.view ==0:
            self.im1 = self.ax.imshow(np.swapaxes(self.X1[:, :, self.ind1], 0, 1), cmap='gray', vmin=0, vmax=2094)
            self.im2 = self.ax.imshow(np.swapaxes(self.Y[:, :, self.ind1], 0, 1), cmap=self.cmap1, alpha=.3, vmin=1,
                                     vmax=6)
            plt.rcParams['hatch.color'] = 'r'
            self.im3 = self.ax.contourf(np.transpose(self.Z[:, :, self.ind1]), hatches=[None, '//', '\\', 'XX'],
                                        colors='none', edges='r', levels=np.arange(5))
            self.ax.set_ylabel('slice %s' % (self.ind1 + 1))
            self.figure2.canvas.draw_idle()
        elif self.view ==1: # sagittal
            self.im1 = self.ax.imshow(np.swapaxes(self.X2[self.ind2, :, :], 0, 1), cmap='gray', vmin=0, vmax=2094)
            self.im2 = self.ax.imshow(np.swapaxes(self.Y[self.ind2, :, :], 0, 1), cmap=self.cmap1, alpha=.3, vmin=1,
                                       vmax=6, extent=[0, self.X1.shape[1], np.round(self.slices1 * self.thickness), 0])
            plt.rcParams['hatch.color'] = 'r'
            self.im3 = self.ax.contourf(np.transpose(self.Z[self.ind2, :, :]), hatches=[None, '//', '\\', 'XX'],
                                     colors='none', edges='r', levels=np.arange(5), extent=[0, self.X1.shape[1],
                                      np.round(self.slices1 * self.thickness), 0])
            self.ax.set_ylabel('slice %s' % (self.ind2 + 1))
        else: # coronal
            self.im1 = self.ax.imshow(np.swapaxes(self.X2[:, self.ind3, :], 0, 1), cmap='gray', vmin=0, vmax=2094)

            self.im2 = self.ax.imshow(np.swapaxes(self.Y[:, self.ind3, :], 0, 1), cmap=self.cmap1, alpha=.3, vmin=1,
                                      vmax=6, extent=[0, self.X1.shape[0], np.round(self.slices1 * self.thickness), 0])
            plt.rcParams['hatch.color'] = 'r'
            self.im3 = self.ax.contourf(np.transpose(self.Z[:, self.ind3, :]), hatches=[None, '//', '\\', 'XX'],
                                        colors='none', edges='r', levels=np.arange(5), extent=[0, self.X1.shape[0],
                                        np.round(self.slices1 * self.thickness), 0])
            self.ax.set_ylabel('slice %s' % (self.ind3 + 1))


        self.draw()