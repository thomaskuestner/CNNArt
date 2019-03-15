import copy
import json
from numbers import Integral

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication
from matplotlib import path, patches, colors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle, PathPatch, Patch
from matplotlib.widgets import ToolHandles, AxesWidget
from skimage import color

CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_SELECT = Qt.ClosedHandCursor

class Canvas(FigureCanvas):
    update_data = pyqtSignal(list)
    gray_data = pyqtSignal(list)
    new_page = pyqtSignal()
    slice_link = pyqtSignal(int)
    grey_link = pyqtSignal(list)
    zoomRequest = pyqtSignal(int)
    scrollRequest = pyqtSignal(int, int)
    newShape = pyqtSignal()
    selectionChanged = pyqtSignal(bool)
    deleteEvent = pyqtSignal()
    MARK, SELECT = list(range(2))

    def __init__(self, param, parent=None):

        self.figure = plt.figure()
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)

        self.figure.set_facecolor("black") # white region outsides the (ax)dicom image

        self.voxel = param.get('image')
        self.shape = param.get('shape')
        self.Y = param.get('color')
        self.Z = param.get('hatch')
        self.mode = param.get('mode')
        self.cmap = param.get('cmap')
        self.hmap = param.get('hmap')
        self.trans = param.get('trans')
        self.param = param
        with open('configGUI/lastWorkspace.json', 'r') as json_data:
            lastState = json.load(json_data)
            self.dim = lastState["Dim"][0]

        if self.mode == 1 or self.mode == 4 or self.mode == 7:
            self.slices = self.voxel.shape[-1]
            self.ind = self.slices // 2
        elif self.mode == 2 or self.mode == 5 or self.mode == 8:
            self.slices = self.voxel.shape[-3]
            self.ind = self.slices // 2
        elif self.mode == 3 or self.mode == 6 or self.mode == 9:
            self.slices = self.voxel.shape[-2]
            self.ind = self.slices//2

        self.time = 0
        try:
            self.timemax = self.voxel.shape[-5]
        except:
            self.timemax = 1
        self.depth = 0
        try:
            self.depthmax = self.voxel.shape[-4]
        except:
            self.depthmax = 1
        self.x_clicked = None
        self.y_clicked = None
        self.wheel_clicked = False
        self.wheel_roll = False
        self.ax1 = self.figure.add_subplot(111)
        self.artist_list = []
        self.mask_class = None

        # for marking labels
        # 1 = left mouse button
        # 2 = center mouse button(scroll wheel)
        # 3 = right mouse button

        self.cursor2D = Cursor(self.ax1, useblit=True, color='blue', linestyle='dashed')
        self.cursor2D.set_active(True)

        self.toggle_selector_RS = RectangleSelector(self.ax1, self.rec_onselect, button=[1], drawtype='box', useblit=True,
                                               minspanx=5, minspany=5, spancoords='pixels')

        self.toggle_selector_ES = EllipseSelector(self.ax1, self.ell_onselect, drawtype='box', button=[1], minspanx=5,
                                             useblit=True, minspany=5, spancoords='pixels')

        self.toggle_selector_LS = LassoSelector(self.ax1, self.lasso_onselect, useblit=True, button=[1])

        self.toggle_selector_ES.set_active(False)
        self.toggle_selector_RS.set_active(False)
        self.toggle_selector_LS.set_active(False)
        self.toggle_selector()

        self.figure.canvas.mpl_connect('key_press_event', self.press_event)
        self.figure.canvas.mpl_connect('button_press_event', self.mouse_clicked)
        self.figure.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.figure.canvas.mpl_connect('button_release_event', self.mouse_release)
        self.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.figure.canvas.mpl_connect('pick_event', self.selectShape)

        self.emitlist = []
        self.emitlist.append(self.ind)
        self.emitlist.append(self.slices)
        self.emitlist.append(self.time)
        self.emitlist.append(self.timemax)
        self.emitlist.append(self.depth)
        self.emitlist.append(self.depthmax)

        self.gchange = []
        self.gchange.append(0)
        self.gchange.append(0)
        # 2 elements

        self.labelmode = self.SELECT
        self.current = None
        self.selectedShape = None  # save the selected shape here
        self._cursor = CURSOR_DEFAULT

        self.is_open_dialog = False
        self.shapeList = []
        self.background = None
        self.to_draw = None
        self.picked = False
        self.moveEvent = None
        self.pressEvent = None
        self._corner_order = ['NW', 'NE', 'SE', 'SW']
        self._edge_order = ['W', 'N', 'E', 'S']
        self.active_handle = None
        self._extents_on_press = None
        self.maxdist = 10
        self.labelon = False
        self.selectedshape_name = None

        self.view_image()

    def set_open_dialog(self, value):
        self.is_open_dialog = value

    def get_open_dialog(self):
        return self.is_open_dialog

    def on_scroll(self, event):
        self.wheel_roll = True
        if event.button == 'up':
            self.ind = (self.ind + 1)
            self.slice_link.emit(0)
        else:
            self.ind = (self.ind - 1)
            self.slice_link.emit(1)
        self.after_scroll()

    def after_scroll(self):
        if self.ind >= self.slices:
            self.ind = 0
        if self.ind <= -1:
            self.ind = self.slices - 1

        self.ax1.clear()
        self.shapeList.clear()
        self.emitlist[0] = self.ind
        self.update_data.emit(self.emitlist)
        self.view_image()

    def timechange(self):

        if self.time >= self.timemax:
            self.time = 0
        if self.time <= -1:
            self.time = self.timemax - 1
        self.emitlist[2] = self.time
        self.update_data.emit(self.emitlist)
        self.view_image()

    def depthchange(self):

        if self.depth >= self.depthmax:
            self.depth = 0
        if self.depth <= -1:
            self.depth = self.depthmax - 1
        self.emitlist[4] = self.depth
        self.update_data.emit(self.emitlist)
        self.view_image()

    def press_event(self, event):
        self.v_min, self.v_max = self.pltc.get_clim()
        if event.key == 'w':
            self.wheel_roll = True
            self.ind = (self.ind + 1)
            self.slice_link.emit(0)
            self.after_scroll()
        elif event.key == 'q':
            self.wheel_roll = True
            self.ind = (self.ind - 1)
            self.slice_link.emit(1)
            self.after_scroll()
        elif event.key == 'left':
            self.wheel_clicked = True
            self.factor1 = -20
            self.factor2 = -20
            self.after_adjust()
        elif event.key == 'right':
            self.wheel_clicked = True
            self.factor1 = 20
            self.factor2 = 20
            self.after_adjust()
        elif event.key == 'down':
            self.wheel_clicked = True
            self.factor1 = 20
            self.factor2 = -20
            self.after_adjust()
        elif event.key == 'up':
            self.wheel_clicked = True
            self.factor1 = -20
            self.factor2 = 20
            self.after_adjust()
        elif event.key == '1':
            # keyboard 1
            self.time = self.time - 1
            self.timechange()

        elif event.key == '2':
            # keyboard 2
            self.time = self.time + 1
            self.timechange()

        elif event.key == '3':
            # keyboard 3
            self.depth = self.depth -1

            self.depthchange()
        elif event.key == '4':
            # keyboard 4
            self.depth = self.depth + 1
            self.depthchange()

        elif event.key == 'enter':
            self.is_open_dialog = True
            shapelist = []
            if self.label_shape() == 1:
                shapelist.append(self.toggle_selector_RS.get_select())
            elif self.label_shape == 2:
                shapelist.append(self.toggle_selector_ES.get_select())
            elif self.label_shape == 3:
                shapelist.append(self.toggle_selector_LS.get_select())
            for j in shapelist:
                for i in j:
                    if i not in shapelist:
                        self.shapeList.append(i)

            for i in self.ax1.patches:
                if type(i) is Rectangle or Ellipse:
                    i.set_picker(True)
                else:
                    i.set_picker(False)

            self.newShape.emit()

        elif event.key == 'delete':

            self.to_draw.set_visible(False)
            self._center_handle = None
            self._edge_handles = None
            self._corner_handles = None
            canvas = self.selectedShape.figure.canvas
            canvas.draw_idle()

            self.df = pandas.read_csv('Markings/marking_records.csv')
            self.df = self.df.drop(self.df.index[self.selectind])
            self.df.to_csv('Markings/marking_records.csv', index=False)

            self.deSelectShape()
            self.deleteEvent.emit()

    def after_adjust(self):
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

    def view_image(self):

        if self.mode == 1:
            self.ax1.axis('off')
            try:
                self.pltc = self.ax1.imshow(self.voxel[self.time, self.depth, :, :, self.ind], cmap='gray',
                                            extent=[0, self.shape[-2], 0, self.shape[-3]])
            except:
                self.pltc = self.ax1.imshow(self.voxel[:, :, self.ind], cmap='gray',
                                            extent=[0, self.shape[-2], 0, self.shape[-3]])
            self.draw_idle()
        elif self.mode == 2:
            self.ax1.axis('off')
            try:
                self.pltc = self.ax1.imshow(self.voxel[self.time, self.depth, self.ind, :, :], cmap='gray',
                                            extent=[0, self.shape[-2], 0, self.shape[-1]], interpolation='sinc')
            except:
                self.pltc = self.ax1.imshow(self.voxel[self.ind, :, :], cmap='gray',
                                            extent=[0, self.shape[-2], 0, self.shape[-1]], interpolation='sinc')
            self.draw_idle()
        elif self.mode == 3:
            self.ax1.axis('off')
            try:
                self.pltc = self.ax1.imshow(self.voxel[self.time, self.depth, :, self.ind, :], cmap='gray',
                                            extent=[0, self.shape[-3], 0, self.shape[-1]], interpolation='sinc')
            except:
                self.pltc = self.ax1.imshow(self.voxel[:, self.ind, :], cmap='gray',
                                            extent = [0, self.shape[-3], 0, self.shape[-1]], interpolation = 'sinc')
            self.draw_idle()
        elif self.mode == 4:
            self.ax1.axis('off')
            try:
                if len(self.cmap) > 1:
                    artists = []
                    patch_color_df = pandas.read_csv('configGUI/patch_color.csv')
                    num_classes = patch_color_df['class'].count()
                    labels = list(patch_color_df.iloc[0:num_classes]['class'])
                    self.pltc = self.ax1.imshow(self.voxel[self.time, self.depth, :, :, self.ind], cmap='gray',
                                                extent=[0, self.shape[-2], 0, self.shape[-3]])
                    if not self.Y == []:
                        mask_shape = list(self.voxel[self.time, self.depth, :, :, self.ind].shape)
                        mask_shape.append(3)
                        self.total_mask = np.zeros(mask_shape)
                        for i in range(len(self.cmap)):
                            mask = color.gray2rgb(self.Y[i][self.time, self.depth, :, :, self.ind])
                            self.cmap[i] = matplotlib.colors.to_rgb(self.cmap[i])
                            self.total_mask += mask*self.cmap[i]
                        self.im2 = self.ax1.imshow(self.total_mask, alpha=self.trans,
                                                   extent=[0, self.shape[-2], 0, self.shape[-3]])
                    mask_shape = list(self.voxel[self.time, self.depth, :, :, self.ind].shape)
                    mask_shape.append(3)
                    self.total_mask = np.zeros(mask_shape)
                    for i in range(len(self.cmap)):
                        mask = color.gray2rgb(self.Z[i][self.time, self.depth, :, :, self.ind])
                        self.cmap[i] = matplotlib.colors.to_rgb(self.cmap[i])
                        self.total_mask += mask * self.cmap[i]
                        artists.append(Patch(facecolor=self.cmap[i], label=labels[i]))
                    self.im3 = self.ax1.imshow(self.total_mask, alpha=self.trans,
                                               extent=[0, self.shape[-2], 0, self.shape[-3]])
                    self.ax1.legend(handles=artists, fontsize='x-small')
                else:
                    self.pltc = self.ax1.imshow(self.voxel[self.time, self.depth, :, :, self.ind], cmap='gray',
                                                extent=[0, self.shape[-2], 0, self.shape[-3]])
                    local_cmap = matplotlib.colors.ListedColormap(self.cmap[0])
                    if not self.Y == []:
                        self.im2 = self.ax1.imshow(self.Y[0][self.time, self.depth, :, :, self.ind],
                                                   cmap=local_cmap, alpha=self.trans, extent=[0, self.shape[-2], 0, self.shape[-3]])
                    self.im3 = self.ax1.contourf(self.Z[0][self.time, self.depth, :, :, self.ind],
                                                 cmap=local_cmap, alpha=self.trans, extent=[0, self.shape[-2], self.shape[-3], 0])
            except:
                if len(self.cmap) > 1:
                    artists = []
                    patch_color_df = pandas.read_csv('configGUI/patch_color.csv')
                    num_classes = patch_color_df['class'].count()
                    labels = list(patch_color_df.iloc[0:num_classes]['class'])
                    self.pltc = self.ax1.imshow(self.voxel[:, :, self.ind], cmap='gray', extent=[0, self.shape[-2], 0, self.shape[-3]])
                    if not self.Y == []:
                        mask_shape = list(self.voxel[:, :, self.ind].shape)
                        mask_shape.append(3)
                        self.total_mask = np.zeros(mask_shape)
                        for i in range(len(self.cmap)):
                            mask = color.gray2rgb(self.Y[i][:, :, self.ind])
                            self.cmap[i] = matplotlib.colors.to_rgb(self.cmap[i])
                            self.total_mask += mask*self.cmap[i]
                        self.im2 = self.ax1.imshow(self.total_mask, alpha=self.trans, extent=[0, self.shape[-2], 0, self.shape[-3]])
                    mask_shape = list(self.voxel[:, :, self.ind].shape)
                    mask_shape.append(3)
                    self.total_mask = np.zeros(mask_shape)
                    for i in range(len(self.cmap)):
                        mask = color.gray2rgb(self.Z[i][:, :, self.ind])
                        self.cmap[i] = matplotlib.colors.to_rgb(self.cmap[i])
                        self.total_mask += mask * self.cmap[i]
                        artists.append(Patch(facecolor=self.cmap[i], label=labels[i]))
                    self.im3 = self.ax1.imshow(self.total_mask, alpha=self.trans, extent=[0, self.shape[-2], 0, self.shape[-3]])
                    self.ax1.legend(handles=artists, fontsize='x-small')
                else:
                    self.pltc = self.ax1.imshow(self.voxel[:, :, self.ind], cmap='gray',
                                                extent=[0, self.shape[-2], 0, self.shape[-3]])
                    local_cmap = matplotlib.colors.ListedColormap(self.cmap[0])
                    if not self.Y == []:
                        self.im2 = self.ax1.imshow(self.Y[0][:, :, self.ind], cmap=local_cmap, alpha=self.trans,
                                                   extent=[0, self.shape[-2], 0, self.shape[-3]])
                    self.im3 = self.ax1.contourf(self.Z[0][:, :, self.ind], cmap=local_cmap, alpha=self.trans,
                                                 extent=[0, self.shape[-2], self.shape[-3], 0])
            self.draw_idle()
        elif self.mode == 5:
            self.ax1.axis('off')
            try:
                if len(self.cmap) > 1:
                    artists = []
                    patch_color_df = pandas.read_csv('configGUI/patch_color.csv')
                    num_classes = patch_color_df['class'].count()
                    labels = list(patch_color_df.iloc[0:num_classes]['class'])
                    self.pltc = self.ax1.imshow(self.voxel[self.time, self.depth, self.ind, :, :], cmap='gray',
                                                extent=[0, self.shape[-2], 0, self.shape[-1]], interpolation='sinc')
                    if not self.Y == []:
                        mask_shape = list(self.voxel[self.time, self.depth, self.ind, :, :].shape)
                        mask_shape.append(3)
                        self.total_mask = np.zeros(mask_shape)
                        for i in range(len(self.cmap)):
                            mask = color.gray2rgb(self.Y[i][self.time, self.depth, self.ind, :, :])
                            self.cmap[i] = matplotlib.colors.to_rgb(self.cmap[i])
                            self.total_mask += mask*self.cmap[i]
                        self.im2 = self.ax1.imshow(self.total_mask, alpha=self.trans)
                    mask_shape = list(self.voxel[self.time, self.depth, self.ind, :, :].shape)
                    mask_shape.append(3)
                    self.total_mask = np.zeros(mask_shape)
                    for i in range(len(self.cmap)):
                        mask = color.gray2rgb(self.Z[i][self.time, self.depth, self.ind, :, :])
                        self.cmap[i] = matplotlib.colors.to_rgb(self.cmap[i])
                        self.total_mask += mask * self.cmap[i]
                        artists.append(Patch(facecolor=self.cmap[i], label=labels[i]))
                    self.im3 = self.ax1.imshow(self.total_mask, alpha=self.trans, extent=[0, self.shape[-2], 0, self.shape[-1]])
                    self.ax1.legend(handles=artists, fontsize='x-small')
                else:
                    self.pltc = self.ax1.imshow(self.voxel[self.time, self.depth, self.ind, :, :], cmap='gray',
                                               extent=[0, self.shape[-2], 0, self.shape[-1]], interpolation='sinc')
                    local_cmap = matplotlib.colors.ListedColormap(self.cmap[0])
                    if not self.Y == []:
                        self.im2 = self.ax1.imshow(self.Y[-1][self.time, self.depth, self.ind, :, :], cmap=local_cmap, alpha=self.trans,
                                                   extent=[0, self.shape[-2], 0, self.shape[-1]])

                    self.im3 = self.ax1.contourf(self.Z[-1][self.time, self.depth, self.ind, :, :], cmap=local_cmap,
                                                 alpha=self.trans, extent=[0, self.shape[-2], self.shape[-1], 0])
            except:
                if len(self.cmap) > 1:
                    artists = []
                    patch_color_df = pandas.read_csv('configGUI/patch_color.csv')
                    num_classes = patch_color_df['class'].count()
                    labels = list(patch_color_df.iloc[0:num_classes]['class'])
                    self.pltc = self.ax1.imshow(self.voxel[self.ind, :, :], cmap='gray',
                                                extent=[0, self.shape[-2], 0, self.shape[-1]], interpolation='sinc')
                    if not self.Y == []:
                        mask_shape = list(self.voxel[self.ind, :, :].shape)
                        mask_shape.append(3)
                        self.total_mask = np.zeros(mask_shape)
                        for i in range(len(self.cmap)):
                            mask = color.gray2rgb(self.Y[i][self.ind, :, :])
                            self.cmap[i] = matplotlib.colors.to_rgb(self.cmap[i])
                            self.total_mask += mask*self.cmap[i]
                        self.im2 = self.ax1.imshow(self.total_mask, alpha=self.trans)
                    mask_shape = list(self.voxel[self.ind, :, :].shape)
                    mask_shape.append(3)
                    self.total_mask = np.zeros(mask_shape)
                    for i in range(len(self.cmap)):
                        mask = color.gray2rgb(self.Z[i][self.ind, :, :])
                        self.cmap[i] = matplotlib.colors.to_rgb(self.cmap[i])
                        self.total_mask += mask * self.cmap[i]
                        artists.append(Patch(facecolor=self.cmap[i], label=labels[i]))
                    self.im3 = self.ax1.imshow(self.total_mask, alpha=self.trans, extent=[0, self.shape[-2], 0, self.shape[-1]])
                    self.ax1.legend(handles=artists, fontsize='x-small')
                else:
                    self.pltc = self.ax1.imshow(self.voxel[self.ind, :, :], cmap='gray',
                                                extent=[0, self.shape[-2], 0, self.shape[-1]], interpolation='sinc')
                    local_cmap = matplotlib.colors.ListedColormap(self.cmap[0])
                    if not self.Y == []:
                        self.im2 = self.ax1.imshow(self.Y[-1][self.ind, :, :], cmap=local_cmap,
                                                   alpha=self.trans,
                                                   extent=[0, self.shape[-2], 0, self.shape[-1]])

                    self.im3 = self.ax1.contourf(self.Z[-1][self.ind, :, :], cmap=local_cmap, alpha=self.trans,
                                                 extent=[0, self.shape[-2], self.shape[-1], 0])
            self.draw_idle()
        elif self.mode == 6:
            self.ax1.axis('off')
            try:
                if len(self.cmap) > 1:
                    artists = []
                    patch_color_df = pandas.read_csv('configGUI/patch_color.csv')
                    num_classes = patch_color_df['class'].count()
                    labels = list(patch_color_df.iloc[0:num_classes]['class'])
                    self.pltc = self.ax1.imshow(self.voxel[self.time, self.depth, :, self.ind, :], cmap='gray',
                                                extent=[0, self.shape[-3], 0, self.shape[-1]], interpolation='sinc')
                    if not self.Y == []:
                        mask_shape = list(self.voxel[self.time, self.depth, :, self.ind, :].shape)
                        mask_shape.append(3)
                        self.total_mask = np.zeros(mask_shape)
                        for i in range(len(self.cmap)):
                            mask = color.gray2rgb(self.Y[i][self.time, self.depth, :, self.ind, :])
                            self.cmap[i] = matplotlib.colors.to_rgb(self.cmap[i])
                            self.total_mask += mask*self.cmap[i]
                        self.im2 = self.ax1.imshow(self.total_mask, alpha=self.trans)
                    mask_shape = list(self.voxel[self.time, self.depth, :, self.ind, :].shape)
                    mask_shape.append(3)
                    self.total_mask = np.zeros(mask_shape)
                    for i in range(len(self.cmap)):
                        mask = color.gray2rgb(self.Z[i][self.time, self.depth, :, self.ind, :])
                        self.cmap[i] = matplotlib.colors.to_rgb(self.cmap[i])
                        self.total_mask += mask * self.cmap[i]
                        artists.append(Patch(facecolor=self.cmap[i], label=labels[i]))
                    self.im3 = self.ax1.imshow(self.total_mask, alpha=self.trans, extent=[0, self.shape[-3], 0, self.shape[-1]])
                    self.ax1.legend(handles=artists, fontsize='x-small')
                else:
                    self.pltc = self.ax1.imshow(self.voxel[self.time, self.depth, :, self.ind, :], cmap='gray',
                                                extent=[0, self.shape[-3], 0, self.shape[-1]], interpolation='sinc')
                    local_cmap = matplotlib.colors.ListedColormap(self.cmap[0])
                    if not self.Y == []:
                        self.im2 = self.ax1.imshow(self.Y[-1][self.time, self.depth, :, self.ind, :], cmap=local_cmap,
                                                   alpha=self.trans,
                                                   extent=[0, self.shape[-3], 0, self.shape[-1]])

                    self.im3 = self.ax1.contourf(self.Z[-1][self.time, self.depth, :, self.ind, :], cmap=local_cmap,
                                                 alpha=self.trans, extent=[0, self.shape[-3], self.shape[-1], 0])
            except:
                if len(self.cmap) > 1:
                    artists = []
                    patch_color_df = pandas.read_csv('configGUI/patch_color.csv')
                    num_classes = patch_color_df['class'].count()
                    labels = list(patch_color_df.iloc[0:num_classes]['class'])
                    self.pltc = self.ax1.imshow(self.voxel[:, self.ind, :], cmap='gray',
                                                extent=[0, self.shape[-3], 0, self.shape[-1]], interpolation='sinc')
                    if not self.Y == []:
                        mask_shape = list(self.voxel[:, self.ind, :].shape)
                        mask_shape.append(3)
                        self.total_mask = np.zeros(mask_shape)
                        for i in range(len(self.cmap)):
                            mask = color.gray2rgb(self.Y[i][:, self.ind, :])
                            self.cmap[i] = matplotlib.colors.to_rgb(self.cmap[i])
                            self.total_mask += mask*self.cmap[i]
                        self.im2 = self.ax1.imshow(self.total_mask, alpha=self.trans)
                    mask_shape = list(self.voxel[:, self.ind, :].shape)
                    mask_shape.append(3)
                    self.total_mask = np.zeros(mask_shape)
                    for i in range(len(self.cmap)):
                        mask = color.gray2rgb(self.Z[i][:, self.ind, :])
                        self.cmap[i] = matplotlib.colors.to_rgb(self.cmap[i])
                        self.total_mask += mask * self.cmap[i]
                        artists.append(Patch(facecolor=self.cmap[i], label=labels[i]))
                    self.im3 = self.ax1.imshow(self.total_mask, alpha=self.trans, extent=[0, self.shape[-3], 0, self.shape[-1]])
                    self.ax1.legend(handles=artists, fontsize='x-small')
                else:
                    self.pltc = self.ax1.imshow(self.voxel[:, self.ind, :], cmap='gray',
                                                extent=[0, self.shape[-3], 0, self.shape[-1]], interpolation='sinc')
                    local_cmap = matplotlib.colors.ListedColormap(self.cmap[0])
                    if not self.Y == []:
                        self.im2 = self.ax1.imshow(self.Y[-1][:, self.ind, :], cmap=local_cmap,
                                                   alpha=self.trans,
                                                   extent=[0, self.shape[-3], 0, self.shape[-1]])

                    self.im3 = self.ax1.contourf(self.Z[-1][:, self.ind, :], cmap=local_cmap, alpha=self.trans,
                                                 extent=[0, self.shape[-3], self.shape[-1], 0])
            self.draw_idle()

        elif self.mode == 7:
            self.ax1.axis('off')
            try:
                self.pltc = self.ax1.imshow(self.voxel[self.time, self.depth, :, :, self.ind], cmap='gray',
                                            extent=[0, self.shape[-2], 0, self.shape[-3]])
                if not self.Y == []:
                    self.im2 = self.ax1.imshow(self.Y[self.time, self.depth, self.ind, :, :], cmap=self.cmap, alpha=self.trans,
                                               extent=[0, self.shape[-2], 0, self.shape[-3]])
            except:
                self.pltc = self.ax1.imshow(self.voxel[:, :, self.ind], cmap='gray',
                                            extent=[0, self.shape[-2], 0, self.shape[-3]])
                if not self.Y == []:
                    self.im2 = self.ax1.imshow(self.Y[:, :, self.ind], cmap=self.cmap, alpha=self.trans,
                                               extent=[0, self.shape[-2], 0, self.shape[-3]])
            self.draw_idle()
        elif self.mode == 8:
            self.ax1.axis('off')
            try:
                self.pltc = self.ax1.imshow(self.voxel[self.time, self.depth, self.ind, :, :], cmap='gray',
                                            extent=[0, self.shape[-2], 0, self.shape[-1]], interpolation='sinc')
                if not self.Y == []:
                    self.im2 = self.ax1.imshow(self.Y[self.time, self.depth, self.ind, :, :], cmap=self.cmap, alpha=self.trans,
                                               extent=[0, self.shape[-2], self.shape[-1], 0])
            except:
                self.pltc = self.ax1.imshow(self.voxel[self.ind, :, :], cmap='gray',
                                            extent=[0, self.shape[-2], 0, self.shape[-1]], interpolation='sinc')
                if not self.Y == []:
                    self.im2 = self.ax1.imshow(self.Y[self.ind, :, :], cmap=self.cmap, alpha=self.trans,
                                               extent=[0, self.shape[-2], self.shape[-1], 0])
            self.draw_idle()
        elif self.mode == 9:
            self.ax1.axis('off')
            try:
                self.pltc = self.ax1.imshow(self.voxel[self.time, self.depth, :, self.ind, :], cmap='gray',
                                            extent=[0, self.shape[-3], 0, self.shape[-1]], interpolation='sinc')
                if not self.Y == []:
                    self.im2 = self.ax1.imshow(self.Y[self.time, self.depth, :, self.ind, :], cmap=self.cmap, alpha=self.trans,
                                               extent=[0, self.shape[-3], self.shape[-1], 0])
            except:
                self.pltc = self.ax1.imshow(self.voxel[:, self.ind, :], cmap='gray',
                                            extent=[0, self.shape[-3], 0, self.shape[-1]], interpolation='sinc')
                if not self.Y == []:
                    self.im2 = self.ax1.imshow(self.Y[:, self.ind, :, 0, 0], cmap=self.cmap, alpha=self.trans,
                                               extent=[0, self.shape[-3], self.shape[-1], 0])
            self.draw_idle()

        self.wheel_roll = False

        v_min, v_max = self.pltc.get_clim()
        self.graylist = []
        self.graylist.append(v_min)
        self.graylist.append(v_max)

        self.new_page.emit()

    def set_selected(self, shape):

        self.deSelectShape()
        self.picked = True
        self.toggle_selector_ES.set_active(False)
        self.toggle_selector_RS.set_active(False)
        self.toggle_selector_LS.set_active(False)
        self.selectedShape = shape
        if type(self.selectedShape) is Rectangle or Ellipse:
            self.selectedShape.set_edgecolor('black')
            self.draw_idle()
            self.set_state(2)
            self.edit_selectedShape(self.selectedShape)
        elif type(self.selectedShape) is PathPatch:
            self.selectedShape.set_edgecolor('black')
            self.draw_idle()
            self.set_state(2)

        self.selectionChanged.emit(True)

    def get_selected(self):
        return self.selectedShape

    def selectShape(self,event):

        self.deSelectShape()
        self.picked = True
        self.toggle_selector_ES.set_active(False)
        self.toggle_selector_RS.set_active(False)
        self.toggle_selector_LS.set_active(False)
        self.selectedShape = event.artist
        if type(self.selectedShape) is Rectangle or Ellipse:
            self.selectedShape.set_edgecolor('black')
            self.draw_idle()
            self.set_state(2)
            self.edit_selectedShape(self.selectedShape)
        elif type(self.selectedShape) is PathPatch:
            self.selectedShape.set_edgecolor('black')
            self.draw_idle()
            self.set_state(2)

        self.selectionChanged.emit(True)

    def update_selectedShape(self):

        self.selectedShape = self.to_draw

        plist = np.ndarray.tolist(self.selectedShape.get_path().vertices)
        plist = ', '.join(str(x) for x in plist)

        self.df = pandas.read_csv('Markings/marking_records.csv')
        if not self.df[self.df['artist']==str(self.selectedShape)].index.values.astype(int) == []:
            self.selectind = self.df[self.df['artist']==str(self.selectedShape)].index.values.astype(int)[0]
        else:
            pass
        color = self.df.iloc[self.selectind]['labelcolor']
        if self.labelon:
            self.setToolTip(self.selectedshape_name)
        if color is np.nan:
            color = colors.to_hex('b', keep_alpha=True)
        self.df.loc[self.selectind, 'path'] = plist

        self.df.to_csv('Markings/marking_records.csv', index=False)
        self.selectedShape.set_facecolor(color)
        self.selectedShape.set_alpha(0.5)

        self.selectionChanged.emit(True)

        canvas = self.selectedShape.figure.canvas
        axes = self.selectedShape.axes
        self.background = canvas.copy_from_bbox(self.selectedShape.axes.bbox)
        canvas.restore_region(self.background)

        axes.draw_artist(self.selectedShape)
        axes.draw_artist(self._corner_handles.artist)
        axes.draw_artist(self._edge_handles.artist)
        axes.draw_artist(self._center_handle.artist)
        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def deSelectShape(self):
        self.picked = False
        if self.selectedShape is not None:
            self.selectedShape.set_edgecolor(None)
            self.selectedShape = None
            self.toggle_selector_ES.set_active(True)
            self.toggle_selector_RS.set_active(True)
            self.toggle_selector_LS.set_active(True)
            self.draw_idle()
        self.selectionChanged.emit(False)

    def edit_selectedShape(self, artist):

        self.to_draw = artist

        xc, yc = self.get_corners()
        self._corner_handles = ToolHandles(self.ax1, xc, yc, marker = 'o', marker_props=self.toggle_selector_RS.get_props(), useblit=True)

        self._corner_handles.set_visible(True)
        xe, ye = self.get_edge_centers()
        self._edge_handles = ToolHandles(self.ax1, xe, ye, marker = 'o', marker_props=self.toggle_selector_RS.get_props(), useblit=True)
        self._edge_handles.set_visible(True)

        xc, yc = self.get_center()
        self._center_handle = ToolHandles(self.ax1, [xc], [yc], marker='o',
                                          marker_props=self.toggle_selector_RS.get_props(), useblit=True)
        self._center_handle.set_visible(True)

        if self.pressEvent is not None:
            c_idx, c_dist = self._corner_handles.closest(self.pressEvent.x, self.pressEvent.y)
            e_idx, e_dist = self._edge_handles.closest(self.pressEvent.x, self.pressEvent.y)
            m_idx, m_dist = self._center_handle.closest(self.pressEvent.x, self.pressEvent.y)

            if m_dist < self.maxdist * 2:
                self.active_handle = 'C'
                self._extents_on_press = self.extents
            elif c_dist > self.maxdist and e_dist > self.maxdist:
                self.active_handle = None
                return
            elif c_dist < e_dist:
                self.active_handle = self._corner_order[c_idx]
            else:
                self.active_handle = self._edge_order[e_idx]

            # Save coordinates of rectangle at the start of handle movement.
            x1, x2, y1, y2 = self.extents
            # Switch variables so that only x2 and/or y2 are updated on move.
            if self.active_handle in ['W', 'SW', 'NW']:
                x1, x2 = x2, self.pressEvent.xdata
            if self.active_handle in ['N', 'NW', 'NE']:
                y1, y2 = y2, self.pressEvent.ydata
            self._extents_on_press = x1, x2, y1, y2

        if self.selectedShape is not None:
            self.update_selectedShape()

    def _rect_bbox(self):
        if type(self.selectedShape) is Rectangle:
            x0 = self.to_draw.get_x()
            y0 = self.to_draw.get_y()
            width = self.to_draw.get_width()
            height = self.to_draw.get_height()
            return x0, y0, width, height
        elif type(self.selectedShape) is Ellipse:
            x, y = self.to_draw.center
            width = self.to_draw.width
            height = self.to_draw.height
            return x - width / 2., y - height / 2., width, height

    def get_corners(self):
        """Corners of rectangle from lower left, moving clockwise."""
        x0, y0, width, height = self._rect_bbox()
        xc = x0, x0 + width, x0 + width, x0
        yc = y0, y0, y0 + height, y0 + height
        return xc, yc

    def get_edge_centers(self):
        """Midpoint of rectangle edges from left, moving clockwise."""
        x0, y0, width, height = self._rect_bbox()
        w = width / 2.
        h = height / 2.
        xe = x0, x0 + w, x0 + width, x0 + w
        ye = y0 + h, y0, y0 + h, y0 + height
        return xe, ye

    def get_center(self):
        """Center of rectangle"""
        x0, y0, width, height = self._rect_bbox()
        return x0 + width / 2., y0 + height / 2.

    @property
    def extents(self):
        """Return (xmin, xmax, ymin, ymax)."""
        x0, y0, width, height = self._rect_bbox()
        xmin, xmax = sorted([x0, x0 + width])
        ymin, ymax = sorted([y0, y0 + height])
        return xmin, xmax, ymin, ymax

    def set_extents(self, extents):
        # Update displayed shape
        self.draw_shape(extents)
        # Update displayed handles
        self._corner_handles.set_data(self.get_corners())
        self._edge_handles.set_data(self.get_edge_centers())
        self._center_handle.set_data(self.get_center())

        canvas = self.to_draw.figure.canvas
        axes = self.to_draw.axes
        canvas.restore_region(self.background)

        axes.draw_artist(self.to_draw)
        axes.draw_artist(self._corner_handles.artist)
        axes.draw_artist(self._edge_handles.artist)
        axes.draw_artist(self._center_handle.artist)
        # blit just the redrawn area
        canvas.blit(axes.bbox)

        self.df = pandas.read_csv('Markings/marking_records.csv')
        if type(self.to_draw) is Rectangle or Ellipse:
            self.df.loc[self.selectind, 'artist'] = self.to_draw
        self.df.to_csv('Markings/marking_records.csv', index=False)

    def draw_shape(self, extents):

        if type(self.selectedShape) is Rectangle:
            x0, x1, y0, y1 = extents
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            xlim = sorted(self.ax1.get_xlim())
            ylim = sorted(self.ax1.get_ylim())

            xmin = max(xlim[0], xmin)
            ymin = max(ylim[0], ymin)
            xmax = min(xmax, xlim[1])
            ymax = min(ymax, ylim[1])

            self.to_draw.set_x(xmin)
            self.to_draw.set_y(ymin)
            self.to_draw.set_width(xmax - xmin)
            self.to_draw.set_height(ymax - ymin)

        elif type(self.selectedShape) is Ellipse:
            x1, x2, y1, y2 = extents
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            center = [x1 + (x2 - x1) / 2., y1 + (y2 - y1) / 2.]
            a = (xmax - xmin) / 2.
            b = (ymax - ymin) / 2.

            self.to_draw.center = center
            self.to_draw.width = 2 * a
            self.to_draw.height = 2 * b

    @property
    def geometry(self):
        """
        Returns numpy.ndarray of shape (2,5) containing
        x (``RectangleSelector.geometry[1,:]``) and
        y (``RectangleSelector.geometry[0,:]``)
        coordinates of the four corners of the rectangle starting
        and ending in the top left corner.
        """
        if hasattr(self.to_draw, 'get_verts'):
            xfm = self.ax1.transData.inverted()
            y, x = xfm.transform(self.to_draw.get_verts()).T
            return np.array([x, y])
        else:
            return np.array(self.to_draw.get_data())
##
    def mouse_clicked(self, event):
        if event.button == 2:
            self.x_clicked = event.x
            self.y_clicked = event.y
            self.wheel_clicked = True
        elif event.button == 3:
            if self.editing():
                self.set_state(1)
            elif self.drawing():
                self.set_state(2)
        elif event.button == 1:
            if self.picked:
                self._edit_on_press(event)

    def _edit_on_press(self, event):
        self.pressEvent = event
        contains, attrd = self.selectedShape.contains(event)
        if not contains: return

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.selectedShape.figure.canvas
        axes = self.selectedShape.axes
        self.to_draw.set_animated(True)

        canvas.draw()

        self.background = canvas.copy_from_bbox(self.selectedShape.axes.bbox)

        #canvas.restore_region(self.background)
        axes.draw_artist(self.to_draw)
        axes.draw_artist(self._corner_handles.artist)
        axes.draw_artist(self._edge_handles.artist)
        axes.draw_artist(self._center_handle.artist)
        # blit just the redrawn area
        canvas.blit(axes.bbox)

        self.df = pandas.read_csv('Markings/marking_records.csv')
        if type(self.to_draw) is Rectangle or Ellipse:
            self.df.loc[self.selectind, 'artist'] = self.to_draw
        self.df.to_csv('Markings/marking_records.csv', index=False)

    def mouse_move(self, event):
        # if self.wheel_clicked:
        self.moveEvent = event

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

        elif event.button == 1:
            if self.drawing():
                self.override_cursor(CURSOR_DRAW)
                return
            elif self.editing():
                self.override_cursor(CURSOR_SELECT)
            else:
                self.override_cursor(CURSOR_DEFAULT)

            if self.picked:
                self._edit_on_move(event)

    def _edit_on_move(self, event):

        # resize an existing shape
        if self.active_handle and not self.active_handle == 'C':
            x1, x2, y1, y2 = self._extents_on_press
            if self.active_handle in ['E', 'W'] + self._corner_order:
                x2 = event.xdata
            if self.active_handle in ['N', 'S'] + self._corner_order:
                y2 = event.ydata

        # move existing shape
        elif (self.active_handle == 'C'and self._extents_on_press is not None):
            x1, x2, y1, y2 = self._extents_on_press
            dx = event.xdata - self.pressEvent.xdata
            dy = event.ydata - self.pressEvent.ydata
            x1 += dx
            x2 += dx
            y1 += dy
            y2 += dy

        # new shape
        else:
            center = [self.pressEvent.xdata, self.pressEvent.ydata]
            center_pix = [self.pressEvent.x, self.pressEvent.y]
            dx = (event.xdata - center[0]) / 2.
            dy = (event.ydata - center[1]) / 2.

            center[0] += dx
            center[1] += dy

            x1, x2, y1, y2 = (center[0] - dx, center[0] + dx,
                              center[1] - dy, center[1] + dy)

        self.set_extents(extents=[x1, x2, y1, y2])


    def current_cursor(self):
        cursor = QApplication.overrideCursor()
        if cursor is not None:
            cursor = cursor.shape()
        return cursor

    def override_cursor(self, cursor):
        self._cursor = cursor
        if self.current_cursor() is None:
            QApplication.setOverrideCursor(cursor)
        else:
            QApplication.changeOverrideCursor(cursor)

    def mouse_release(self, event):
        if event.button == 1:
            x = int(event.xdata)
            y = int(event.ydata)

            if self.mode > 3 and self.mode <= 6:
                try:
                    pixel = self.total_mask[x, y, :]
                except:
                    pixel = [0, 0, 0]
                pixel_color = matplotlib.colors.to_hex(pixel)
                color_hex = []
                patch_color_df = pandas.read_csv('configGUI/patch_color.csv')
                count = patch_color_df['color'].count()
                for i in range(count):
                    color_hex.append(matplotlib.colors.to_hex(patch_color_df.iloc[i]['color']))
                try:
                    ind = color_hex.index(str(pixel_color))
                    self.mask_class = patch_color_df.iloc[ind]['class']
                except:
                    pass
                if self.mask_class is not None and self.labelon:
                    self.setToolTip(self.mask_class)

            if not self.labelon:
                self.setToolTip(
                    'Press Enter to choose label\nClick Rectangle or Ellipse to edit\nPress Delete to remove mark')
            else:
                if self.new_shape():
                    self.setToolTip(self.selectedshape_name)

        elif event.button == 2:
            self.wheel_clicked = False

        elif self.picked and event.button == 1:

            self._edit_on_release(event)

    def _edit_on_release(self, event):

        self.picked = False
        self.set_state(1)
        # turn off the rect animation property and reset the background
        self.to_draw.set_animated(False)
        canvas = self.to_draw.figure.canvas
        axes = self.to_draw.axes
        self.background = canvas.copy_from_bbox(self.selectedShape.axes.bbox)
        canvas.restore_region(self.background)

        axes.draw_artist(self.to_draw)
        axes.draw_artist(self._corner_handles.artist)
        axes.draw_artist(self._edge_handles.artist)
        axes.draw_artist(self._center_handle.artist)
        # blit just the redrawn area
        canvas.blit(axes.bbox)

        self.df = pandas.read_csv('Markings/marking_records.csv')
        if type(self.to_draw) is Rectangle or Ellipse:
            self.df.loc[self.selectind, 'artist'] = self.to_draw
        self.df.to_csv('Markings/marking_records.csv', index=False)

    def set_labelon(self, labelon):
        self.labelon = labelon

    def set_toolTip(self, name):
        if self.labelon:
            self.setToolTip(name)

    def set_cursor2D(self,cursoron):
        self.cursor2D.set_visible(cursoron)

    def set_cursor_position(self, x, y):
        self.cursor2D.set_position(x, y)

    def set_facecolor(self,color):
        if self.label_shape()==1:
            self.toggle_selector_RS.set_facecolor(color)
        elif self.label_shape()==2:
            self.toggle_selector_ES.set_facecolor(color)
        elif self.label_shape()==3:
            self.toggle_selector_LS.set_facecolor(color)

    def drawing(self):
        return self.labelmode == self.MARK

    def editing(self):
        return self.labelmode == self.SELECT

    def set_state(self, value):
        if value == 1:
            self.labelmode = self.MARK
        elif value == 2:
            self.labelmode = self.SELECT
        else:
            self.labelmode = None

    def set_greyscale(self, glist):
        v_min = glist[0]
        v_max = glist[1]
        self.pltc.set_clim(vmin=v_min, vmax=v_max)
        glist[0] = v_min
        glist[1] = v_max
        self.gray_data.emit(glist)
        self.figure.canvas.draw()


    def set_color(self):

        self.voxel = self.param.get('image')
        self.shape = self.param.get('shape')
        self.Y = self.param.get('color')
        self.Z = self.param.get('hatch')
        self.mode = self.param.get('mode')
        self.cmap = self.param.get('cmap')
        self.hmap = self.param.get('hmap')
        self.trans = self.param.get('trans')
        self.view_image()

    def set_transparency(self, value):

        if self.mode > 3:
            self.trans = value
            self.view_image()
        else:
            for item in self.ax1.get_children():
                if not type(item)== AxesImage:
                    self.artist_list.append(item)
            for artist in self.artist_list:
                artist.set_alpha(value)
                self.figure.canvas.draw()

    def linked_slice(self, data):
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
            self.view_image()

        else:
            pass

    def linked_grey(self, glist):
        if self.wheel_clicked == False:
            v_min, v_max = self.pltc.get_clim()
            __vmin = glist[0]
            __vmax = glist[1]
            v_min += __vmin
            v_max += __vmax
            self.pltc.set_clim(vmin=v_min, vmax=v_max)
            # if type(v_min)=='int':
            try:
                self.graylist[0] = v_min.round(2)
                self.graylist[1] = v_max.round(2)
            except:
                self.graylist[0] = v_min
                self.graylist[1] = v_max
            self.gray_data.emit(self.graylist)
            self.figure.canvas.draw()
        else:
            pass

    def set_cursor(self, crosson):
        if crosson:
            if self.toggle_selector_RS.active:
                self.toggle_selector_RS.set_cursor(True)
                self.toggle_selector_ES.set_cursor(False)
                self.toggle_selector_LS.set_cursor(False)

            elif self.toggle_selector_ES.active:
                self.toggle_selector_RS.set_cursor(False)
                self.toggle_selector_ES.set_cursor(True)
                self.toggle_selector_LS.set_cursor(False)

            elif self.toggle_selector_LS.active:
                self.toggle_selector_RS.set_cursor(False)
                self.toggle_selector_ES.set_cursor(False)
                self.toggle_selector_LS.set_cursor(True)
        else:
            self.toggle_selector_RS.set_cursor(False)
            self.toggle_selector_ES.set_cursor(False)
            self.toggle_selector_LS.set_cursor(False)

    def rec_onselect(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        rect = Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=True,
                         alpha=.2, edgecolor=None)
        plist = np.ndarray.tolist(rect.get_path().vertices)
        plist = ', '.join(str(x) for x in plist)
        self.ax1.add_patch(rect)
        self.figure.canvas.draw()
        if self.mode == 1 or self.mode == 4 or self.mode == 7:
            onslice = 'Z %s' % (self.ind + 1)
        elif self.mode == 2 or self.mode == 5 or self.mode == 8:
            onslice = 'X %s' % (self.ind + 1)
        elif self.mode == 3 or self.mode == 6 or self.mode == 9:
            onslice = 'Y %s' % (self.ind + 1)

        self.df = pandas.read_csv('Markings/marking_records.csv')
        df_size = pandas.DataFrame.count(self.df)
        df_rows = df_size['artist']
        self.df.loc[df_rows, 'artist'] = rect
        self.df.loc[df_rows, 'labelshape'] = 'rec'
        self.df.loc[df_rows, 'slice'] = onslice
        self.df.loc[df_rows, 'path'] = plist
        self.df.loc[df_rows, 'status'] = 0
        self.df.to_csv('Markings/marking_records.csv', index=False)

    def ell_onselect(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                      width=np.abs(x1 - x2), height=np.abs(y1 - y2), alpha=.2, edgecolor=None)
        plist = np.ndarray.tolist(ell.get_path().vertices)
        plist = ', '.join(str(x) for x in plist)
        self.ax1.add_patch(ell)
        self.figure.canvas.draw()
        if self.mode == 1 or self.mode == 4 or self.mode == 7:
            onslice = 'Z %s' % (self.ind + 1)
        elif self.mode == 2 or self.mode == 5 or self.mode == 8:
            onslice = 'X %s' % (self.ind + 1)
        elif self.mode == 3 or self.mode == 6 or self.mode == 9:
            onslice = 'Y %s' % (self.ind + 1)

        self.df = pandas.read_csv('Markings/marking_records.csv')
        df_size = pandas.DataFrame.count(self.df)
        df_rows = df_size['artist']
        self.df.loc[df_rows, 'artist'] = ell
        self.df.loc[df_rows, 'labelshape'] = 'ell'
        self.df.loc[df_rows, 'slice'] = onslice
        self.df.loc[df_rows, 'path'] = plist
        self.df.loc[df_rows, 'status'] = 0
        self.df.to_csv('Markings/marking_records.csv', index=False)

    def lasso_onselect(self, verts):
        self.pathlasso = path.Path(verts)
        patch = patches.PathPatch(self.pathlasso, fill=True, alpha=.2, edgecolor=None)
        self.ax1.add_patch(patch)
        self.figure.canvas.draw()
        if self.mode == 1 or self.mode == 4 or self.mode == 7:
            onslice = 'Z %s' % (self.ind+1)
        elif self.mode == 2 or self.mode == 5 or self.mode == 8:
            onslice = 'X %s' % (self.ind+1)
        elif self.mode == 3 or self.mode == 6 or self.mode == 9:
            onslice = 'Y %s' % (self.ind+1)
        self.df = pandas.read_csv('Markings/marking_records.csv')
        df_size = pandas.DataFrame.count(self.df)
        df_rows = df_size['artist']
        self.df.loc[df_rows, 'artist'] = patch
        plist = np.ndarray.tolist(self.pathlasso.vertices)
        plist = ', '.join(str(x) for x in plist)
        self.df.loc[df_rows, 'labelshape'] = 'lasso'
        self.df.loc[df_rows,'slice'] = onslice
        self.df.loc[df_rows,'path'] = plist
        self.df.loc[df_rows, 'status'] = 0
        self.df.to_csv('Markings/marking_records.csv', index=False)

    def rec_toggle_selector_on(self):
        self.toggle_selector_RS.set_active(True)
        self.figure.canvas.draw()

    def rec_toggle_selector_off(self):
        self.toggle_selector_RS.set_active(False)

    def ell_toggle_selector_on(self):
        self.toggle_selector_ES.set_active(True)
        self.figure.canvas.draw()

    def ell_toggle_selector_off(self):
        self.toggle_selector_ES.set_active(False)

    def lasso_toggle_selector_on(self):
        self.toggle_selector_LS.set_active(True)
        self.figure.canvas.draw()

    def lasso_toggle_selector_off(self):
        self.toggle_selector_LS.set_active(False)

    def toggle_selector(self):
        if self.toggle_selector_RS.active:
            self.toggle_selector_ES.set_active(False)
            self.toggle_selector_RS.set_active(True)
            self.toggle_selector_LS.set_active(False)
        elif self.toggle_selector_ES.active:
            self.toggle_selector_ES.set_active(True)
            self.toggle_selector_RS.set_active(False)
            self.toggle_selector_LS.set_active(False)
        elif self.toggle_selector_LS.active:
            self.toggle_selector_ES.set_active(False)
            self.toggle_selector_RS.set_active(False)
            self.toggle_selector_LS.set_active(True)
        else:
            self.toggle_selector_ES.set_active(False)
            self.toggle_selector_RS.set_active(False)
            self.toggle_selector_LS.set_active(False)

    def label_shape(self):
        if self.toggle_selector_RS.is_new_shape():
            return 1
        elif self.toggle_selector_ES.is_new_shape():
            return 2
        elif self.toggle_selector_LS.is_new_shape():
            return 3
        else:
            return 0

    def new_shape(self):
        if self.toggle_selector_RS.active:
            return self.toggle_selector_RS.is_new_shape()
        elif self.toggle_selector_ES.active:
            return self.toggle_selector_ES.is_new_shape()
        elif self.toggle_selector_LS.active:
            return self.toggle_selector_LS.is_new_shape()
        else:
            return False

class Cursor(AxesWidget):

    def __init__(self, ax, horizOn=True, vertOn=True, useblit=False,
                 **lineprops):

        AxesWidget.__init__(self, ax)

        self.connect_event('button_press_event', self.press)
        self.connect_event('draw_event', self.clear)

        self.visible = False
        self.horizOn = horizOn
        self.vertOn = vertOn
        self.useblit = useblit and self.canvas.supports_blit

        if self.useblit:
            lineprops['animated'] = True
        self.lineh = ax.axhline(ax.get_ybound()[0], visible=True, **lineprops)
        self.linev = ax.axvline(ax.get_xbound()[0], visible=True, **lineprops)

        self.background = None
        self.needclear = False

    def set_active(self, active):
        self._active = self.active
        AxesWidget.set_active(self, active)
        if active:
            self.update_background(None)

    def set_visible(self, visible):
        self.visible = visible
        self.linev.set_visible(self.visible)
        self.lineh.set_visible(self.visible)

        self._update()

    def set_position(self,x,y):
        self.linev.set_xdata((x, x))
        self.lineh.set_ydata((y, y))
        self.linev.set_visible(self.visible and self.vertOn)
        self.lineh.set_visible(self.visible and self.horizOn)

        self._update()

    def update_background(self, event):
        """force an update of the background"""
        # If you add a call to `ignore` here, you'll want to check edge case:
        # `release` can call a draw event even when `ignore` is True.
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def clear(self, event):
        """clear the cursor"""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.linev.set_visible(False)
        self.lineh.set_visible(False)

    def press(self, event):

        if event.inaxes != self.ax:
            self.linev.set_visible(False)
            self.lineh.set_visible(False)

            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True

        self.linev.set_xdata((event.xdata, event.xdata))
        self.lineh.set_ydata((event.ydata, event.ydata))
        self.linev.set_visible(self.visible and self.vertOn)
        self.lineh.set_visible(self.visible and self.horizOn)

        self._update()

    def _update(self):

        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)
            self.canvas.blit(self.ax.bbox)

        else:

            self.canvas.draw_idle()

        return False

class _SelectorWidget(AxesWidget):

    def __init__(self, ax, onselect, useblit=False, button=None,
                 state_modifier_keys=None):
        AxesWidget.__init__(self, ax)

        self.visible = True
        self.onselect = onselect
        self.useblit = useblit and self.canvas.supports_blit
        self.connect_default_events()

        self.state_modifier_keys = dict(move=' ', clear='escape',
                                        square='shift', center='control')
        self.state_modifier_keys.update(state_modifier_keys or {})

        self.background = None
        self.artists = []

        if isinstance(button, Integral):
            self.validButtons = [button]
        else:
            self.validButtons = button

        # will save the data (position at mouseclick)
        self.eventpress = None
        # will save the data (pos. at mouserelease)
        self.eventrelease = None
        self._prev_event = None
        self.state = set()
        self._active = True

        def get_active(self):
            return self._active

        self.active = property(get_active, lambda self, active: self.set_active(active),
                          doc="Is the widget active?")

    def set_active(self, active):
        self._active = self.active
        AxesWidget.set_active(self, active)
        if active:
            self.update_background(None)

    def update_background(self, event):
        """force an update of the background"""
        # If you add a call to `ignore` here, you'll want to check edge case:
        # `release` can call a draw event even when `ignore` is True.
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def connect_default_events(self):
        """Connect the major canvas events to methods."""
        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('button_press_event', self.press)
        self.connect_event('button_release_event', self.release)
        self.connect_event('draw_event', self.update_background)
        self.connect_event('key_press_event', self.on_key_press)
        self.connect_event('key_release_event', self.on_key_release)
        self.connect_event('scroll_event', self.on_scroll)

    def ignore(self, event):
        """return *True* if *event* should be ignored"""
        if not self.active or not self.ax.get_visible():
            return True

        # If canvas was locked
        if not self.canvas.widgetlock.available(self):
            return True

        if not hasattr(event, 'button'):
            event.button = None

        # Only do rectangle selection if event was triggered
        # with a desired button
        if self.validButtons is not None:
            if event.button not in self.validButtons:
                return True

        # If no button was pressed yet ignore the event if it was out
        # of the axes
        if self.eventpress is None:
            return event.inaxes != self.ax

        # If a button was pressed, check if the release-button is the
        # same.
        if event.button == self.eventpress.button:
            return False

        # If a button was pressed, check if the release-button is the
        # same.
        return (event.inaxes != self.ax or
                event.button != self.eventpress.button)

    def update(self):
        """draw using newfangled blit or oldfangled draw depending on
        useblit

        """
        if not self.ax.get_visible():
            return False

        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            for artist in self.artists:
                self.ax.draw_artist(artist)
            self.canvas.blit(self.ax.bbox)

        else:
            self.canvas.draw_idle()

        return False

    def _get_data(self, event):
        """Get the xdata and ydata for event, with limits"""
        if event.xdata is None:
            return None, None
        x0, x1 = self.ax.get_xbound()
        y0, y1 = self.ax.get_ybound()
        xdata = max(x0, event.xdata)
        xdata = min(x1, xdata)
        ydata = max(y0, event.ydata)
        ydata = min(y1, ydata)
        return xdata, ydata

    def _clean_event(self, event):
        """Clean up an event

        Use prev event if there is no xdata
        Limit the xdata and ydata to the axes limits
        Set the prev event
        """
        if event.xdata is None:
            event = self._prev_event
        else:
            event = copy.copy(event)
        event.xdata, event.ydata = self._get_data(event)

        self._prev_event = event
        return event

    def press(self, event):
        """Button press handler and validator"""
        if not self.ignore(event):
            event = self._clean_event(event)
            self.eventpress = event
            self._prev_event = event
            key = event.key or ''
            key = key.replace('ctrl', 'control')
            # move state is locked in on a button press
            if key == self.state_modifier_keys['move']:
                self.state.add('move')
            self._press(event)
            return True
        return False

    def _press(self, event):
        """Button press handler"""
        pass

    def release(self, event):
        """Button release event handler and validator"""
        if not self.ignore(event) and self.eventpress:
            event = self._clean_event(event)
            self.eventrelease = event
            self._release(event)
            self.eventpress = None
            self.eventrelease = None
            self.state.discard('move')
            return True
        return False

    def _release(self, event):
        """Button release event handler"""
        pass

    def onmove(self, event):
        """Cursor move event handler and validator"""
        if not self.ignore(event) and self.eventpress:
            event = self._clean_event(event)
            self._onmove(event)
            return True
        return False

    def _onmove(self, event):
        """Cursor move event handler"""
        pass

    def on_scroll(self, event):
        """Mouse scroll event handler and validator"""
        if not self.ignore(event):
            self._on_scroll(event)

    def _on_scroll(self, event):
        """Mouse scroll event handler"""
        pass

    def on_key_press(self, event):
        """Key press event handler and validator for all selection widgets"""
        if self.active:
            key = event.key or ''
            key = key.replace('ctrl', 'control')
            if key == self.state_modifier_keys['clear']:
                for artist in self.artists:
                    artist.set_visible(False)
                self.update()
                return
            for (state, modifier) in self.state_modifier_keys.items():
                if modifier in key:
                    self.state.add(state)
            self._on_key_press(event)

    def _on_key_press(self, event):
        """Key press event handler - use for widget-specific key press actions.
        """
        pass

    def on_key_release(self, event):
        """Key release event handler and validator"""
        if self.active:
            key = event.key or ''
            for (state, modifier) in self.state_modifier_keys.items():
                if modifier in key:
                    self.state.discard(state)
            self._on_key_release(event)

    def _on_key_release(self, event):
        """Key release event handler"""
        pass

    def set_visible(self, visible):
        """ Set the visibility of our artists """
        self.visible = visible
        for artist in self.artists:
            artist.set_visible(visible)

    def get_select(self):
        for artist in self.artists:
            select = []
            if artist not in select:
                select.append(artist)
            return select

class RectangleSelector(_SelectorWidget):

    _shape_klass = Rectangle
    new_shape = False

    def __init__(self, ax, onselect, drawtype='box',
                 minspanx=None, minspany=None, useblit=False,
                 lineprops=None, rectprops=None, spancoords='data',
                 button=None, maxdist=10, marker_props=None,
                 state_modifier_keys=None):


        _SelectorWidget.__init__(self, ax, onselect, useblit=useblit,
                                 button=button,
                                 state_modifier_keys=state_modifier_keys)

        self.to_draw = None
        self.visible = True
        self.cursor_on = False
        self.lineh = ax.axhline(ax.get_ybound()[0], visible=False, color='blue', linestyle='dashed')
        self.linev = ax.axvline(ax.get_xbound()[0], visible=False, color='blue', linestyle='dashed')

        if drawtype == 'none':
            drawtype = 'line'                        # draw a line but make it
            self.visible = False                     # invisible

        if drawtype == 'box':
            if rectprops is None:
                rectprops = dict(edgecolor=None,alpha=0.2, fill=True)

            rectprops['animated'] = self.useblit
            self.rectprops = rectprops
            self.to_draw = self._shape_klass((0, 0), 0, 1, visible=False,
                                             **self.rectprops)
            self.ax.add_patch(self.to_draw)
        if drawtype == 'line':
            if lineprops is None:
                lineprops = dict(color='black', linestyle='-',
                                 linewidth=2, alpha=0.5)
            lineprops['animated'] = self.useblit
            self.lineprops = lineprops
            self.to_draw = Line2D([0, 0], [0, 0], visible=False,
                                  **self.lineprops)
            self.ax.add_line(self.to_draw)

        self.minspanx = minspanx
        self.minspany = minspany

        if spancoords not in ('data', 'pixels'):
            raise ValueError("'spancoords' must be 'data' or 'pixels'")

        self.spancoords = spancoords
        self.drawtype = drawtype

        self.maxdist = maxdist

        if rectprops is None:
            self.props = dict(mec='r')
        else:
            self.props = dict(mec=rectprops.get('edgecolor', 'r'))
        self._corner_order = ['NW', 'NE', 'SE', 'SW']
        xc, yc = self.corners
        self._corner_handles = ToolHandles(self.ax, xc, yc, marker_props=self.props,
                                           useblit=self.useblit)

        self._edge_order = ['W', 'N', 'E', 'S']
        xe, ye = self.edge_centers
        self._edge_handles = ToolHandles(self.ax, xe, ye, marker='s',
                                         marker_props=self.props,
                                         useblit=self.useblit)

        xc, yc = self.center
        self._center_handle = ToolHandles(self.ax, [xc], [yc], marker='s',
                                          marker_props=self.props,
                                          useblit=self.useblit)

        self.active_handle = None

        if self.active_handle is not None:
            self.artists = [self.to_draw, self._center_handle.artist,
                        self._corner_handles.artist,
                        self._edge_handles.artist]
        else:
            self.artists = [self.to_draw]

        self._extents_on_press = None

    def _press(self, event):

        self.new_shape = False
        """on button press event"""
        # make the drawed box/line visible get the click_event-coordinates,
        # button, ...

        self.active_handle = None
        if self.active_handle is None:
            # Clear previous rectangle before drawing new rectangle.
            self.update()

        x = event.xdata
        y = event.ydata
        self.extents = x, x, y, y
        self.press_pos_x = event.x
        self.press_pos_y = event.y

        self.set_visible(self.visible)

    def _release(self, event):

        self.new_shape = True
        # cursor
        self.linev.set_visible(False)
        self.lineh.set_visible(False)

        # update the eventpress and eventrelease with the resulting extents
        x1, x2, y1, y2 = self.extents
        self.eventpress.xdata = x1
        self.eventpress.ydata = y1
        xy1 = self.ax.transData.transform_point([x1, y1])
        self.eventpress.x, self.eventpress.y = xy1

        self.eventrelease.xdata = x2
        self.eventrelease.ydata = y2
        xy2 = self.ax.transData.transform_point([x2, y2])
        self.eventrelease.x, self.eventrelease.y = xy2

        if self.spancoords == 'data':
            xmin, ymin = self.eventpress.xdata, self.eventpress.ydata
            xmax, ymax = self.eventrelease.xdata, self.eventrelease.ydata
            # calculate dimensions of box or line get values in the right
            # order
        elif self.spancoords == 'pixels':
            xmin, ymin = self.eventpress.x, self.eventpress.y
            xmax, ymax = self.eventrelease.x, self.eventrelease.y
        else:
            raise ValueError('spancoords must be "data" or "pixels"')

        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        spanx = xmax - xmin
        spany = ymax - ymin
        xproblems = self.minspanx is not None and spanx < self.minspanx
        yproblems = self.minspany is not None and spany < self.minspany

        # check if drawn distance (if it exists) is not too small in
        # either x or y-direction
        if self.drawtype != 'none' and (xproblems or yproblems):
            for artist in self.artists:
                artist.set_visible(False)
            self.update()
            return

        # call desired function
        self.onselect(self.eventpress, self.eventrelease)
        self.update()
        return False

    def _onmove(self, event):

        # handle cursor
        self.linev.set_xdata((event.xdata, event.xdata))
        self.lineh.set_ydata((event.ydata, event.ydata))
        self.linev.set_visible(self.cursor_on)
        self.lineh.set_visible(self.cursor_on)

        # resize an existing shape
        if self.active_handle and not self.active_handle == 'C':
            x1, x2, y1, y2 = self._extents_on_press
            if self.active_handle in ['E', 'W'] + self._corner_order:
                x2 = event.xdata
            if self.active_handle in ['N', 'S'] + self._corner_order:
                y2 = event.ydata

        # move existing shape
        elif (self.active_handle == 'C'and self._extents_on_press is not None):
            x1, x2, y1, y2 = self._extents_on_press
            dx = event.xdata - self.eventpress.xdata
            dy = event.ydata - self.eventpress.ydata
            x1 += dx
            x2 += dx
            y1 += dy
            y2 += dy

        # new shape
        else:
            center = [self.eventpress.xdata, self.eventpress.ydata]
            center_pix = [self.eventpress.x, self.eventpress.y]
            dx = (event.xdata - center[0]) / 2.
            dy = (event.ydata - center[1]) / 2.

            center[0] += dx
            center[1] += dy

            x1, x2, y1, y2 = (center[0] - dx, center[0] + dx,
                              center[1] - dy, center[1] + dy)

        self.extents = x1, x2, y1, y2

    def update(self):
        """draw using newfangled blit or oldfangled draw depending on
        useblit

        """
        if not self.ax.get_visible():
            return False

        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            for artist in self.artists:
                self.ax.draw_artist(artist)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)

            self.canvas.blit(self.ax.bbox)

        else:
            self.canvas.draw_idle()
        return False

    def set_cursor(self, cursor_on):
        self.cursor_on = cursor_on
        self.linev.set_visible(self.cursor_on)
        self.lineh.set_visible(self.cursor_on)

    def is_new_shape(self):
        return self.new_shape

    def get_shape(self):
        rect = []
        for i in self.ax.patches:
            if i not in rect:
                rect.append(i)
        return rect

    def get_props(self):
        return self.props

    def set_facecolor(self,color):
        self.to_draw.set_facecolor(color)
        self.update()

    @property
    def _rect_bbox(self):
        if self.drawtype == 'box':
            x0 = self.to_draw.get_x()
            y0 = self.to_draw.get_y()
            width = self.to_draw.get_width()
            height = self.to_draw.get_height()
            return x0, y0, width, height
        else:
            x, y = self.to_draw.get_data()
            x0, x1 = min(x), max(x)
            y0, y1 = min(y), max(y)
            return x0, y0, x1 - x0, y1 - y0

    @property
    def corners(self):
        """Corners of rectangle from lower left, moving clockwise."""
        x0, y0, width, height = self._rect_bbox
        xc = x0, x0 + width, x0 + width, x0
        yc = y0, y0, y0 + height, y0 + height
        return xc, yc

    @property
    def edge_centers(self):
        """Midpoint of rectangle edges from left, moving clockwise."""
        x0, y0, width, height = self._rect_bbox
        w = width / 2.
        h = height / 2.
        xe = x0, x0 + w, x0 + width, x0 + w
        ye = y0 + h, y0, y0 + h, y0 + height
        return xe, ye

    @property
    def center(self):
        """Center of rectangle"""
        x0, y0, width, height = self._rect_bbox
        return x0 + width / 2., y0 + height / 2.

    @property
    def extents(self):
        """Return (xmin, xmax, ymin, ymax)."""
        x0, y0, width, height = self._rect_bbox
        xmin, xmax = sorted([x0, x0 + width])
        ymin, ymax = sorted([y0, y0 + height])
        return xmin, xmax, ymin, ymax

    @extents.setter
    def extents(self, extents):
        # Update displayed shape
        self.draw_shape(extents)
        # Update displayed handles
        self._corner_handles.set_data(*self.corners)
        self._edge_handles.set_data(*self.edge_centers)
        self._center_handle.set_data(*self.center)
        self.set_visible(self.visible)
        self.update()

    def draw_shape(self, extents):
        x0, x1, y0, y1 = extents
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        xlim = sorted(self.ax.get_xlim())
        ylim = sorted(self.ax.get_ylim())

        xmin = max(xlim[0], xmin)
        ymin = max(ylim[0], ymin)
        xmax = min(xmax, xlim[1])
        ymax = min(ymax, ylim[1])

        if self.drawtype == 'box':
            self.to_draw.set_x(xmin)
            self.to_draw.set_y(ymin)
            self.to_draw.set_width(xmax - xmin)
            self.to_draw.set_height(ymax - ymin)

        elif self.drawtype == 'line':
            self.to_draw.set_data([xmin, xmax], [ymin, ymax])

    @property
    def geometry(self):
        """
        Returns numpy.ndarray of shape (2,5) containing
        x (``RectangleSelector.geometry[1,:]``) and
        y (``RectangleSelector.geometry[0,:]``)
        coordinates of the four corners of the rectangle starting
        and ending in the top left corner.
        """
        if hasattr(self.to_draw, 'get_verts'):
            xfm = self.ax.transData.inverted()
            y, x = xfm.transform(self.to_draw.get_verts()).T
            return np.array([x, y])
        else:
            return np.array(self.to_draw.get_data())

class EllipseSelector(RectangleSelector):

    _shape_klass = Ellipse
    new_shape = False

    def draw_shape(self, extents):
        x1, x2, y1, y2 = extents
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        center = [x1 + (x2 - x1) / 2., y1 + (y2 - y1) / 2.]
        a = (xmax - xmin) / 2.
        b = (ymax - ymin) / 2.

        if self.drawtype == 'box':
            self.to_draw.center = center
            self.to_draw.width = 2 * a
            self.to_draw.height = 2 * b
        else:
            rad = np.deg2rad(np.arange(31) * 12)
            x = a * np.cos(rad) + center[0]
            y = b * np.sin(rad) + center[1]
            self.to_draw.set_data(x, y)

    @property
    def _rect_bbox(self):
        if self.drawtype == 'box':
            x, y = self.to_draw.center
            width = self.to_draw.width
            height = self.to_draw.height
            return x - width / 2., y - height / 2., width, height
        else:
            x, y = self.to_draw.get_data()
            x0, x1 = min(x), max(x)
            y0, y1 = min(y), max(y)
            return x0, y0, x1 - x0, y1 - y0

class LassoSelector(_SelectorWidget):

    new_shape = False

    def __init__(self, ax, onselect=None, useblit=True, lineprops=None,
                 button=None):
        _SelectorWidget.__init__(self, ax, onselect, useblit=useblit,
                                 button=button)

        self.verts = None
        self.patch = None
        self.cursor_on = False
        self.lineh = ax.axhline(ax.get_ybound()[0], visible=False, color='blue', linestyle='dashed')
        self.linev = ax.axvline(ax.get_xbound()[0], visible=False, color='blue', linestyle='dashed')

        if lineprops is None:
            lineprops = dict()
        if useblit:
            lineprops['animated'] = True
        self.line = Line2D([], [], **lineprops)
        self.line.set_visible(False)
        self.ax.add_line(self.line)
        self.artists = [self.line]

    def _press(self, event):
        self.verts = [self._get_data(event)]
        self.line.set_visible(True)
        self.new_shape = False
        self.update()

    def _release(self, event):
        # cursor
        self.linev.set_visible(False)
        self.lineh.set_visible(False)
        if self.verts is not None:
            self.verts.append(self._get_data(event))
            self.onselect(self.verts)
        self.line.set_data([[], []])
        self.line.set_visible(False)
        self.verts = None
        self.update()
        # open labeldialog
        self.new_shape = True

    def _onmove(self, event):

        # handle cursor
        self.linev.set_xdata((event.xdata, event.xdata))
        self.lineh.set_ydata((event.ydata, event.ydata))
        self.linev.set_visible(self.cursor_on)
        self.lineh.set_visible(self.cursor_on)
        if self.verts is None:
            return
        self.verts.append(self._get_data(event))

        self.line.set_data(list(zip(*self.verts)))

        self.update()

    def update(self):
        """draw using newfangled blit or oldfangled draw depending on
        useblit

        """
        pathlasso = path.Path(self.verts)
        self.patch = patches.PathPatch(pathlasso, fill=True, alpha=.2, edgecolor=None)
        if not self.ax.get_visible():
            return False

        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            for artist in self.artists:
                self.ax.draw_artist(artist)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)

            self.canvas.blit(self.ax.bbox)

        else:
            self.canvas.draw_idle()
        return False

    def set_cursor(self, cursor_on):
        self.cursor_on = cursor_on
        self.linev.set_visible(self.cursor_on)
        self.lineh.set_visible(self.cursor_on)

    def is_new_shape(self):
        return self.new_shape

    def get_select(self):
        return self.patch

    def set_facecolor(self, color):
        self.patch.set_facecolor(color)
        self.update()