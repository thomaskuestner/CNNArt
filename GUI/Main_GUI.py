from tkinter import *
import os
import numpy as np
import shelve
import scipy.io
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import LassoSelector, RectangleSelector, EllipseSelector
from matplotlib import path
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from MRT_Layer import MRT_Layer
from MRT_array_set import MRT_array
#from ChooseData_For_CNN2 import CNN_Preprocessing
#from time import *
import dicom
import dicom_numpy
from PIL import Image, ImageTk
#import matlab.engine

class Artefact_Labeling_GUI():

    def __init__(self):
        self.window = Tk()
        self.window.title("Artefact Labeling")
        self.window.geometry('1425x770') #self.window.geometry('1920x1080')
        self.window.configure(background="gray38")

        self.mrt_layer_set = []
        self.optionlist = []
        self.mrt_layer_names = []

        # TODO: replace by dbinfo = DatabaseInfo()
        # self.mrt_model = dbinfo.get_mrt_model
        self.mrt_model = {'t1_tse_tra_fs_Becken_0008': '0008',
                          't1_tse_tra_fs_Becken_Motion_0010': '0010',
                          't1_tse_tra_fs_mbh_Leber_0004': '0004',
                          't1_tse_tra_fs_mbh_Leber_Motion_0005': '0005',
                          't1_tse_tra_Kopf_0002': '0002',
                          't1_tse_tra_Kopf_Motion_0003': '0003','t2_tse_tra_fs_Becken_0009': '0009',
                          't2_tse_tra_fs_Becken_Motion_0011': '0011',
                          't2_tse_tra_fs_Becken_Shim_xz_0012': '0012',
                          't2_tse_tra_fs_navi_Leber_0006': '0006',
                          't2_tse_tra_fs_navi_Leber_Shim_xz_0007': '0007'}
        self.mrt_smodel = {'t1_tse_tra_fs_Becken_0008': 'Becken', 't1_tse_tra_fs_Becken_Motion_0010': 'Becken',
                           't1_tse_tra_fs_mbh_Leber_0004': 'Leber', 't1_tse_tra_fs_mbh_Leber_Motion_0005': 'Leber',
                           't1_tse_tra_Kopf_0002': 'Kopf', 't1_tse_tra_Kopf_Motion_0003': 'Kopf',
                           't2_tse_tra_fs_Becken_0009': 'Becken', 't2_tse_tra_fs_Becken_Motion_0011': 'Becken',
                           't2_tse_tra_fs_Becken_Shim_xz_0012': 'Becken', 't2_tse_tra_fs_navi_Leber_0006': 'Leber',
                           't2_tse_tra_fs_navi_Leber_Shim_xz_0007': 'Leber'}
        self.mrt_artefact = {'t1_tse_tra_fs_Becken_0008': '', 't1_tse_tra_fs_Becken_Motion_0010': 'Move',
                             't1_tse_tra_fs_mbh_Leber_0004': '', 't1_tse_tra_fs_mbh_Leber_Motion_0005': 'Move',
                             't1_tse_tra_Kopf_0002': '', 't1_tse_tra_Kopf_Motion_0003': 'Move',
                             't2_tse_tra_fs_Becken_0009': '', 't2_tse_tra_fs_Becken_Motion_0011': 'Move',
                             't2_tse_tra_fs_Becken_Shim_xz_0012': 'Shim', 't2_tse_tra_fs_navi_Leber_0006': '',
                             't2_tse_tra_fs_navi_Leber_Shim_xz_0007': 'Shim'}
        self.col_list = {"red": "1", "green": "2", "blue": "3"}
        self.res_col_list = {"1": "red", "2": "green", "3": "blue"}
        self.artefact_list = ["Movement-Artefact", "Shim-Artefact", "Noise-Artefact"]
        self.mark_list = ["Rectangle", "Ellipse", "Lasso"]
        self.list_of_images = ["icons/Move.png", "icons/Rectangle.png", "icons/Ellipse.png", "icons/Lasso.png", "icons/Lupe.png", "icons/3D.png"]

        # TODO: adapt paths
        #self.sFolder = "C:/Users/Sebastian Milde/Pictures/MRT"#
        self.sFolder = 'D:' + os.sep + 'med_data' + os.sep + 'MRPhysics' + os.sep + 'newProtocol'
        #self.Path_marking = "C:/Users/Sebastian Milde/Pictures/Universitaet/Masterarbeit/Markings/"
        self.Path_marking = 'D:' + os.sep + 'med_data' + os.sep + 'MRPhysics' + os.sep + 'Markings'
        self.proband = os.listdir(self.sFolder)
        self.artefact = os.listdir(self.sFolder + os.sep + '01_ab' + os.sep + 'dicom_sorted')

        self.panel = Label(self.window, bg="black")

        self.rahmen1 = Frame(self.window, bg= "gray65", bd=3, relief="sunken")
        self.rahmen1.place(x = 5, y = 5, width = 215, height = 250)

        self.rahmen2 = Frame(self.window, bg= "gray65", bd=3, relief="sunken")
        self.rahmen2.place(x=5, y=260, width=215, height=200)

        self.text_r1 = Text(self.rahmen1, height = 10, width = 200)
        self.text_r1.insert(INSERT,"Choose MRT-Layer")

        # Define buttons and option menus
        image = Image.open("icons/open.png")
        img_open = ImageTk.PhotoImage(image)
        label = Label(image=img_open)
        label.image = img_open
        self.load_button = Button(self.rahmen1, image=img_open, text="Load MRT-Layer", font=('Arial', 11, 'bold'),
                                  bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT,
                                  command=self.load_MRT)

        self.path_entry = Entry(self.rahmen1)
        self.path_entry.place(x=5, y=50, width=200, height=25)
        self.path_entry.insert(0,self.sFolder)

        self.tool_buttons = []
        self.activated_button = 0
        self.button1_activated = True
        self.x_clicked = None
        self.y_clicked = None
        self.mouse_second_clicked = False

        def onClick(i):
            old_button = self.activated_button
            self.activated_button = i
            if self.activated_button == 0:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(False)
                self.button1_activated = True
            if self.activated_button == 2:
                toggle_selector.ES.set_active(True)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(False)
                self.button1_activated = False
            if self.activated_button == 1:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(True)
                toggle_selector.LS.set_active(False)
                self.button1_activated = False
            if self.activated_button == 3:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(True)
                self.button1_activated = False
            if old_button == i:
                pass
            else:
                self.tool_buttons[old_button].configure(relief=RAISED)
                self.tool_buttons[self.activated_button].configure(relief=SUNKEN)
            return

        column = 5
        for i in range(0, len(self.list_of_images), 1):
            image = Image.open(self.list_of_images[i])
            img_tool = ImageTk.PhotoImage(image)
            label = Label(image=img_tool)
            label.image = img_tool
            b = Button(self.rahmen2, image = img_tool, bg="gray56", borderwidth=2, command=lambda i=i : onClick(i))
            if i == 5:
                row = 45
                column = 5
            else:
                row = 5

            b.place(x = column, y = row)
            column += 40
            self.tool_buttons.append(b)
            self.tool_buttons[self.activated_button].configure(relief=SUNKEN)

        self.proband_str = StringVar(self.window)
        self.proband_str.set(self.proband[0])
        self.proband_option = OptionMenu(self.rahmen1, self.proband_str, *self.proband)
        self.proband_option.config(bg = "gray56")

        self.artefact_str = StringVar(self.window)
        self.artefact_str.set(self.artefact[0])
        self.artefact_option = OptionMenu(self.rahmen1, self.artefact_str, *self.artefact)
        self.artefact_option.config(bg = "gray56")

        self.layer = StringVar(self.window)
        self.layer.set("(empty)")
        self.option_layers = OptionMenu(self.rahmen1, self.layer, "(empty)", command = self.change_layer)
        self.option_layers.config(bg = "gray56")

        self.chooseArtefact = StringVar(self.window)
        self.chooseArtefact.set(self.artefact_list[0])
        self.chooseArtefact_option = OptionMenu(self.rahmen2, self.chooseArtefact, *self.artefact_list)
        self.chooseArtefact_option.config(bg = "gray56")

        self.set_cnn = Button(self.window, text="Settings for CNN", bg="gray56",font=('Arial', 11, 'bold'),
                              command = self.chooseData)
        image = Image.open("icons/save.png")
        img = ImageTk.PhotoImage(image)
        label = Label(image=img)
        label.image = img
        image = Image.open("icons/exit.png")
        img_exit = ImageTk.PhotoImage(image)
        label = Label(image=img_exit)
        label.image = img_exit

        self.save_button = Button(self.window, image = img, text="Save", bg="gray56",font=('Arial', 11, 'bold'), borderwidth=4,compound = LEFT)
        self.exit_button = Button(self.window, image = img_exit, text="Exit", bg="gray56",font=('Arial', 11, 'bold'), borderwidth=4,compound = LEFT,command=self.exit)

        self.number_mrt_label = Label(self.window, bg="#000000", fg="white", font="Aral 23 bold")
        self.art_mod_label = Label(self.panel, bg="#000000", fg="white", font="Aral 12 bold")
        self.hint_label = Label(self.window, text = "Please load MRT-Layer",bg="#000000", fg="white", font="Aral 25 bold")

        self.load_button.place(x=5, y=5, width=200, height=40) #self.load_button.place(x=5, y=5, width=200, height=40)

        self.proband_option.place(x=5, y=80, width=200, height=50) #self.proband_option.place(x=5, y=55, width=200, height=50)
        self.artefact_option.place(x=5, y=135, width=200, height=50) # self.artefact_option.place(x=5, y=105, width=200, height=50)
        self.option_layers.place(x=5, y=190, width=200, height=50) #self.option_layers.place(x=5, y=155, width=200, height=50)
        self.chooseArtefact_option.place(x=5, y=85, width=200, height=50)

        self.set_cnn.place(x=12.5, y=470, width=200, height=40)
        self.exit_button.place(x=12, y=725, width=200, height=40)
        self.save_button.place(x=12, y=680, width=200, height=40)
        self.panel.place(x=225, y=5, width=1204, height=764)

        # Visualisation
        self.fig = plt.figure(dpi=50)
        #self.fig.patch.set_facecolor('black')
        self.ax = plt.gca()
        self.pltc = None
        #self.ax.set_axis_bgcolor('black')
        self.ax.text(0.5,0.5,'To label MRT-Artefacts load MRT-Layer', horizontalalignment = 'center', verticalalignment = 'center', color = 'white', fontsize = 20, transform = self.ax.transAxes)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.show()
        self.canvas.get_tk_widget().place(x=600, y=250)  # expand=1

        def lasso_onselect(verts):
            print (verts)
            p = path.Path(verts)
            current_mrt_layer = self.get_currentLayerNumber()
            proband = self.art_mod_label['text'][10:self.art_mod_label['text'].find('\n')]
            model = self.art_mod_label['text'][
                    self.art_mod_label['text'].find('\n') + 9:len(self.art_mod_label['text'])]
            print(proband)
            print(model)
            saveFile = shelve.open(self.Path_marking + proband + ".slv", writeback=True)
            print(p)
            patch = None
            col_str = None
            if self.chooseArtefact.get() == self.artefact_list[0]:
                col_str = "31"
                patch = patches.PathPatch(p, fill=False, edgecolor='red', lw=2)
            elif self.chooseArtefact.get() == self.artefact_list[1]:
                col_str = "32"
                patch = patches.PathPatch(p, fill=False, edgecolor='green', lw=2)
            elif self.chooseArtefact.get() == self.artefact_list[2]:
                col_str = "33"
                patch = patches.PathPatch(p, fill=False, edgecolor='blue', lw=2)
            self.ax.add_patch(patch)
            layer_name = model

            #if saveFile.has_key(layer_name):
            if layer_name in saveFile:
                number_str = str(self.mrt_layer_set[current_mrt_layer].get_current_Number()) + "_" + col_str + "_" + str(len(self.ax.patches) - 1)
                saveFile[layer_name].update({number_str: p})
            else:
                number_str = str(
                    self.mrt_layer_set[current_mrt_layer].get_current_Number()) + "_" + col_str + "_" + str(
                    len(self.ax.patches) - 1)
                saveFile[layer_name] = {number_str: p}

            saveFile.close()
            self.fig.canvas.draw_idle()

        def ronselect(eclick, erelease):
            'eclick and erelease are matplotlib events at press and release'
            col_str = None
            rect = None
            ell = None
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            current_mrt_layer = self.get_currentLayerNumber()
            proband = self.art_mod_label['text'][10:self.art_mod_label['text'].find('\n')]
            model = self.art_mod_label['text'][
                    self.art_mod_label['text'].find('\n') + 9:len(self.art_mod_label['text'])]
            print(proband)
            print(model)
            saveFile = shelve.open(self.Path_marking + proband +".slv", writeback=True)
            p = np.array(([x1,y1,x2,y2]))

            layer_name =  model

            if toggle_selector.RS.active and not toggle_selector.ES.active:
                if self.chooseArtefact.get() == self.artefact_list[0]:
                    col_str = "11"
                    rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=False,
                                         edgecolor="red", lw=2)
                elif self.chooseArtefact.get() == self.artefact_list[1]:
                    col_str = "12"
                    rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=False,
                                         edgecolor="green", lw=2)
                elif self.chooseArtefact.get() == self.artefact_list[2]:
                    col_str = "13"
                    rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=False,
                                         edgecolor="blue", lw=2)
                self.ax.add_patch(rect)
            elif toggle_selector.ES.active and not toggle_selector.RS.active:
                if self.chooseArtefact.get() == self.artefact_list[0]:
                    col_str = "21"
                    ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                                  width=np.abs(x1 - x2), height=np.abs(y1 - y2), edgecolor="red", fc='None', lw=2)
                elif self.chooseArtefact.get() == self.artefact_list[1]:
                    col_str = "22"
                    ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                                  width=np.abs(x1 - x2), height=np.abs(y1 - y2), edgecolor="green", fc='None', lw=2)
                elif self.chooseArtefact.get() == self.artefact_list[2]:
                    col_str = "23"
                    ell = Ellipse(xy=(min(x1, x2) + np.abs(x1 - x2) / 2, min(y1, y2) + np.abs(y1 - y2) / 2),
                                  width=np.abs(x1 - x2), height=np.abs(y1 - y2), edgecolor="blue", fc='None', lw=2)
                self.ax.add_patch(ell)

            #if saveFile.has_key(layer_name):
            if layer_name in saveFile:
                number_str = str(self.mrt_layer_set[current_mrt_layer].get_current_Number()) + "_" + col_str + "_" + str(len(self.ax.patches) - 1)
                saveFile[layer_name].update({number_str: p})
            else:
                number_str = str(
                    self.mrt_layer_set[current_mrt_layer].get_current_Number()) + "_" + col_str + "_" + str(
                    len(self.ax.patches) - 1)
                saveFile[layer_name] = {number_str: p}
            saveFile.close()
            print(' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata))
            print(' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata))
            print(' used button   : ', eclick.button)

        def toggle_selector(event):
            print(' Key pressed.')
            if self.activated_button == 2 and not toggle_selector.ES.active and (
                        toggle_selector.LS.active or toggle_selector.RS.active):
                toggle_selector.ES.set_active(True)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(False)
            if self.activated_button == 1 and not toggle_selector.RS.active and (
                        toggle_selector.LS.active or toggle_selector.ES.active):
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(True)
                toggle_selector.LS.set_active(False)
            if self.activated_button == 3 and (
                        toggle_selector.ES.active or toggle_selector.RS.active) and not toggle_selector.LS.active:
                toggle_selector.ES.set_active(False)
                toggle_selector.RS.set_active(False)
                toggle_selector.LS.set_active(True)

        toggle_selector.RS = RectangleSelector(self.ax, ronselect, button=[1]) #drawtype='box', useblit=False, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True

        toggle_selector.ES = EllipseSelector(self.ax, ronselect, drawtype='line', button=[1],  minspanx=5, minspany=5, spancoords='pixels',
                                               interactive=True) #drawtype='line', minspanx=5, minspany=5, spancoords='pixels', interactive=True

        toggle_selector.LS = LassoSelector(self.ax, lasso_onselect, button=[1])

        toggle_selector.ES.set_active(False)
        toggle_selector.RS.set_active(False)
        toggle_selector.LS.set_active(False)
        self.fig.canvas.mpl_connect('key_press_event', self.click)
        self.fig.canvas.mpl_connect('button_press_event', self.mouse_clicked)
        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self.mouse_release)
        #self.fig.canvas.mpl_connect('key_press_event', toggle_selector)
        #lasso = LassoSelector(self.ax, self.onselect, button=[1])
        #self.hint_label.place(x=550, y=200, width = 400, height = 300)
        self.window.mainloop()

    def load_MRT(self):
        mrt_layer_name = str(self.proband_str.get()) + "_" + self.mrt_model[str(self.artefact_str.get())]
        filenames_list = []
        if mrt_layer_name in self.mrt_layer_names:
            pass
        else:
            self.mrt_layer_names.append(mrt_layer_name)
            PathDicom = self.sFolder + "/" + self.proband_str.get() + "/dicom_sorted/" + self.artefact_str.get() + "/"

            #load Dicom_Array
            files = sorted([os.path.join(PathDicom, file) for file in os.listdir(PathDicom)], key=os.path.getctime)
            datasets = [dicom.read_file(f) \
                        for f in files]

            try:
                voxel_ndarray, pixel_space = dicom_numpy.combine_slices(datasets)
                voxel_ndarray_matlab = voxel_ndarray.transpose(1, 0, 2)
            except dicom_numpy.DicomImportException:
                # invalid DICOM data
                raise

            dx, dy, dz = 1.0, 1.0, pixel_space[2][2] #pixel_space[0][0], pixel_space[1][1], pixel_space[2][2]
            pixel_array = voxel_ndarray[:, :, 0]
            x_1d = dx * np.arange(voxel_ndarray.shape[0])
            y_1d = dy * np.arange(voxel_ndarray.shape[1])
            z_1d = dz * np.arange(voxel_ndarray.shape[2])

            smodel = self.mrt_smodel[str(self.artefact_str.get())]
            sartefact = self.mrt_artefact[str(self.artefact_str.get())]
            mrt_layer = MRT_Layer(voxel_ndarray.shape[0], voxel_ndarray.shape[1], voxel_ndarray.shape[2],
                                  sartefact, smodel, x_1d,y_1d, z_1d,
                                  voxel_ndarray, mrt_layer_name, voxel_ndarray_matlab)
            self.mrt_layer_set.append(mrt_layer)
            self.optionlist.append("(" + str(len(self.mrt_layer_set)) + ") " + mrt_layer_name + ": " + str(
                voxel_ndarray.shape[0]) + " x " + str(
                voxel_ndarray.shape[1]))
            self.refresh_option(self.optionlist)

            self.number_mrt_label.config(text="1/" + str(mrt_layer.get_number_mrt()))
            self.number_mrt_label.place(x=1330, y=10)
            self.art_mod_label.config(text = "Proband:  " + str(self.proband_str.get()) + "\nModel:  " + str(self.artefact_str.get()))
            self.art_mod_label.place(x = 5, y = 5)
            plt.cla()
            plt.gca().set_aspect('equal')  # plt.axes().set_aspect('equal')
            plt.xlim(0, voxel_ndarray.shape[0]*dx)
            plt.ylim(voxel_ndarray.shape[1]*dy, 0)
            plt.set_cmap(plt.gray())
            self.pltc = plt.pcolormesh(x_1d, y_1d, np.swapaxes(pixel_array, 0, 1), vmin=0, vmax=2094)

            # load
            File_Path = self.Path_marking + self.proband_str.get() +".slv"
            loadFile = shelve.open(File_Path)
            number_Patch = 0
            cur_no = "0"
            #if loadFile.has_key(self.artefact_str.get()):
            if self.artefact_str.get() in loadFile:
                layer = loadFile[self.artefact_str.get()]
                '''while layer.has_key(cur_no + "_11_" + str(number_Patch)) \
                        or layer.has_key(cur_no + "_12_" + str(number_Patch)) \
                        or layer.has_key(cur_no + "_13_" + str(number_Patch)) \
                        or layer.has_key(cur_no + "_21_" + str(number_Patch)) \
                        or layer.has_key(cur_no + "_22_" + str(number_Patch)) \
                        or layer.has_key(cur_no + "_23_" + str(number_Patch)) \
                        or layer.has_key(cur_no + "_31_" + str(number_Patch)) \
                        or layer.has_key(cur_no + "_32_" + str(number_Patch)) \
                        or layer.has_key(cur_no + "_33_" + str(number_Patch)):
                '''
                while (cur_no + "_11_" + str(number_Patch)) in layer \
                        or (cur_no + "_12_" + str(number_Patch)) in layer \
                        or (cur_no + "_13_" + str(number_Patch)) in layer \
                        or (cur_no + "_21_" + str(number_Patch)) in layer \
                        or (cur_no + "_22_" + str(number_Patch)) in layer \
                        or (cur_no + "_23_" + str(number_Patch)) in layer \
                        or (cur_no + "_31_" + str(number_Patch)) in layer \
                        or (cur_no + "_32_" + str(number_Patch)) in layer \
                        or (cur_no + "_33_" + str(number_Patch)) in layer:
                    patch = None
                    #if layer.has_key(cur_no + "_11_" + str(number_Patch)):
                    if (cur_no + "_11_" + str(number_Patch)) in layer:
                        p = layer[cur_no + "_11_" + str(number_Patch)]
                        patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]),
                                              np.abs(p[1] - p[3]), fill=False,
                                              edgecolor="red", lw=2)
                    #elif layer.has_key(cur_no + "_12_" + str(number_Patch)):
                    elif (cur_no + "_12_" + str(number_Patch)) in layer:
                        p = layer[cur_no + "_12_" + str(number_Patch)]
                        patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]),
                                              np.abs(p[1] - p[3]), fill=False,
                                              edgecolor="green", lw=2)
                    #elif layer.has_key(cur_no + "_13_" + str(number_Patch)):
                    elif (cur_no + "_13_" + str(number_Patch)):
                        p = layer[cur_no + "_13_" + str(number_Patch)]
                        patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]),
                                              np.abs(p[1] - p[3]), fill=False,
                                              edgecolor="blue", lw=2)
                    #elif layer.has_key(cur_no + "_21_" + str(number_Patch)):
                    elif (cur_no + "_21_" + str(number_Patch)) in layer:
                        p = layer[cur_no + "_21_" + str(number_Patch)]
                        patch = Ellipse(
                            xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                            width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="red", fc='None', lw=2)
                    #elif layer.has_key(cur_no + "_22_" + str(number_Patch)):
                    elif (cur_no + "_22_" + str(number_Patch)) in layer:
                        p = layer[cur_no + "_22_" + str(number_Patch)]
                        patch = Ellipse(
                            xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                            width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="green", fc='None', lw=2)
                    #elif layer.has_key(cur_no + "_23_" + str(number_Patch)):
                    elif (cur_no + "_23_" + str(number_Patch)) in layer:
                        p = layer[cur_no + "_23_" + str(number_Patch)]
                        patch = Ellipse(
                            xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                            width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="blue", fc='None', lw=2)
                    #elif layer.has_key(cur_no + "_31_" + str(number_Patch)):
                    elif (cur_no + "_31_" + str(number_Patch)) in layer:
                        p = layer[cur_no + "_31_" + str(number_Patch)]
                        patch = patches.PathPatch(p, fill=False, edgecolor='red', lw=2)
                    #elif layer.has_key(cur_no + "_32_" + str(number_Patch)):
                    elif (cur_no + "_32_" + str(number_Patch)) in layer:
                        p = layer[cur_no + "_32_" + str(number_Patch)]
                        patch = patches.PathPatch(p, fill=False, edgecolor='green', lw=2)
                    #elif layer.has_key(cur_no + "_33_" + str(number_Patch)):
                    elif (cur_no + "_33_" + str(number_Patch)) in layer:
                        p = layer[cur_no + "_33_" + str(number_Patch)]
                        patch = patches.PathPatch(p, fill=False, edgecolor='blue', lw=2)
                    self.ax.add_patch(patch)
                    number_Patch += 1

            self.fig.canvas.draw()

    def click(self, event):
        current_mrt_layer = self.get_currentLayerNumber()
        if event.key == 'right':
            if self.mrt_layer_set[current_mrt_layer].get_current_Number() < self.mrt_layer_set[
                current_mrt_layer].get_number_mrt() - 1:
                self.mrt_layer_set[current_mrt_layer].increase_current_Number()
                plt.cla()
                plt.xlim(0, self.mrt_layer_set[current_mrt_layer].get_mrt_width())
                plt.ylim(self.mrt_layer_set[current_mrt_layer].get_mrt_height(), 0)
                self.pltc = plt.pcolormesh(self.mrt_layer_set[current_mrt_layer].get_x_arange(), self.mrt_layer_set[current_mrt_layer].get_y_arange(), np.swapaxes(self.mrt_layer_set[current_mrt_layer].get_current_Slice(), 0, 1), vmin=0, vmax=2094)
                self.number_mrt_label.config(
                    text=str(self.mrt_layer_set[current_mrt_layer].get_current_Number() + 1) + "/" + str(
                        self.mrt_layer_set[current_mrt_layer].get_number_mrt()))
                self.loadMark()
                self.fig.canvas.draw_idle()

        elif event.key == 'left':
            if self.mrt_layer_set[current_mrt_layer].get_current_Number() > 0:
                self.mrt_layer_set[current_mrt_layer].decrease_current_Number()
                plt.cla()
                plt.xlim(0, self.mrt_layer_set[current_mrt_layer].get_mrt_width())
                plt.ylim(self.mrt_layer_set[current_mrt_layer].get_mrt_height(), 0)
                self.pltc = plt.pcolormesh(self.mrt_layer_set[current_mrt_layer].get_x_arange(),
                               self.mrt_layer_set[current_mrt_layer].get_y_arange(),
                               np.swapaxes(self.mrt_layer_set[current_mrt_layer].get_current_Slice(), 0, 1), vmin=0,
                               vmax=2094)
                self.number_mrt_label.config(
                    text=str(self.mrt_layer_set[current_mrt_layer].get_current_Number() + 1) + "/" + str(
                        self.mrt_layer_set[current_mrt_layer].get_number_mrt()))
                self.loadMark()
                self.fig.canvas.draw_idle()

    def mouse_clicked(self, event):
        if event.button == 2 and self.button1_activated:
            self.x_clicked = event.xdata
            self.y_clicked = event.ydata
            self.mouse_second_clicked = True

        elif event.button==3:
            current_mrt_layer = self.get_currentLayerNumber()
            art_mod = self.layer.get()
            proband = self.art_mod_label['text'][10:self.art_mod_label['text'].find('\n')]
            model = self.art_mod_label['text'][self.art_mod_label['text'].find('\n')+9:len(self.art_mod_label['text'])]
            print(proband)
            print(model)
            deleteMark = shelve.open(self.Path_marking + proband +".slv", writeback=True)
            layer = deleteMark[model]
            cur_no = str(self.mrt_layer_set[current_mrt_layer].get_current_Number())
            cur_pa = str(len(self.ax.patches)-1)
            #if layer.has_key(cur_no + "_11_" + cur_pa):
            if (cur_no + "_11_" + cur_pa) in layer:
                 del deleteMark[model][cur_no + "_11_" + cur_pa]
            #elif layer.has_key(cur_no + "_12_" + cur_pa):
            elif (cur_no + "_12_" + cur_pa) in layer:
                 del deleteMark[model][cur_no + "_12_" + cur_pa]
            #elif layer.has_key(cur_no + "_13_" + cur_pa):
            elif (cur_no + "_13_" + cur_pa) in layer:
                 del deleteMark[model][cur_no + "_13_" + cur_pa]
            #elif layer.has_key(cur_no + "_21_" + cur_pa):
            elif (cur_no + "_21_" + cur_pa) in layer:
                 del deleteMark[model][cur_no + "_21_" + cur_pa]
            #elif layer.has_key(cur_no + "_22_" + cur_pa):
            elif (cur_no + "_22_" + cur_pa) in layer:
                 del deleteMark[model][cur_no + "_22_" + cur_pa]
            #elif layer.has_key(cur_no + "_23_" + cur_pa):
            elif (cur_no + "_23_" + cur_pa) in layer:
                 del deleteMark[model][cur_no + "_23_" + cur_pa]
            #elif layer.has_key(cur_no + "_31_" + cur_pa):
            elif (cur_no + "_31_" + cur_pa) in layer:
                 del deleteMark[model][cur_no + "_31_" + cur_pa]
            #elif layer.has_key(cur_no + "_32_" + cur_pa):
            elif (cur_no + "_32_" + cur_pa) in layer:
                 del deleteMark[model][cur_no + "_32_" + cur_pa]
            #elif layer.has_key(cur_no + "_33_" + cur_pa):
            elif (cur_no + "_33_" + cur_pa) in layer:
                 del deleteMark[model][cur_no + "_33_" + cur_pa]

            deleteMark.close()
            plt.cla()
            plt.xlim(0, self.mrt_layer_set[current_mrt_layer].get_mrt_width())
            plt.ylim(self.mrt_layer_set[current_mrt_layer].get_mrt_height(), 0)
            self.pltc = plt.pcolormesh(self.mrt_layer_set[current_mrt_layer].get_x_arange(),
                           self.mrt_layer_set[current_mrt_layer].get_y_arange(),
                           np.swapaxes(self.mrt_layer_set[current_mrt_layer].get_current_Slice(), 0, 1), vmin=0,
                           vmax=2094)
            self.loadMark()
            self.fig.canvas.draw_idle()

    def mouse_move(self, event):
        if self.button1_activated and self.mouse_second_clicked:
            factor = 10
            __x = event.xdata - self.x_clicked
            __y = event.ydata - self.y_clicked
            print(__x)
            print(__y)
            v_min, v_max = self.pltc.get_clim()
            print(v_min, v_max)
            if __x >= 0 and __y >= 0:
                print("h")
                __vmin = np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = np.abs(__x) * factor - np.abs(__y) * factor
            elif __x < 0 and __y >= 0:
                print("h")
                __vmin = -np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor - np.abs(__y) * factor

            elif __x < 0 and __y < 0:
                print("h")
                __vmin = -np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor + np.abs(__y) * factor

            else:
                print("h")
                __vmin = np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = np.abs(__x) * factor + np.abs(__y) * factor

            v_min += __vmin
            v_max += __vmax
            print(v_min, v_max)
            self.pltc.set_clim(vmin=v_min, vmax=v_max)
            self.fig.canvas.draw()

    def mouse_release(self, event):
        if event.button == 2:
            self.mouse_second_clicked = False

    def loadMark(self):
        current_mrt_layer = self.get_currentLayerNumber()
        proband = self.art_mod_label['text'][10:self.art_mod_label['text'].find('\n')]
        model = self.art_mod_label['text'][self.art_mod_label['text'].find('\n') + 9:len(self.art_mod_label['text'])]
        print(proband)
        print(model)
        loadFile = shelve.open(self.Path_marking + proband +".slv")
        number_Patch = 0
        #if loadFile.has_key(model):
        if model in loadFile:
            layer_name = model
            layer = loadFile[layer_name]
            cur_no = str(self.mrt_layer_set[current_mrt_layer].get_current_Number())

            '''while layer.has_key(cur_no + "_11_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_12_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_13_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_21_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_22_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_23_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_31_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_32_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_33_" + str(number_Patch)): '''
            while (cur_no + "_11_" + str(number_Patch)) in layer \
                    or (cur_no + "_12_" + str(number_Patch)) in layer \
                    or (cur_no + "_13_" + str(number_Patch)) in layer \
                    or (cur_no + "_21_" + str(number_Patch)) in layer \
                    or (cur_no + "_22_" + str(number_Patch)) in layer \
                    or (cur_no + "_23_" + str(number_Patch)) in layer \
                    or (cur_no + "_31_" + str(number_Patch)) in layer \
                    or (cur_no + "_32_" + str(number_Patch)) in layer \
                    or (cur_no + "_33_" + str(number_Patch)) in layer:
                patch = None
                #if layer.has_key(cur_no+ "_11_" + str(number_Patch)):
                if (cur_no + "_11_" + str(number_Patch)) in layer:
                    p = layer[cur_no+ "_11_" + str(number_Patch)]
                    print(p)
                    patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]), np.abs(p[1] - p[3]), fill=False,
                                         edgecolor="red", lw=2)
                #elif layer.has_key(cur_no+ "_12_" + str(number_Patch)):
                elif (cur_no + "_12_" + str(number_Patch)) in layer:
                    p = layer[cur_no+ "_12_" + str(number_Patch)]
                    patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]), np.abs(p[1] - p[3]), fill=False,
                                         edgecolor="green", lw=2)
                #elif layer.has_key(cur_no+ "_13_" + str(number_Patch)):
                elif (cur_no + "_13_" + str(number_Patch)) in layer:
                    p = layer[cur_no+ "_13_" + str(number_Patch)]
                    patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]), np.abs(p[1] - p[3]), fill=False,
                                         edgecolor="blue", lw=2)
                #elif layer.has_key(cur_no + "_21_" + str(number_Patch)):
                elif (cur_no + "_21_" + str(number_Patch)) in layer:
                    p = layer[cur_no + "_21_" + str(number_Patch)]
                    patch = Ellipse(xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                                  width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="red", fc='None', lw=2)
                #elif layer.has_key(cur_no + "_22_" + str(number_Patch)):
                elif (cur_no + "_22_" + str(number_Patch)) in layer:
                    p = layer[cur_no + "_22_" + str(number_Patch)]
                    patch = Ellipse(xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                                    width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="green", fc='None', lw=2)
                #elif layer.has_key(cur_no + "_23_" + str(number_Patch)):
                elif (cur_no + "_23_" + str(number_Patch)) in layer:
                    p = layer[cur_no + "_23_" + str(number_Patch)]
                    patch = Ellipse(xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                                    width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="blue", fc='None', lw=2)
                #elif layer.has_key(cur_no + "_31_" + str(number_Patch)):
                elif (cur_no + "_31_" + str(number_Patch)) in layer:
                    p = layer[cur_no + "_31_" + str(number_Patch)]
                    patch = patches.PathPatch(p, fill=False, edgecolor='red', lw=2)
                #elif layer.has_key(cur_no + "_32_" + str(number_Patch)):
                elif (cur_no + "_32_" + str(number_Patch)) in layer:
                    p = layer[cur_no + "_32_" + str(number_Patch)]
                    patch = patches.PathPatch(p, fill=False, edgecolor='green', lw=2)
                #elif layer.has_key(cur_no + "_33_" + str(number_Patch)):
                elif (cur_no + "_33_" + str(number_Patch)) in layer:
                    p = layer[cur_no + "_33_" + str(number_Patch)]
                    patch = patches.PathPatch(p, fill=False, edgecolor='blue', lw=2)
                self.ax.add_patch(patch)
                number_Patch += 1

    def refresh_option(self, new_list):
        self.layer.set(new_list[len(new_list) - 1])
        self.option_layers['menu'].delete(0, 'end')
        for choice in new_list:
            self.option_layers['menu'].add_command(label=choice, command=lambda value=choice: self.layer.set(value))


    def change_layer(self, value):
        num = int(value.find(")"))
        print(value)
        current_mrt_layer = 0
        if num == 2:
            current_mrt_layer = int(self.layer.get()[1:2]) - 1
        elif num == 3:
            current_mrt_layer = int(self.layer.get()[1:3]) - 1

        ArrayDicom = self.mrt_layer_set[current_mrt_layer].get_Dicom_array()
        plt.cla()
        plt.xlim(0, self.mrt_layer_set[current_mrt_layer].get_mrt_width())
        plt.ylim(self.mrt_layer_set[current_mrt_layer].get_mrt_height(), 0)
        plt.pcolormesh(self.mrt_layer_set[current_mrt_layer].get_x_arange(),
                       self.mrt_layer_set[current_mrt_layer].get_y_arange(), np.rot90(
                ArrayDicom[:, :, 0]))
        self.number_mrt_label.config(text=str("1/" + str(self.mrt_layer_set[current_mrt_layer].get_number_mrt())))
        loadFile = shelve.open("mark_mrt_layer.slv")
        number_Patch = 0
        #if loadFile.has_key(self.mrt_layer_set[current_mrt_layer].get_model_name()):
        if (self.mrt_layer_set[current_mrt_layer].get_model_name()) in loadFile:
            layer_name = self.mrt_layer_set[current_mrt_layer].get_model_name()
            layer = loadFile[layer_name]
            '''while layer.has_key(str(self.mrt_layer_set[current_mrt_layer].get_current_Number()) + "_" + str(
                    number_Patch)):'''
            while (str(self.mrt_layer_set[current_mrt_layer].get_current_Number()) + "_" + str(number_Patch)) in layer:
                num_str = str(self.mrt_layer_set[current_mrt_layer].get_current_Number()) + "_" + str(number_Patch)
                p = loadFile[layer_name][num_str]
                patch = None
                if self.chooseArtefact.get() == self.artefact_list[0]:
                    patch = patches.PathPatch(p, fill=False, edgecolor='red', lw=2)
                elif self.chooseArtefact.get() == self.artefact_list[1]:
                    patch = patches.PathPatch(p, fill=False, edgecolor='green', lw=2)
                elif self.chooseArtefact.get() == self.artefact_list[2]:
                    patch = patches.PathPatch(p, fill=False, edgecolor='blue', lw=2)
                self.ax.add_patch(patch)
                number_Patch += 1
        self.fig.canvas.draw_idle()

    def get_currentLayerNumber(self):
        num = int(self.layer.get().find(")"))
        current_mrt_layer = 0
        if num == 2:
            current_mrt_layer = int(self.layer.get()[1:2]) - 1
        elif num == 3:
            current_mrt_layer = int(self.layer.get()[1:3]) - 1
        return current_mrt_layer

    def chooseData(self):
        process = CNN_Preprocessing(self.sFolder, self.artefact, self.proband)

    def save(self):
        matFile_set = []
        mrt_image_size = {}
        for number_layer in range(0, len(self.mrt_layer_set), 1):
            image_size = str(self.mrt_layer_set[number_layer].get_mrt_width()) + "x" + str(
                self.mrt_layer_set[number_layer].get_mrt_height())
            #if mrt_image_size.has_key(image_size):
            if image_size in mrt_image_size:
                matFile_set[int(mrt_image_size[image_size])].concatenate_arrays(
                    self.mrt_layer_set[number_layer].get_Dicom_array(), self.mrt_layer_set[number_layer].get_mask())
            else:
                mrt_array_set = MRT_array(self.mrt_layer_set[number_layer].get_mrt_width(),
                                          self.mrt_layer_set[number_layer].get_mrt_height(),
                                          self.mrt_layer_set[number_layer].get_Dicom_array(),
                                          self.mrt_layer_set[number_layer].get_mask())
                dic_number = len(mrt_image_size)
                dic_image = {image_size: dic_number}
                mrt_image_size.update(dic_image)
                matFile_set.append(mrt_array_set)

        matFile = {}
        for number_mat in range(0, len(matFile_set), 1):
            dicom_name = "Dicom" + str(matFile_set[number_mat].get_mrt_width()) + "_" + str(
                matFile_set[number_mat].get_mrt_height())
            mask_name = "Mask" + str(matFile_set[number_mat].get_mrt_width()) + "_" + str(
                matFile_set[number_mat].get_mrt_height())
            matF = {dicom_name: matFile_set[number_mat].get_dicom_set(),
                    mask_name: matFile_set[number_mat].get_labeling_mask()}
            matFile.update(matF)

        scipy.io.savemat("Dicom_mask.mat", matFile)

    def exit(self):
        self.window.destroy()

if __name__ == "__main__":
    m = Artefact_Labeling_GUI()
