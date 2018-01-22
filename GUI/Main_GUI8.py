from Tkinter import *
import ttk
import Pmw
from PIL import Image, ImageTk
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import LassoSelector, RectangleSelector, EllipseSelector
from matplotlib import path
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from DataPreprocessing import*
from DatasetSplit import*
from MRT_Layer import*
from cnn_main import*
import scipy.io as sio
from tkFileDialog import askopenfilename
import h5py
from Unpatching import*
import matplotlib as mpl

class MainGUI():

    def __init__(self):
        self.root = Tk()
        self.root.title('PatchBased Labeling')
        self.root.geometry('1425x770')
        self.root.configure(background="gray38")
        # Path
        self.sFolder = "/home/d1224/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/"
        self.Path_markings = "/home/d1224/no_backup/d1224/PatchbasedLabeling Results/Markings/"
        self.Path_train_results = "/home/d1224/no_backup/d1224/PatchbasedLabeling Results/results/"
        self.Path_data = "/home/d1224/no_backup/d1224/PatchbasedLabeling Results/Training and Test data/"

        Pmw.initialise()
        self.list_of_images = ["Move.png", "Rectangle.png", "Ellipse.png", "Lasso.png", "3D.png"]
        self.artefact_list = ["Movement-Artefact", "Shim-Artefact", "Noise-Artefact"]
        self.tool_buttons = []
        self.mrt_layer_names = []
        self.mrt_layer_set = []
        self.optionlist = []
        self.activated_button = 0
        self.button1_activated = True
        self.x_clicked = None
        self.y_clicked = None
        self.mouse_second_clicked = False
        self.nb = Pmw.NoteBook(self.root)
        self.p1 = self.nb.add('DICOM-Image Processing')
        self.p2 = self.nb.add('Data Preprocessing')
        self.p3 = self.nb.add('Convolution Neural Network')
        self.p4 = self.nb.add('Viewing results')

        self.nb.pack(padx=5, pady=5, fill=BOTH, expand=1)

        # GUI first tab: DICOM-Image Processing
        #################################################################################################################################################################################
        self.proband = os.listdir(self.sFolder)
        self.proband.sort()
        self.model = os.listdir(self.sFolder + self.proband[0] + "/dicom_sorted")
        self.model.sort()

        self.rahmen1 = Frame(self.p1, bg="gray65", bd=3, relief="sunken")
        self.rahmen1.place(x=5, y=5, width=215, height=250)
        self.rahmen2 = Frame(self.p1, bg="gray65", bd=3, relief="sunken")
        self.rahmen2.place(x=5, y=260, width=215, height=165)

        image = Image.open("open.png")
        img_open = ImageTk.PhotoImage(image)
        label = Label(image=img_open)
        label.image = img_open
        self.load_button = Button(self.rahmen1, image=img_open, text="Load MRT-Layer", font=('Arial', 11, 'bold'),
                                  bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT, command = self.load_MRT)
        self.load_button.place(x=5, y=5, width=200, height=40)
        self.path_entry = Entry(self.rahmen1)
        self.path_entry.place(x=88, y=50, width=115, height=25)
        self.path_entry.insert(0, self.sFolder)

        self.proband_str = StringVar(self.p1)
        self.proband_str.set(self.proband[0])
        self.proband_option = OptionMenu(self.rahmen1, self.proband_str, *self.proband)
        self.proband_option.config(bg="gray56")

        self.artefact_str = StringVar(self.p1)
        self.artefact_str.set(self.model[0])
        self.artefact_option = OptionMenu(self.rahmen1, self.artefact_str, *self.model)
        self.artefact_option.config(bg="gray56")

        self.layer = StringVar(self.p1)
        self.layer.set("(empty)")
        self.option_layers = OptionMenu(self.rahmen1, self.layer, "(empty)")
        self.option_layers.config(bg="gray56")

        self.proband_option.place(x=5, y=80, width=200,
                                  height=50)  # self.proband_option.place(x=5, y=55, width=200, height=50)
        self.artefact_option.place(x=5, y=135, width=200,
                                   height=50)  # self.artefact_option.place(x=5, y=105, width=200, height=50)
        self.option_layers.place(x=5, y=190, width=200,
                                 height=50)  # self.option_layers.place(x=5, y=155, width=200, height=50)

        self.path_label = Label(self.rahmen1, bg="gray65", font="Aral 10 bold", text="DICOM Path")
        self.path_label.place(x=5, y=52)

        self.draw_label = Label(self.rahmen2, bg="gray65", font="Aral 10 bold", text="Choose Drawing Tools")
        self.art_label = Label(self.rahmen2, bg="gray65", font="Aral 10 bold", text="Choose artefact type")
        self.draw_label.place(x=5, y=5)
        self.art_label.place(x=5, y=80)

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
            b = Button(self.rahmen2, image=img_tool, bg="gray56", borderwidth=2, command=lambda i=i: onClick(i))
            if i == 5:
                row = 45
                column = 5
            else:
                row = 30

            b.place(x=column, y=row)
            column += 40
            self.tool_buttons.append(b)
            self.tool_buttons[self.activated_button].configure(relief=SUNKEN)

        self.chooseArtefact = StringVar(self.root)
        self.chooseArtefact.set(self.artefact_list[0])
        self.chooseArtefact_option = OptionMenu(self.rahmen2, self.chooseArtefact, *self.artefact_list)
        self.chooseArtefact_option.config(bg="gray56")
        self.chooseArtefact_option.place(x=5, y=105, width=200, height=50)

        self.panel = Label(self.p1, bg="black")
        self.panel2 = Label(self.p1, bg="LightSteelBlue4")
        self.panel.place(x=225, y=5, width=1204, height=764)
        self.panel2.place(x=225, y=5, width=1204, height=50)
        self.art_mod_label = Label(self.p1, bg="LightSteelBlue4", fg="white", font="Aral 12 bold")
        self.number_mrt_label = Label(self.p1, bg="LightSteelBlue4", fg="white", font="Aral 12 bold")
        self.v_min_label = Label(self.p1, bg="LightSteelBlue4", fg="white", font="Aral 9 bold")
        self.v_max_label = Label(self.p1, bg="LightSteelBlue4", fg="white", font="Aral 9 bold")

        self.fig = plt.figure(dpi=50)
        # self.fig.patch.set_facecolor('black')
        self.ax = plt.gca()
        self.pltc = None
        # self.ax.set_axis_bgcolor('black')
        self.ax.text(0.5, 0.5, 'To label MRT-Artefacts load MRT-Layer', horizontalalignment='center',
                     verticalalignment='center', color='white', fontsize=20, transform=self.ax.transAxes)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.p1)
        self.canvas.show()
        self.canvas.get_tk_widget().place(x=600, y=250)  # expand=1

        def lasso_onselect(verts):
            print (verts)
            p = path.Path(verts)
            current_mrt_layer = self.get_currentLayerNumber()
            proband = self.mrt_layer_set[current_mrt_layer].get_proband()
            model = self.mrt_layer_set[current_mrt_layer].get_model()
            print(proband)
            print(model)
            saveFile = shelve.open(self.Path_markings + proband + ".slv", writeback=True)
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
            if saveFile.has_key(layer_name):
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
            col_str = None
            rect = None
            ell = None
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            current_mrt_layer = self.get_currentLayerNumber()
            proband = self.mrt_layer_set[current_mrt_layer].get_proband()
            model = self.mrt_layer_set[current_mrt_layer].get_model()
            print(proband)
            print(model)
            saveFile = shelve.open(self.Path_markings + proband +".slv", writeback=True)
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

            if saveFile.has_key(layer_name):
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

        # GUI second tab: Data Preprocessing
        #############################################################################################################################################################################
        self.list_var = []
        self.list_varl = []
        self.list_proband = []
        self.list_modell = []

        self.tab2_rahmen1 = Frame(self.p2, bg="gray55", bd=3, relief="sunken")
        self.tab2_rahmen1.place(x=5, y=20, width=400, height=600)
        self.tab2_rahmen2 = Frame(self.p2, bg="gray55", bd=3, relief="sunken")
        self.tab2_rahmen2.place(x=450, y=20, width=540, height=600)
        self.lf_result = LabelFrame(self.p2, text='Patching-Results', font="Aral 12 bold")
        self.lf_result.place(x=1000, y=20, width=400, height=600)

        self.start_pre_button = Button(self.p2, text="Start Data Preprocessing", font=('Arial', 11, 'bold'),
                                       bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT,
                                       command=self.fData_Preprocessing)
        self.start_pre_button.place(x=1200, y=680, width=200, height=40)

        self.progress_label = Label(self.p2, font="Aral 10 bold", text="RigidPatching...")
        self.progress_label.place(x=5, y=655)
        self.progress = ttk.Progressbar(self.p2, orient="horizontal", length=600, mode="determinate")
        self.progress.place(x=5, y=680, width=1190, height=40)

        self.lf_proband = LabelFrame(self.tab2_rahmen1, text='Proband', bg="gray55", font="Aral 12 bold")
        self.lf_proband.place(x=20, y=10, width=360, height=200)
        self.lf_model = LabelFrame(self.tab2_rahmen1, text='Model', bg="gray55", font="Aral 12 bold")
        self.lf_model.place(x=20, y=220, width=360, height=350)
        col = 1
        row = 1
        for prob in range(0, len(self.proband), 1):
            var = IntVar()
            self.list_var.append(var)
            chk = Checkbutton(self.lf_proband, bg="gray55", text=self.proband[prob], font=('Arial', 11, 'bold'),
                                  variable=self.list_var[prob])  # tab2_rahmen1
            chk.grid(row=row + 1, column=col + 1, sticky=W)
            self.list_proband.append(chk)
            if col == 5:
                col = 1
                row += 1
            else:
                col += 1

        for mod in range(0, len(self.model), 1):
            var1 = IntVar()
            self.list_varl.append(var1)
            if self.model[mod] == "FastView_0001":
                chk = Checkbutton(self.lf_model, bg="gray55", text=self.model[mod], font=('Arial', 10, 'bold'),
                                  variable=self.list_varl[mod], state=DISABLED)  # state = DISABLED
                # chk.state(['!selected'])
                chk.grid(row=mod + 1, column=1, sticky=W)
                self.list_modell.append(chk)
            else:
                chk = Checkbutton(self.lf_model, bg="gray55", text=self.model[mod], font=('Arial', 10, 'bold'),
                                  variable=self.list_varl[mod])
                # chk.state(['!selected'])
                chk.grid(row=mod + 1, column=1, sticky=W)
                self.list_modell.append(chk)

        self.ch_patching_process = ["Rigid-Patching", "Adaptive-Splitting"]
        self.ch_patch_size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.ch_patch_overlap = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.ch_data_splitting = ["normal", "normal_rand", "crossvalidation_data", "crossvalidation_patient"]
        self.ch_dimension = ["2D", "3D"]

        self.lf_patching = LabelFrame(self.tab2_rahmen2, text='Patching', bg="gray55", font="Aral 12 bold")
        self.lf_patching.place(x=20, y=10, width=500, height=200)
        self.lf_data_split = LabelFrame(self.tab2_rahmen2, text='Data Splitting', bg="gray55", font="Aral 12 bold")
        self.lf_data_split.place(x=20, y=220, width=500, height=350)

        # heading_frame1 = Label(tab2_rahmen2, text="Data Preprocessing", bg="gray55", font="Aral 14 bold")
        # heading_frame1.place(x=55, y=2)

        self.Patching_process = Label(self.lf_patching, text="Choose Patching-Process", bg="gray55",
                                      font="Aral 10 bold")
        self.Patching_process.place(x=5, y=5)
        self.patching_process = StringVar(self.root)
        self.patching_process.set(self.ch_patching_process[0])
        self.patching_process_option = OptionMenu(self.lf_patching, self.patching_process, *self.ch_patching_process)
        self.patching_process_option.place(x=15, y=30, width=200, height=40)

        self.patch_size = Label(self.lf_patching, text="Choose Patch-Size (2D or 3D)", bg="gray55", font="Aral 10 bold")
        self.patch_size.place(x=5, y=80)
        self.patchsize1 = StringVar(self.root)
        self.patchsize1.set(self.ch_patch_size[3])
        self.patchsize1_option = OptionMenu(self.lf_patching, self.patchsize1, *self.ch_patch_size)
        self.patchsize1_option.place(x=15, y=105, width=50, height=40)

        self.patchsize2 = StringVar(self.root)
        self.patchsize2.set(self.ch_patch_size[3])
        self.patchsize2_option = OptionMenu(self.lf_patching, self.patchsize2, *self.ch_patch_size)
        self.patchsize2_option.place(x=90, y=105, width=50, height=40)

        self.patchsize3 = StringVar(self.root)
        self.patchsize3.set(self.ch_patch_size[3])
        self.patchsize3_option = OptionMenu(self.lf_patching, self.patchsize3, *self.ch_patch_size)
        self.patchsize3_option.place(x=165, y=105, width=50, height=40)

        self.patch_overlap = Label(self.lf_patching, text="Choose Patch-Overlap", bg="gray55", font="Aral 10 bold")
        self.patch_overlap.place(x=270, y=5)

        self.patchOver = StringVar(self.root)
        self.patchOver.set(self.ch_patch_overlap[5])
        self.patchOver_option = OptionMenu(self.lf_patching, self.patchOver, *self.ch_patch_overlap)
        self.patchOver_option.place(x=285, y=30, width=200, height=40)

        self.path_result_button = Button(self.lf_data_split, text="Choose Path", font=('Arial', 9, 'bold'),
                                       bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT,
                                       command=self.open_explorer)
        self.path_result_button.place(x=390, y=20, width=100, height=25)
        self.path_result_entry = Entry(self.lf_data_split)
        self.path_result_entry.place(x=10, y=20, width=370, height=25)

        self.split_data = Label(self.lf_data_split, text="Data-Splitting", bg="gray55", font="Aral 10 bold")
        self.split_data.place(x=5, y=65)

        self.data_split = StringVar(self.root)
        self.data_split.set(self.ch_data_splitting[0])
        self.data_split_option = OptionMenu(self.lf_data_split, self.data_split, *self.ch_data_splitting)
        self.data_split_option.place(x=15, y=90, width=200, height=40)

        self.split_val = Label(self.lf_data_split, text="Split-Ratio", bg="gray55", font="Aral 10 bold")
        self.split_val.place(x=5, y=140)

        self.split_rat = StringVar(self.root)
        self.split_rat.set(self.ch_patch_overlap[1])
        self.split_rat_option = OptionMenu(self.lf_data_split, self.split_rat, *self.ch_patch_overlap)
        self.split_rat_option.place(x=15, y=165, width=200, height=40)

        self.split_val_lab = Label(self.lf_data_split, text="Split-Ratio for Labeling", bg="gray55", font="Aral 10 bold")
        self.split_val_lab.place(x=5, y=215)

        self.split_rat_lab = StringVar(self.root)
        self.split_rat_lab.set(self.ch_patch_overlap[0])
        self.split_rat_lab_option = OptionMenu(self.lf_data_split, self.split_rat_lab, *self.ch_patch_overlap)
        self.split_rat_lab_option.place(x=15, y=240, width=200, height=40)

        self.dim_lab = Label(self.lf_patching, text="Dimension", bg="gray55", font="Aral 10 bold")
        self.dim_lab.place(x=270, y=80)

        self.dim_rat_lab = StringVar(self.root)
        self.dim_rat_lab.set(self.ch_dimension[0])
        self.dim_rat_lab_option = OptionMenu(self.lf_patching, self.dim_rat_lab, *self.ch_dimension)
        self.dim_rat_lab_option.place(x=285, y=105, width=200, height=40)

        # GUI tab3: CNN
        #####################################################################################################################################################

        self.ch_cnn_model = ["motion_head", "motion_abd", "motion_all", "shim", "noise", "allData",
                             "3D"]
        self.ch_parameter_optim = ["none", "grid"]
        self.ch_batchSize = [32, 64, 128]
        self.ch_learning_rate = [0.1, 0.01, 0.05, 0.005, 0.001]
        self.ch_epoch = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        self.cnn_method = ["Training", "Prediction"]

        self.rahmen1_p3 = Frame(self.p3, bg="gray55", bd=3, relief="sunken")
        self.rahmen1_p3.place(x=5, y=5, width=400, height=600)
        self.lf_cnn = LabelFrame(self.rahmen1_p3, text='Parameter for CNN', bg="gray55", font="Aral 12 bold")
        self.lf_cnn.place(x=20, y=20, width=360, height=560)
        self.path_cnn_button = Button(self.lf_cnn, text="Choose Path", font=('Arial', 9, 'bold'),
                                         bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT,
                                         command=self.open_explorer)
        self.path_cnn_button.place(x=240, y=20, width=100, height=25)
        self.path_cnn_entry = Entry(self.lf_cnn)
        self.path_cnn_entry.place(x=10, y=20, width=220, height=25)

        self.Cnn_model = Label(self.lf_cnn, text="CNN-Model", bg="gray55", font="Aral 10 bold")
        self.Cnn_model.place(x=5, y=60)
        self.cnn_model = StringVar(self.root)
        self.cnn_model.set(self.ch_cnn_model[0])
        self.cnn_model_option = OptionMenu(self.lf_cnn, self.cnn_model, *self.ch_cnn_model)
        self.cnn_model_option.place(x=80, y=85, width=200, height=40)

        self.para_optim = Label(self.lf_cnn, text="Parameter-Optimization", bg="gray55", font="Aral 10 bold")
        self.para_optim.place(x=5, y=135)
        self.para_opt = StringVar(self.root)
        self.para_opt.set(self.ch_parameter_optim[0])
        self.para_opt_option = OptionMenu(self.lf_cnn, self.para_opt, *self.ch_parameter_optim)
        self.para_opt_option.place(x=80, y=160, width=200, height=40)

        self.batch = Label(self.lf_cnn, text="Batch-Size", bg="gray55", font="Aral 10 bold")
        self.batch.place(x=5, y=210)
        self.batch_opt = StringVar(self.root)
        self.batch_opt.set(self.ch_batchSize[1])
        self.batch_opt_option = OptionMenu(self.lf_cnn, self.batch_opt, *self.ch_batchSize)
        self.batch_opt_option.place(x=80, y=235, width=200, height=40)

        self.learn = Label(self.lf_cnn, text="Learning-Rate", bg="gray55", font="Aral 10 bold")
        self.learn.place(x=5, y=285)
        self.learn_opt = StringVar(self.root)
        self.learn_opt.set(self.ch_learning_rate[1])
        self.learn_opt_option = OptionMenu(self.lf_cnn, self.learn_opt, *self.ch_learning_rate)
        self.learn_opt_option.place(x=80, y=310, width=200, height=40)

        self.epoch = Label(self.lf_cnn, text="Epoche", bg="gray55", font="Aral 10 bold")
        self.epoch.place(x=5, y=360)
        self.epoch_opt = StringVar(self.root)
        self.epoch_opt.set(self.ch_epoch[2])
        self.epoch_opt_option = OptionMenu(self.lf_cnn, self.epoch_opt, *self.ch_epoch)
        self.epoch_opt_option.place(x=80, y=385, width=200, height=40)

        self.method = Label(self.lf_cnn, text="CNN-Method", bg="gray55", font="Aral 10 bold")
        self.method.place(x=5, y=435)
        self.method_opt = StringVar(self.root)
        self.method_opt.set(self.cnn_method[0])
        self.method_opt_option = OptionMenu(self.lf_cnn, self.method_opt, *self.cnn_method)
        self.method_opt_option.place(x=80, y=460, width=200, height=40)

        self.start_button_frame2 = Button(self.p3, text="Start Training CNN", font=('Arial', 11, 'bold'),
                                          bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT, command = self.start_cnn)
        self.start_button_frame2.place(x=1200, y=680, width=200, height=40)

        self.progress_label_p3 = Label(self.p3, font="Aral 10 bold", text="Training...")
        self.progress_label_p3.place(x=5, y=655)
        self.progress_p3 = ttk.Progressbar(self.p3, orient="horizontal", length=600, mode="determinate")
        self.progress_p3.place(x=5, y=680, width=1190, height=40)

        #######################################################################################################################################################

        # GUI tab4: Visualize Results
        self.rahmen1_p4 = Frame(self.p4, bg="gray65", bd=3, relief="sunken")
        self.rahmen1_p4.place(x=5, y=5, width=215, height=115)
        self.view_label_acc = Label(self.rahmen1_p4, bg="gray65", font="Aral 12 bold", text="View Accuracy and Loss")
        self.view_label_acc.place(x=2, y=2)
        self.path_acc_button = Button(self.rahmen1_p4, text="Choose Path", font=('Arial', 9, 'bold'),
                                      bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT,
                                      command=self.open_explorer)
        self.path_acc_button.place(x=5, y=30, width=100, height=25)
        self.path_acc_entry = Entry(self.rahmen1_p4)
        self.path_acc_entry.place(x=105, y=30, width=100, height=25)
        self.plot_acc_loss = Button(self.rahmen1_p4, text="Plot Accuracy and Loss", font=('Arial', 10, 'bold'),
                                          bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT,
                                          command=self.start_cnn)
        self.plot_acc_loss.place(x=5, y=65, width=200, height=40)

        self.rahmen2_p4 = Frame(self.p4, bg="gray65", bd=3, relief="sunken")
        self.rahmen2_p4.place(x=5, y=130, width=215, height=280)
        self.view_label_map = Label(self.rahmen2_p4, bg="gray65", font="Aral 12 bold", text="View Probability-Map")
        self.view_label_map.place(x=2, y=2, height = 30)
        self.proband_str_view = StringVar(self.p1)
        self.proband_str_view.set(self.proband[0])
        self.proband_option_view = OptionMenu(self.rahmen2_p4, self.proband_str_view, *self.proband)
        self.proband_option_view.config(bg="gray56")

        self.artefact_str_view = StringVar(self.p1)
        self.artefact_str_view.set(self.model[0])
        self.artefact_option_view = OptionMenu(self.rahmen2_p4, self.artefact_str_view, *self.model)
        self.artefact_option_view.config(bg="gray56")
        self.proband_option_view.place(x=5, y = 35, width = 200, height = 50)
        self.artefact_option_view.place(x=5, y = 90, width = 200, height = 50)

        self.path_pred_button = Button(self.rahmen2_p4, text="Choose Prediction File", font=('Arial', 9, 'bold'),
                                      bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT,
                                      command=self.open_explorer)
        self.path_pred_button.place(x=5, y=150, width=160, height=25)
        self.path_pred_entry = Entry(self.rahmen2_p4)
        self.path_pred_entry.place(x=5, y=180, width=200, height=25)
        self.plot_map = Button(self.rahmen2_p4, text="Plot Probability-Map", font=('Arial', 10, 'bold'),
                                    bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT,
                                    command=self.plotProbMap)
        self.plot_map.place(x=5, y=215, width=200, height=40)

        self.rahmen3_p4 = Frame(self.p4, bg="gray65", bd=3, relief="sunken")
        self.rahmen3_p4.place(x=5, y=420, width=215, height=100)
        self.view_label_konfusion = Label(self.rahmen3_p4, bg="gray65", font="Aral 12 bold", text="View Konfusionsmatrix")
        self.view_label_konfusion.place(x=2, y=2, height=30)

        self.panel_p4 = Canvas(self.p4, width=1204, height=764, bg="black")
        self.panel2_p4 = Canvas(self.p4, width=1204, height=50, bg="LightSteelBlue4")
        self.panel_p4.place(x=225, y=5)
        self.panel2_p4.place(x=225, y=5)
        self.art_mod_label_p4 = Label(self.p4, bg="LightSteelBlue4", fg="white", font="Aral 12 bold",
                                   text="Proband: 01_ab,   Model: t1_tse_tra_Kopf_0002")
        self.number_mrt_label_p4 = Label(self.p4, bg="LightSteelBlue4", fg="white", font="Aral 12 bold", text="1/40")
        self.art_mod_label_p4.place(x=227, y=7)
        self.number_mrt_label_p4.place(x=1350, y=7)

        #self.fig2, self.ax2 = plt.subplots(figsize=(16, 0.3), dpi=60)
        #self.fig2.subplots_adjust(left=0, bottom=0, right=1, top=1,
        #                     wspace=0, hspace=0)
        #cmap = 'jet'
        #norm = mpl.colors.Normalize(vmin=5, vmax=10)
        #cb1 = mpl.colorbar.ColorbarBase(self.ax2, cmap=cmap,
         #                               norm=norm,
          #                              orientation='horizontal')
        #plt.axis('off')

        #canvas2 = FigureCanvasTkAgg(self.fig2, master=self.panel2_p4)
        #canvas2.get_tk_widget().configure(background='LightSteelBlue4', highlightcolor='LightSteelBlue4',
         #                                highlightbackground='LightSteelBlue4')
        #canvas2.show()
        #canvas2.get_tk_widget().place(x=100, y=29)  # expand=1

        self.root.mainloop()

    def load_MRT(self):
        mrt_layer_name = str(self.proband_str.get()) + "_" + str(self.artefact_str.get())
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
                print(voxel_ndarray.shape)
            except dicom_numpy.DicomImportException:
                # invalid DICOM data
                raise

            dx, dy, dz = 1.0, 1.0, pixel_space[2][2] #pixel_space[0][0], pixel_space[1][1], pixel_space[2][2]
            pixel_array = voxel_ndarray[:, :, 0]
            x_1d = dx * np.arange(voxel_ndarray.shape[0])
            y_1d = dy * np.arange(voxel_ndarray.shape[1])
            z_1d = dz * np.arange(voxel_ndarray.shape[2])

            model = self.artefact_str.get()
            proband = self.proband_str.get()
            mrt_layer = MRT_Layer(voxel_ndarray.shape[0], voxel_ndarray.shape[1], voxel_ndarray.shape[2],
                                  proband, model, x_1d,y_1d, z_1d, 0, 2000,
                                  voxel_ndarray, mrt_layer_name)
            self.mrt_layer_set.append(mrt_layer)
            self.optionlist.append("(" + str(len(self.mrt_layer_set)) + ") " + mrt_layer_name + ": " + str(
                voxel_ndarray.shape[0]) + " x " + str(
                voxel_ndarray.shape[1]))
            self.refresh_option(self.optionlist)

            self.number_mrt_label.config(text="1/" + str(mrt_layer.get_number_mrt()))
            self.art_mod_label.config(text = "Proband:  " + str(self.proband_str.get()) + ",      Model:  " + str(self.artefact_str.get())) #text="Proband: 01_ab,   Model: t1_tse_tra_Kopf_0002"
            self.v_min_label.config(text = "0")
            self.v_max_label.config(text = "2000")
            self.art_mod_label.place(x=227, y=7)
            self.number_mrt_label.place(x=1350, y=7)
            self.v_min_label.place(x=240, y=30)
            self.v_max_label.place(x=1330, y=30)
            plt.cla()
            plt.gca().set_aspect('equal')  # plt.axes().set_aspect('equal')
            plt.xlim(0, voxel_ndarray.shape[0]*dx)
            plt.ylim(voxel_ndarray.shape[1]*dy, 0)
            plt.set_cmap(plt.gray())
            self.pltc = plt.pcolormesh(x_1d, y_1d, np.swapaxes(pixel_array, 0, 1), vmin=0, vmax=2094)

            # load
            File_Path = self.Path_markings + self.proband_str.get() +".slv"
            loadFile = shelve.open(File_Path)
            number_Patch = 0
            cur_no = "0"
            if loadFile.has_key(self.artefact_str.get()):
                layer = loadFile[self.artefact_str.get()]
                while layer.has_key(cur_no + "_11_" + str(number_Patch)) or layer.has_key(cur_no + "_12_" + str(number_Patch)) or layer.has_key(cur_no + "_13_" + str(number_Patch)) or layer.has_key(cur_no + "_21_" + str(number_Patch)) or layer.has_key(cur_no + "_22_" + str(number_Patch)) or layer.has_key(cur_no + "_23_" + str(number_Patch)) or layer.has_key(cur_no + "_31_" + str(number_Patch)) or layer.has_key(cur_no + "_32_" + str(number_Patch)) or layer.has_key(cur_no + "_33_" + str(number_Patch)):
                    patch = None
                    if layer.has_key(cur_no + "_11_" + str(number_Patch)):
                        p = layer[cur_no + "_11_" + str(number_Patch)]
                        patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]),
                                              np.abs(p[1] - p[3]), fill=False,
                                              edgecolor="red", lw=2)
                    elif layer.has_key(cur_no + "_12_" + str(number_Patch)):
                        p = layer[cur_no + "_12_" + str(number_Patch)]
                        patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]),
                                              np.abs(p[1] - p[3]), fill=False,
                                              edgecolor="green", lw=2)
                    elif layer.has_key(cur_no + "_13_" + str(number_Patch)):
                        p = layer[cur_no + "_13_" + str(number_Patch)]
                        patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]),
                                              np.abs(p[1] - p[3]), fill=False,
                                              edgecolor="blue", lw=2)
                    elif layer.has_key(cur_no + "_21_" + str(number_Patch)):
                        p = layer[cur_no + "_21_" + str(number_Patch)]
                        patch = Ellipse(
                            xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                            width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="red", fc='None', lw=2)
                    elif layer.has_key(cur_no + "_22_" + str(number_Patch)):
                        p = layer[cur_no + "_22_" + str(number_Patch)]
                        patch = Ellipse(
                            xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                            width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="green", fc='None', lw=2)
                    elif layer.has_key(cur_no + "_23_" + str(number_Patch)):
                        p = layer[cur_no + "_23_" + str(number_Patch)]
                        patch = Ellipse(
                            xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                            width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="blue", fc='None', lw=2)
                    elif layer.has_key(cur_no + "_31_" + str(number_Patch)):
                        p = layer[cur_no + "_31_" + str(number_Patch)]
                        patch = patches.PathPatch(p, fill=False, edgecolor='red', lw=2)
                    elif layer.has_key(cur_no + "_32_" + str(number_Patch)):
                        p = layer[cur_no + "_32_" + str(number_Patch)]
                        patch = patches.PathPatch(p, fill=False, edgecolor='green', lw=2)
                    elif layer.has_key(cur_no + "_33_" + str(number_Patch)):
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
                v_min = self.mrt_layer_set[current_mrt_layer].get_v_min()
                v_max = self.mrt_layer_set[current_mrt_layer].get_v_max()
                self.pltc = plt.pcolormesh(self.mrt_layer_set[current_mrt_layer].get_x_arange(), self.mrt_layer_set[current_mrt_layer].get_y_arange(), np.swapaxes(self.mrt_layer_set[current_mrt_layer].get_current_Slice(), 0, 1), vmin=v_min, vmax=v_max)
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
                v_min = self.mrt_layer_set[current_mrt_layer].get_v_min()
                v_max = self.mrt_layer_set[current_mrt_layer].get_v_max()
                self.pltc = plt.pcolormesh(self.mrt_layer_set[current_mrt_layer].get_x_arange(),
                               self.mrt_layer_set[current_mrt_layer].get_y_arange(),
                               np.swapaxes(self.mrt_layer_set[current_mrt_layer].get_current_Slice(), 0, 1), vmin=v_min,
                               vmax=v_max)
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
            proband = self.mrt_layer_set[current_mrt_layer].get_proband()
            model = self.mrt_layer_set[current_mrt_layer].get_model()
            print(proband)
            print(model)
            deleteMark = shelve.open(self.Path_markings + proband +".slv", writeback=True)
            layer = deleteMark[model]
            cur_no = str(self.mrt_layer_set[current_mrt_layer].get_current_Number())
            cur_pa = str(len(self.ax.patches)-1)
            if layer.has_key(cur_no + "_11_" + cur_pa):
                 del deleteMark[model][cur_no + "_11_" + cur_pa]
            elif layer.has_key(cur_no + "_12_" + cur_pa):
                 del deleteMark[model][cur_no + "_12_" + cur_pa]
            elif layer.has_key(cur_no + "_13_" + cur_pa):
                 del deleteMark[model][cur_no + "_13_" + cur_pa]
            elif layer.has_key(cur_no + "_21_" + cur_pa):
                 del deleteMark[model][cur_no + "_21_" + cur_pa]
            elif layer.has_key(cur_no + "_22_" + cur_pa):
                 del deleteMark[model][cur_no + "_22_" + cur_pa]
            elif layer.has_key(cur_no + "_23_" + cur_pa):
                 del deleteMark[model][cur_no + "_23_" + cur_pa]
            elif layer.has_key(cur_no + "_31_" + cur_pa):
                 del deleteMark[model][cur_no + "_31_" + cur_pa]
            elif layer.has_key(cur_no + "_32_" + cur_pa):
                 del deleteMark[model][cur_no + "_32_" + cur_pa]
            elif layer.has_key(cur_no + "_33_" + cur_pa):
                 del deleteMark[model][cur_no + "_33_" + cur_pa]

            deleteMark.close()
            plt.cla()
            plt.xlim(0, self.mrt_layer_set[current_mrt_layer].get_mrt_width())
            plt.ylim(self.mrt_layer_set[current_mrt_layer].get_mrt_height(), 0)
            v_min = self.mrt_layer_set[current_mrt_layer].get_v_min()
            v_max = self.mrt_layer_set[current_mrt_layer].get_v_max()
            self.pltc = plt.pcolormesh(self.mrt_layer_set[current_mrt_layer].get_x_arange(),
                           self.mrt_layer_set[current_mrt_layer].get_y_arange(),
                           np.swapaxes(self.mrt_layer_set[current_mrt_layer].get_current_Slice(), 0, 1), vmin=v_min,
                           vmax=v_max)
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
            self.v_min_label.config(text=str(v_min))
            self.v_max_label.config(text=str(v_max))
            self.pltc.set_clim(vmin=v_min, vmax=v_max)
            self.fig.canvas.draw()

    def mouse_release(self, event):
        current_mrt_layer = self.get_currentLayerNumber()
        self.mrt_layer_set[current_mrt_layer].set_v_min(int(self.v_min_label['text']))
        self.mrt_layer_set[current_mrt_layer].set_v_max(int(self.v_max_label['text']))
        if event.button == 2:
            self.mouse_second_clicked = False

    def loadMark(self):
        current_mrt_layer = self.get_currentLayerNumber()
        proband = self.mrt_layer_set[current_mrt_layer].get_proband()
        model = self.mrt_layer_set[current_mrt_layer].get_model()
        print(proband)
        print(model)
        loadFile = shelve.open(self.Path_markings + proband +".slv")
        number_Patch = 0
        if loadFile.has_key(model):
            layer_name = model
            layer = loadFile[layer_name]
            cur_no = str(self.mrt_layer_set[current_mrt_layer].get_current_Number())

            while layer.has_key(cur_no + "_11_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_12_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_13_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_21_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_22_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_23_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_31_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_32_" + str(number_Patch)) or layer.has_key(
                                    cur_no + "_33_" + str(number_Patch)):
                patch = None
                if layer.has_key(cur_no+ "_11_" + str(number_Patch)):
                    p = layer[cur_no+ "_11_" + str(number_Patch)]
                    print(p)
                    patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]), np.abs(p[1] - p[3]), fill=False,
                                         edgecolor="red", lw=2)
                elif layer.has_key(cur_no+ "_12_" + str(number_Patch)):
                    p = layer[cur_no+ "_12_" + str(number_Patch)]
                    patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]), np.abs(p[1] - p[3]), fill=False,
                                         edgecolor="green", lw=2)
                elif layer.has_key(cur_no+ "_13_" + str(number_Patch)):
                    p = layer[cur_no+ "_13_" + str(number_Patch)]
                    patch = plt.Rectangle((min(p[0], p[2]), min(p[1], p[3])), np.abs(p[0] - p[2]), np.abs(p[1] - p[3]), fill=False,
                                         edgecolor="blue", lw=2)
                elif layer.has_key(cur_no + "_21_" + str(number_Patch)):
                    p = layer[cur_no + "_21_" + str(number_Patch)]
                    patch = Ellipse(xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                                  width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="red", fc='None', lw=2)
                elif layer.has_key(cur_no + "_22_" + str(number_Patch)):
                    p = layer[cur_no + "_22_" + str(number_Patch)]
                    patch = Ellipse(xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                                    width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="green", fc='None', lw=2)
                elif layer.has_key(cur_no + "_23_" + str(number_Patch)):
                    p = layer[cur_no + "_23_" + str(number_Patch)]
                    patch = Ellipse(xy=(min(p[0], p[2]) + np.abs(p[0] - p[2]) / 2, min(p[1], p[3]) + np.abs(p[1] - p[3]) / 2),
                                    width=np.abs(p[0] - p[2]), height=np.abs(p[1] - p[3]), edgecolor="blue", fc='None', lw=2)
                elif layer.has_key(cur_no + "_31_" + str(number_Patch)):
                    p = layer[cur_no + "_31_" + str(number_Patch)]
                    patch = patches.PathPatch(p, fill=False, edgecolor='red', lw=2)
                elif layer.has_key(cur_no + "_32_" + str(number_Patch)):
                    p = layer[cur_no + "_32_" + str(number_Patch)]
                    patch = patches.PathPatch(p, fill=False, edgecolor='green', lw=2)
                elif layer.has_key(cur_no + "_33_" + str(number_Patch)):
                    p = layer[cur_no + "_33_" + str(number_Patch)]
                    patch = patches.PathPatch(p, fill=False, edgecolor='blue', lw=2)
                self.ax.add_patch(patch)
                number_Patch += 1

    def refresh_option(self, new_list):
        self.layer.set(new_list[len(new_list) - 1])
        self.option_layers['menu'].delete(0, 'end')
        for choice in new_list:
            self.option_layers['menu'].add_command(label=choice, command=lambda value=choice: self.layer.set(value))

    def get_currentLayerNumber(self):
        num = int(self.layer.get().find(")"))
        current_mrt_layer = 0
        if num == 2:
            current_mrt_layer = int(self.layer.get()[1:2]) - 1
        elif num == 3:
            current_mrt_layer = int(self.layer.get()[1:3]) - 1
        return current_mrt_layer


    def open_explorer(self):
        name = askopenfilename()
        self.path_cnn_entry.insert(0, name)
        print(self.path_cnn_entry.get())


    def fData_Preprocessing(self):
        allPatchesList = []
        allLabelsList = []
        proband_list = []
        model_list = []
        nbAllPatches = 0
        if self.dim_rat_lab.get() == '2D':
            patch_Size = np.array(([int(self.patchsize1.get()), int(self.patchsize2.get())]))  # dType = float
        else:
            patch_Size = np.array(([int(self.patchsize1.get()), int(self.patchsize2.get()), int(self.patchsize3.get())]))
        patch_overlap = float(self.patchOver.get())
        for pnd in range(0, len(self.proband), 1):
            if self.list_var[pnd].get() == 1:
                for mod in range(0, len(self.model), 1):
                    if self.list_varl[mod].get() == 1:
                        proband = self.list_proband[pnd]['text']
                        proband_list.append(proband)
                        model = self.list_modell[mod]['text']
                        model_list.append(model)
                        print(proband, model)
                        dPatches, dLabel, nbPatches = fPreprocessData(self.Path_markings, self.sFolder, proband, model, patch_Size,
                                                                      patch_overlap, float(self.split_rat_lab.get()), self.dim_rat_lab.get())
                        nbAllPatches = nbAllPatches + nbPatches
                        allPatchesList.append(dPatches)
                        allLabelsList.append(dLabel)

        if self.dim_rat_lab.get() == '2D':
            allPatches = np.zeros((patch_Size[0], patch_Size[1], nbAllPatches), dtype=float)
        else:
            allPatches = np.zeros((patch_Size[0], patch_Size[1],  patch_Size[2], nbAllPatches), dtype=float)

        allLabels = np.zeros((nbAllPatches), dtype=float)
        firstIndex = 0
        for iIndex in range(0, len(allPatchesList),1):
            if self.dim_rat_lab.get() == '2D':
                allPatches[:, :, firstIndex:firstIndex + allPatchesList[iIndex].shape[2]] = allPatchesList[iIndex]
                allLabels[firstIndex:firstIndex + allPatchesList[iIndex].shape[2]] = allLabelsList[iIndex]
                firstIndex = firstIndex + allPatchesList[iIndex].shape[2]
            else:
                allPatches[:, :, :, firstIndex:firstIndex + allPatchesList[iIndex].shape[3]] = allPatchesList[iIndex]
                allLabels[firstIndex:firstIndex + allPatchesList[iIndex].shape[3]] = allLabelsList[iIndex]
                firstIndex = firstIndex + allPatchesList[iIndex].shape[3]

            #allPatches[:,:,firstIndex:firstIndex+allPatchesList[iIndex].shape[2]]= allPatchesList[iIndex]
            #allLabels[firstIndex:firstIndex+allPatchesList[iIndex].shape[2]] = allLabelsList[iIndex]
            #firstIndex = firstIndex + allPatchesList[iIndex].shape[2]
        Path = "C:/Users/Sebastian Milde/Pictures/Universitaet/Masterarbeit/Labels_Konfusion/Kopft1_mitLabel_2D.h5"
        with h5py.File(Path, 'w') as hf:
         #   hf.create_dataset('AllPatches', data=allPatches)
            hf.create_dataset('AllLabels', data=allLabels)

        #print(allLabels)

        #label_name = "AllLabels"
        #patches_name = "AllPatches"
        #matFile = {label_name: allLabels,patches_name: allPatches}
        #scipy.io.savemat("Dicom_mask.mat", matFile)
        print("Rigid3D Done")
        # Data Splitting
        #if self.dim_rat_lab.get() == '2D':
        #    fSplitDataset(self.Path_data, proband_list, model_list, allPatches, allLabels, self.data_split.get(),
       #                   patch_Size, patch_overlap, float(self.split_rat.get()))
        #else:
         #   fSplitDataset3D(self.Path_data, proband_list, model_list, allPatches, allLabels, self.data_split.get(),
          #                patch_Size, patch_overlap, float(self.split_rat.get()))

    def start_cnn(self):
        sPathOut = self.Path_train_results
        model = self.cnn_model.get()
        batchSize = int(self.batch_opt.get())
        learn_rate = float(self.learn_opt.get())
        epoch = int(self.epoch_opt.get())
        CNN_execute(self.path_cnn_entry.get(), sPathOut, batchSize, learn_rate, epoch, model)

    def plotProbMap(self):

        canvas = FigureCanvasTkAgg(self.fig3, master=self.p4)
        canvas.show()
        canvas.get_tk_widget().place(x=600, y=250)  # expand=1
        PathDicom = self.sFolder + "/" + self.proband_str_view.get() + "/dicom_sorted/" + self.artefact_str_view.get() + "/"
        files = sorted([os.path.join(PathDicom, file) for file in os.listdir(PathDicom)], key=os.path.getctime)
        datasets = [dicom.read_file(f) \
                    for f in files]
        try:
            voxel_ndarray, pixel_space = dicom_numpy.combine_slices(datasets)
            print(voxel_ndarray.shape)
        except dicom_numpy.DicomImportException:
            # invalid DICOM data
            raise

        dx, dy, dz = 1.0, 1.0, pixel_space[2][2]  # pixel_space[0][0], pixel_space[1][1], pixel_space[2][2]
        pixel_array = voxel_ndarray[:, :, 0]
        x_1d = dx * np.arange(voxel_ndarray.shape[0])
        y_1d = dy * np.arange(voxel_ndarray.shape[1])
        z_1d = dz * np.arange(voxel_ndarray.shape[2])
        PatchSize = np.array((40.0, 40.0))
        PatchOverlay = 0.9
        #Path = "/home/d1224/no_backup/d1224/Kopft1_05_withoutlabel_testma_val_ab_4040_lr_0.001_bs_64_pred.mat"
        #conten = sio.loadmat(Path)
        # print(conten)
        #prob_test = conten['prob_pre']
        #prob_test = prob_test[0:prob_test.shape[0] / 2, :]
        #self.imglay = fUnpatch2D(prob_test, PatchSize, PatchOverlay, voxel_ndarray.shape, 1)  # fUnpatch2D
        with h5py.File("/home/d1224/no_backup/d1224/Unpatching_Kopf.h5", 'r') as hf:
            imglay = hf['unpatch'][:]
        # matFile = {'upatching': self.imglay, 'probabilities': prob_test}
        # sio.savemat("/home/d1224/no_backup/d1224/Unpatching/Beckent1aufBeckent13D_Unpatching_mask.mat", matFile)
        current_Number = 0
        LayImg = imglay[:, :, current_Number]
        self.x_1d = dx * np.arange(voxel_ndarray.shape[0])
        self.y_1d = dy * np.arange(voxel_ndarray.shape[1])
        self.z_1d = dz * np.arange(voxel_ndarray.shape[2])
        self.fig = plt.figure(dpi=100)
        self.ax3 = plt.gca()
        self.fig.canvas.mpl_connect('key_press_event', self.click)
        plt.cla()
        plt.gca().set_aspect('equal')  # plt.axes().set_aspect('equal')
        plt.xlim(0, voxel_ndarray.shape[0] * dx)
        plt.ylim(voxel_ndarray.shape[1] * dy, 0)
        plt.set_cmap(plt.gray())
        print(pixel_array.shape)
        print(LayImg.shape)
        pltc = plt.imshow(np.swapaxes(pixel_array, 0, 1), vmin=0,
                               vmax=1000)  # x_1d, y_1d, np.swapaxes(pixel_array, 0, 1), vmin=0, vmax=2094
        pltc_lay = plt.imshow(np.swapaxes(LayImg, 0, 1), cmap='jet', alpha=0.4, vmin=0,
                                   vmax=1)
        self.fig3.canvas.draw_idle()


if __name__ == "__main__":
    m = MainGUI()
