from Tkinter import *
import ttk
import os
import dicom
import dicom_numpy
import shelve
import numpy as np
import scipy.io
from utils.Patching import*
from Training_Test_Split import *
from Data_Preprocessing import*
import cProfile

class CNN_Preprocessing():
    def __init__(self, sFolder, artefact, proband):
        # TODO: replace by DatabaseInfo
        self.mrt_model = {'t1_tse_tra_fs_Becken_0008': 'Becken_0008',
                          't1_tse_tra_fs_Becken_Motion_0010': 'Becken_Motion_0010',
                          't1_tse_tra_fs_mbh_Leber_0004': 'Leber_0004',
                          't1_tse_tra_fs_mbh_Leber_Motion_0005': 'Leber_Motion_0005',
                          't1_tse_tra_Kopf_0002': 'Kopf_0002',
                          't1_tse_tra_Kopf_Motion_0003': 'Kopf_Motion_0003', 't2_tse_tra_fs_Becken_0009': 'Becken_0009',
                          't2_tse_tra_fs_Becken_Motion_0011': 'Becken_Motion_0011',
                          't2_tse_tra_fs_Becken_Shim_xz_0012': 'Becken_Shim_0012',
                          't2_tse_tra_fs_navi_Leber_0006': 'Leber_0006',
                          't2_tse_tra_fs_navi_Leber_Shim_xz_0007': 'Leber_Shim_0007'}
        self.col_list = {"1": "red", "2": "green", "3": "blue"}
        self.Folder = sFolder
        self.artefact = artefact
        self.proband = proband
        self.sFolder = "C:/Users/Sebastian Milde/Pictures/MRT" # TODO: adapt path!
        self.proband = os.listdir(self.sFolder)
        self.model = os.listdir(self.sFolder + "/ab/dicom_sorted")
        self.win = Tk()
        self.win.title("Preprocessing for CCN")
        self.win.geometry('1280x610')
        self.win.configure(background="gray30")

        self.frame1 = Frame(self.win, bg = "gray55", bd=3, width=300, height=500, relief="sunken")
        self.frame2 = Frame(self.win, bg = "gray55", bd=3, width=300, height=500, relief="sunken")
        self.frame3 = Frame(self.win, bg = "gray55", bd=3, width=300, height=500, relief="sunken")
        self.frame4 = Frame(self.win, bg = "gray55", bd=3, width=300, height=500, relief="sunken")
        self.frame1.grid(row = 1, column = 2, padx = 10, pady = 10)
        self.frame2.grid(row=1, column=3, padx = 10)
        self.frame3.grid(row=1, column=1, padx = 10)
        self.frame4.grid(row=1, column=4, padx = 10)

        # Design Frame 1
        self.ch_patching_process = ["Rigid-Patching", "Adaptive-Splitting"]
        self.ch_patch_size = [10,20,30,40,50,60,70,80,90,100]
        self.ch_patch_overlap = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        self.ch_data_splitting = ["normal", "crossvalidation_data", "crossvalidation_patient"]

        self.heading_frame1 = Label(self.frame1, text="Data Preprocessing", bg = "gray55",  font="Aral 14 bold")
        self.heading_frame1.place(x = 55, y = 2)

        self.Patching_process = Label(self.frame1, text="Patching-Process", bg="gray55", font="Aral 10 bold")
        self.Patching_process.place(x = 5, y = 40)
        self.patching_process = StringVar(self.win)
        self.patching_process.set(self.ch_patching_process[0])
        self.patching_process_option = OptionMenu(self.frame1, self.patching_process, *self.ch_patching_process)
        self.patching_process_option.place(x=50, y=65, width=200, height=40)

        self.patch_size = Label(self.frame1, text = "Patch-Size (3D)", bg="gray55", font="Aral 10 bold")
        self.patch_size.place(x= 5,y=115)
        self.patchsize1 = StringVar(self.win)
        self.patchsize1.set(self.ch_patch_size[3])
        self.patchsize1_option = OptionMenu(self.frame1, self.patchsize1, *self.ch_patch_size)
        self.patchsize1_option.place(x=50, y=140, width=50, height=40)

        self.patchsize2 = StringVar(self.win)
        self.patchsize2.set(self.ch_patch_size[3])
        self.patchsize2_option = OptionMenu(self.frame1, self.patchsize2, *self.ch_patch_size)
        self.patchsize2_option.place(x=125, y=140, width=50, height=40)

        self.patchsize3 = StringVar(self.win)
        self.patchsize3.set(self.ch_patch_size[3])
        self.patchsize3_option = OptionMenu(self.frame1, self.patchsize3, *self.ch_patch_size)
        self.patchsize3_option.place(x=200, y=140, width=50, height=40)

        self.patch_overlap = Label(self.frame1, text="Patch-Overlap", bg="gray55", font="Aral 10 bold")
        self.patch_overlap.place(x=5, y=190)

        self.patchOver = StringVar(self.win)
        self.patchOver.set(self.ch_patch_overlap[5])
        self.patchOver_option = OptionMenu(self.frame1, self.patchOver, *self.ch_patch_overlap)
        self.patchOver_option.place(x=50, y=215, width=200, height=40)

        self.split_data = Label(self.frame1, text="Data-Splitting", bg="gray55", font="Aral 10 bold")
        self.split_data.place(x=5, y=265)

        self.data_split = StringVar(self.win)
        self.data_split.set(self.ch_data_splitting[0])
        self.data_split_option = OptionMenu(self.frame1, self.data_split, *self.ch_data_splitting)
        self.data_split_option.place(x=50, y=290, width=200, height=40)

        self.split_val = Label(self.frame1, text="Split-Ratio", bg="gray55", font="Aral 10 bold")
        self.split_val.place(x=5, y=340)

        self.split_rat = StringVar(self.win)
        self.split_rat.set(self.ch_patch_overlap[1])
        self.split_rat_option = OptionMenu(self.frame1, self.split_rat, *self.ch_patch_overlap)
        self.split_rat_option.place(x=50, y=365, width=200, height=40)

        self.split_val_lab = Label(self.frame1, text="Split-Ratio for Labeling", bg="gray55", font="Aral 10 bold")
        self.split_val_lab.place(x=5, y=415)

        self.split_rat_lab = StringVar(self.win)
        self.split_rat_lab.set(self.ch_patch_overlap[0])
        self.split_rat_lab_option = OptionMenu(self.frame1, self.split_rat_lab, *self.ch_patch_overlap)
        self.split_rat_lab_option.place(x=50, y=440, width=200, height=40)


        self.start_button_frame1 = Button(self.win, text="Start Data Preprocessing", font=('Arial', 11, 'bold'),
                                  bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT, command = self.preprocess_data)

        self.start_button_frame1.place(x=380, y=515, width=200, height=40)

        # Design Frame 2
        self.ch_cnn_model = ["motion_head.py", "motion_abd.py", "motion_all.py", "shim.py", "noise.py", "allData.py", "3D.py"]
        self.ch_parameter_optim = ["none", "grid"]
        self.ch_batchSize = [32, 64, 128]
        self.ch_learning_rate = [0.1, 0.01, 0.05, 0.005, 0.001]
        self.ch_epoch = [100,200,300,400,500,600,700,800,900,1000]
        self.cnn_method = ["Training", "Prediction"]

        self.heading_frame2 = Label(self.frame2, text="Parameter for CNN", bg="gray55", font="Aral 14 bold")
        self.heading_frame2.place(x=55, y=2)

        self.Cnn_model = Label(self.frame2, text="CNN-Model", bg="gray55", font="Aral 10 bold")
        self.Cnn_model.place(x=5, y=40)
        self.cnn_model = StringVar(self.win)
        self.cnn_model.set(self.ch_cnn_model[0])
        self.cnn_model_option = OptionMenu(self.frame2, self.cnn_model, *self.ch_cnn_model)
        self.cnn_model_option.place(x=50, y=65, width=200, height=40)

        self.para_optim = Label(self.frame2, text="Parameter-Optimization", bg="gray55", font="Aral 10 bold")
        self.para_optim.place(x=5, y=115)
        self.para_opt = StringVar(self.win)
        self.para_opt.set(self.ch_parameter_optim[0])
        self.para_opt_option = OptionMenu(self.frame2, self.para_opt, *self.ch_parameter_optim)
        self.para_opt_option.place(x=50, y=140, width=200, height=40)

        self.batch = Label(self.frame2, text="Batch-Size", bg="gray55", font="Aral 10 bold")
        self.batch.place(x=5, y=190)
        self.batch_opt = StringVar(self.win)
        self.batch_opt.set(self.ch_batchSize[1])
        self.batch_opt_option = OptionMenu(self.frame2, self.batch_opt, *self.ch_batchSize)
        self.batch_opt_option.place(x=50, y=215, width=200, height=40)

        self.learn = Label(self.frame2, text="Learning-Rate", bg="gray55", font="Aral 10 bold")
        self.learn.place(x=5, y=265)
        self.learn_opt = StringVar(self.win)
        self.learn_opt.set(self.ch_learning_rate[1])
        self.learn_opt_option = OptionMenu(self.frame2, self.learn_opt, *self.ch_learning_rate)
        self.learn_opt_option.place(x=50, y=290, width=200, height=40)

        self.epoch = Label(self.frame2, text="Epoche", bg="gray55", font="Aral 10 bold")
        self.epoch.place(x=5, y=340)
        self.epoch_opt = StringVar(self.win)
        self.epoch_opt.set(self.ch_epoch[2])
        self.epoch_opt_option = OptionMenu(self.frame2, self.epoch_opt, *self.ch_epoch)
        self.epoch_opt_option.place(x=50, y=365, width=200, height=40)

        self.method = Label(self.frame2, text="CNN-Method", bg="gray55", font="Aral 10 bold")
        self.method.place(x=5, y=415)
        self.method_opt = StringVar(self.win)
        self.method_opt.set(self.cnn_method[0])
        self.method_opt_option = OptionMenu(self.frame2, self.method_opt, *self.cnn_method)
        self.method_opt_option.place(x=50, y=440, width=200, height=40)

        self.start_button_frame2 = Button(self.win, text="Start Training CNN", font=('Arial', 11, 'bold'),
                                          bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT)

        self.start_button_frame2.place(x=700, y=515, width=200, height=40)

        # Design Frame 3
        self.heading_frame3 = Label(self.frame3, text="Data Selection", bg="gray55", font="Aral 14 bold")
        self.heading_frame3.place(x=75, y=2)
        self.frame31 = Frame(self.frame3, bd=3, relief="sunken")
        self.heading_frame31 = Label(self.frame3, text="Proband", bg="gray55", font="Aral 10 bold")
        self.heading_frame31.place(x=2,y=25)
        self.frame32 = Frame(self.frame3, bd=3, width=260, relief="sunken")
        self.heading_frame32 = Label(self.frame3, text="MRT-Layer", bg="gray55", font="Aral 10 bold")
        self.heading_frame32.place(x = 2, y = 185)
        self.frame31.place(x=2, y=50)
        self.frame32.place(x=2, y=210)
        self.list_var = []
        self.list_varl = []
        self.list_proband = []
        self.list_modell = []

        prob = 0
        for row in range(0, len(self.proband)/3, 1):
            for col in range(0, 3, 1):
                    var = IntVar()
                    self.list_var.append(var)
                    chk = ttk.Checkbutton(self.frame31, text=self.proband[prob], variable=self.list_var[prob])
                    # chk.state(['!alternate'])
                    chk.state(['!selected'])
                    chk.grid(row=row + 1, column=col + 1, sticky=W)
                    self.list_proband.append(chk)
                    prob+=1

        for mod in range(0, len(self.model), 1):
            var1 = IntVar()
            self.list_varl.append(var1)
            if self.model[mod] == "FastView_0001":
                chk = ttk.Checkbutton(self.frame32, text=self.model[mod], variable=self.list_varl[mod]) #state = DISABLED
                chk.state(['!selected'])
                chk.grid(row=mod + 1, column=1, sticky=W)
                self.list_modell.append(chk)
            else:
                chk = ttk.Checkbutton(self.frame32, text=self.model[mod], variable=self.list_varl[mod])
                chk.state(['!selected'])
                chk.grid(row=mod + 1, column=1, sticky=W)
                self.list_modell.append(chk)

        # Design Frame 4 ----> Visualization results
        self.heading_frame4 = Label(self.frame4, text="Visualization of Training-Results", bg="gray55", font="Aral 12 bold")
        self.heading_frame4.place(x=15, y=2)
        self.plot_acc_loss = Button(self.frame4, text = "Plotting accuracy and loss", font=('Arial', 11, 'bold'), bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT)
        self.plot_mean_std = Button(self.frame4, text="Plotting mean and standard deviation", font=('Arial', 11, 'bold'),
                                    bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT)
        self.vis_overlay = Button(self.frame4, text="Visualize Overlay", font=('Arial', 11, 'bold'),
                                    bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT)
        self.conf_matrix = Button(self.frame4, text="Confusion Matrix", font=('Arial', 11, 'bold'),
                                    bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT)

        self.plot_acc_loss.place(x=50, y=30, width=200, height=40)
        self.plot_mean_std.place(x=50, y=90, width=200, height=40)
        self.vis_overlay.place(x=50, y=150, width=200, height=40)
        self.conf_matrix.place(x=50, y=210, width=200, height=40)

        self.canc_button = Button(self.win, text="Cancel", font=('Arial', 11, 'bold'),
                                          bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT, command = exit)
        self.canc_button.place(x=380, y=560, width=200, height=40)

        self.win.mainloop()

    def preprocess_data(self):
        allLabels = []
        allPatches = None
        patch_Size = np.array(([int(self.patchsize1.get()), int(self.patchsize2.get())]))  # dType = float
        patch_overlap = float(self.patchOver.get())
        for pnd in range(0, len(self.proband), 1):
            if self.list_proband[pnd].state() == () or self.list_proband[pnd].state() == ('focus',):
                print("not selected")
            else:
                for mod in range(0, len(self.artefact), 1):
                    if self.list_modell[mod].state() == () or self.list_proband[pnd].state() == ('focus',):
                        print("not selected")
                    else:
                        proband = self.list_proband[pnd]['text']
                        model = self.list_modell[mod]['text']
                        print(proband, model)
                        dPatches, dLabel = fPreprocessData(self.sFolder, proband, model, patch_Size, patch_overlap, float(self.split_rat_lab.get()))
                        cProfile.run("fPreprocessData(self.sFolder, proband, model, patch_Size, patch_overlap, float(self.split_rat_lab.get()))")

                        if allPatches is None:
                            allPatches = dPatches
                            print(allPatches.shape)
                        else:
                            allPatches = np.concatenate((allPatches, dPatches), axis=2)
                            print(allPatches.shape)
                        allLabels = np.concatenate((allLabels, dLabel))
                        print(allLabels.shape)

        Path = "C:/Users/Sebastian Milde/Pictures/Universitaet/Masterarbeit/Patches and Labels/test_patches.h5" # TODO: adapt path!
        with h5py.File(Path, 'w') as hf:
            hf.create_dataset('AllPatches', data=allPatches)
            hf.create_dataset('AllLabels', data=allLabels)

        label_name = "AllLabels"
        patches_name = "AllPatches"
        matFile = {label_name: allLabels,patches_name: allPatches}
        scipy.io.savemat("Dicom_mask.mat", matFile)
        # Data Splitting
        fSplitDataset(allPatches, allLabels, self.data_split.get(), patch_Size, patch_overlap, float(self.split_rat.get()))
        print("Done")

    def exit(self):
        self.win.destroy()

if __name__ == "__main__":
    cnn = CNN_Preprocessing()