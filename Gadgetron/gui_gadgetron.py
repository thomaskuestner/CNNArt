import tkinter
import tkinter.messagebox
import tkinter.font as tkfont
import tkinter.filedialog as tkfile
import os
import shutil
import subprocess


class Window(object):
    def __init__(self):
        self.path = ''
        self.tk = tkinter.Tk()
        self.sysfont = tkfont.Font(size=2, weight=tkfont.BOLD)
        self.tk.title('CNNArt with Gadgetron')
        self.tk.geometry('500x500')

        self.dockerPath, self.modelPath, self.weightPath, self.mriPath = "", "", "", ""

        self.buildDockerMsg = tkinter.Message(
            master=self.tk, width=1000, text="CNNArt Path:", justify="left")
        self.buildDockerMsg.grid(row=0, column=1)
        self.chooseModelMsg = tkinter.Message(
            master=self.tk, width=1000, text="Model Path:", justify="left")
        self.chooseModelMsg.grid(row=1, column=1)
        self.runGadgetMsg = tkinter.Message(
            master=self.tk, width=1000, text="MRI H5 Path:", justify="left")
        self.runGadgetMsg.grid(row=2, column=1)
        self.runMsg = tkinter.Message(
            master=self.tk, width=1000, text="Run Message is in the shell", justify="left")
        self.runMsg.grid(row=3, column=1)

        tkinter.Button(master=self.tk, text='Choose CNNArt Path',
                       command=self.buildDocker, height=5, width=30).grid(row=0, column=0)
        tkinter.Button(master=self.tk, text='Choose model\'s JSON AND H5',
                       command=self.chooseModel, height=5, width=30).grid(row=1, column=0)
        tkinter.Button(master=self.tk, text='Choose MRI file',
                       command=self.runGadget, height=5, width=30).grid(row=2, column=0)
        tkinter.Button(master=self.tk, text='Run', command=self.run,
                       height=5, width=30).grid(row=3, column=0)
        self.tk.mainloop()

    def buildDocker(self):
        try:
            docker_path = tkfile.askdirectory()
            print("Log: buildDocker => CNNArt path =>",
                  type(docker_path), docker_path)
            self.buildDockerMsg.configure(text="CNNArt path =>"+docker_path)
            assert(os.path.split(docker_path)[1] == 'CNNArt')
            self.dockerPath = docker_path
        except AssertionError:
            warningInfo = docker_path + ' is Not CNNArt Path. Try again...'
            print(warningInfo)
            tkinter.messagebox.showwarning(
                title="buildDocker Warning", message=warningInfo)

    def chooseModel(self):
        try:
            nn_path = tkfile.askopenfilenames()
            print("Log: chooseModel => JSON and H5 path =>",
                  type(nn_path), nn_path)
            self.chooseModelMsg.configure(
                text="JSON and H5 path =>\n"+"\n".join(nn_path))
            assert(len(nn_path) == 2)
            if nn_path[0].endswith('.h5'):
                weight_path = nn_path[0]
                model_path = nn_path[1]
            else:
                weight_path = nn_path[1]
                model_path = nn_path[0]
            assert(model_path.endswith('.json')
                   and weight_path.endswith('.h5'))
            self.modelPath, self.weightPath = model_path, weight_path
        except AssertionError:
            warningInfo = 'NN files must end with h5 and json. The file you have chosen are ' + \
                '\n'.join(nn_path)
            print(warningInfo)
            tkinter.messagebox.showwarning(
                title="chooseModel Warning", message=warningInfo)

    def runGadget(self):
        try:
            mri_path = tkfile.askopenfilename()
            print("Log: runGadget => MRI H5 path =>", type(mri_path), mri_path)
            self.runGadgetMsg.configure(text="MRI path =>"+"".join(mri_path))
            assert(mri_path.endswith('.h5'))
        except AssertionError:
            warningInfo = path + ' is Not H5 File. Try again...'
            print(warningInfo)
            tkinter.messagebox.showwarning(
                title="buildDocker Warning", message=warningInfo)
        self.mriPath = mri_path

    def run(self):
        os.system('cd '+os.path.join(self.dockerPath, 'Gadgetron') +
                  '&& sudo docker build -t cnnart_gadgetron .')

        os.mkdir('mydocker')
        shutil.copy(self.modelPath, os.path.join(os.path.curdir, 'mydocker'))
        shutil.copy(self.weightPath, os.path.join(os.path.curdir, 'mydocker'))

        shutil.copy(self.mriPath, os.path.join(os.path.curdir, 'mydocker'))
        os.chdir('mydocker')
        os.system(
            'sudo docker run -t --name gt1 --detach --volume $(pwd):/opt/data cnnart_gadgetron')
        os.system('sudo docker exec -ti gt1 bash')


win = Window()
