import tkinter
import tkinter.font as tkfont
import tkinter.filedialog as tkfile
import os 
import shutil, subprocess

class Window(object):
    def __init__(self):
        self.path = ''
        self.tk = tkinter.Tk()
        self.sysfont = tkfont.Font(size=2, weight=tkfont.BOLD)
        self.tk.title('CNNArt with Gadgetron')
        self.tk.geometry('500x300+300+300') 

        # self.label = tkinter.Label(master=self.tk, text='Choose CNNArt path', font=self.sysfont)
        # self.label.pack()
        self.build_docker = tkinter.Button(master=self.tk, text='Choose CNNArt Path', command=self.buildDocker)
        self.build_docker.pack()
        self.choose_model = tkinter.Button(master=self.tk, text='Choose NN json and h5', font=self.sysfont, command=self.chooseModel)
        self.choose_model.pack()
        self.choose_mri = tkinter.Button(master=self.tk, text='Choose MRI file', font=self.sysfont, command=self.runGadget)
        self.choose_mri.pack()
        # self.cnnart_path = tkinter.Label(master=self.tk, text=self.path, font=self.sysfont)
        # self.cnnart_path.pack()
        self.tk.mainloop()

    def buildDocker(self):
        while True:
            try:
                path = tkfile.askdirectory()
                assert(os.path.split(path)[1] == 'CNNArt')
                break
            except AssertionError:
                print(path + ' is Not CNNArt Path. Try again...')
        os.system('cd '+os.path.join(path, 'Gadgetron') + '&& sudo docker build -t cnnart_gadgetron .')

    def chooseModel(self):
        while True:
            try:
                nn_path = tkfile.askopenfilenames()
                assert(len(nn_path) == 2)
                if nn_path[0].endswith('.h5'):
                    weight_path = nn_path[0]
                    model_path = nn_path[1]
                else:
                    weight_path = nn_path[1]
                    model_path = nn_path[0]
                assert(model_path.endswith('.json') and weight_path.endswith('.h5'))
                break
            except AssertionError:
                print('NN files must end with h5 and json. The file you have chosen are ' + str(nn_path))
        os.mkdir('mydocker')
        shutil.copy(model_path, os.path.join(os.path.curdir, 'mydocker'))
        shutil.copy(weight_path, os.path.join(os.path.curdir, 'mydocker'))

    def runGadget(self):
        while True:
            try:
                mri_path = tkfile.askopenfilename()
                assert(mri_path.endswith('.h5'))
                break
            except AssertionError:
                print(path + ' is Not H5 File. Try again...')
        shutil.copy(mri_path, os.path.join(os.path.curdir, 'mydocker'))
        os.chdir('mydocker')
        os.system('sudo docker run -t --name gt1 --detach --volume $(pwd):/opt/data cnnart_gadgetron')
        os.system('sudo docker exec -ti gt1 bash')
        exit()
win = Window()
