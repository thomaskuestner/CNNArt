from tkinter import *
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import numpy as np
import os
import dicom
import dicom_numpy
import matplotlib as mpl

def zoom_in_move(event):
    global x_zclicked, y_zclicked
    dx = event.x - x_zclicked
    dy = event.y - y_zclicked
    x_zclicked = event.x
    y_zclicked = event.y
    dpi = fig.dpi+10
    fig.set_dpi(dpi)
    fig.set_size_inches(6.4,4.9)
    #ax = canvas.figure.axes[0]
    #ax.set_xlim(dpi)
    #ax.set_ylim(dpi)
    canvas = FigureCanvasTkAgg(fig, master=panel)
    canvas.get_tk_widget().place(x=20, y=100)  # x=225, y=55, width = 1204, height = 764
    fig.canvas.mpl_connect('button_press_event', mouse_clicked)
    fig.canvas.draw()
    canvas.show()
    print(fig.get_size_inches())
    print(fig.dpi)

def zoom_in_clicked(event):
    global x_zclicked, y_zclicked
    x_zclicked = event.x
    y_zclicked = event.y
    print (x_zclicked, y_zclicked)

def zoom_out():
    dpi = fig.dpi-10
    print(dpi)
    fig.set_dpi(dpi)
    fig.set_size_inches(6.4,4.9, forward=True)
    #ax = canvas.figure.axes[0]
    #ax.set_xlim(dpi)
    #ax.set_ylim(dpi)
    canvas = FigureCanvasTkAgg(fig, master=panel)
    # canvas.show()
    canvas.get_tk_widget().place(x=20, y=100)  # x=225, y=55, width = 1204, height = 764
    fig.canvas.mpl_connect('button_press_event', mouse_clicked)
    canvas.draw()
    fig.canvas.draw()
    canvas.show()
    print(fig.get_size_inches())
    print(fig.dpi)

def mouse_clicked( event):
    x_clicked = event.xdata
    y_clicked = event.ydata
    print(x_clicked, y_clicked)

def panel_mouse_clicked(event):
    global x_first, y_first
    print (event.x, event.y)
    x_first = event.x
    y_first = event.y

def panel_mouse_move(event):
    global x_canvas, y_canvas
    global x_first, y_first
    print (event.x, event.y)
    dx = event.x - x_first
    dy = event.y - y_first
    x_first = event.x
    y_first = event.y
    x_canvas = x_canvas + dx
    y_canvas = y_canvas + dy
    canvas.get_tk_widget().place(x=x_canvas, y=y_canvas)
    fig.canvas.draw()
    print(dx, dy)

def mouse_clicked_second(event):
    global x_second_clicked, y_second_clicked
    x_second_clicked, y_second_clicked = event.x, event.y

def mouse_move_second(event):
    global x_second_clicked, y_second_clicked
    factor = 10
    __x = event.x - x_second_clicked
    __y = event.y - y_second_clicked
    x_second_clicked = event.x
    y_second_clicked = event.y
    v_min, v_max = pltc.get_clim()
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

    if (float(__vmin - __vmax)) / (v_max - v_min+0.001) > 1:
        nmb = (float(__vmin - __vmax)) / (v_max - v_min+0.001) + 1
        __vmin = (float(__vmin - __vmax)) / nmb * (__vmin / (__vmin - __vmax))
        __vmax = (float(__vmin - __vmax)) / nmb * (__vmax / (__vmin - __vmax))

    v_min += __vmin
    v_max += __vmax
    v_min_label.config(text=str(v_min.round(2)))
    v_max_label.config(text=str(v_max.round(2)))
    pltc.set_clim(vmin=v_min.round(2), vmax=v_max.round(2))
    fig.canvas.draw()

def change_values_plot(event):
    def get_min_max():
        #v_min = entry_min.get()
        #v_max = entry_max.get()
        pltc.set_clim(vmin=float(V_min.get()), vmax=float(V_max.get()))
        v_min_label.config(text=V_min.get())
        v_max_label.config(text=V_max.get())
        fig.canvas.draw()
        New_Win.quit()
        New_Win.destroy()

    def cancel_button():
        New_Win.quit()
        New_Win.destroy()

    New_Win = Toplevel()
    New_Win.title('Choose min max for Colorbar')
    New_Win.geometry('200x155')
    New_Win.configure(background="gray55")
    v_min_label = Label(New_Win, font="Aral 12 bold", bg="gray55", text="Min")
    v_max_label = Label(New_Win, font="Aral 12 bold", bg="gray55", text="Max")
    v_min_label.place(x=10, y=10)
    v_max_label.place(x=10, y=60)
    V_min = StringVar()
    V_max = StringVar()
    entry_min = Entry(New_Win, textvariable = V_min)
    entry_min.place(x=12, y=35, width=178, height=25)
    entry_min.insert(0, 0)

    entry_max = Entry(New_Win, textvariable = V_max)
    entry_max.place(x=12, y=85, width=178, height=25)
    entry_max.insert(0, 2000)

    ok_button = Button(New_Win, text="Ok", font=('Arial', 10, 'bold'),
                            bg="gray40", activeforeground='grey', borderwidth=4, compound=LEFT,
                            command=get_min_max)
    ok_button.place(x=75, y=120, width=55, height=30)
    cancel_button = Button(New_Win, text="Cancel", font=('Arial', 10, 'bold'),
                                bg="gray40", activeforeground='grey', borderwidth=4, compound=LEFT, command=exit)
    cancel_button.place(x=135, y=120, width=55, height=30)

root = Tk()
root.title('PatchBased Labeling')
root.geometry('1425x770')
root.configure(background="gray38")

#panel = Label(root, bg="black")
#panel.place(x=225, y=5, width=1204, height=764)
panel = Canvas(root, width=1204,height=764, bg="black")
panel.place(x=225, y=5)
panel.bind("<Button-1>", panel_mouse_clicked)
panel.bind("<B1-Motion>", panel_mouse_move)
panel.bind("<Button-3>", zoom_in_clicked)
panel.bind("<B3-Motion>", zoom_in_move)
panel.bind("<Button-2>", mouse_clicked_second)
panel.bind("<B2-Motion>", mouse_move_second)
panel2 = Canvas(root, bg="LightSteelBlue4", borderwidth = 0)
panel2.place(x=225, y=5, width=1204, height=50)
v_min_label = Label(panel2, bg="LightSteelBlue4", fg="white", font="Aral 9 bold", text = "0.00")
v_max_label = Label(panel2, bg="LightSteelBlue4", fg="white", font="Aral 9 bold", text = "2000.00")
v_min_label.place(x=10, y=20)
v_max_label.place(x=1100, y=20)
#zoom_in_button = Button(root,  text="zoom in", font=('Arial', 11, 'bold'),
 #                                 bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT, command = zoom_in)
#zoom_in_button.place(x=5, y=5, width=200, height=40)
zoom_out_button = Button(root,  text="zoom out", font=('Arial', 11, 'bold'),
                                  bg="gray56", activeforeground='grey', borderwidth=4, compound=LEFT, command = zoom_out)
zoom_out_button.place(x=5, y=60, width=200, height=40)

PathDicom = "C:\\Users\hansw\Videos\\artefacts\\03\dicom_sorted\\t1_tse_tra_Kopf_Motion_0003"

            #load Dicom_Array
files = sorted([os.path.join(PathDicom, file) for file in os.listdir(PathDicom)], key=os.path.getctime)
datasets = [dicom.read_file(f) \
                        for f in files]

try:
        voxel_ndarray, pixel_space = dicom_numpy.combine_slices(datasets)
except dicom_numpy.DicomImportException:
                # invalid DICOM data
        raise

dx, dy, dz = 1.0, 1.0, pixel_space[2][2] #pixel_space[0][0], pixel_space[1][1], pixel_space[2][2]
pixel_array = voxel_ndarray[:, :, 0]
x_1d = dx * np.arange(voxel_ndarray.shape[0])
y_1d = dy * np.arange(voxel_ndarray.shape[1])
z_1d = dz * np.arange(voxel_ndarray.shape[2])
fig = plt.figure(figsize=(6.4,4.9), dpi=70)
print(fig.get_size_inches())
#fig.patch.set_facecolor('black')
ax = plt.gca()
print(ax)
#ax.set_axis_bgcolor('black')
#ax.text(0.5, 0.5, 'To label MRT-Artefacts load MRT-Layer', horizontalalignment='center',
 #                    verticalalignment='center', color='white', fontsize=20, transform=self.ax.transAxes)
 # expand=1
plt.cla()
plt.gca().set_aspect('equal')  # plt.axes().set_aspect('equal')
plt.xlim(0, voxel_ndarray.shape[0]*dx)
plt.ylim(voxel_ndarray.shape[1]*dy, 0)
plt.set_cmap(plt.gray())
pltc = plt.pcolormesh(x_1d, y_1d, np.swapaxes(pixel_array, 0, 1), vmin=0, vmax=2094)

canvas = FigureCanvasTkAgg(fig, master=panel)
#canvas.show()
canvas.get_tk_widget().place(x=20, y=100)  # x=225, y=55, width = 1204, height = 764
x_canvas = 20
y_canvas = 100
fig.canvas.mpl_connect('button_press_event', mouse_clicked)
#canvas._tkcanvas.place(x = -90, y = -90)
print(fig.get_size_inches())
#canvas._tkcanvas.pack()
#toolbar = NavigationToolbar2TkAgg(canvas, root)
#canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)
fig2, ax2 = plt.subplots(figsize = (17,0.3), dpi = 60)
#fig2.canvas.
#fig2.patch.set_facecolor('LightSteelBlue4')
#fig2.patch.set_facecolor('LightSteelBlue4')
fig2.subplots_adjust(left=0, bottom=0, right=1, top=1,
                wspace=0, hspace=0)
cmap = 'jet'# mpl.cm.jet
norm = mpl.colors.Normalize(vmin=5, vmax=10)

cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
plt.axis('off')
canvas2 = FigureCanvasTkAgg(fig2, master=panel2)
canvas2.get_tk_widget().configure(background = 'LightSteelBlue4', highlightcolor = 'LightSteelBlue4', highlightbackground = 'LightSteelBlue4')
canvas2.show()
canvas2.get_tk_widget().place(x=50, y=25)
fig2.canvas.mpl_connect('button_press_event', change_values_plot)

root.mainloop()
