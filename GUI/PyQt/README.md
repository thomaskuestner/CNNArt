#CNNArtGUI

This GUI is designed with PyQt5 and works for Python >2.7 and >3.5.

## GUI calling
Qt_main.py 
to open the GUI, this GUI can be run independent from CNNArt.

This GUI contains three tabs:

 The first tab: 
	image viewing: flexible layout, 2D\3D mode, images can be loaded from python workspace
	prediction overlay: colors and hatches can be modified
	labeling: 3 selector(rectangle, ellipse, lasso), the name and color of the labels can be modified

 The second tab: 
	network training: data preprocessing, parameter setting
			  accuracy\loss curves can be dynamically plotted

 The third tab: 
	network structure: the network structure can be loaded from h5 file
	feature maps and filters viewing
