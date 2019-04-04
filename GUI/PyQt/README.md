# CNNArtGUI

The the original code repository is

https://github.com/thomaskuestner/CNNArt

This GUI is designed with PyQt5 and works for Python >3.5.

## GUI calling
run `Qt_main.py` to open the GUI. <br/>
This GUI can be run independent from CNNArt. <br/>
Before running, please change path in param.yml. <br/>
Call methods <br/>
`from Qt_main import imagine`<br/>
`imagine.main()`<br/>
`imagine.main('pathOfImageFile','pathOfImageDirectory', ...)`<br/>
`imagine.main(image array)`<br/>

## Requirements
`python3.6 -m pip install jsonschema pickleshare h5py keras matplotlib pandas scipy protobuf tensorflow pyYAML PyQt5-stubs PyQt5 numpy pandas argparse hyperas hyperopt dicom pydicom dicom-numpy pydot pyqtgraph python-csv gtabview pyqtdeploy nibabel Pillow xtract scikit-learn scikit-image seaborn`

## GUI features
- data viewing (DICOM, natural scene images)
- preprocessing: labeling/annotation, patching, data augmentation, data splitting
- network training: hyperparameter setting and network selection
- test data evaluation: online accuracy/loss plots, confusion matrix, derived test metrics
- network visualization: architecture, feature maps, kernel weights, deep visualization

### Data viewing
#### Image Viewing
- flexible layout, 2D/3D mode, images can be loaded from python workspace
- prediction overlay: colors and hatches can be modified
#### Result Showing
#### labeling
- 3 selector(rectangle, ellipse, lasso), the name and color of the labels can be modified

### Network training
#### data preprocessing, parameter setting
- accuracy/loss curves can be dynamically plotted
#### network training
#### network test

### Network visualization 
- network structure: the network structure can be loaded from h5 file
- view_image feature maps and kernels
- envoke deep visualization from a test data image
