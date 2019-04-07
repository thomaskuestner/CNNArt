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
#### Image and result viewing
- flexible layout, 2D/3D mode, images can be loaded from python workspace
- prediction overlay: colors and hatches can be modified
- assistant tools: ROI selector, cursor inspector
#### Labeling
- 3 selector(rectangle, ellipse, lasso), feasible labeling and editing
- the name and color of the labels can be modified
- connected label list: showing label name, show on/off label, sorting label names
### Network training
#### Data preprocessing, parameter setting
- perform patching datasets with different splitting methods, labeling methods, storage modes, using segmentation masks.
- setting data path and edtting patch size, overlay, datasets splitting ratio, number of folds for cross validation
- select datasets for training/validation/test
#### Network training
- choose hyperparameters: batch size, learning rate, weight decay, (Nestrov) momentum, number of epochs
- using data augumentation with all kinds of methods
- accuracy/loss curves can be dynamically plotted
- setting learning output and save network model, model weights, and training information
- comprehensive network interface 
### Network visualization 
#### Network visualization
- network structure: the network structure can be loaded from h5 file
- show feature maps and kernels of each layer
- show subsection
#### Network testing
- load test datasets from workspace or path
- load model from workspace or path
- showing test evaluation
- show segmentation mask, artifacts
