# CNNArtGUI

The the original code repository is

https://github.com/thomaskuestner/CNNArt

This GUI is designed with PyQt5 and works for Python >3.5.

## GUI calling
run `imagine.py` to open the GUI. <br/>
This GUI can be run independent from CNNArt. <br/>
Before running, please change path in param_GUI.py
`from imagine import imagin_main`
`imagine_main('pathOfImageFile','pathOfImageDirectory', ...)`<br/>
`imagine_main(image array)`<br/>

## Requirements
`python3.6 -m pip install jsonschema pickleshare h5py keras matplotlib pandas scipy protobuf tensorflow pyYAML PyQt5-stubs PyQt5 numpy pandas argparse hyperas hyperopt graphviz dicom pydicom dicom-numpy pydot pyqtgraph python-csv gtabview pyqtdeploy nibabel Pillow xtract scikit-learn scikit-image seaborn`

## GUI features
- data viewing (DICOM, natural scene images)
- pre-processing: labeling/annotation, patching, data augmentation, data splitting for 2D and 3D data
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
#### Data pre-processing, parameter setting
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
## File Format
### Image file format
The GUI supports a wide range of image formats to view, including 
- normal image formats .CUR, .ICNS, .SVG, .TGA, .BMP, .WEBP, .GIF,.JPG, .JPEG, .PNG, .PBM, .PGM, .PPM,.TIFF,.XBM; 
- a single DICOM image, including .IMA, .DCM; 
- a folder with all DICOM images;
- a single file in .NII;
- a single file in.MAT, .NPY where stores image array; for 5D array, channel-last
- image arrays with multi-dimensions from 2D to 5D
### Result file format
- in .MAT, .NPY or .NPZ
- in the result file, masking shape should be in the same shape as original image
- the original image array should be saved with the item key 'img', or image path with the key 'img_path'
- the preferred colors for each class in thr key 'color'
### Network format
network model in HDF5
### Datasets format
datasets in HDF5

## Configuration
- In the file configuration/predefined_classes.txt, it is able to edit frequently used classes for manual labeling.
- The file networks/network_generic.py, it is a template for creating a network suitable for this GUI. After that, it should be saved with proper name in the path networks/SUBDIRS/...                   
- In the config/database_generic.py, it is a template for creating a database information in .CSV and saving in the path config/database/..., which is used in GUI. 
- In the file config/param_GUI.yml
  - change datasets, masking, output path   
  - change datasets information
  - directory separation symbol should be '/' independent of OS   
                                                                                                                             