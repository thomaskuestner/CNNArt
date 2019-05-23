# CNNArt [![Build Status](https://semaphoreci.com/api/v1/thomaskuestner/cnnart/branches/master/shields_badge.svg)](https://semaphoreci.com/thomaskuestner/cnnart) [![Waffle.io - Columns and their card count](https://badge.waffle.io/thomaskuestner/CNNArt.svg?columns=all)](https://waffle.io/thomaskuestner/CNNArt)
### Automatic and reference-free MR artifact detection
- localization and quantification of artifacts (motion, magnetic field inhomogeneity and noise) in binary or multi-class setting
- correction of motion-induced artifacts (rigid and non-rigid motion)

### Visualization of trained network architectures
- visualize the trained kernels and feature maps
- deep visualization: significance map of trained network content, backpropagate most-likely input patch and sparse attractor points of a test image

### GUI
easy-to-use graphical interface for medical deep learning
- 2D/3D data viewer
- data preprocessing: labeling, patching, data augmentation, data splitting
- network training: parameter setting, training/validation/test set selection, call to DL backend (keras, Tensorflow, ...)
- test data evaluation: accuracy/loss plots, confusion matrix and derived metrics
- network visualization: kernel weights, feature maps and deep visualization

## Usage
Install the requirements
```shell
$ python3 -m pip install -r requirements.txt
```

### direct
1. define database layout in `config/database/_NAME_OF_DATABASE_.csv` (as specified in param.yml -> MRdatabase)
2. edit parameters in `config/param.yml`
3. run code via `main.py`

### GUI
training/prediction can also be invoked from the GUI. Please adapt `mainGUI_Template.py` according to your needs   
`Qt_main.py`

### calling structure
`main.py ==> model.fTrain()/fPredict()`

## Networks
Network | Artifact type detection | Publication
------------ | ------------- | -------------
CNN2D | motion_rigid <br/> motion_non-rigid <br/> motion_both | 1, 7
CNN3D | motion_rigid <br/> motion_non-rigid <br/> motion_both | 2, 6
MNetArt | motion_rigid <br/> motion_non-rigid <br/> motion_both | 2, 4
VNetArt | motion_rigid <br/> motion_non-rigid <br/> motion_both | 2, 4, 5
DenseNet | motion_both <br/> inhomogeneity <br/> noise |
DenseResNet | motion_both <br/> inhomogeneity <br/> noise | 3
ResNet | motion_both <br/> inhomogeneity <br/> noise |
GoogleNet | motion_both <br/> inhomogeneity |
InceptionNet | motion_both <br/> inhomogeneity <br/> noise | 3
VGGNet | motion_both <br/> inhomogeneity |

## References
1. [Küstner, T., Liebgott, A., Mauch, L., Martirosian, P., Bamberg, F., Nikolaou, K., Yang B., Schick F. & Gatidis, S. (2017). Automated reference-free detection of motion artifacts in magnetic resonance images. Magnetic Resonance Materials in Physics, Biology and Medicine, 1-14.](https://link.springer.com/article/10.1007/s10334-017-0650-z)<br/>
2. Küstner, T., Jandt, M., Liebgott, A., Mauch, L., Martirosian, P., Bamberg, F., Nikolaou, K., Gatidis, S., Schick, F. & Yang, B. (2018). Automatic Motion Artifact Detection for Whole-Body Magnetic Resonance Imaging. Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).<br/>
3. Küstner, T., Liu, K., Liebgott, A., Mauch, L., Martirosian, P., Bamberg, F., Nikolaou, K., Yang, B., Schick, F. & Gatidis, S. (2018). Simultaneous detection and identification of MR artifact types in whole-body imaging. Proceedings of the International Society for Magnetic Resonance in Medicine (ISMRM).<br/>
4. Küstner, T., Jandt, M., Liebgott, A., Mauch, L., Martirosian, P., Bamberg, F., Nikolaou, K., Gatidis, S., Yang, B. & Schick, F. (2018). Motion artifact quantification and localization for whole-body MRI. Proceedings of the International Society for Magnetic Resonance in Medicine (ISMRM).<br/>
5. Liebgott, A., Milde, S., Jandt, M., Mauch, L., Martirosian, P., Bamberg, F., Schick, F., Nikolaou, K., Yang, B., Gatidis, S. & Küstner, T. (2018). Impact of Labeling Process on Automated Motion Artifact Detection in Whole-Body MR Images with a Deep Learning Approach: A Comparative Study. Proceedings of the ISMRM Workshop on Machine Learning.<br/>
6. Küstner, T., Liegbott, A., Mauch, L., Martirosian, P., Schick, F., Bamberg, F., Nikolaou, K., Yang, B. & Gatidisi, S. (2017). Automatic reference-free motion artifact detection and quantification in T1-weighted MR images of the head and abdomen. Proceedings of the Annual Scientific Meeting (ESMRMB).<br/>
7. Küstner, T., Liebgott, A., Mauch, L., Martirosian, P., Nikolaou, K., Schick, F., Yang, B. & Gatidis, S. (2017). Automatic reference-free detection and quantification of MR image artifacts in human examinations due to motion. Proceedings of the International Society for Magnetic Resonance in Medicine (ISMRM).
