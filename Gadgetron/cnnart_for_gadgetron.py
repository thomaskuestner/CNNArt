import numpy as np
import sys
import numpy as np                  # for algebraic operations, matrices
import h5py
import scipy.io as sio              # I/O
import os                      # operating system
import argparse
import sys, os
sys.path.append(os.getcwd()) 
sys.path.append('/opt/data/CNNArt')
from utils.data import *
from utils.dlnetwork import *
from utils.Label import Label
import datetime
import yaml
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from utils.RigidPatching import fRigidPatching3D_maskLabeling
from utils.RigidUnpatching import fUnpatchSegmentation
from main_minimal import fParseConfig 
def gadget_cnnart(img_matrix):
    cfg = fParseConfig('/opt/data/CNNArt/config/param_gadgetron.yml')
    print('==== In cnnart_for_gadgetron, config loaded ====')
    data = Data(cfg)
    # patch and split into training, val, test set
    print('==== In cnnart_for_gadgetron, Data() set ====')
    assert 'Gadgetron' in cfg.keys() and cfg['Gadgetron']['useGadgetron'] == True

    # img_matrix = np.load(cfg['Gadgetron']['sPathIn'])
    img_matrix = np.squeeze(img_matrix)
    data.X_test = fRigidPatching3D_maskLabeling(
        dicom_numpy_array=img_matrix, 
        patchSize=[data.patchSizeX, data.patchSizeY,data.patchSizeZ], 
        patchOverlap=cfg['patchOverlap'], 
        mask_numpy_array=np.zeros(img_matrix.shape), 
        ratio_labeling=0.5,
        dataset=1)
    print('==== In cnnart_for_gadgetron, rigidPatching3D finished ====')
    
    data.X_test = np.moveaxis(data.X_test, -1, 0)
    data.X_test = np.reshape(data.X_test, data.X_test.shape+(1,))
    data.Y_test = np.zeros(data.X_test.shape)

    print('==== Import Networks ====')
    dlnetwork = Dlnetwork(cfg)

    print('==== Artifact detection ====')
    # dynamic loading of corresponding model
    if data.storeMode == 'STORE_TFRECORD':
        sModel = 'networks.FullyConvolutionalNetworks.motion.VResFCN_3D_Upsampling_final_Motion_Binary_tf'
        import networks.FullyConvolutionalNetworks.motion.VResFCN_3D_Upsampling_final_Motion_Binary_tf as cnnModel
    else:
        sModel = 'networks.FullyConvolutionalNetworks.motion.VResFCN_3D_Upsampling_final_Motion_Binary'
        import networks.FullyConvolutionalNetworks.motion.VResFCN_3D_Upsampling_final_Motion_Binary as cnnModel
    predictions = cnnModel.fPredict(X_test=data.X_test,
                        Y_test=data.Y_test,
                        Y_segMasks_test=np.zeros(data.X_test.shape),
                        sModelPath=dlnetwork.savemodel,
                        batch_size=dlnetwork.batchSize,
                        usingClassification=dlnetwork.usingClassification,
                        usingSegmentationMasks=data.usingSegmentationMasks,
                        dlnetwork=dlnetwork)
    np.save('prediction.npy', predictions['prob_pre'])                    
    print('==== Result plotting ====')
    if hasattr(data, 'X_test'):
        data.patchSizePrediction = [data.X_test.shape[1], data.X_test.shape[2], data.X_test.shape[3]]
        data.patchOverlapPrediction = data.patchOverlap
    else:
        data.patchSizePrediction = data.patchSize
        data.patchOverlapPrediction = data.patchOverlap
    unpatched_img_foreground = fUnpatchSegmentation(predictions['prob_pre'],
                                                    patchSize=data.patchSizePrediction,
                                                    patchOverlap=data.patchOverlapPrediction,
                                                    actualSize=img_matrix.shape,
                                                    iClass=1)
    unpatched_img_background = fUnpatchSegmentation(predictions['prob_pre'],
                                                    patchSize=data.patchSizePrediction,
                                                    patchOverlap=data.patchOverlapPrediction,
                                                    actualSize=img_matrix.shape,
                                                    iClass=0)

    unpatched_img_foreground, unpatched_img_background = sign_func(unpatched_img_foreground), sign_func(unpatched_img_background)
    np.save('unpatched_img_foreground.npy', unpatched_img_foreground)
    np.save('unpatched_img_background.npy', unpatched_img_background)
    print('==== All finished ====')
    return unpatched_img_background

def sign_func(matrix):
    matrix[matrix > 0.5*matrix.max()] = matrix.max()
    matrix[matrix <= 0.5*matrix.max()] = matrix.min()
    return matrix