'''
Copyright: 2016-2019 Thomas Kuestner (thomas.kuestner@med.uni-tuebingen.de) under Apache2 license
@author: Thomas Kuestner
'''
# imports
import sys
import numpy as np                  # for algebraic operations, matrices
import h5py
import scipy.io as sio              # I/O
import os                      # operating system
import argparse
from utils.data import *
from utils.dlnetwork import *
from utils.Label import Label
import datetime
import yaml
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def fParseConfig(sFile):
    # get config file
    with open(sFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg


def fArtDetection(data, dlnetwork, sMode):
    # set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(data.iGPU)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    #if (data.storeMode != 'STORE_TFRECORD') & (sMode != 'plotting'):
            ################################################################################################################
            # debug!
            # for i in range(data.X_train.shape[0]):
            #
            #     plt.subplot(141)
            #     plt.imshow(data.X_train[i, :, :, 4, 0])
            #
            #     plt.subplot(142)
            #     plt.imshow(data.Y_segMasks_train[i, :, :, 4, 0])
            #
            #     plt.subplot(143)
            #     plt.imshow(data.Y_segMasks_train[i, :, :, 4, 1])
            #
            #     #plt.subplot(144)
            #     #plt.imshow(data.Y_segMasks_train[i, :, :, 4, 2])
            #
            #     plt.show()
            #
            #     print(i)

            ###################################################################################################################

    # output folder
    data.outPutFolderDataPath = data.pathOutput + os.sep + dlnetwork.neuralNetworkModel + "_"
    if data.patchingMode == 'PATCHING_2D':
        data.outPutFolderDataPath += "2D" + "_" + str(data.patchSizeX) + "x" + str(data.patchSizeY)
    elif data.patchingMode == 'PATCHING_3D':
        data.outPutFolderDataPath += "3D" + "_" + str(data.patchSizeX) + "x" + str(data.patchSizeY) + \
                                     "x" + str(data.patchSizeZ)

    data.outPutFolderDataPath += "_" + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')

    if not os.path.exists(data.outPutFolderDataPath):
        os.makedirs(data.outPutFolderDataPath)

    if not os.path.exists(data.outPutFolderDataPath + os.sep + 'checkpoints'):
        os.makedirs(data.outPutFolderDataPath + os.sep + 'checkpoints')

    # summarize cnn and training
    # TODO
    #data.create_cnn_training_summary(dlnetwork.neuralNetworkModel, data.outPutFolderDataPath)

    # segmentation FCN training
    # dynamic loading of corresponding model
    if data.storeMode == 'STORE_TFRECORD':
        sModel = 'networks.FullyConvolutionalNetworks.motion.VResFCN_3D_Upsampling_final_Motion_Binary_tf'
        import networks.FullyConvolutionalNetworks.motion.VResFCN_3D_Upsampling_final_Motion_Binary_tf as cnnModel
    else:
        sModel = 'networks.FullyConvolutionalNetworks.motion.VResFCN_3D_Upsampling_final_Motion_Binary'
        import networks.FullyConvolutionalNetworks.motion.VResFCN_3D_Upsampling_final_Motion_Binary as cnnModel
    #cnnModel = __import__(sModel, globals(), locals(), ['fTrain', 'fPredict'], 0)
    # dynamic module loading with specified functions and with absolute importing (level=0) -> work in both Python2 and Python3

    if sMode == 'training':
        if data.storeMode == 'STORE_TFRECORD':
            # only TFRecord processing in training
            cnnModel.fTrain(datagenerator=data.datagenerator,
                            X_test=data.X_test,
                            y_test=data.Y_test,
                            Y_segMasks_test=data.Y_segMasks_test,
                            sOutPath=data.outPutFolderDataPath,
                            patchSize=[data.patchSizeX, data.patchSizeY, data.patchSizeZ],
                            batchSize=dlnetwork.batchSize,
                            learningRate=dlnetwork.learningRate,
                            iEpochs=dlnetwork.epochs,
                            dlnetwork=dlnetwork)
        else:
            if dlnetwork.trainMode == 'ARRAY':
                cnnModel.fTrain(X_train=data.X_train,
                                y_train=data.Y_train,
                                Y_segMasks_train=data.Y_segMasks_train,
                                X_valid=data.X_validation,
                                y_valid=data.Y_validation,
                                Y_segMasks_valid=data.Y_segMasks_validation,
                                X_test=data.X_test,
                                y_test=data.Y_test,
                                Y_segMasks_test=data.Y_segMasks_test,
                                sOutPath=data.outPutFolderDataPath,
                                patchSize=[data.patchSizeX, data.patchSizeY, data.patchSizeZ],
                                batchSize=dlnetwork.batchSize,
                                learningRate=dlnetwork.learningRate,
                                iEpochs=dlnetwork.epochs,
                                dlnetwork=dlnetwork)
            else: # GENERATOR
                cnnModel.fTrain(X_train=data.datasetOutputPath + os.sep + 'train',
                                X_valid=data.datasetOutputPath + os.sep + 'validation',
                                X_test=data.datasetOutputPath + os.sep + 'test',
                                sOutPath=data.outPutFolderDataPath,
                                patchSize=[data.patchSizeX, data.patchSizeY, data.patchSizeZ],
                                batchSize=dlnetwork.batchSize,
                                learningRate=dlnetwork.learningRate,
                                iEpochs=dlnetwork.epochs,
                                dlnetwork=dlnetwork)

        print('==== Network training finished ====')
        if data.plotresults:
            # prepare test data output
            predictions = fgetpredictions(data.outPutFolderDataPath, data.usingSegmentationMasks,
                                          dlnetwork.usingClassification)


    elif sMode == 'prediction':  # prediction
        predictions = cnnModel.fPredict(X_test=data.X_test,
                          Y_test=Y_test,
                          Y_segMasks_test=data.Y_segMasks_test,
                          sModelPath=dlnetwork.savemodel,
                          batch_size=dlnetwork.batchSize,
                          usingClassification=dlnetwork.usingClassification,
                          usingSegmentationMasks=data.usingSegmentationMasks,
                          dlnetwork=dlnetwork)

        print('==== Network testing finished ====')

    else:  # plotting
        # load from pre-trained network run, the predicted outputs
        predictions = fgetpredictions(data.outPutFolderDataPath, data.usingSegmentationMasks, dlnetwork.usingClassification, data.plotTestFile)

    # result preparation
    if data.plotresults | (sMode == 'plotting'):
        print('==== Result plotting ====')
        if data.usingSegmentationMasks:
            data.handlepredictionssegmentation(predictions)
        else:
            data.handlepredictions(predictions)


def fgetpredictions(sOutPath, usingSegmentationMasks, usingClassification, plotTestFile=None):
    # save names
    if plotTestFile is None:
        _, sPath = os.path.splitdrive(sOutPath)
        sPath, sFilename = os.path.split(sPath)
        sFilename, sExt = os.path.splitext(sFilename)
        model_name = sOutPath + os.sep + sFilename
    else:
        model_name = plotTestFile

    dataIn = sio.loadmat(model_name)
    predictions = {}

    if usingSegmentationMasks:
        predictions['prob_pre'] = dataIn['segmentation_predictions']
        if usingClassification:
            predictions['classification_predictions'] = dataIn['classification_predictions']
            predictions['loss_test'] = dataIn['loss_test']
            predictions['segmentation_output_loss_test'] = dataIn['segmentation_output_loss_test']
            predictions['classification_output_loss_test'] = dataIn['classification_output_loss_test']
            predictions['segmentation_output_dice_coef_test'] = dataIn['segmentation_output_dice_coef_test']
            predictions['classification_output_acc_test'] = dataIn['classification_output_acc_test']
        else:
            predictions['score_test'] = dataIn['score_test']
            predictions['acc_test'] = dataIn['acc_test']
    else:
        predictions['confusion_matrix'] = dataIn['confusion_matrix']
        predictions['classification_report'] = dataIn['classification_report']
        predictions['prob_pre'] = dataIn['prob_test']

    return predictions


if __name__ == "__main__":  # for command line call
    # input parsing
    parser = argparse.ArgumentParser(description='''CNN artifact detection''', epilog='''(c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de''')
    parser.add_argument('-c', '--config', nargs = 1, type = str, help='path to config file', default= 'config/param_minimal.yml')
    parser.add_argument('-i','--inPath', nargs = 1, type = str, help='input path to *.mat of stored patches', default= '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/in.mat')
    parser.add_argument('-o','--outPath', nargs = 1, type = str, help='output path to the file used for storage (subfiles _model, _weights, ... are automatically generated)', default= '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/out' )
    parser.add_argument('-m','--model', nargs = 1, type = str, choices =['motion_head_CNN2D', 'motion_abd_CNN2D', 'motion_all_CNN2D', 'motion_CNN3D', 'motion_MNetArt', 'motion_VNetArt', 'multi_DenseResNet', 'multi_InceptionNet'], help='select CNN model', default='motion_2DCNN_head' )
    parser.add_argument('-t','--train', dest='train', action='store_true', help='if set -> training | if not set -> prediction' )
    parser.add_argument('-p','--paraOptim', dest='paraOptim', type = str, choices = ['grid','hyperas','none'], help='parameter optimization via grid search, hyper optimization or no optimization', default = 'none')
    parser.add_argument('-b', '--batchSize', nargs='*', dest='batchSize', type=int, help='batchSize', default=64)
    parser.add_argument('-l', '--learningRates', nargs='*', dest='learningRate', type=int, help='learningRate', default=0.0001)
    parser.add_argument('-e', '--epochs', nargs=1, dest='epochs', type=int, help='epochs', default=300)

    args = parser.parse_args()

    # parse input
    #cfg = fParseConfig(args.config[0])
    cfg = fParseConfig('config/param_minimal_tk.yml')

    data = Data(cfg)

    # patch and split into training, val, test set
    if (cfg['sMode'] == 'training') | (cfg['sMode'] == 'prediction'):
        data.generateDataset()

    # get network parameters
    dlnetwork = Dlnetwork(cfg)

    print('==== Artifact detection ====')
    fArtDetection(data, dlnetwork, cfg['sMode'])
