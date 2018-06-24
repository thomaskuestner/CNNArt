# -*- coding: utf-8 -*-
"""
Visualize CNNs

@author: Thomas Kuestner
reference to hadim https://gist.github.com/hadim/9fedb72b54eb3bc453362274cd347a6a
"""
import theano
import theano.tensor as T
import os
import os.path
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import glob
import yaml
import h5py
from DatabaseInfo import DatabaseInfo
import ..utilsGUI.DataPreprocessing as datapre
from keras.models import load_model
from .deepvis.network_visualization import make_mosaic,plot_feature_map,plot_all_feature_maps,get_weights_mosaic,plot_weights,plot_all_weights,on_click
from keras.utils import plot_model
import keras.optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau




# initalize stuff
sTypeVis = 'weights' # deep: lukas implementation, else: weights of first layer
lShow = False

# parse parameters

folderPath = os.path.dirname(os.path.split(os.path.abspath(__file__))[0])
cfgPath= os.path.dirname(os.path.dirname(__file__))
cfgPath= os.path.join(cfgPath,'config','param.yml')

with open(cfgPath, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)


# default database: MRPhysics with ['newProtocol','dicom_sorted']
dbinfo = DatabaseInfo(cfg['MRdatabase'],cfg['subdirs'],folderPath)

patchSize = cfg['patchSize']
batchSize = cfg['batchSize'][0]

sOutsubdir = cfg['subdirs'][2]
sOutPath = cfg['selectedDatabase']['pathout'] + os.sep + ''.join(map(str,patchSize)).replace(" ", "") + os.sep + sOutsubdir
sNetworktype = cfg['network'].split("_")
model_name = cfg['selectedDatabase']['bestmodel'][sNetworktype[2]]


model_path = sOutPath + model_name + '_model.h5'
model = load_model(model_path)
# model = load_model('/no_backup/d1240/CNNArt/results/4040/testout4040_lr_0.001_bs_128_model.h5')

plot_model(model, to_file='model.png', show_layer_names=True,rankdir='TB')

if sTypeVis == 'deep':

    #Perform the visualization:
    class_idx = 0
    reg_param = 1/(2e-4)
    output    = model.get_output()
    input     = model.input

    cost 	   = -T.sum(T.log(output[:,class_idx]+1e-8))
    gradient = theano.tensor.grad(cost, input)
    calcGrad = theano.function([input], gradient)
    calcCost = theano.function([input], cost)


    #1. Use Deep Visualization
    #define the cost function (negative log-likelihood for class with class_idx:
    dv     = network_visualization.DeepVisualizer(calcGrad, calcCost, np.random.uniform(0,1.0, size=(1,1,patchSize[0,0],patchSize[0,1])), alpha = reg_param)
    resultDV = dv.optimize(np.random.uniform(0,1.0, size=(1,1,patchSize[0,0],patchSize[0,1])))

    if lShow:
        plt.figure(1)
        plt.title('deep visualizer')
        plt.imshow(resultDV.reshape(patchSize[0],patchSize[1]))
        plt.show()

    print('Saving deep visualization')
    sio.savemat(sSaveName[0] + '_DV.mat', {'resultDV': resultDV})

    #2. Use subset selection:
    step_size = 0.019
    reg_param = 1/(2e-4)

    #data_c = test[100:110] # extract images from the examples as initial point
    resultAll = []
    for i in range(0,len(test),10):
        print('### Patch %d/%d ###' % (i, len(test)))
        data_c = test[i:i+10]
        oss_v  = network_visualization.SubsetSelection(calcGrad, calcCost, data_c, alpha = reg_param, gamma = step_size)
        result = oss_v.optimize(np.random.uniform(0,1.0, size=data_c.shape))
        resultAll.append(result)
        #resultAll = np.concatenate((resultAll,result), axis=0)
        if lShow:
            plt.figure(2)
            plt.title('subset selection')
            plt.imshow(result[0].reshape(40,40))
            plt.show()

    print('Saving subset selection')
    sio.savemat(sSaveName[0] + '_SS.mat', {'resultSS': resultAll})
    #sio.savemat(sDataTest + os.sep + 'visualize_out.mat', {'result': resultAll})

elif sTypeVis == 'keras_weight':
    X_test = np.zeros((0, patchSize[0], patchSize[1]))
    y_test = np.zeros(0)
    for iImg in range(0, len(cfg['lPredictImg'])):
        # patches and labels of reference/artifact
        tmpPatches, tmpLabels = datapre.fPreprocessData(cfg['lPredictImg'][iImg], cfg['patchSize'], cfg['patchOverlap'],
                                                        1, cfg['sLabeling'])
        X_test = np.concatenate((X_test, tmpPatches), axis=0)
        y_test = np.concatenate((y_test, cfg['lLabelPredictImg'][iImg] * tmpLabels), axis=0)

    weight_name = sOutPath + model_name + '_weights.h5'
    opti = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]

    model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])

    model.load_weights(weight_name)

    X_test = np.expand_dims(X_test, axis=1)
    y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32) - 1)]).T

    model.evaluate(X_test, y_test, batch_size=batchSize)
    model.predict(X_test, batchSize, 1)
    # _ = plot_feature_map(model, 0, X_test[:4], n=2)
    _ = plot_all_feature_maps(model, X_test[:3], n=2)
    _[0].show()

elif sTypeVis == 'weights':
    layers_to_show = []
    n =[]

    for i, layer in enumerate(model.layers[:]):
        if hasattr(layer, "weights"):
            if len(layer.weights) == 0:
                continue
            weights = layer.weights[0].container.data
            if weights.ndim == 4:
                layers_to_show.append((i, layer))


    for i, (layer_id, layer) in enumerate(layers_to_show):
        w = layer.weights[0].container.data
        w = np.transpose(w, (3, 2, 0, 1))

        # n define the maximum number of weights to display
        n.append(w.shape[0])



    ########################################################
    #choose plot one layer's weight or plot all the weights#
    ########################################################

    #_=plot_weights(model, 0,n=n[-1])

    _ = plot_all_weights(model, n=n[-1])
    _.show()



