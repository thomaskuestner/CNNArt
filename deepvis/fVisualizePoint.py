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
from keras.models import load_model
from network_visualization import make_mosaic




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

sOutsubdir = cfg['subdirs'][2]
sOutPath = cfg['selectedDatabase']['pathout'] + os.sep + ''.join(map(str,patchSize)).replace(" ", "") + os.sep + sOutsubdir
sNetworktype = cfg['network'].split("_")
model_name = cfg['selectedDatabase']['bestmodel'][sNetworktype[2]]


model_path = sOutPath + model_name + '_model.h5'
model = load_model(model_path)
# model = load_model('/no_backup/d1240/CNNArt/results/4040/testout4040_lr_0.001_bs_128_model.h5')

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
    dataTrain = sio.loadmat(sDataTrain)
    X_train = dataTrain['X_train']
    y_train = dataTrain['y_train']
    ##########
    ##  not working
    ###########
    #convout1 = model.layers[1].output
    from random import randint

    img_to_visualize = randint(0, len(y_train) - 1)

    # Generate function to visualize first layer
    convout1_f = theano.function([model.get_input(train=False)], model.layers[1].get_output(train=False))
    convolutions = convout1_f(X_train[img_to_visualize: img_to_visualize + 1,:,:,:])

   #matplotlib inline
    # The non-magical version of the previous line is this:
    # get_ipython().magic(u'matplotlib inline')
    imshow = plt.imshow  # alias
    #plt.title("Image used: #%d (digit=%d)" % (img_to_visualize, y_train[img_to_visualize]))
    #imshow(X_train[img_to_visualize])

    plt.title("First convolution:")
    imshow(convolutions[0][0])

elif sTypeVis == 'weights':
    #visualize weight vectors

    # record the layers which have 'weights'
    layers_to_show = []

    for i, layer in enumerate(model.layers[:]):
        if hasattr(layer, "weights"):
            if len(layer.weights) == 0:
                continue
        w = layer.weights[0].container.data
        if w.ndim == 4:
            layers_to_show.append((i, layer))


    for i, (layer_id, layer) in enumerate(layers_to_show):
        w = layer.weights[0].container.data
        w = np.transpose(w, (3, 2, 0, 1))


        # n define the maximum number of weights to display
        n = w.shape[0]

        # Create the mosaic of weights
        nrows = int(np.round(np.sqrt(n)))
        ncols = int(nrows)

        if nrows ** 2 < n:
            ncols += 1

        #filters = [w[0, :, :] for w in w]
        fig = plt.figure(figsize=(15, 15))
        plt.suptitle("The Weights of Layer #{} called '{}' of type {}".format(layer_id, layer.name, layer.__class__.__name__))

        mosaic = make_mosaic(w[:, 0], nrows, ncols, border=1)

        im = plt.imshow(mosaic)



plt.show()


