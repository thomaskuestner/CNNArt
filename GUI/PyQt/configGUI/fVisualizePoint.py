# -*- coding: utf-8 -*-
"""
Visualize CNNs

@author: Thomas Kuestner
"""
import theano
import theano.tensor as T
import keras
import os
from keras.models import model_from_json
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import argparse

#load dataset:
from config.PATH import DLART_OUT_PATH
from configGUI.network_visualization import DeepVisualizer, SubsetSelection


dataPath = DLART_OUT_PATH + os.sep + '4040/patient/02'
sDataTest = DLART_OUT_PATH
if os.name == 'posix':
    sDataTrain = dataPath + os.sep + 'crossVal02_4040.mat'
else:
    sDataTest = DLART_OUT_PATH

#modelPath = dataPath + os.sep + 'outcrossVal024040_lr_0.0001_bs_64'

parser = argparse.ArgumentParser(description='''CNN feature visualization''', epilog='''(c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de''')
parser.add_argument('-i','--inPath', nargs = 1, type = str, help='input path to *.mat', default= DLART_OUT_PATH)
args = parser.parse_args()

# initalize stuff
sTypeVis = 'deep' # deep: lukas implementation, else: weights of first layer
lShow = False

#data = sio.loadmat(sDataTest + os.sep + 'visualize.mat')
data = sio.loadmat(args.inPath[0] + os.sep + 'visualize.mat')

#train  = data['X_train']
#ltrain = data['y_train']

test  = data['X_test']
ltest = data['y_test']
sSaveName = data['sSavefile']
modelPath = data['sOptiModel']
patchSize = data['patchSize']

#load the model:
#reg=1e-5
model = model_from_json(open(modelPath[0] + '_json').read())
model.load_weights(modelPath[0] + '_weights.h5')

#opti = keras.optimizers.RMSprop(lr=0.0003, rho=0.9, epsilon=1e-06)
#opti = keras.optimizers.SGD(lr=0.01, momentum=1e-8, decay=0.1, nesterov=True);
opti = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=opti)

#predict one image:
preds_prob = model.predict(test, batch_size=64, verbose=1)

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
    dv     = DeepVisualizer(calcGrad, calcCost, np.random.uniform(0,1.0, size=(1,1,patchSize[0,0],patchSize[0,1])), alpha = reg_param)
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
        oss_v  = SubsetSelection(calcGrad, calcCost, data_c, alpha = reg_param, gamma = step_size)
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
    w = model.layers[0].W.get_value()
    filters = [w[0,:,:] for w in w]
    plt.figure(3)
    plt.title('Weights layer 1')
    for i in range(len(filters)):
        plt.subplot(8,4,i+1)
        plt.imshow(filters[i])
