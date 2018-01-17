# -*- coding: utf-8 -*-
"""
Visualize CNNs

@author: Thomas Kuestner
"""
import theano
import theano.tensor as T
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import glob
import yaml
import h5py
from DatabaseInfo import DatabaseInfo
import utils.DataPreprocessing as datapre
import utils.Training_Test_Split as ttsplit
import cnn_main



def make_mosaic(im, nrows, ncols, border=1):

    import numpy.ma as ma

    nimgs = len(im)
    imshape = im[0].shape

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    im
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = im[i]

    return mosaic


#load dataset:
if os.name == 'posix':
    dataPath = '/scratch/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Headcross/4040/patient/02'
    sDataTest = '/scratch/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Headcross'
    sDataTrain = dataPath + os.sep + 'crossVal02_4040.mat'
else:
    dataPath = 'W:\ImageSimilarity\Databases\MRPhysics\CNN\Headcross\4040\patient\02'
    sDataTest = 'W:\ImageSimilarity\Databases\MRPhysics\CNN\Headcross'

#modelPath = dataPath + os.sep + 'outcrossVal024040_lr_0.0001_bs_64'

parser = argparse.ArgumentParser(description='''CNN feature visualization''', epilog='''(c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de''')
parser.add_argument('-i','--inPath', nargs = 1, type = str, help='input path to *.mat', default= '/scratch/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Headcross')
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
