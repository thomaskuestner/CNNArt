import keras.backend as K
from keras.utils.vis_utils import plot_model, model_to_dot
import network_visualization
from network_visualization import *
# import theano
# import theano.tensor as T
# import tensorflow
# import tensorflow.tensor as T
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
from keras.models import load_model

def plot_subset_mosaic( im, nrows, ncols, fig, **kwargs):
    # Set default matplotlib parameters
    if not 'interpolation' in kwargs.keys():
        kwargs['interpolation'] = "none"

    if not 'cmap' in kwargs.keys():
        kwargs['cmap'] = "gray"

    # im = np.squeeze(im, axis=1)
    nimgs = len(im)

    imshape = im[0].shape

    mosaic = np.zeros(imshape)

    for i in range(nimgs):

        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.set_xlim(0, imshape[0] - 1)
        ax.set_ylim(0, imshape[1] - 1)


        mosaic = im[i]


        ax.imshow(mosaic, **kwargs)
        ax.set_axis_off()
    #fig.suptitle("Subset Selection of Patch #{} in Layer '{}'".format(ind, feature_map))
    # fig.canvas.mpl_connect('button_press_event', on_click)
    return fig

def simpleName(inputName):
    if "/" in inputName:
        inputName = inputName.split("/")[0]
        if ":" in inputName:
            inputName = inputName.split(':')[0]
    elif ":" in inputName:
        inputName = inputName.split(":")[0]
        if "/" in inputName:
            inputName = inputName.split('/')[0]

    return inputName

modelDimension =''
# modelName="testout404010_sf051_VNet_MS_lr_0.0005_bs_64_model.h5" # 3D With merge
# modelName='testout404010_lr_0.0005_bs_128_model.h5' # 3D Without merge
modelName="testout8080_lr_0.0001_bs_128_model.h5" # 2D Without merge

model = load_model(modelName)
#plot_model(model,'M3DModel.png',show_shapes=False)

twoInput=False
modelDimension=''
# h=h5py.File("normal404010sf051.h5","r") # 3D With merge
# h=h5py.File('crossVal404010.h5','r') # 3D Without merge
h=h5py.File("normal8080.h5","r") # 2D Without merge

# the number of the input
for i in h:
    if i=='X_test_p2' or i=='y_test_p2':
        twoInput=True
        break
# X_test=h['X_test'][:,432:540,:,:,:]
# y_test=h['y_test'][:,432:540]

if h['X_test'].ndim == 4:
    modelDimension = '2D'
    X_test = h['X_test'][:, 2052:2160, :, :]
    X_test = np.transpose(np.array(X_test), (1, 0, 2, 3))
    y_test = h['y_test'][:, 2052:2160]
    y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32) - 1)]).T
    y_test = np.squeeze(y_test, axis=1)
    if twoInput:
        X_test_p2 = h['X_test_p2'][:, 2052:2160, :, :]
        X_test_p2 = np.transpose(np.array(X_test_p2), (1, 0, 2, 3))
        y_test_p2 = h['y_test_p2'][:, 2052:2160]
        y_test_p2 = np.asarray([y_test_p2[:], np.abs(np.asarray(y_test_p2[:], dtype=np.float32) - 1)]).T
        y_test_p2 = np.squeeze(y_test_p2, axis=1)
elif h['X_test'].ndim == 5:
    modelDimension = '3D'
    X_test=h['X_test'][:,0:20,:,:,:]
    X_test = np.transpose(np.array(X_test), (1, 0, 2, 3, 4))
    y_test = h['y_test'][:, 0:20]
    y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32) - 1)]).T
    y_test = np.squeeze(y_test, axis=1)
    if twoInput:
        X_test_p2 = h['X_test_p2'][:, 0:20, :, :, :]
        X_test_p2 = np.transpose(np.array(X_test_p2), (1, 0, 2, 3, 4))
        y_test_p2 = h['y_test_p2'][:, 0:20]
        y_test_p2 = np.asarray([y_test_p2[:], np.abs(np.asarray(y_test_p2[:], dtype=np.float32) - 1)]).T
        y_test_p2 = np.squeeze(y_test_p2, axis=1)
else:
    print('the dimension of X_test should be 4 or 5')


# batchSize=5
# score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batchSize, verbose=1)
# prob_pre = model.predict(X_test, batch_size=batchSize, verbose=1)


# h5name=modelDimension+'.h5'
# h3=h5py.File(h5name,'w')
h3=h5py.File('M2D_WithoutMergettt.h5','w')
# h3=h5py.File('M3D_WithMergettt.h5','w')
modelname=h3.create_dataset('modelName',data=modelName)
numberOfTheInput=h3.create_dataset('twoInput',data=twoInput)
modelDimension=h3.create_dataset('modelDimension',data=modelDimension)

Xtest = h3.create_dataset('subset_selection', data=X_test)
if twoInput:
    Xtestp2 = h3.create_dataset('subset_selection_2', data=X_test_p2)


layer_index_name={} # used to save each layer's name and the related index
for i,layer in enumerate(model.layers):
    layer_index_name[layer.name]=i

activation={}
layer_outputValue={}

for i,layer in enumerate(model.input_layers):

    # get_activations = K.function([model.layers[layer_index_name[layer.name]].input, K.learning_phase()],
    #                              [model.layers[layer_index_name[layer.name]].output, ])

    get_activations = K.function([layer.input, K.learning_phase()],
                                 [layer.output, ])

    if i==0:
        layer_outputValue[layer.name]= get_activations([X_test, 0])[0]
    elif i==1:
        layer_outputValue[layer.name] = get_activations([X_test_p2, 0])[0]
    else:
        print('no output of the input layer is created')


act=h3.create_group('activations')
for i,layer in enumerate(model.layers):
    # input_len=layer.input.len()
    if hasattr(layer.input,"__len__"):
        if len(layer.input)==2:
            inputLayerNameList = []
            for ind_li,layerInput in enumerate(layer.input):
                inputLayerNameList.append(simpleName(layerInput.name))

            get_activations = K.function([layer.input[0],layer.input[1], K.learning_phase()], [layer.output, ])
            layer_outputValue[layer.name] = get_activations([layer_outputValue[inputLayerNameList[0]],
                                                             layer_outputValue[inputLayerNameList[1]],0])[0]
            a = act.create_dataset(layer.name, data=layer_outputValue[layer.name])

        elif len(layer.input)==3:
            inputLayerNameList = []
            for ind_li, layerInput in enumerate(layer.input):
                inputLayerNameList.append(simpleName(layerInput.name))

            get_activations = K.function([layer.input[0], layer.input[1], layer.input[2],K.learning_phase()], [layer.output, ])
            layer_outputValue[layer.name] = get_activations([layer_outputValue[inputLayerNameList[0]],
                                                             layer_outputValue[inputLayerNameList[1]],
                                                             layer_outputValue[inputLayerNameList[2]],0])[0]
            a = act.create_dataset(layer.name, data=layer_outputValue[layer.name])

        elif len(layer.input)==4:
            inputLayerNameList = []
            for ind_li, layerInput in enumerate(layer.input):
                inputLayerNameList.append(simpleName(layerInput.name))

            get_activations = K.function([layer.input[0], layer.input[1], layer.input[2],layer.input[3], K.learning_phase()],
                                         [layer.output, ])
            layer_outputValue[layer.name] = get_activations([layer_outputValue[inputLayerNameList[0]],
                                                             layer_outputValue[inputLayerNameList[1]],
                                                             layer_outputValue[inputLayerNameList[2]],
                                                             layer_outputValue[inputLayerNameList[3]],0])[0]
            a = act.create_dataset(layer.name, data=layer_outputValue[layer.name])

        elif len(layer.input)==5:
            inputLayerNameList = []
            for ind_li, layerInput in enumerate(layer.input):
                inputLayerNameList.append(simpleName(layerInput.name))

            get_activations = K.function(
                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],K.learning_phase()],
                [layer.output, ])
            layer_outputValue[layer.name] = get_activations([layer_outputValue[inputLayerNameList[0]],
                                                             layer_outputValue[inputLayerNameList[1]],
                                                             layer_outputValue[inputLayerNameList[2]],
                                                             layer_outputValue[inputLayerNameList[3]],
                                                             layer_outputValue[inputLayerNameList[4]],0])[0]
            a = act.create_dataset(layer.name, data=layer_outputValue[layer.name])

        else:
            print('the number of input is more than 5')

    else:
        get_activations = K.function([layer.input, K.learning_phase()], [layer.output, ])
        inputLayerName = simpleName(layer.input.name)
        layer_outputValue[layer.name] = get_activations([layer_outputValue[inputLayerName], 0])[0]
        a = act.create_dataset(layer.name, data=layer_outputValue[layer.name])

dot = model_to_dot(model, show_shapes=False, show_layer_names=True, rankdir='TB')
# layers_by_depth = model.layers_by_depth
if hasattr(model,"layers_by_depth"):
    layers_by_depth = model.layers_by_depth
elif hasattr(model.model,"layers_by_depth"):
    layers_by_depth = model.model.layers_by_depth
else:
    print('the model or model.model should contain parameter layers_by_depth')


layer_by_depth=h3.create_group('layer_by_depth')
weights=h3.create_group('weights')

maxCol = 0
maxRow = len(layers_by_depth)
# Save the structure,weights the layers' names in .h5 file
for i in range(len(layers_by_depth)):
    i_layer = layer_by_depth.create_group(str(i)) #the No i layer  in the model
    for ind,layer in enumerate(layers_by_depth[i]): # the layers in No i layer in the model
        if maxCol < ind:
            maxCow = ind
        i_ind=i_layer.create_group(str(ind))
        layer_name=i_ind.create_dataset('layer_name', data=layer.name)
        if len(layer.weights)==0:
            w=0
        else:
            # w=layer.weights[0].container.data
            # sess_i = tf.InteractiveSession()
            w = layer.weights[0]
            init = tf.global_variables_initializer()
            with tf.Session() as sess_i:
                sess_i.run(init)
                # print(sess_i.run(w))
                w=sess_i.run(w)

        weights.create_dataset(layer.name,data=w) # save the weights



# class_idx = 0
# reg_param = 1 / (2e-4)
#
#
# input = model.input #tensor
# cost  = -K.sum(K.log(input[:,class_idx]+1e-8))  #tensor
# gradient = K.gradients(cost, input)  #list
#
# sess = tf.InteractiveSession()
# calcCost = network_visualization.TensorFlowTheanoFunction([input], cost)
# calcGrad = network_visualization.TensorFlowTheanoFunction([input], gradient)
#
# #2. Use subset selection:
#
# step_size = 0.19
# reg_param = 0.0000001
#
# test=X_test[:]
# data_c = test
# oss_v = network_visualization.SubsetSelection(calcGrad, calcCost, data_c, alpha=reg_param, gamma=step_size)
# result = oss_v.optimize(np.random.uniform(0, 1.0, size=data_c.shape))
#
# result=result * test
# result[result>0]=1

# fig = plt.figure()
# fig.suptitle('Xc')
#
# result=np.squeeze(result,axis=1)
# result =result[14]
# result=np.transpose(result, (2,0,1))
# nimgs = len(result)
# nrows = int(np.round(np.sqrt(nimgs)))
# ncols = int(nrows)
# if (nrows ** 2) < nimgs:
#     ncols += 1
#
# fig = plot_subset_mosaic(result, nrows, ncols, fig)
# plt.show()

# h3.create_dataset('subset_selection', data=result)
h3.close()