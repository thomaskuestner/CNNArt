import numpy as np
import h5py
import keras
from fSetGPU import*

#######################################################################################################################
# Preprocessing of loaded data:                                                                                       #
# To train Convolutional Neuronal Networks (CNNs) it's important to preprocess Patches and labels.                    #
# The CNN get's the following Parameter:                                                                              #
# X_train: 4D array -----> (number of patches, channel, patchSize[0], patchSize[1])                                   #
# y_train: array([[0., 1.], [0., 1.], [1., 0.], [0., 1.]]) ----> (number of labels, categorical variable)             #
# X_test: 4D array -----> (number of patches, channel, patchSize[0], patchSize[1])                                    #
# y_test: array([[0., 1.], [0., 1.], [1., 0.], [0., 1.]]) ----> (number of labels, categorical variable)              #
# patchSize:  array([[ 100.,  100.]])                                                                                 #
#######################################################################################################################

def fPreprocessDataForCNN(Path):
    dData = dict()
    with h5py.File(Path, 'r') as hf:
        X_train = hf['X_train'][:]
        X_test = hf['X_test'][:]
        y_train = hf['y_train'][:]
        y_test = hf['y_test'][:]
        patchSize = hf['patchSize'][:]

    X_train = np.expand_dims(X_train, axis=1) #axis =1
    X_test = np.expand_dims(X_test, axis=1) # axis = 1
    print(X_train.shape, X_test.shape)
    y_train = y_train == 1
    y_train = y_train.astype(int)
    y_test = y_test == 1
    y_test = y_test.astype(int)
    y_train = keras.utils.to_categorical(y_train)
    #y_train = np.asarray([y_train, np.abs(np.asarray(y_train, dtype=np.float32) - 1)]).T
    y_test = keras.utils.to_categorical(y_test)
    #y_test = np.asarray([y_test, np.abs(np.asarray(y_test, dtype=np.float32) - 1)]).T
    print(y_train, y_test)

    patchSize = np.array(([patchSize]), dtype = np.float32)

    dData['X_train'] = X_train
    dData['X_test'] = X_test
    dData['y_train'] = y_train
    dData['y_test'] = y_test
    dData['patchSize'] = patchSize

    return dData


#with open(os.path.expanduser('~')+'\\.keras\\keras.json','w') as f:
 #   new_settings = """{\r\n
  #  "epsilon": 1e-07,\r\n
   # "image_data_format": "channels_first",\n
   # "backend": "theano",\r\n
   # "floatx": "float32"\r\n
   # }"""
   # f.write(new_settings)


def CNN_execute(sPathIn, sPathOut, batchSize, learn_rate, epoch, model):
    fSetGPU()
    keras.backend.set_image_data_format('channels_first')
    dData = fPreprocessDataForCNN(sPathIn)
    cnnModel = __import__(model, globals(), locals(), ['createModel', 'fTrain', 'fPredict'], -1)
    cnnModel.fTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sPathOut, dData['patchSize'],batchSize=batchSize, learningRate=learn_rate,
                    iEpochs=epoch)
