import numpy as np
import keras
import os
import h5py

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, input_path, batch_size=16, dim=(64, 64, 16), usingClassification=True, n_channels=1, n_classes=2, shuffle=True, list_IDs=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.input_path = input_path
        if list_IDs is None:
            # parse input directory
            self.list_IDs = [f for f in os.listdir(input_path) if f.endswith('.hdf5')]  # only take hdf5 files
        else:
            self.list_IDs = list_IDs
        self.usingClassification = usingClassification
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype='f')  # ATTENTION: Assuming channels_last!
        YSeg = np.empty((self.batch_size, *self.dim, self.n_channels), dtype='f')
        Y = np.empty((self.batch_size, self.n_classes), dtype='f')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            f = h5py.File(self.input_path + os.sep + ID, 'r')
            X[i,] = np.array(f.get('X'))
            Y[i,] = np.array(f.get('Y'))
            YSeg[i,] = np.array(f.get('Y_segMasks'))
            f.close()

        if self.usingClassification:
            return X, {'segmentation_output': YSeg, 'classification_output': Y}
        else:
            return X, Y