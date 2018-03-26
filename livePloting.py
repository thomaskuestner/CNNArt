
from __future__ import print_function
import numpy as np
np.random.seed(1337)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i =0
        self.x =[]
        self.logs =[]
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        x = range(len(self.losses))
        plt.figure()
        # acc

        plt.plot(self.x, self.accuracy['epoch'], 'r', label='train acc')
        # loss
        plt.plot(self.x, self.losses['epoch'], 'g', label='train loss')

            # val_acc
        plt.plot(self.x, self.val_acc['epoch'], 'b', label='val acc')
        # val_loss
        plt.plot(self.x, self.val_loss['epoch'], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

        self.i = self.i+1
'''
    def loss_plot(self, loss_type):
        x = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(x, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(x, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(x, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(x, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
'''

batch_size = 128
nb_classes = 10
nb_epoch = 2


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = LossHistory()


model.fit(X_train, Y_train,
            batch_size=batch_size, nb_epoch=nb_epoch,
            verbose=1,
            validation_data=(X_test, Y_test),
            callbacks=[history])


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

