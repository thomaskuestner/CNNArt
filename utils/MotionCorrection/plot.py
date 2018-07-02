import keras
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class PlotLosses(keras.callbacks.Callback):
    def __init__(self, file_name):
        self.file_name = file_name

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        plt.plot(self.x, self.losses, 'r')
        plt.plot(self.x, self.val_losses, 'b')
        plt.legend(['train', 'test'], loc='upper left')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig(self.file_name)