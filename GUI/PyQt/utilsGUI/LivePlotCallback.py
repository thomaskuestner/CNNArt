
import keras

class LivePlotCallback(keras.callbacks.Callback):

    def __init__(self, dlart_handle):
        self.dlart_handle = dlart_handle

    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs.get('loss'))
        self.train_accuracy.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        self.dlart_handle.livePlotTrainingPerformance(train_acc=self.train_accuracy,
                                                      val_acc=self.val_acc,
                                                      train_loss=self.train_loss,
                                                      val_loss=self.val_loss)


