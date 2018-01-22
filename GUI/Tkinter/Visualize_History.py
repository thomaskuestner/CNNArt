import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np

Path = "C:/Users/Sebastian Milde/Documents/MATLAB/IQA/Codes_FeatureLearning/bestModels/head_4040_lr_0.0001_bs_64.mat"
conten = sio.loadmat(Path)
acc = np.squeeze(conten['acc'])
val_acc = np.squeeze(conten['val_acc'])
loss = np.squeeze(conten['loss'])
val_loss = np.squeeze(conten['val_loss'])
print(acc)
print(val_acc)
print(loss)
print(val_loss)
plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochen')
#plt.legend(['Train', 'Test'], loc = 'upper.left')
plt.show()

plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epochen')
#plt.legend(['Train', 'Test'], loc = 'upper.left')
plt.show()
