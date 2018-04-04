from utils.Unpatching import *
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
import dicom
import dicom_numpy as dicom_np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import copy


path = 'D:/med_data/MRPhysics/MA Results/FCN/FCN 3D-VResFCN-Upsampling final_3D_64x64x32_2018-03-30_10-47/FCN 3D-VResFCN-Upsampling final_3D_64x64x32_2018-03-30_10-47.mat'
preds = sio.loadmat(path)

# segmentation output
seg_dice_train = preds['segmentation_output_dice_coef_training']
seg_loss_train = preds['segmentation_output_loss_training']

seg_dice_val = preds['segmentation_output_dice_coef_val']
seg_loss_val = preds['segmentation_output_loss_val']


# classification output
clas_acc_train = preds['classification_output_acc_training']
clas_loss_train = preds['classification_output_loss_training']

clas_acc_val = preds['classification_output_acc_val']
clas_loss_val = preds['classification_output_loss_val']

# plot
epochs = np.arange(1, seg_dice_train.shape[1]+1)
fontsize = 12

# plot segmentation results
ax1 = plt.subplot(221)
plt.text(0.5, 1.08, 'Segmentation Performance',
         horizontalalignment='center',
         fontsize=18,
         transform = ax1.transAxes)
ax1.plot(epochs, np.squeeze(seg_dice_train), 'b')
ax1.plot(epochs, np.squeeze(seg_dice_val), 'r')
ax1.set_xlabel('Epochs', fontsize=fontsize)
ax1.set_ylabel('Dice Coefficient', fontsize=fontsize)
ax1.grid(b=True, which='both')
ax1.legend(['Training Dice Coefficient', 'Validation Dice Coefficient'], fontsize=fontsize)

ax3 = plt.subplot(223)
ax3.plot(epochs, np.squeeze(seg_loss_train), 'b')
ax3.plot(epochs, np.squeeze(seg_loss_val), 'r')
ax3.set_xlabel('Epochs', fontsize=fontsize)
ax3.set_ylabel('Dice Loss', fontsize=fontsize)
ax3.grid(b=True, which='both')
ax3.legend(['Training Dice Loss', 'Validation Dice Loss'], fontsize=fontsize)

# plot classification results
ax2 = plt.subplot(222)
plt.text(0.5, 1.08, 'Multi-class Classification Performance',
         horizontalalignment='center',
         fontsize=18,
         transform = ax2.transAxes)
ax2.plot(epochs, np.squeeze(clas_acc_train), 'b')
ax2.plot(epochs, np.squeeze(clas_acc_val), 'r')
ax2.set_xlabel('Epochs', fontsize=fontsize)
ax2.set_ylabel('Accuracy', fontsize=fontsize)
ax2.grid(b=True, which='both')
ax2.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=fontsize)

ax4 = plt.subplot(224)
ax4.plot(epochs, np.squeeze(clas_loss_train), 'b')
ax4.plot(epochs, np.squeeze(clas_loss_val), 'r')
ax4.set_xlabel('Epochs', fontsize=fontsize)
ax4.set_ylabel('Loss', fontsize=fontsize)
ax4.grid(b=True, which='both')
ax4.legend(['Training Loss', 'Validation Loss'], fontsize=fontsize)

plt.show(block=True)