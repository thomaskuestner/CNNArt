import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import seaborn as sn
import pandas as pd


path = 'D:/med_data/MRPhysics/MA Results/Output_Learning-9.3.18/Multiclass SE-ResNet-56_2D_64x64_2018-03-07_11-48/model_predictions.mat'

mat = sio.loadmat(path)

confusion_matrix = mat['confusion_matrix']
classification_report = mat['classification_report']
stri = classification_report[0]
print(stri)

sum_all = np.array(np.sum(confusion_matrix, axis=0))

all = np.zeros((len(sum_all), len(sum_all)))
for i in range(all.shape[0]):
     all[i,:]=sum_all

confusion_matrix = np.divide(confusion_matrix, all)

df_cm = pd.DataFrame(confusion_matrix,
                     index = [i for i in "ABCDEFGHIJK"],
                     columns = [i for i in "ABCDEFGHIJK"], )

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

sn.heatmap(df_cm, annot=True, fmt='.4f')

plt.show(block=True)

print()


