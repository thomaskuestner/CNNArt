import numpy as np

from DeepLearningArt.DLArt_GUI.Dataset import *

datasets = {
        't1_tse_tra_Kopf_0002': Dataset('t1_tse_tra_Kopf_0002', None,'ref', 'head', 't1'),
        't1_tse_tra_Kopf_Motion_0003': Dataset('t1_tse_tra_Kopf_Motion_0003', None, 'motion','head', 't1'),
        't1_tse_tra_fs_mbh_Leber_0004': Dataset('t1_tse_tra_fs_mbh_Leber_0004',None,'ref','abdomen', 't1'),
        't1_tse_tra_fs_mbh_Leber_Motion_0005': Dataset('t1_tse_tra_fs_mbh_Leber_Motion_0005', None, 'motion', 'abdomen', 't1'),
        't2_tse_tra_fs_navi_Leber_0006': Dataset('t2_tse_tra_fs_navi_Leber_0006',None,'ref','abdomen', 't2'),
        't2_tse_tra_fs_navi_Leber_Shim_xz_0007': Dataset('t2_tse_tra_fs_navi_Leber_Shim_xz_0007', None, 'shim', 'abdomen', 't2'),
        't1_tse_tra_fs_Becken_0008': Dataset('t1_tse_tra_fs_Becken_0008', None, 'ref', 'pelvis', 't1'),
        't2_tse_tra_fs_Becken_0009': Dataset('t2_tse_tra_fs_Becken_0009', None, 'ref', 'pelvis', 't2'),
        't1_tse_tra_fs_Becken_Motion_0010': Dataset('t1_tse_tra_fs_Becken_Motion_0010', None, 'motion', 'pelvis', 't1'),
        't2_tse_tra_fs_Becken_Motion_0011': Dataset('t2_tse_tra_fs_Becken_Motion_0011', None, 'motion', 'pelvis', 't2'),
        't2_tse_tra_fs_Becken_Shim_xz_0012': Dataset('t2_tse_tra_fs_Becken_Shim_xz_0012', None, 'shim', 'pelvis', 't2')
    }

label = 111

for i in datasets:
    set = datasets[i]
    if set.getDatasetLabel() == label:
        print(str(set.getPathdata()))


from sklearn.metrics import confusion_matrix

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
cm = confusion_matrix(y_true, y_pred)

print()