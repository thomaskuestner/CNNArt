import h5py
import numpy as np

def  read_mat_image (path):
    arrays = {}
    f = h5py.File(path)
    for k, v in f.items():
        arrays[k] = np.array(v)
        return arrays['dImg']


