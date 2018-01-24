import sys
import shelve
import json

import pandas as pd
import numpy as np
from matplotlib import path as Path

a = np.asarray([1, 2 , 3, 4, 5, 6, 7, 8, 9])
dict = {}
h = a[3]
dict["Y123"] = int(h)
i = a[7]
dict["Y98"] = int(i)

with open(("D:/med_data/MRPhysics/DeepLearningArt_Output/P1_D1_PM0_X40_Y40_O0.5_L0_S1/test.json"), 'w') as fp:
    json.dump(dict, fp)
