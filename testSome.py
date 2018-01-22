import sys
import shelve
import json
import numpy as np
from matplotlib import path as Path

with open('D:/datasets/train2014/annotations_trainval2014/annotations/instances_train2014.json', 'r') as fp:
    file = json.load(fp)


with open('D:/datasets/train2014/annotations_trainval2014/annotations/instances_train2014_2.json', 'w')as fp:
    json.dump(file, fp, indent=4)









