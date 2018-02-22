############################################
# !!! Run this script with Python 2.7 !!!! #
############################################
"""
@author: Yannick Wilhelm
@email: yannick.wilhelm@gmx.de
@date: January 2018
"""


import sys
import os
import shelve
import json
import numpy
from matplotlib import path

# the dict structure for the json decoder:
#jsondict = {'model':
#                       {'15_31_0':
#                                       {'vertices': [[1, 0],[1, 2], [5, 6]], 'codes': None},
#                        '23_31_0':
#                                       {'vertices': [[10, 5],[1, 6]], 'codes': None}
#               },
#           'model2':
#                      {'23_31_0':
#                                       {'vertices': [[1, 8],[5, 5]], 'codes': None},
#                      {'12_32_0':
#                                       {'points': [23 1  2 5]}}
#               }
#           }
#

# directories etc.....
dir = "Markings" + os.sep
patient = "15_yb"

sPath = dir + patient + ".slv"
jPath = dir + patient + ".json"

# dict
jsonDict = {}

# load shelve file
inShelve = shelve.open(sPath)

for key in inShelve.keys():
    dataOfFirstKey = inShelve[key]
    innerDict = {}
    for key2 in dataOfFirstKey:
        path = dataOfFirstKey[key2]     # path is an instance of matplotlib path.Path
        innerInnerDict = {}
        try:
            curVertices = numpy.ndarray.tolist(path.vertices)     # attribute of path.Path
            curCodes = path.codes           # attribute of path.Path
            innerInnerDict['vertices'] = curVertices
            innerInnerDict['codes'] = curCodes
        except:
            curPoints = numpy.ndarray.tolist(path)
            innerInnerDict['points'] = curPoints


        innerDict[key2] = innerInnerDict

    jsonDict[key] = innerDict

with open(jPath, 'w') as fp:
    # save dict in one line in json file
    json.dump(jsonDict, fp, indent=4)

    # for human friendly reading
    #json.dump(jsonDict, fp, indent=4)