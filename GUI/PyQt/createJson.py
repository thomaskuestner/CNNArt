import json, os

# colorsets = {'class2':{'colors':['blue', 'red'], 'trans':[0.3]},
#             'class8':{'hatches':[None, '//', '\\', '**']},
#             'class11':{'colors':['blue', 'purple', 'cyan', 'yellow', 'green'], 'hatches':[None, '//', '\\', 'XX'], 'trans':[0.3]}}

# workspace = {'mode':[],'layout':[None, None], 'listA':[], 'Probs':[], 'Hatches':[], 'NrClass':[], 'Pathes':[], 'NResults':[], 'Corres':[]}

labels = {'names':['label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8'],
          'colors':['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink'],
          'path':['C:/Users/hansw/Desktop/Ma_code/PyQt/Markings']}
wFile = json.dumps(labels)
with open('editlabel.json', 'w') as json_data:
    json_data.write(wFile)

#  to print json
# import json
# with open('C:/Users/hansw/Desktop/Ma_code/PyQt/Markings/01_ab.json', 'r') as json_data:
#     infos = json.load(json_data)
#     for key in infos.keys():
#         print(key)
