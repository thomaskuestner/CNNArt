import json, os

# colorsets = {'class2':{'colors':['blue', 'red'], 'trans':[0.3]},
#             'class8':{'hatches':[None, '//', '\\', '**']},
#             'class11':{'colors':['blue', 'purple', 'cyan', 'yellow', 'green'], 'hatches':[None, '//', '\\', 'XX'], 'trans':[0.3]}}

workspace = {'mode':[],'layout':[None, None], 'listA':[], 'Probs':[], 'Hatches':[], 'NrClass':[], 'Pathes':[], 'NResults':[], 'Corres':[]}

wFile = json.dumps(workspace)
with open('lastWorkspace.json', 'w') as json_data:
    json_data.write(wFile)
