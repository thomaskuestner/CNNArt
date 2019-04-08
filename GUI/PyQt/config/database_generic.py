# # generate a database information file
import csv
import os

# Here type in the name of database
database_name = 'to_be_defined'

column = ['pathdata', 'pathlabel', 'artefact', 'bodyregion']

# Here type in the sequences name of each patient
pathdada = ['to be defined',
            'to be defined',
            '...']

# Here type in the label name for each corresponding sequence
pathlabel = ['None',
             'None',
             '...']

# Here type in the artefact for each corresponding sequence
artefact = ['None',
            'None',
            '...']

# Here type in the bodyregion for each corresponding sequence
bodyregion = ['None',
              'None',
              '...']

data = {'pathdata': pathdada,
        'pathlabel': pathlabel,
        'artefact': artefact,
        'bodyregion': bodyregion}

directory = 'database' + os.sep + database_name
if not os.path.isdir(directory):
    os.makedirs(directory)

fullpath = directory + os.sep  + database_name + '.csv'
with open(fullpath, 'w') as f:
    for key in data.keys():
        f.write("%s,%s\n" % (key, data[key]))
