## define path information, change path in the file param_GUI.yml
## define path as constant.

import yaml
with open('config/param_GUI.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

DATASETS = cfg['MRdatabase']

SUBDIRS = cfg['subdirs']

PATH_OUT = cfg['pathout']

GPU_HOME_PATH = cfg['gpu_home']

LABEL_PATH = cfg['label_path']

LEARNING_OUT = cfg['output_learning']

GRAPHVIZ_PATH = cfg['graphviz_path']