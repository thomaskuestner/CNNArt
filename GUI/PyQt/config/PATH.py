## define path information, change path in the file param_GUI.yml
## define path as constant.

import yaml
with open('config/param_GUI.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

DATASETS = cfg['MRdatabase']

SUBDIRS = cfg['subdirs']

PATH_OUT = cfg['pathout']

PREDICT_IMG_PATH = cfg['lPredictImg']

DLART_OUT_PATH = cfg['DLart_output_path']

GPU_HOME_PATH = cfg['gpu_home']

PREDICT_RESULT_PATH = cfg['predict_result']

TEST_PATH = cfg['test_path']

CNN_PATH = cfg['cnn_path']

MARKING_PATH = cfg['marking_path']

LABEL_PATH = cfg['label_path']

LEARNING_OUT = cfg['output_learning']

CHECK_PATH = cfg['check_point']