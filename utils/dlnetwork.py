'''
Copyright: 2016-2019 Thomas Kuestner (thomas.kuestner@med.uni-tuebingen.de) under Apache2 license
@author: Thomas Kuestner
'''

class Dlnetwork:

    def __init__(self, cfg):

        # network and parameters
        self.neuralNetworkModel = cfg['network']
        self.usingClassification = cfg['usingClassification']  # use classification output on deepest layer
        self.savedmodel = cfg['sSaveModel']
        self.batchSize = cfg['batchSize']
        self.learningRate = cfg['learningRate']
        self.epochs = cfg['epochs']
        self.optimizer = cfg['optimizer']['algorithm']
        # SGD
        self.momentum = cfg['optimizer']['momentum']
        # SGD, RMSProp, ADAGRAD, ADADELTA, ADAM
        self.weightdecay = cfg['optimizer']['weightdecay']
        # SGD
        self.nesterov = cfg['optimizer']['nesterov']
