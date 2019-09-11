'''
Copyright: 2016-2019 Thomas Kuestner (thomas.kuestner@med.uni-tuebingen.de) under Apache2 license
@author: Thomas Kuestner
'''

class Dlnetwork:

    def __init__(self, cfg):

        # network and parameters
        self.neuralNetworkModel = cfg['network']
        self.usingClassification = cfg['usingClassification']  # use classification output on deepest layer
        self.savemodel = cfg['sSaveModel']
        self.batchSize = cfg['batchSize']
        self.learningRate = cfg['learningRate']
        self.epochs = cfg['epochs']
        self.optimizer = cfg['optimizer']['algorithm']
        # train with array or as a generator
        self.trainMode = cfg['trainMode']
        if cfg['storeMode'] == 'STORE_TFRECORD':  # you need to use generator
            self.trainMode = 'GENERATOR'
        if cfg['storeMode'] == 'STORE_HDF5':  # you need to use array processing
            self.trainMode = 'ARRAY'
        # SGD
        self.momentum = cfg['optimizer']['momentum']
        # SGD, RMSProp, ADAGRAD, ADADELTA, ADAM
        self.weightdecay = cfg['optimizer']['weightdecay']
        # SGD
        self.nesterov = cfg['optimizer']['nesterov']

        # data augmentation
        self.dataAugm_featurewise_center = cfg['dataAugmentation']['featurewise_center']
        self.dataAugm_samplewise_center = cfg['dataAugmentation']['samplewise_center']
        self.dataAugm_featurewise_std_normalization = cfg['dataAugmentation']['featurewise_std_normalization']
        self.dataAugm_samplewise_std_normalization = cfg['dataAugmentation']['samplewise_std_normalization']
        self.dataAugm_zca_whitening = cfg['dataAugmentation']['zca_whitening']
        self.dataAugm_zca_epsilon = cfg['dataAugmentation']['zca_epsilon']
        self.dataAugm_rotation_range = cfg['dataAugmentation']['rotation_range']
        self.dataAugm_width_shift_range = cfg['dataAugmentation']['width_shift_range']
        self.dataAugm_height_shift_range = cfg['dataAugmentation']['height_shift_range']
        self.dataAugm_shear_range = cfg['dataAugmentation']['shear_range']
        self.dataAugm_zoom_range = cfg['dataAugmentation']['zoom_range']
        self.dataAugm_channel_shift_range = cfg['dataAugmentation']['channel_shift_range']
        self.dataAugm_fill_mode = cfg['dataAugmentation']['fill_mode']
        self.dataAugm_cval = cfg['dataAugmentation']['cval']
        self.dataAugm_horizontal_flip = cfg['dataAugmentation']['horizontal_flip']
        self.dataAugm_vertical_flip = cfg['dataAugmentation']['vertical_flip']
        self.dataAugm_rescale = cfg['dataAugmentation']['rescale']
        self.dataAugm_histogram_equalization = cfg['dataAugmentation']['histogram_equalization']
        self.dataAugm_contrast_stretching = cfg['dataAugmentation']['contrast_stretching']
        self.dataAugm_adaptive_equalization = cfg['dataAugmentation']['adaptive_equalization']

        if (self.dataAugm_featurewise_center) | (self.dataAugm_samplewise_center) | (self.dataAugm_featurewise_std_normalization) | \
            (self.dataAugm_samplewise_std_normalization) | (self.dataAugm_zca_whitening) | (self.dataAugm_horizontal_flip) | \
            (self.dataAugm_vertical_flip) | (self.dataAugm_rescale) | (self.dataAugm_histogram_equalization) | \
            (self.dataAugm_contrast_stretching) | (self.dataAugm_adaptive_equalization):
            self.dataAugmentation = True
        else:
            self.dataAugmentation = False