
def fParseConfig(sFile):
    # get config file
    with open(sFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg


if __name__ == "__main__": # for command line call
    # input parsing
    parser = argparse.ArgumentParser(description='''CNN artifact detection''', epilog='''(c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de''')
    parser.add_argument('-c', '--config', nargs = 1, type = str, help='path to config file', default= 'config/param.yml')
    parser.add_argument('-i','--inPath', nargs = 1, type = str, help='input path to *.mat of stored patches', default= '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/in.mat')
    parser.add_argument('-o','--outPath', nargs = 1, type = str, help='output path to the file used for storage (subfiles _model, _weights, ... are automatically generated)', default= '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/out' )
    parser.add_argument('-m','--model', nargs = 1, type = str, choices =['motion_head_CNN2D', 'motion_abd_CNN2D', 'motion_all_CNN2D', 'motion_CNN3D', 'motion_MNetArt', 'motion_VNetArt', 'multi_DenseResNet', 'multi_InceptionNet'], help='select CNN model', default='motion_2DCNN_head' )
    parser.add_argument('-t','--train', dest='train', action='store_true', help='if set -> training | if not set -> prediction' )
    parser.add_argument('-p','--paraOptim', dest='paraOptim', type = str, choices = ['grid','hyperas','none'], help='parameter optimization via grid search, hyper optimization or no optimization', default = 'none')
    parser.add_argument('-b', '--batchSize', nargs='*', dest='batchSize', type=int, help='batchSize', default=64)
    parser.add_argument('-l', '--learningRates', nargs='*', dest='learningRate', type=int, help='learningRate', default=0.0001)
    parser.add_argument('-e', '--epochs', nargs=1, dest='epochs', type=int, help='epochs', default=300)

    args = parser.parse_args()

    # parse input
    cfg = fParseConfig(args.config[0])

    lTrain = cfg['lTrain']  # training or prediction
    lSave = cfg['lSave']  # save intermediate test, training sets
    lCorrection = cfg['lCorrection']  # artifact correction or classification
    sPredictModel = cfg['sPredictModel']  # choose trained model used in prediction
    # initiate info objects
    # default database: MRPhysics with ['newProtocol','dicom_sorted']
    dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])


    # load/create input data
    patchSize = cfg['patchSize']
    if cfg['sSplitting'] == 'normal':
        sFSname = 'normal'
    elif cfg['sSplitting'] == 'crossvalidation_data':
        sFSname = 'crossVal_data'
        nFolds = cfg['nFolds']
    elif cfg['sSplitting'] == 'crossvalidation_patient':
        sFSname = 'crossVal'

    # set ouput path
    sOutsubdir = cfg['subdirs'][2]
    sOutPath = cfg['selectedDatabase']['pathout'] + os.sep + ''.join(map(str, patchSize)).replace(" ",
                                                                                                  "") + os.sep + sOutsubdir + str(
        patchSize[0]) + str(patchSize[1])  # + str(ind_split) + '_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
    if len(patchSize) == 3:
        sOutPath = sOutPath + str(patchSize[2])
    if sTrainingMethod != "None":
        if sTrainingMethod != "ScaleJittering":
            sOutPath = sOutPath + '_sf' + ''.join(map(str, lScaleFactor)).replace(" ", "").replace(".", "")
            sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + 'sf' + ''.join(
                map(str, lScaleFactor)).replace(" ", "").replace(".", "") + '.h5'
        else:
            sOutPath = sOutPath + '_sj'
            sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + 'sj' + '.h5'
    else:
        sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str, patchSize)).replace(" ", "") + '.h5'

    if lCorrection:
        #########################
        ## Artifact Correction ##
        #########################
        correction.run(cfg, dbinfo)

    elif lTrain:
        ########################
        ## artifact detection ##
        ## ---- training ---- ##
        ########################
        fTrainArtDetection()


    else:
        ########################
        ## artifact detection ##
        ## --- prediction --- ##
        ########################
        fPredictArtDetection()