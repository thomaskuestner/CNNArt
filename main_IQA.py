import os
import tensorflow as tf
import glob
import sys
import yaml
from DatabaseInfo import DatabaseInfo, NAKOInfo
import pathlib

sys.path.append('/home/d1274/PycharmProjects/NAKO_transfer_learning')
from tensorflow.keras import backend as K
from tensorflow.python import keras

from networks.multiclass.CNN3D import multiclass_3D_SE_ResNet
from utils import generator
from utils import Patching
from utils.tfrecord.medio import convert_tf, read_image

config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

import tensorflow.keras.optimizers as optimizers

from tensorflow.keras.callbacks import (
    CSVLogger,
    ModelCheckpoint
)

if __name__ == '__main__':

    # use different param.yml if with sys.argv
    if len(sys.argv) > 1:
        param_yml = sys.argv[1] + '.yml'
    else:
        param_yml = 'param.yml'

    # get config file
    with open('config' + os.sep + param_yml, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'], cfg['sDatabaseRootPath'])

    # check if there are tfrecord files existing, if not, creating corresponding one
    original_dataset_path = os.path.join(cfg['sDatabaseRootPath'], cfg['MRdatabase'])
    print('original database path: ', original_dataset_path)
    for ipat, pat in enumerate(dbinfo.lPats):
        # the expected tfrecord saved format: med_data/NAKO/NAKO_IQA/Q1/....tfrecord

        if not os.path.exists(os.path.join(cfg['tfrecordsPath'], pat, dbinfo.sSubDirs[0])):
            if (pat == 'Results'):
                continue

            # tfrecords not yet created
            for iseq, seq in enumerate(dbinfo.lImgData):
                # create tfrecords

                # example result: /home/d1274/med_data/NAKO/NAKO_IQA/Q1/dicom_sorted/3D_GRE_TRA_bh_F_COMPOSED_0015.mat
                subject_path = os.path.join(original_dataset_path, pat + os.sep + dbinfo.sSubDirs[1]+ os.sep + seq.sPath + '.mat')

                image = read_image.read_mat_image(subject_path)

                # example result: /home/d1274/med_data/NAKO/NAKO_IQA_tf/Q1/3D_GRE_TRA_bh_F_COMPOSED_0015.tfrecord
                tf_save_path = os.path.join(cfg['tfrecordsPath'], pat, dbinfo.sSubDirs[0], seq.sPath + '.tfrecord')
                tf_save_pathlib = pathlib.Path(tf_save_path)
                tf_save_pathlib.parent.mkdir(parents=True, exist_ok=True)
                convert_tf.im2tfrecord(image=image, path=tf_save_path)


    # begin training
    print('begin creating the input dataset')


    epoch = cfg['epoch']
    path = cfg['tfrecordsPath']
    batch_size = cfg['batch_size']
    image_shape = cfg ['image_shape']
    patch_shape = cfg['patch_shape']
    start = cfg['start']
    num_images_loaded = cfg['num_images_loaded']
    overlap = cfg['overlap']
    num_classes = cfg['num_classes']
    mean_value = cfg['mean_value']
    std_value = cfg['std_value']
    test_groups = cfg['test_groups']
    lShuffleTraining = cfg['lShuffleTraining']

    db_tf = NAKOInfo(cfg['MRTfrecordDatabase'], cfg['subdirs'], cfg['sDatabaseRootPath'])

    train_files, train_labels, eva_files, eva_labels = db_tf.get_train_eval_files( pattern='_F_',
                                                                                   test_groups=test_groups,
                                                                                   train_eval_ratio = 0.85,
                                                                                   lShuffleTraining=lShuffleTraining)

    # there are two ways to extract the image, start from the beginning bprder or start from the ned border
    patches_per_image_1 = len(Patching.compute_patch_indices(image_shape=image_shape,
                                      patch_size=patch_shape,
                                      overlap=overlap,
                                      start=start,
                                      order = True))
    patches_per_image_2 = len(Patching.compute_patch_indices(image_shape=image_shape,
                                                            patch_size=patch_shape,
                                                            overlap=overlap,
                                                            start=start,
                                                            order = False))
    assert patches_per_image_1 == patches_per_image_2
    patches_per_image = patches_per_image_1


    steps_per_epoch = int(len(train_files) * patches_per_image / batch_size)
    validata_steps = int(len(eva_files) * patches_per_image / batch_size)
    print('expected step per epoch: ' , steps_per_epoch)


    # construct the model
    model, _ = multiclass_3D_SE_ResNet.createModel(patchSize=patch_shape, numClasses=3)
    print(model.summary())

    # optimize way ???
    learning_rate = 0.1
    decay_rate = learning_rate / epoch
    momentum = 0.8
    sgd = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(loss="categorical_crossentropy",
                  optimizer="sgd",
                  metrics=['acc'])


    def get_callbacks(model_file, logging_file=None, early_stopping_patience=None,
                      initial_learning_rate=0.01, lr_change_mode=None, verbosity=1):
        callbacks = list()

        # save the model
        callbacks.append(ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, mode='auto'))

        # records the basic metrics
        callbacks.append(CSVLogger(logging_file, append=True))
        return callbacks


    model_file = os.path.join('/home/d1274/no_backup/d1274/NAKO_transfer_learning_model/NAKO_IQA_model',
                              'simple_test_1.h5')
    logging_file = 'simple_test_1.log'

    train_generator = generator.tfdata_generator(file_lists=train_files, label_lists=train_labels,
                                                 is_training=True,
                                                 image_shape= image_shape,
                                                 patch_size=patch_shape,
                                                 start=start,
                                                 batch_size=batch_size,
                                                 overlap = overlap,
                                                 num_imgaes_loaded=num_images_loaded,
                                                 num_classes=num_classes,
                                                 mean_value= mean_value,
                                                 std_value=std_value)

    val_generator = generator.tfdata_generator(file_lists=eva_files, label_lists=eva_labels,
                                               is_training=False,
                                               image_shape= image_shape,
                                               patch_size=patch_shape,
                                               start= start,
                                               overlap = overlap,
                                               batch_size=batch_size,
                                               num_classes=num_classes,
                                               mean_value=mean_value,
                                               std_value=std_value)

    res = model.fit(
        train_generator.make_one_shot_iterator(),
        steps_per_epoch=steps_per_epoch,
        epochs=epoch,
        validation_data=val_generator.make_one_shot_iterator(),
        validation_steps=validata_steps,
        callbacks=get_callbacks(model_file=model_file, logging_file=logging_file))

    print('the result: ', res)
    

