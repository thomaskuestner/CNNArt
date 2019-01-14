import os
import tensorflow as tf
import glob
import sys

sys.path.append('/home/d1274/PycharmProjects/NAKO_transfer_learning')
from tensorflow.keras import backend as K
from tensorflow.python import keras

from networks.multiclass.CNN3D import multiclass_3D_SE_ResNet
from utils import get_train_eval_files_NAKO_IQA
from utils import generator
from utils import Patching

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
    print('begin creating the input dataset')


    epoch = 200
    path = '/home/d1274/med_data/NAKO/NAKO_IQA_tf'


    batch_size = 48
    image_shape = [236, 320, 260]
    patch_shape = [64, 64, 64]
    start = [20, 20, 20]
    num_images_loaded = 30
    overlap = 32

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

    train_files, train_labels, eva_files, eva_labels = get_train_eval_files_NAKO_IQA.get_train_eval_files(path, pattern='_F_')

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
                                                 num_imgaes_loaded=num_images_loaded)

    val_generator = generator.tfdata_generator(file_lists=eva_files, label_lists=eva_labels,
                                               is_training=False,
                                               image_shape= image_shape,
                                               patch_size=patch_shape,
                                               start= start,
                                               overlap = overlap,
                                               batch_size=batch_size)

    res = model.fit(
        train_generator.make_one_shot_iterator(),
        steps_per_epoch=steps_per_epoch,
        epochs=epoch,
        validation_data=val_generator.make_one_shot_iterator(),
        validation_steps=validata_steps,
        callbacks=get_callbacks(model_file=model_file, logging_file=logging_file))

    print('the result: ', res)