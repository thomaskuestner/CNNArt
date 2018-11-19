#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file tries to parse the image into the tfrecord and save it in the disk
# Then reload the tfrecord into the tensorflow session and see if works

import tensorflow as tf
import nibabel
from medio import convert_tf, read_image

if __name__ == '__main__':

    # define some example values
    path_tf = '/home/d1274/no_backup/d1274/tmp_data/scan.tfrecord'
    b_custom = False

    # path_image = '/home/d1274/med_data/NAKO/NAKO_IQA/Q1/dicom_sorted/3D_GRE_TRA_bh_W_COMPOSED_0014.mat'
    path_image = '/home/d1274/med_data/NAKO/NAKO_IQA/Q1/dicom_sorted/3D_GRE_TRA_fb_deep_F_COMPOSED_0043.mat'

    image= read_image.read_mat_image(path=path_image)
    convert_tf.im2tfrecord(image=image, path=path_tf)

    # begin construct input pipeline
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)

    # custom functions are also possible via dataset.map & tf.py_func
    dataset = dataset.map(map_func=convert_tf.parse_function)

    dataset_batched = dataset.batch(batch_size=1)
    training_dataset = dataset_batched

    iterator = training_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # dummy "model"
    result = next_element

    with tf.Session() as sess:

        training_filenames = [path_tf]
        sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

        while True:
            try:
                value = sess.run(result)
                print(value.shape)
                print(value[0].shape)

                # nibabel.viewers.OrthoSlicer3D(value[0, ...]).show()
                nibabel.viewers.OrthoSlicer3D(value[0, ...]).show()

            except tf.errors.OutOfRangeError:
                print('finished')
                break
