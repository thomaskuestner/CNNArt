#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import os

from medio import convert_tf
from medio import parse_tf


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    num_parallel_calls = 4
    # slice_size the slice you cut
    slice_size = [64, 64, 64]

    # define some example values
    data_dir = '/home/d1274/no_backup/d1274/data'
    b_viewer = True
    b_verbose = True

    # choose & fetch all required data / discard subjects missing crucial data
    list_images = parse_tf.fetch_paths(data_dir, '_F_')
    #list_images = ['/home/d1274/no_backup/d1274/data/Q8/3D_GRE_TRA_bh_F_COMPOSED_0015.tfrecord' for _ in range(20)]

    def parse_label(path_name):
        if path_name.find('_deep_') != -1:
            return 2
        elif path_name.find('bh') != -1:
            return 0
        elif path_name.find('fb') != -1:
            return 1

    print(list_images)
    list_labels = [parse_label(x) for x in list_images]
    #print(list_labels)
    #list_labels =  tf.data.Dataset.from_tensor_slices(list_labels)


    # generate placeholders that receive paths of type str
    images = tf.placeholder(tf.string, shape=[None])
    labels= tf.placeholder(tf.int32, shape=[None])

    # create dataset for your needs
    dataset_image = tf.data.TFRecordDataset(images)
    dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((dataset_image, dataset_labels))

    # Note: this could be rewritten as one map call
    # map parse function to each zipped element

    dataset = dataset.map(
        map_func=lambda a, b: (convert_tf.parse_function(a), b),
        num_parallel_calls=num_parallel_calls)



    # ensure correct dims

    dataset = dataset.map(
        map_func=lambda a, b: (tf.slice(a, [0,0,0], slice_size),
                                  b,),
        num_parallel_calls=num_parallel_calls)


    dataset = dataset.map(
        map_func=lambda a, b: (tf.expand_dims(a, -1),
                               b,),
        num_parallel_calls=num_parallel_calls)



    # here one could use shuffle, repeat, prefetch, ...
    dataset = dataset.repeat()
    dataset_batched = dataset.batch(batch_size=16)
    dataset_batched = dataset_batched.prefetch(buffer_size=128)

    iterator = dataset_batched.make_initializable_iterator()
    next_element = iterator.get_next()

    # dummy "model"
    result = next_element

    # simple session example
    with tf.Session() as sess:

        sess.run(iterator.initializer,
                 feed_dict={images: list_images, labels: list_labels})

        while True:
            try:
                t0 = time.time()
                #print('....')
                batched_images, train_labels = sess.run(result)

                print('elapsed loading time: ', time.time() - t0)
                #print(batched_images[0, ...].max(), batched_labels_1[0, ...].max(), batched_labels_2[0, ...].max())

                print(batched_images.shape)
                print(train_labels)

            except tf.errors.OutOfRangeError:
                print('finished')
                break


