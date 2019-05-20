#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import os

from utils.tfrecord.medio import convert_tf
from utils.tfrecord.medio import parse_tf

def get_patches(image, num_patches=30, patch_size=64):
    """Get `num_patches` random crops from the image"""
    patches = []
    for i in range(num_patches):
        patch = tf.random_crop(image, [patch_size, patch_size, patch_size, 1 ])
        patches.append(patch)

    patches = tf.stack(patches)
    assert patches.get_shape().dims == [num_patches, patch_size, patch_size, patch_size, 1]
    return patches

def parse_label(path_name):
    if path_name.find('_deep_') != -1:
        return 2
    elif path_name.find('bh') != -1:
        return 0
    elif path_name.find('fb') != -1:
        return 1


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    num_parallel_calls = 4

    # define some example values
    data_dir = '/home/s1304/no_backup/s1304/data'
    b_viewer = True
    b_verbose = True

    # choose & fetch all required data / discard subjects missing crucial data
    list_images = parse_tf.fetch_paths(data_dir, '_F_')
    print(list_images)
    list_labels = [parse_label(x) for x in list_images]

    num_samples = len(list_images)
    num_patches = 30  # number of patches to extract from each image
    patch_size = 64   # size of the patches per axis
    buffer_size = 5*num_patches  # shuffle patches from 5 different big images
    batch_size = 64


    # This function splits a whole image into many splits and return
    get_patches_fn = lambda image: get_patches(image,
                                               num_patches=num_patches,
                                               patch_size=patch_size)

    # begin input pipeline constructs
    # generate placeholders that receive paths of type str
    images = tf.placeholder(tf.string, shape=[None])
    labels= tf.placeholder(tf.int32, shape=[None])

    # create dataset for your needs
    dataset_image = tf.data.TFRecordDataset(images)
    dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((dataset_image, dataset_labels))

    dataset = dataset.shuffle(buffer_size=num_samples)

    # Note: this could be rewritten as one map call
    # map parse function to each zipped element
    dataset = dataset.map(
        map_func=lambda a, b: (convert_tf.parse_function(a), b),
        num_parallel_calls=num_parallel_calls)

    # This is for single channel, extand the last axis as the channel axis
    dataset = dataset.map(
        map_func=lambda a, b: (tf.expand_dims(a, -1),
                               b,),
        num_parallel_calls=num_parallel_calls)

    dataset = dataset.map(map_func= lambda a,b : (get_patches_fn(a),
                                                  [b for _ in range(num_patches)]),
                          num_parallel_calls= num_parallel_calls)
    dataset = dataset.apply(tf.contrib.data.unbatch())

    # here one could use shuffle, repeat, prefetch, ...
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset_batched = dataset.batch(batch_size=batch_size)
    dataset_batched = dataset_batched.prefetch(buffer_size=128)
    dataset_batched = dataset_batched.repeat()

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


